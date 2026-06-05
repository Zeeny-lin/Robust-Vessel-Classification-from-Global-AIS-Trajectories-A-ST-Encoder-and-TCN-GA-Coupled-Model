"""
Function: Trajectory image + MVFFNet training + progress bar + post-voting merge (ablation - MVFFNet)
Model   : MVFFNet (Multi-View Feature Fusion Network)
Input   : Trajectory images (PNG)
          Save validation TOP-5 checkpoints and weighted voting
          Early stopping supported
"""

import os
import glob
import heapq
import random
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as tv_models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# -------------- Device --------------
REQUIRE_CUDA = True
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install GPU-enabled PyTorch/CUDA runtime.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if mp.current_process().name == "MainProcess":
        print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")
else:
    if mp.current_process().name == "MainProcess":
        print(f"Using device: {device}")

# -------------- Paths --------------
DATA_ROOT = r"D:\日常项目\25.08论文\data\data\data_classify"
IMAGE_ROOT = r"D:\日常项目\25.08论文\data\data\image"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

TRAIN_IMAGE_DIR = os.path.join(IMAGE_ROOT, "train")
VAL_IMAGE_DIR = os.path.join(IMAGE_ROOT, "val")
TEST_IMAGE_DIR = os.path.join(IMAGE_ROOT, "test")

RESULT_DIR = r"D:\日常项目\25.08论文\resultprocess\MVFFNet"
os.makedirs(RESULT_DIR, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULT_DIR, f"run_{current_time}")
os.makedirs(RUN_DIR, exist_ok=True)

LOG_FILE = os.path.join(RUN_DIR, "training_log.txt")
MODEL_DIR = os.path.join(RUN_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Torch cache directory (avoid permission issues in user-home cache path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TORCH_CACHE_DIR = os.path.join(SCRIPT_DIR, ".torch_cache")
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)
torch.hub.set_dir(TORCH_CACHE_DIR)

CLASS_NAMES = ["Bulk Carrier", "Container Ship", "Fishing", "Oil Tanker"]
CLASS_MAP = {
    "Bulk Carrier": 0,
    "Container Ship": 1,
    "Fishing": 2,
    "Oil Tanker": 3,
}

# -------------- Hyperparameters --------------
SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True

MODEL_NAME = "MVFFNet"
DROPOUT_RATE = 0.2
PRETRAINED_BACKBONE = True
BACKBONE_LR_RATIO = 0.3

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-6

TOTAL_EPOCHS = 80
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 0.001
TOPK_MODELS = 5
LABEL_SMOOTHING = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True
USE_CLASS_WEIGHTS = False
USE_ONECYCLE_LR = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------- MVFFNet --------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, groups=1):
        super().__init__()
        if p is None:
            p = ((k - 1) // 2) * d
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MVFFFusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch_k3 = ConvBNAct(in_ch, out_ch, k=3, s=1)
        self.branch_k5 = ConvBNAct(in_ch, out_ch, k=5, s=1)
        self.branch_dil = ConvBNAct(in_ch, out_ch, k=3, s=1, d=2)
        self.fuse = ConvBNAct(out_ch * 3, out_ch, k=1, s=1, p=0)
        self.shortcut = nn.Identity() if in_ch == out_ch else ConvBNAct(in_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x):
        b1 = self.branch_k3(x)
        b2 = self.branch_k5(x)
        b3 = self.branch_dil(x)
        y = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return F.relu(y + self.shortcut(x), inplace=True)


class MVFFNetClassifier(nn.Module):
    def __init__(self, num_classes, model_name="mvffnet", pretrained=False, dropout_rate=0.2):
        super().__init__()
        _ = model_name

        # ResNet18 backbone (pretrained if available, fallback to random init).
        if pretrained:
            try:
                backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:
                print(f"Warning: failed to load pretrained ResNet18 weights, fallback to random init: {e}")
                backbone = tv_models.resnet18(weights=None)
        else:
            backbone = tv_models.resnet18(weights=None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64ch
        self.layer2 = backbone.layer2  # 128ch
        self.layer3 = backbone.layer3  # 256ch
        self.layer4 = backbone.layer4  # 512ch

        self.proj1 = ConvBNAct(64, 128, k=1, s=1, p=0)
        self.proj2 = ConvBNAct(128, 128, k=1, s=1, p=0)
        self.proj3 = ConvBNAct(256, 128, k=1, s=1, p=0)
        self.proj4 = ConvBNAct(512, 128, k=1, s=1, p=0)

        self.fuse = nn.Sequential(
            ConvBNAct(512, 256, k=3, s=1),
            MVFFFusionBlock(256, 256),
            MVFFFusionBlock(256, 256),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def get_param_groups(self, base_lr: float, backbone_lr_ratio: float = 0.3):
        backbone_params = (
            list(self.stem.parameters())
            + list(self.layer1.parameters())
            + list(self.layer2.parameters())
            + list(self.layer3.parameters())
            + list(self.layer4.parameters())
        )
        head_params = (
            list(self.proj1.parameters())
            + list(self.proj2.parameters())
            + list(self.proj3.parameters())
            + list(self.proj4.parameters())
            + list(self.fuse.parameters())
            + list(self.classifier.parameters())
        )
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_ratio},
            {"params": head_params, "lr": base_lr},
        ]

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        target_size = c2.shape[-2:]
        p1 = F.interpolate(self.proj1(c1), size=target_size, mode="bilinear", align_corners=False)
        p2 = self.proj2(c2)
        p3 = F.interpolate(self.proj3(c3), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(c4), size=target_size, mode="bilinear", align_corners=False)

        x = self.fuse(torch.cat([p1, p2, p3, p4], dim=1))
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        return self.classifier(feat)


# -------------- Dataset --------------
class ShipTrajectoryImageDataset(Dataset):
    def __init__(self, image_dir, data_dir=None, is_train=False):
        self.images = []
        self.labels = []
        self.filenames = []
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
                    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        for ship_type in CLASS_NAMES:
            ship_image_dir = os.path.join(image_dir, ship_type)
            if not os.path.exists(ship_image_dir):
                print(f"Warning: Image directory {ship_image_dir} does not exist")
                continue

            png_files = glob.glob(os.path.join(ship_image_dir, "*.png"))
            for png_file in png_files:
                try:
                    filename = os.path.splitext(os.path.basename(png_file))[0]
                    self.images.append(png_file)
                    self.labels.append(CLASS_MAP[ship_type])
                    self.filenames.append(filename)
                except Exception as e:
                    print(f"Error loading {png_file}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long), self.filenames[idx]


# -------------- Early Stopping --------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


# -------------- Train / Eval --------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, epoch, total_epochs):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

    for batch_idx, (batch_X, batch_Y, filenames) in enumerate(pbar):
        batch_X = batch_X.to(device, non_blocking=(device.type == "cuda"))
        batch_Y = batch_Y.to(device, non_blocking=(device.type == "cuda"))
        if REQUIRE_CUDA and batch_idx == 0 and batch_X.device.type != "cuda":
            raise RuntimeError("Training batch is not on CUDA. GPU training is not active.")
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda" and USE_AMP)):
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and USE_ONECYCLE_LR:
            scheduler.step()

        batch_acc = (outputs.argmax(1) == batch_Y).float().mean().item()
        running_loss += loss.item() * batch_Y.size(0)
        running_acc += batch_acc * batch_Y.size(0)
        n += batch_Y.size(0)

        avg_loss = running_loss / n
        avg_acc = running_acc / n
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{avg_acc:.4f}", "Batch": f"{batch_idx+1}/{len(loader)}"})

    return running_loss / n, running_acc / n


def evaluate_model(model, loader, criterion, epoch, total_epochs, mode="Val"):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [{mode}]", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

    with torch.no_grad():
        for batch_idx, (batch_X, batch_Y, filenames) in enumerate(pbar):
            batch_X = batch_X.to(device, non_blocking=(device.type == "cuda"))
            batch_Y = batch_Y.to(device, non_blocking=(device.type == "cuda"))
            with autocast(device_type="cuda", enabled=(device.type == "cuda" and USE_AMP)):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

            preds = outputs.argmax(1)
            batch_acc = (preds == batch_Y).float().mean().item()

            running_loss += loss.item() * batch_Y.size(0)
            running_acc += batch_acc * batch_Y.size(0)
            n += batch_Y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())

            avg_loss = running_loss / n
            avg_acc = running_acc / n
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{avg_acc:.4f}", "Batch": f"{batch_idx+1}/{len(loader)}"})

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return running_loss / n, accuracy, precision, recall, f1, all_preds, all_labels


# -------------- TOP-K Saver --------------
class TopkSaver:
    def __init__(self, k=5, save_dir=MODEL_DIR):
        self.k = k
        self.heap = []  # (f1_score, epoch, path)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def push(self, f1_val, epoch, model):
        path = os.path.join(self.save_dir, f"epoch_{epoch}_f1_{f1_val:.4f}.pth")
        torch.save(model.state_dict(), path)
        heapq.heappush(self.heap, (f1_val, epoch, path))
        if len(self.heap) > self.k:
            _, _, old_path = heapq.heappop(self.heap)
            if os.path.exists(old_path):
                os.remove(old_path)

    def best_checkpoints(self):
        return sorted(self.heap, key=lambda x: -x[0])


# -------------- Test & Integration --------------
def test_model(model, test_loader, model_name=""):
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []

    pbar = tqdm(test_loader, desc=f"Testing {model_name}", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

    with torch.no_grad():
        for batch_idx, (batch_X, batch_Y, filenames) in enumerate(pbar):
            batch_X = batch_X.to(device, non_blocking=(device.type == "cuda"))
            batch_Y = batch_Y.to(device, non_blocking=(device.type == "cuda"))
            with autocast(device_type="cuda", enabled=(device.type == "cuda" and USE_AMP)):
                outputs = model(batch_X)
            preds = outputs.argmax(1)
            batch_acc = (preds == batch_Y).float().mean().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())
            all_filenames.extend(filenames)

            pbar.set_postfix({"Batch": f"{batch_idx+1}/{len(test_loader)}", "Batch_Acc": f"{batch_acc:.4f}"})

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_val = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    results_df = pd.DataFrame(
        {
            "traj_no": all_filenames,
            "test_shiptype": [CLASS_NAMES[p] for p in all_preds],
            "real_shiptype": [CLASS_NAMES[l] for l in all_labels],
        }
    )

    return accuracy, precision, recall, f1_val, results_df


def integrate_results(test_results_list, weights):
    vote_counts = {}
    real_type_map = {}

    for (f1_val, epoch, _), results_df in zip(weights, test_results_list):
        weight = f1_val
        for _, row in results_df.iterrows():
            traj_no = row["traj_no"]
            pred_type = row["test_shiptype"]
            real_type = row["real_shiptype"]

            if traj_no not in vote_counts:
                vote_counts[traj_no] = {}
            if pred_type not in vote_counts[traj_no]:
                vote_counts[traj_no][pred_type] = 0
            vote_counts[traj_no][pred_type] += weight
            real_type_map[traj_no] = real_type

    integrated_results = []
    for traj_no, votes in vote_counts.items():
        final_pred = max(votes.items(), key=lambda x: x[1])[0]
        integrated_results.append(
            {"traj_no": traj_no, "test_shiptype": final_pred, "real_shiptype": real_type_map.get(traj_no, "Unknown")}
        )

    return pd.DataFrame(integrated_results)


def integrate_by_shipno(integrated_df, test_dir):
    integrated_df = integrated_df.copy()
    traj_stem = integrated_df["traj_no"].astype(str).str.replace(r"\.csv$", "", regex=True)
    integrated_df["base_shipno"] = traj_stem.str.replace(r"_\d+$", "", regex=True)

    trajectory_lengths = []
    for traj_no in integrated_df["traj_no"]:
        traj_stem = os.path.splitext(str(traj_no))[0]
        candidates = [f"{traj_stem}.csv", str(traj_no)]
        for ship_type in CLASS_NAMES:
            found = False
            for cand in candidates:
                csv_path = os.path.join(test_dir, ship_type, cand)
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        trajectory_lengths.append(len(df))
                    except Exception:
                        trajectory_lengths.append(1)
                    found = True
                    break
            if found:
                break
            if ship_type == CLASS_NAMES[-1]:
                trajectory_lengths.append(1)

    integrated_df["length"] = trajectory_lengths

    def resolve_group(df):
        votes = {}
        for _, row in df.iterrows():
            pred_type = row["test_shiptype"]
            weight = row["length"]
            votes[pred_type] = votes.get(pred_type, 0) + weight
        return max(votes.items(), key=lambda x: x[1])[0]

    final_df = integrated_df.groupby("base_shipno").apply(resolve_group).reset_index(name="test_shiptype")
    real_df = (
        integrated_df.groupby("base_shipno")["real_shiptype"]
        .agg(lambda s: s.value_counts().index[0] if len(s.value_counts()) > 0 else "Unknown")
        .reset_index()
    )
    final_df = final_df.merge(real_df, on="base_shipno", how="left")
    final_df["real_shiptype"] = final_df["real_shiptype"].fillna("Unknown")
    final_df = final_df.rename(columns={"base_shipno": "shipno"})
    return final_df


# -------------- Main --------------
def main():
    set_seed(SEED)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("MVFFNet Training Log (Image-based Classification)\n")
        f.write(f"Start Time: {datetime.now()}\n")
        f.write("=" * 50 + "\n")

    def log_message(message):
        print(message)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    hyperparams = {
        "MODEL_NAME": MODEL_NAME,
        "IMAGE_SIZE": IMAGE_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY and device.type == "cuda",
        "SEED": SEED,
        "PRETRAINED_BACKBONE": PRETRAINED_BACKBONE,
        "BACKBONE_LR_RATIO": BACKBONE_LR_RATIO,
        "DROPOUT_RATE": DROPOUT_RATE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "MIN_LEARNING_RATE": MIN_LEARNING_RATE,
        "TOTAL_EPOCHS": TOTAL_EPOCHS,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "EARLY_STOP_MIN_DELTA": EARLY_STOP_MIN_DELTA,
        "TOPK_MODELS": TOPK_MODELS,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
        "USE_AMP": USE_AMP,
        "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
        "USE_ONECYCLE_LR": USE_ONECYCLE_LR,
        "DEVICE": str(device),
    }
    if device.type == "cuda":
        hyperparams["GPU_NAME"] = torch.cuda.get_device_name(0)
        hyperparams["TORCH_CUDA_VERSION"] = torch.version.cuda

    log_message("Hyperparameters:")
    for k, v in hyperparams.items():
        log_message(f"  {k}: {v}")

    log_message("Loading image datasets...")
    train_dataset = ShipTrajectoryImageDataset(TRAIN_IMAGE_DIR, is_train=True)
    val_dataset = ShipTrajectoryImageDataset(VAL_IMAGE_DIR, is_train=False)
    test_dataset = ShipTrajectoryImageDataset(TEST_IMAGE_DIR, is_train=False)

    log_message(f"Train samples: {len(train_dataset)}")
    log_message(f"Val samples: {len(val_dataset)}")
    log_message(f"Test samples: {len(test_dataset)}")

    common_loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    model = MVFFNetClassifier(
        num_classes=len(CLASS_NAMES),
        model_name="mvffnet",
        pretrained=PRETRAINED_BACKBONE,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    if REQUIRE_CUDA and not next(model.parameters()).is_cuda:
        raise RuntimeError("Model is not on CUDA device. Please check your environment/device settings.")

    class_counts = np.bincount(np.array(train_dataset.labels), minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = (class_counts.sum() / (len(CLASS_NAMES) * (class_counts + 1e-8))).astype(np.float32)
    class_weights = np.sqrt(class_weights)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    log_message(f"Class counts: {class_counts.tolist()}")
    if USE_CLASS_WEIGHTS:
        log_message(f"Class weights: {[round(float(w), 4) for w in class_weights.tolist()]}")
    else:
        log_message("Class weights: disabled")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor if USE_CLASS_WEIGHTS else None,
        label_smoothing=LABEL_SMOOTHING,
    )
    if PRETRAINED_BACKBONE:
        params = model.get_param_groups(LEARNING_RATE, backbone_lr_ratio=BACKBONE_LR_RATIO)
    else:
        params = model.parameters()
    optimizer = optim.AdamW(params, lr=LEARNING_RATE, betas=(0.9, 0.99), weight_decay=WEIGHT_DECAY)
    if USE_ONECYCLE_LR:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            epochs=TOTAL_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.10,
            div_factor=10.0,
            final_div_factor=100.0,
            anneal_strategy="cos",
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_EPOCHS, eta_min=MIN_LEARNING_RATE
        )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and USE_AMP))

    saver = TopkSaver(k=TOPK_MODELS)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, verbose=True)

    log_message("\nStarting training...")
    best_f1 = 0.0
    total_epochs = TOTAL_EPOCHS
    actual_epochs = total_epochs

    for epoch in range(total_epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, epoch, total_epochs
        )
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate_model(
            model, val_loader, criterion, epoch, total_epochs, "Val"
        )
        if not USE_ONECYCLE_LR:
            scheduler.step()

        print("\n" + "=" * 60)
        print(f"Epoch {epoch+1:02d}/{total_epochs} - RESULTS:")
        print(f"  Train - Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"          Precision: {val_precision:.4f}, Recall(Macro): {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"          LR: {scheduler.get_last_lr()[0]:.6f}")
        print("=" * 60 + "\n")

        log_message(f"Epoch {epoch+1:02d}:")
        log_message(f"  Train - Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}")
        log_message(
            f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"Precision: {val_precision:.4f}, Recall(Macro): {val_recall:.4f}, F1: {val_f1:.4f}"
        )
        log_message(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_f1 > best_f1:
            best_f1 = val_f1

        saver.push(val_f1, epoch + 1, model)

        early_stopping(val_f1)
        if early_stopping.early_stop:
            actual_epochs = epoch + 1
            log_message(f"Early stopping triggered at epoch {epoch+1}")
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    log_message("\n" + "=" * 50)
    log_message(f"Training completed after {actual_epochs} epochs")
    log_message("Testing with top-5 models...")

    ckpts = saver.best_checkpoints()
    log_message(f"Using top {len(ckpts)} models for testing:")

    test_results = []
    test_metrics = []

    for i, (f1_val, epoch, path) in enumerate(ckpts):
        print(f"\nTesting Model {i+1}: Epoch {epoch}, Val F1: {f1_val:.4f}")
        log_message(f"Model {i+1}: Epoch {epoch}, Val F1: {f1_val:.4f}")

        model.load_state_dict(torch.load(path, map_location=device))
        accuracy, precision, recall, test_f1, results_df = test_model(model, test_loader, f"Model_{epoch}")

        print(f"\nTest Results for Epoch {epoch}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall(Macro): {recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")

        test_csv_path = os.path.join(RUN_DIR, f"test_results_epoch_{epoch}.csv")
        results_df.to_csv(test_csv_path, index=False, encoding="utf-8")

        test_metrics.append(
            {
                "epoch": epoch,
                "val_f1": f1_val,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall_macro": recall,
                "test_f1": test_f1,
            }
        )
        test_results.append(results_df)
        log_message(
            f"  Test Results - Acc: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall(Macro): {recall:.4f}, F1: {test_f1:.4f}"
        )

    test_metrics_df = pd.DataFrame(test_metrics)
    test_metrics_df.to_csv(os.path.join(RUN_DIR, "test_metrics_summary.csv"), index=False)

    log_message("\n" + "=" * 50)
    log_message("Performing weighted voting integration...")
    print("\nPerforming weighted voting integration...")

    integrated_df = integrate_results(test_results, ckpts)
    final_integrated_df = integrate_by_shipno(integrated_df, TEST_DIR)

    integrated_csv_path = os.path.join(RUN_DIR, "integrated_result.csv")
    final_integrated_df.to_csv(integrated_csv_path, index=False, encoding="utf-8")

    eval_df = final_integrated_df[final_integrated_df["real_shiptype"] != "Unknown"].copy()
    if eval_df.empty:
        raise RuntimeError("All integrated samples have Unknown real_shiptype. Please check test data file-name mapping.")

    integrated_accuracy = accuracy_score(eval_df["real_shiptype"], eval_df["test_shiptype"])
    integrated_precision = precision_score(
        eval_df["real_shiptype"], eval_df["test_shiptype"], average="weighted", zero_division=0
    )
    integrated_recall = recall_score(
        eval_df["real_shiptype"], eval_df["test_shiptype"], average="macro", zero_division=0
    )
    integrated_f1 = f1_score(
        eval_df["real_shiptype"], eval_df["test_shiptype"], average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print("FINAL INTEGRATED RESULTS:")
    print(f"Accuracy: {integrated_accuracy:.4f}")
    print(f"Precision: {integrated_precision:.4f}")
    print(f"Recall(Macro): {integrated_recall:.4f}")
    print(f"F1 Score: {integrated_f1:.4f}")
    print("=" * 60)

    log_message("\nFinal Integrated Results:")
    log_message(f"Accuracy: {integrated_accuracy:.4f}")
    log_message(f"Precision: {integrated_precision:.4f}")
    log_message(f"Recall(Macro): {integrated_recall:.4f}")
    log_message(f"F1 Score: {integrated_f1:.4f}")

    final_metrics = {
        "integrated_accuracy": integrated_accuracy,
        "integrated_precision": integrated_precision,
        "integrated_recall_macro": integrated_recall,
        "integrated_f1": integrated_f1,
        "actual_epochs": actual_epochs,
        "early_stopping_triggered": actual_epochs < total_epochs,
    }

    pd.DataFrame([final_metrics]).to_csv(os.path.join(RUN_DIR, "final_integrated_metrics.csv"), index=False)

    log_message(f"\nAll results saved to: {RUN_DIR}")
    log_message(f"Training completed at: {datetime.now()}")
    log_message(f"Actual training epochs: {actual_epochs}")
    print(f"\nAll results saved to: {RUN_DIR}")
    print(f"Actual training epochs: {actual_epochs}")


if __name__ == "__main__":
    main()
