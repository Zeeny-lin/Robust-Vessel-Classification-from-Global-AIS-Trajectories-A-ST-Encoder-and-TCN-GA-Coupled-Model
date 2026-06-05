"""
Experiment: TCN + global-attention ablation without spatial and temporal encoders.

Pipeline:
1. Use raw features directly: (lat, lon, sog, cog, delta_h, day_frac)
2. Sequence model: TCN
3. Global attention on TCN output (no spatial encoder, no temporal encoder)
4. Ship-level evaluation: weighted top-k ensemble voting
"""

import glob
import heapq
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -------------------- Runtime --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "data", "data_classify")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

RESULT_DIR = os.path.join(BASE_DIR, "resultprocess", "Ablation_TCN_GlobalAttention_NoSpatialTemporalEncoder")
os.makedirs(RESULT_DIR, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULT_DIR, f"run_{current_time}")
MODEL_DIR = os.path.join(RUN_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
LOG_FILE = os.path.join(RUN_DIR, "training_log.txt")


# -------------------- Config --------------------
CLASS_NAMES = ["Bulk Carrier", "Container Ship", "Fishing", "Oil Tanker"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

MAX_SEQ_LEN = 300
BATCH_SIZE = 16
EPOCHS = 30

INPUT_FEATURE_DIM = 6
TCN_CHANNELS = [64, 128]
GLOBAL_ATTN_HIDDEN = 64
GLOBAL_ATTN_HEADS = 4

LR = 1e-3
SEED = 42


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- Model --------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GlobalAttention(nn.Module):
    """Global self-attention on TCN sequence features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = GLOBAL_ATTN_HIDDEN,
        num_heads: int = GLOBAL_ATTN_HEADS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        feat = self.in_proj(x)
        attended, attn_weights = self.self_attn(feat, feat, feat, need_weights=True)
        attended = self.out_norm(attended + feat)
        attended = self.out_proj(attended)
        return attended, attn_weights


class TcnGlobalAttentionModel(nn.Module):
    """Raw features + TCN + global attention, without spatial/temporal encoders."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = INPUT_FEATURE_DIM,
        tcn_channels: list[int] | None = None,
        global_attn_hidden: int = GLOBAL_ATTN_HIDDEN,
        global_attn_heads: int = GLOBAL_ATTN_HEADS,
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = TCN_CHANNELS

        print("\n=== Model: TCN + Global Attention (No Spatial/Temporal Encoder) ===")
        print("Input used by model: lat, lon, sog, cog, delta_h, day_frac (raw)")

        self.feature_norm = nn.LayerNorm(input_dim)
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=tcn_channels,
            kernel_size=2,
            dropout=0.2,
        )
        self.global_attention = GlobalAttention(
            input_dim=tcn_channels[-1],
            hidden_dim=global_attn_hidden,
            num_heads=global_attn_heads,
            dropout=dropout,
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], tcn_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels[-1] // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths=None, return_attention_weights: bool = False):
        raw_feat = self.feature_norm(x[:, :, :INPUT_FEATURE_DIM])
        tcn_out = self.tcn(raw_feat.permute(0, 2, 1)).permute(0, 2, 1)
        attended, global_weights = self.global_attention(tcn_out)
        pooled = self.global_pool(attended.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier(pooled)

        if return_attention_weights:
            return logits, {"global": global_weights}
        return logits


# -------------------- Data --------------------
def clean_and_extract_features(df: pd.DataFrame, max_len: int = MAX_SEQ_LEN) -> pd.DataFrame | None:
    try:
        df.columns = [c.lower().strip() for c in df.columns]

        # Core features always expected; fill missing with zeros for robustness.
        for col in ["lat", "lon", "sog", "cog", "delta_h", "day_frac"]:
            if col not in df.columns:
                df[col] = 0.0

        for col in ["lat", "lon", "sog", "cog", "delta_h", "day_frac"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # If raw timestamp exists, recompute temporal encoding from postime.
        if "postime" in df.columns:
            postime_numeric = pd.to_numeric(df["postime"], errors="coerce")
            if postime_numeric.notna().mean() > 0.8:
                df["postime"] = pd.to_datetime(postime_numeric, unit="s", errors="coerce")
            else:
                df["postime"] = pd.to_datetime(df["postime"], errors="coerce")
            df = df.sort_values("postime").reset_index(drop=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["lat"] = df["lat"].interpolate(method="linear", limit_direction="both")
        df["lon"] = df["lon"].interpolate(method="linear", limit_direction="both")
        df["sog"] = df["sog"].fillna(0.0)
        df["cog"] = df["cog"].fillna(0.0)

        if "postime" in df.columns:
            df["delta_h"] = df["postime"].diff().dt.total_seconds() / 3600.0
            if len(df) > 0:
                df.loc[0, "delta_h"] = 0.0
            df["delta_h"] = df["delta_h"].fillna(0.0)
            df["day_frac"] = (
                df["postime"].dt.hour * 3600
                + df["postime"].dt.minute * 60
                + df["postime"].dt.second
            ) / 86400.0

        df["delta_h"] = pd.to_numeric(df["delta_h"], errors="coerce").fillna(0.0)
        df["day_frac"] = pd.to_numeric(df["day_frac"], errors="coerce").fillna(0.0)
        df.fillna(0.0, inplace=True)

        result_df = df[["lat", "lon", "sog", "cog", "delta_h", "day_frac"]].astype(float)
        if len(result_df) > max_len:
            indices = np.linspace(0, len(result_df) - 1, max_len, dtype=int)
            result_df = result_df.iloc[indices]
        return result_df
    except Exception:
        return None


class ShipTrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, max_seq_len: int = MAX_SEQ_LEN):
        self.X: list[np.ndarray] = []
        self.y: list[int] = []
        self.filenames: list[str] = []
        self.max_seq_len = max_seq_len

        print(f"Scanning directory: {data_dir}")
        for ship_type in CLASS_NAMES:
            ship_dir = os.path.join(data_dir, ship_type)
            if not os.path.exists(ship_dir):
                continue

            total_files = 0
            kept_files = 0
            for csv_file in glob.glob(os.path.join(ship_dir, "*.csv")):
                total_files += 1
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = [c.lower().strip() for c in df.columns]
                    cleaned = clean_and_extract_features(df, max_len=self.max_seq_len)
                    if cleaned is None or len(cleaned) <= 5:
                        continue

                    data_np = cleaned.values
                    if np.isnan(data_np).any() or np.isinf(data_np).any():
                        continue

                    self.X.append(data_np)
                    self.y.append(CLASS_MAP[ship_type])
                    self.filenames.append(os.path.basename(csv_file))
                    kept_files += 1
                except Exception:
                    continue
            print(f"  {ship_type}: kept {kept_files}/{total_files}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
            self.filenames[idx],
        )


def pad_collate_fn(batch):
    seqs, labels, filenames = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return seqs, torch.tensor(labels), None, lengths, filenames


# -------------------- Train / Eval --------------------
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improvement = score - self.best_score if self.mode == "max" else self.best_score - score
        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


class TopkSaver:
    def __init__(self, k: int = 3, save_dir: str = MODEL_DIR):
        self.k = k
        self.heap: list[tuple[float, int, str]] = []
        self.save_dir = save_dir

    def push(self, f1_val: float, epoch: int, model: nn.Module) -> None:
        path = os.path.join(self.save_dir, f"epoch_{epoch}_f1_{f1_val:.4f}.pth")
        torch.save(model.state_dict(), path)
        heapq.heappush(self.heap, (f1_val, epoch, path))
        if len(self.heap) > self.k:
            _, _, old_path = heapq.heappop(self.heap)
            if os.path.exists(old_path):
                os.remove(old_path)

    def best_checkpoints(self) -> list[tuple[float, int, str]]:
        return sorted(self.heap, key=lambda x: -x[0])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", bar_format="{l_bar}{bar:10}{r_bar}")

    for batch_x, batch_y, _, lengths, _ in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, lengths)
        loss = criterion(outputs, batch_y)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc = (outputs.argmax(1) == batch_y).float().mean().item()
        running_loss += loss.item() * batch_y.size(0)
        running_acc += acc * batch_y.size(0)
        n += batch_y.size(0)
        pbar.set_postfix({"Loss": running_loss / max(1, n), "Acc": running_acc / max(1, n)})

    return running_loss / (n + 1e-8), running_acc / (n + 1e-8)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    mode: str = "Val",
):
    model.eval()
    running_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y, _, lengths, _ in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_y.size(0)
            n += batch_y.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    print(f"[{mode}] Epoch {epoch}: Loss={running_loss / n:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    return running_loss / n, acc, f1, all_preds, all_labels


def ensemble_predict(models_with_weights, test_loader):
    ship_votes = defaultdict(lambda: np.zeros(len(CLASS_NAMES)))
    ship_true_labels = {}

    print("\nRunning Ensemble Prediction (TCN+GlobalAttention)...")
    for f1_weight, _, model_path in models_with_weights:
        model = TcnGlobalAttentionModel(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print(f"  -> Inferencing with model (Val F1: {f1_weight:.4f})...")
        with torch.no_grad():
            for batch_x, batch_y, _, lengths, filenames in tqdm(test_loader, desc="Inference"):
                batch_x = batch_x.to(device)
                outputs = model(batch_x, lengths)
                preds = outputs.argmax(1).cpu().numpy()
                true_labels = batch_y.numpy()

                for fname, pred_cls, true_lbl in zip(filenames, preds, true_labels):
                    ship_id = fname.split("_")[0] if "_" in fname else fname
                    ship_votes[ship_id][pred_cls] += f1_weight
                    ship_true_labels[ship_id] = true_lbl
    return ship_votes, ship_true_labels


def calculate_final_metrics(ship_votes, ship_true_labels):
    final_preds, final_true, ship_ids = [], [], []
    for ship_id, votes in ship_votes.items():
        final_preds.append(int(np.argmax(votes)))
        final_true.append(int(ship_true_labels[ship_id]))
        ship_ids.append(ship_id)

    acc = accuracy_score(final_true, final_preds)
    prec = precision_score(final_true, final_preds, average="weighted", zero_division=0)
    rec = recall_score(final_true, final_preds, average="weighted", zero_division=0)
    f1 = f1_score(final_true, final_preds, average="weighted", zero_division=0)

    report_dict = classification_report(final_true, final_preds, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    detail_df = pd.DataFrame(
        {
            "ShipID": ship_ids,
            "True_Type": [CLASS_NAMES[i] for i in final_true],
            "Pred_Type": [CLASS_NAMES[i] for i in final_preds],
            "Is_Correct": [p == t for p, t in zip(final_preds, final_true)],
        }
    )
    return acc, prec, rec, f1, report_df, detail_df


def main():
    set_seed(SEED)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Training Start: {datetime.now()}\n")
        f.write(f"Path: {DATA_ROOT}\n")
        f.write(f"Max Sequence Length: {MAX_SEQ_LEN}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Input Feature Dim: {INPUT_FEATURE_DIM}\n")
        f.write(f"TCN Channels: {TCN_CHANNELS}\n")
        f.write(f"Global Attention Hidden Dim: {GLOBAL_ATTN_HIDDEN}\n")
        f.write(f"Global Attention Heads: {GLOBAL_ATTN_HEADS}\n")
        f.write("Spatial encoder: disabled\n")
        f.write("Temporal encoder: disabled\n")
        f.write("Attention: global only\n")

    def log(msg: str):
        print(msg)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("Loading data for TCN+GlobalAttention ablation (no spatial and temporal encoders)...")
    train_ds = ShipTrajectoryDataset(TRAIN_DIR, max_seq_len=MAX_SEQ_LEN)
    val_ds = ShipTrajectoryDataset(VAL_DIR, max_seq_len=MAX_SEQ_LEN)
    test_ds = ShipTrajectoryDataset(TEST_DIR, max_seq_len=MAX_SEQ_LEN)
    log(f"Samples - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    if len(train_ds) == 0:
        log("Error: no training data found.")
        return

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    model = TcnGlobalAttentionModel(
        num_classes=len(CLASS_NAMES),
        tcn_channels=TCN_CHANNELS,
        global_attn_hidden=GLOBAL_ATTN_HIDDEN,
        global_attn_heads=GLOBAL_ATTN_HEADS,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total Parameters: {total_params:,}")
    log(f"Trainable Parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    saver = TopkSaver(k=3, save_dir=MODEL_DIR)
    stopper = EarlyStopping(patience=7, mode="max")

    log("\n=== Starting Training: TCN + GlobalAttention ===")
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, epoch)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_dl, criterion, epoch)
        scheduler.step()
        log(
            f"Ep {epoch}: Tr_Loss={tr_loss:.4f}, Tr_Acc={tr_acc:.4f}, "
            f"Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}, Val_F1={val_f1:.4f}"
        )
        saver.push(val_f1, epoch, model)
        if stopper(val_f1):
            log("Early stopping triggered.")
            break

    log("\n=== Starting Ensemble Evaluation (Ship-level) ===")
    best_models = saver.best_checkpoints()
    if not best_models:
        current_model_path = os.path.join(MODEL_DIR, "current_model.pth")
        torch.save(model.state_dict(), current_model_path)
        best_models = [(0.0, 0, current_model_path)]

    ship_votes, ship_true_labels = ensemble_predict(best_models, test_dl)
    acc, prec, rec, f1, report_df, detail_df = calculate_final_metrics(ship_votes, ship_true_labels)

    print("\n" + "=" * 54)
    print("FINAL SHIP-LEVEL ENSEMBLE RESULTS (TCN + GlobalAttention, No Spatial/Temporal Encoders):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 54)

    log(f"\nFinal Ship Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    report_df.to_csv(os.path.join(RUN_DIR, "final_classification_report.csv"))
    detail_df.to_csv(os.path.join(RUN_DIR, "final_ship_predictions.csv"), index=False)

    config = {
        "model_type": "TcnGlobalAttentionModel",
        "description": "Raw 6D features + TCN + global attention, without spatial encoder or temporal encoder.",
        "input_params": {"input_dim": INPUT_FEATURE_DIM},
        "tcn_params": {"channels": TCN_CHANNELS, "kernel_size": 2},
        "global_attention_params": {"hidden_dim": GLOBAL_ATTN_HIDDEN, "num_heads": GLOBAL_ATTN_HEADS},
        "features_used": ["lat", "lon", "sog", "cog", "delta_h", "day_frac"],
        "max_seq_len": MAX_SEQ_LEN,
        "final_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        },
    }
    with open(os.path.join(RUN_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    log(f"\nAll results saved to: {RUN_DIR}")
    log("Training completed successfully.")


if __name__ == "__main__":
    main()
