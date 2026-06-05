"""
功能   : 轨迹 + 时间特征 训练 + 进度条 + 后处理投票合并 (Baseline - BP-AdaBoost)
         模型：BP-AdaBoost（BP神经网络作为弱分类器 + AdaBoost集成）
输入   : 从 [lat, lon, sog, cog, delta_h, day_frac] 6维特征中提取更多统计特征
参考   : Frontiers in Marine Science 2025 BP-AdaBoost 方法（多类别BP输出 + AdaBoost权重更新）
"""

import os
import glob
from pathlib import Path
import math
import heapq
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# -------------- Device --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------- 路径配置 --------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_CANDIDATES = [
    PROJECT_ROOT / 'data' / 'data' / 'process_seg',
    Path(r'D:\日常项目\25.08论文\data\data\process_seg'),
    Path(r'D:\日常项目\25.08论文\data\data\process_seg'),
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if cand.exists():
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    DATA_ROOT = DATA_ROOT_CANDIDATES[0]

TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'val'
TEST_DIR = DATA_ROOT / 'test'

RESULT_DIR = PROJECT_ROOT / 'result' / 'BP_AdaBoost_FeatureEngineering'
os.makedirs(RESULT_DIR, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(str(RESULT_DIR), f"run_{current_time}")
os.makedirs(RUN_DIR, exist_ok=True)

LOG_FILE = os.path.join(RUN_DIR, "training_log.txt")
MODEL_DIR = os.path.join(RUN_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Oil Tanker']
CLASS_MAP = {k: v for v, k in enumerate(CLASS_NAMES)}

# -------------- BP-AdaBoost 超参（参考论文设置）--------------
BP_HIDDEN = 64
BP_HIDDEN_2 = 32
BP_LR = 0.001
BP_EPOCHS = 400
BP_BATCH = 256
BP_DROPOUT = 0.2
BP_L2 = 1e-4

N_WEAK = 30
EARLY_STOP_PATIENCE = 5
BP_RETRY = 3
USE_SAMME_R = True
BOOST_LR = 0.2  # shrinkage to stabilize boosting
MIN_F1_DELTA = 0.002
ERR_STOP = 0.6
ERR_PATIENCE = 2

# -------------- 特征工程函数 --------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两个经纬度点之间的哈弗辛距离（公里）"""
    R = 6371  # 地球半径（公里）
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def calculate_trajectory_features(trajectory_data):
    """
    从轨迹数据中提取特征
    输入: trajectory_data (numpy array) - [lat, lon, sog, cog, delta_h, day_frac]
    输出: 特征向量
    """
    if len(trajectory_data) == 0:
        return np.zeros(32)

    basic_features = []

    lats = trajectory_data[:, 0]
    lons = trajectory_data[:, 1]

    if len(lats) > 1:
        total_distance = 0
        distances = []
        for i in range(1, len(lats)):
            dist = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            distances.append(dist)
            total_distance += dist

        basic_features.extend([
            total_distance,
            np.mean(distances) if distances else 0,
            np.std(distances) if distances else 0,
            np.max(distances) if distances else 0,
            np.min(distances) if distances else 0,
        ])
    else:
        basic_features.extend([0, 0, 0, 0, 0])

    sogs = trajectory_data[:, 2]
    basic_features.extend([
        np.mean(sogs),
        np.std(sogs),
        np.max(sogs),
        np.min(sogs),
        np.median(sogs),
        len([s for s in sogs if s > 10]) / len(sogs) if len(sogs) > 0 else 0,
    ])

    cogs = trajectory_data[:, 3]
    cog_rad = np.radians(cogs)
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)
    mean_sin = np.mean(cog_sin)
    mean_cos = np.mean(cog_cos)
    mean_direction = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

    basic_features.extend([
        mean_direction,
        np.std(cogs),
        len([c for c in np.diff(cogs) if abs(c) > 30]) / max(len(cogs)-1, 1),
    ])

    delta_hs = trajectory_data[:, 4]
    basic_features.extend([
        np.mean(delta_hs),
        np.std(delta_hs),
        np.max(delta_hs),
        np.min(delta_hs),
        np.sum(np.abs(delta_hs)),
    ])

    day_fracs = trajectory_data[:, 5]
    if len(day_fracs) > 1:
        duration = day_fracs[-1] - day_fracs[0]
        time_intervals = np.diff(day_fracs)
        basic_features.extend([
            duration,
            np.mean(time_intervals),
            np.std(time_intervals),
            len(day_fracs) / max(duration, 0.001),
        ])
    else:
        basic_features.extend([0, 0, 0, 0])

    if len(lats) > 2:
        start_end_dist = haversine_distance(lats[0], lons[0], lats[-1], lons[-1])
        efficiency = start_end_dist / total_distance if total_distance > 0 else 0
        direction_changes = []
        for i in range(1, len(cogs)-1):
            change = min(abs(cogs[i] - cogs[i-1]), 360 - abs(cogs[i] - cogs[i-1]))
            direction_changes.append(change)
        basic_features.extend([
            start_end_dist,
            efficiency,
            np.mean(direction_changes) if direction_changes else 0,
            np.std(direction_changes) if direction_changes else 0,
        ])
    else:
        basic_features.extend([0, 0, 0, 0])

    basic_features.extend([
        len(trajectory_data),
        np.percentile(sogs, 25) if len(sogs) > 0 else 0,
        np.percentile(sogs, 75) if len(sogs) > 0 else 0,
        np.percentile(cogs, 25) if len(cogs) > 0 else 0,
        np.percentile(cogs, 75) if len(cogs) > 0 else 0,
    ])

    return np.array(basic_features)


# -------------- 数据集类 --------------
class ShipTrajectoryFeatureDataset:
    def __init__(self, data_dir):
        self.X = []
        self.y = []
        self.filenames = []
        self.required = ['lat', 'lon', 'sog', 'cog', 'delta_h', 'day_frac']

        for ship_type in CLASS_NAMES:
            ship_dir = os.path.join(str(data_dir), ship_type)
            if not os.path.exists(ship_dir):
                continue

            csv_files = glob.glob(os.path.join(ship_dir, '*.csv'))
            print(f"Processing {len(csv_files)} files in {ship_type}...")

            for csv_file in tqdm(csv_files, desc=f"Extracting features - {ship_type}"):
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = [c.lower() for c in df.columns]
                    for col in self.required:
                        if col not in df.columns:
                            df[col] = 0.0
                    trajectory = df[self.required].astype(float).values
                    if len(trajectory) > 0:
                        features = calculate_trajectory_features(trajectory)
                        self.X.append(features)
                        self.y.append(CLASS_MAP[ship_type])
                        self.filenames.append(os.path.basename(csv_file))
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")

        self.X = np.array(self.X, dtype=np.float32)
        if len(self.X) > 0:
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def get_data(self):
        return self.X, self.y, self.filenames


# -------------- BP 弱分类器 --------------
class BPWeakNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, num_classes, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_bp_weak(X, y, sample_weight, input_dim, num_classes):
    model = BPWeakNet(input_dim, BP_HIDDEN, BP_HIDDEN_2, num_classes, dropout=BP_DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BP_LR, weight_decay=BP_L2)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Bootstrap resample by AdaBoost weights (more stable for weak learners)
    n = len(X)
    prob = sample_weight / (sample_weight.sum() + 1e-8)
    idx = np.random.choice(n, size=n, replace=True, p=prob)
    X_res = X[idx]
    y_res = y[idx]

    X_t = torch.tensor(X_res, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_res, dtype=torch.long, device=device)
    w_t = torch.ones(len(X_res), dtype=torch.float32, device=device)

    n = len(X_res)
    indices = np.arange(n)
    avg_epoch_loss = None
    avg_epoch_loss_unweighted = None
    for ep in range(BP_EPOCHS):
        np.random.shuffle(indices)
        running_loss, running_loss_unw, seen = 0.0, 0.0, 0
        pbar = tqdm(range(0, n, BP_BATCH), desc=f"  BP Epoch {ep+1}/{BP_EPOCHS}", leave=False)
        for i in pbar:
            batch_idx = indices[i:i+BP_BATCH]
            xb = X_t[batch_idx]
            yb = y_t[batch_idx]
            wb = w_t[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss_vec = criterion(logits, yb)
            loss = (loss_vec * wb).sum() / (wb.sum() + 1e-8)
            loss_unw = loss_vec.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(batch_idx)
            running_loss_unw += loss_unw.item() * len(batch_idx)
            seen += len(batch_idx)
            if seen > 0:
                pbar.set_postfix({'loss_w': f"{running_loss/seen:.4f}", 'loss': f"{running_loss_unw/seen:.4f}"})
        if seen > 0:
            avg_epoch_loss = running_loss / seen
            avg_epoch_loss_unweighted = running_loss_unw / seen

    return model, avg_epoch_loss, avg_epoch_loss_unweighted


# -------------- AdaBoost (SAMME) --------------
class BP_AdaBoost:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learners = []
        self.alphas = []

    def add_learner(self, learner, alpha):
        self.learners.append(learner)
        self.alphas.append(alpha)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        if not self.learners:
            raise ValueError("No learners in ensemble.")
        scores = np.zeros((len(X), self.num_classes), dtype=np.float32)
        for alpha, learner in zip(self.alphas, self.learners):
            learner.eval()
            with torch.no_grad():
                logits = learner(torch.tensor(X, dtype=torch.float32, device=device))
                probs = F.softmax(logits, dim=1).cpu().numpy()
            scores += alpha * np.log(probs + 1e-8)
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def state_dict(self):
        return {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'alphas': self.alphas,
            'learners': [l.state_dict() for l in self.learners],
        }

    @staticmethod
    def load_state(state, hidden_dim=BP_HIDDEN, hidden_dim2=BP_HIDDEN_2):
        model = BP_AdaBoost(state['input_dim'], state['num_classes'])
        model.alphas = state['alphas']
        for sd in state['learners']:
            net = BPWeakNet(state['input_dim'], hidden_dim, hidden_dim2, state['num_classes'], dropout=BP_DROPOUT).to(device)
            net.load_state_dict(sd)
            model.learners.append(net)
        return model


# -------------- TOP-K 保存 --------------
class TopkSaver:
    def __init__(self, k=5, save_dir=MODEL_DIR):
        self.k = k
        self.heap = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def push(self, f1_val, epoch, model_state, scaler):
        model_path = os.path.join(self.save_dir, f'epoch_{epoch}_f1_{f1_val:.4f}.pth')
        scaler_path = os.path.join(self.save_dir, f'epoch_{epoch}_scaler.pkl')
        torch.save(model_state, model_path)
        joblib.dump(scaler, scaler_path)
        heapq.heappush(self.heap, (f1_val, epoch, model_path, scaler_path))
        if len(self.heap) > self.k:
            _, _, old_model, old_scaler = heapq.heappop(self.heap)
            if os.path.exists(old_model):
                os.remove(old_model)
            if os.path.exists(old_scaler):
                os.remove(old_scaler)

    def best_checkpoints(self):
        return sorted(self.heap, key=lambda x: -x[0])


# -------------- 训练与评估 --------------
def train_and_validate(ensemble, X_train, y_train, X_val, y_val, epoch, sample_weight):
    # 1. Train weak learner (retry to avoid err >= 1 - 1/K)
    K = len(CLASS_NAMES)
    best = None
    best_err = 1.0
    for retry in range(BP_RETRY):
        seed = 42 + epoch * 10 + retry
        torch.manual_seed(seed)
        np.random.seed(seed)
        learner, train_loss, train_loss_unw = train_bp_weak(X_train, y_train, sample_weight, input_dim=X_train.shape[1], num_classes=K)

        learner.eval()
        with torch.no_grad():
            logits = learner(torch.tensor(X_train, dtype=torch.float32, device=device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        incorrect = (preds != y_train).astype(np.float32)
        err = np.sum(sample_weight * incorrect) / (np.sum(sample_weight) + 1e-8)

        if err < best_err:
            best_err = err
            best = (learner, preds, incorrect)

        if err < 1.0 - 1.0 / K:
            break

    if best is None:
        return ensemble, sample_weight, None, None, None, None, None, 1.0

    learner, preds, incorrect = best
    train_acc = (preds == y_train).mean()
    err = best_err

    if USE_SAMME_R:
        # SAMME.R: update weights using probabilities (no error threshold)
        learner.eval()
        with torch.no_grad():
            logits = learner(torch.tensor(X_train, dtype=torch.float32, device=device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
        p_true = probs[np.arange(len(y_train)), y_train]
        p_true = np.clip(p_true, 1e-8, 1.0)
        # shrinkage to avoid weight explosion
        sample_weight = sample_weight * np.exp(-BOOST_LR * (K - 1) / K * np.log(p_true))
        sample_weight = np.clip(sample_weight, 1e-8, 1.0)
        sample_weight = sample_weight / (np.sum(sample_weight) + 1e-8)
        alpha = BOOST_LR
    else:
        # SAMME threshold
        if err <= 1e-8:
            alpha = 1.0
        else:
            if err >= 1.0 - 1.0 / K:
                return ensemble, sample_weight, None, None, None, None, None, err
            alpha = math.log((1 - err) / err) + math.log(K - 1)

        # Update weights
        sample_weight = sample_weight * np.exp(alpha * incorrect)
        sample_weight = sample_weight / (np.sum(sample_weight) + 1e-8)

    # 4. Add learner
    ensemble.add_learner(learner, alpha)

    # 5. Validate
    val_preds = ensemble.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)

    return ensemble, sample_weight, val_acc, val_precision, val_recall, val_f1, val_preds, err, train_acc, train_loss, train_loss_unw


def test_model(ensemble, X_test, y_test, filenames):
    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    results_df = pd.DataFrame({
        'traj_no': filenames,
        'test_shiptype': [CLASS_NAMES[p] for p in preds],
        'real_shiptype': [CLASS_NAMES[l] for l in y_test]
    })
    return acc, prec, rec, f1, results_df


def integrate_results(test_results_list, weights):
    vote_counts = {}
    for (f1_val, epoch, _, _), results_df in zip(weights, test_results_list):
        weight = f1_val
        for _, row in results_df.iterrows():
            traj_no = row['traj_no']
            pred_type = row['test_shiptype']
            if traj_no not in vote_counts:
                vote_counts[traj_no] = {}
            vote_counts[traj_no][pred_type] = vote_counts[traj_no].get(pred_type, 0) + weight

    integrated_results = []
    for traj_no, votes in vote_counts.items():
        final_pred = max(votes.items(), key=lambda x: x[1])[0]
        integrated_results.append({'traj_no': traj_no, 'test_shiptype': final_pred})
    return pd.DataFrame(integrated_results)


def integrate_by_shipno(integrated_df, test_dir):
    integrated_df['base_shipno'] = integrated_df['traj_no'].str.replace(r'_\d+\.csv$', '', regex=True)

    trajectory_lengths = []
    for traj_no in integrated_df['traj_no']:
        for ship_type in CLASS_NAMES:
            csv_path = os.path.join(str(test_dir), ship_type, traj_no)
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    trajectory_lengths.append(len(df))
                    break
                except Exception:
                    trajectory_lengths.append(1)
            else:
                if ship_type == CLASS_NAMES[-1]:
                    trajectory_lengths.append(1)

    integrated_df['length'] = trajectory_lengths

    def resolve_group(df):
        votes = {}
        for _, row in df.iterrows():
            pred_type = row['test_shiptype']
            weight = row['length']
            votes[pred_type] = votes.get(pred_type, 0) + weight
        return max(votes.items(), key=lambda x: x[1])[0]

    def get_real_shiptype(base_shipno):
        for ship_type in CLASS_NAMES:
            pattern = os.path.join(str(test_dir), ship_type, f"{base_shipno}_*.csv")
            if glob.glob(pattern):
                return ship_type
        return "Unknown"

    final_df = integrated_df.groupby('base_shipno').apply(resolve_group).reset_index(name='test_shiptype')
    final_df['real_shiptype'] = final_df['base_shipno'].apply(get_real_shiptype)
    final_df = final_df.rename(columns={'base_shipno': 'shipno'})
    return final_df


# -------------- 主流程 --------------
def main():
    with open(LOG_FILE, 'w') as f:
        f.write("BP-AdaBoost Training Log\n")
        f.write(f"Start Time: {datetime.now()}\n")
        f.write("="*50 + "\n")

    def log(msg):
        print(msg)
        with open(LOG_FILE, 'a') as f:
            f.write(msg + "\n")

    # 1. 加载数据并提取特征
    log("Loading datasets and extracting features...")
    log(f"DATA_ROOT: {DATA_ROOT}")
    log(f"DATA_ROOT exists: {DATA_ROOT.exists()}")
    train_dataset = ShipTrajectoryFeatureDataset(TRAIN_DIR)
    X_train, y_train, train_filenames = train_dataset.get_data()
    val_dataset = ShipTrajectoryFeatureDataset(VAL_DIR)
    X_val, y_val, val_filenames = val_dataset.get_data()
    test_dataset = ShipTrajectoryFeatureDataset(TEST_DIR)
    X_test, y_test, test_filenames = test_dataset.get_data()

    log(f"Train samples: {len(X_train)}")
    log(f"Val samples: {len(X_val)}")
    log(f"Test samples: {len(X_test)}")
    if len(X_train) == 0:
        log("Error: No training data found. Please check DATA_ROOT path.")
        return
    log(f"Feature dimension: {X_train.shape[1]}")

    # 2. 标准化
    scaler = StandardScaler()
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 初始化AdaBoost
    ensemble = BP_AdaBoost(input_dim=X_train.shape[1], num_classes=len(CLASS_NAMES))
    saver = TopkSaver(k=5)

    # 4. 训练循环
    log("\nStarting BP-AdaBoost training...")
    best_f1 = 0.0
    patience = 0
    err_bad = 0
    sample_weight = np.ones(len(X_train), dtype=np.float32) / len(X_train)

    for epoch in range(1, N_WEAK + 1):
        ensemble, sample_weight, val_acc, val_prec, val_rec, val_f1, _, err, tr_acc, tr_loss, tr_loss_unw = train_and_validate(
            ensemble, X_train, y_train, X_val, y_val, epoch, sample_weight)

        if val_f1 is None:
            log(f"Epoch {epoch}: weak learner error too high (err={err:.4f}), stop.")
            if not USE_SAMME_R:
                break
            else:
                continue

        log(f"Epoch {epoch:02d}: Train Loss(w)={tr_loss:.4f}, Train Loss={tr_loss_unw:.4f}, Train Acc={tr_acc:.4f}, Val Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}, err={err:.4f}, alpha={BOOST_LR:.2f}")

        saver.push(val_f1, epoch, ensemble.state_dict(), scaler)

        if val_f1 > best_f1 + MIN_F1_DELTA:
            best_f1 = val_f1
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                log("Early stopping triggered.")
                break

        if err > ERR_STOP:
            err_bad += 1
        else:
            err_bad = 0
        if err_bad >= ERR_PATIENCE:
            log(f"Stopping: weak learner error too high for {ERR_PATIENCE} consecutive rounds (err={err:.4f}).")
            break

    # 5. 使用TOP-K模型进行测试
    log("\nTesting with top-5 ensembles...")
    ckpts = saver.best_checkpoints()
    if not ckpts:
        log("No valid ensemble saved (all weak learners too weak). Please adjust BP settings.")
        return
    test_results = []
    test_metrics = []

    for i, (f1_val, epoch, model_path, scaler_path) in enumerate(ckpts):
        log(f"Model {i+1}: Epoch {epoch}, Val F1: {f1_val:.4f}")
        state = torch.load(model_path, map_location=device, weights_only=True)
        ensemble_loaded = BP_AdaBoost.load_state(state)
        scaler_loaded = joblib.load(scaler_path)

        X_test_s = scaler_loaded.transform(X_test_raw)
        accuracy, precision, recall, test_f1, results_df = test_model(
            ensemble_loaded, X_test_s, y_test, test_filenames)

        test_csv_path = os.path.join(RUN_DIR, f"test_results_epoch_{epoch}.csv")
        results_df.to_csv(test_csv_path, index=False, encoding='utf-8')

        test_metrics.append({
            'epoch': epoch,
            'val_f1': f1_val,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall_macro': recall,
            'test_f1': test_f1
        })
        test_results.append(results_df)

        log(f"  Test Results - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {test_f1:.4f}")

    pd.DataFrame(test_metrics).to_csv(os.path.join(RUN_DIR, "test_metrics_summary.csv"), index=False)

    # 6. 加权投票集成
    log("\nPerforming weighted voting integration...")
    integrated_df = integrate_results(test_results, ckpts)
    final_integrated_df = integrate_by_shipno(integrated_df, TEST_DIR)

    integrated_csv_path = os.path.join(RUN_DIR, "integrated_result.csv")
    final_integrated_df.to_csv(integrated_csv_path, index=False, encoding='utf-8')

    integrated_accuracy = accuracy_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'])
    integrated_precision = precision_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'],
                                         average='weighted', zero_division=0)
    integrated_recall = recall_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'],
                                   average='macro', zero_division=0)
    integrated_f1 = f1_score(final_integrated_df['real_shiptype'], final_integrated_df['test_shiptype'],
                            average='weighted', zero_division=0)

    log("\nFinal Integrated Results:")
    log(f"Accuracy: {integrated_accuracy:.4f}")
    log(f"Precision: {integrated_precision:.4f}")
    log(f"Recall: {integrated_recall:.4f}")
    log(f"F1 Score: {integrated_f1:.4f}")

    final_metrics_df = pd.DataFrame([{
        'integrated_accuracy': integrated_accuracy,
        'integrated_precision': integrated_precision,
        'integrated_recall_macro': integrated_recall,
        'integrated_f1': integrated_f1
    }])
    final_metrics_df.to_csv(os.path.join(RUN_DIR, "final_integrated_metrics.csv"), index=False)

    log(f"\nAll results saved to: {RUN_DIR}")


if __name__ == '__main__':
    main()
