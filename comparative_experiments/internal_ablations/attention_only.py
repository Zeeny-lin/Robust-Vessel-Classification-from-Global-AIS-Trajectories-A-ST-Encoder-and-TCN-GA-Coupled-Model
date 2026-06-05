"""
消融实验: 仅注意力机制 (无TCN, 无Space2Vec, 无时间编码)
数据流: lat,lon -> Linear(2->64), sog,cog -> Linear(2->32) -> 拼接(96) -> 三层注意力(Spatial+Temporal+Cross) -> 分类头
目的: 验证在去除TCN、Space2Vec和时间编码(delta_h, day_frac)的情况下，仅使用基础特征和注意力机制的效果
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import heapq
from collections import defaultdict
from datetime import datetime

# -------------- Device Configuration --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------- Path Configuration --------------
DATA_ROOT = r'D:\日常项目\25.8论文\data\data\process_seg'

TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# 统一输出到 Ablation_Attention_Only
RESULT_DIR = r'D:\日常项目\25.8论文\resultprocess\Ablation_Attention_Only'
os.makedirs(RESULT_DIR, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULT_DIR, f"run_{current_time}")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_FILE = os.path.join(RUN_DIR, "training_log.txt")
MODEL_DIR = os.path.join(RUN_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Oil Tanker']
CLASS_MAP = {k: v for v, k in enumerate(CLASS_NAMES)}

MAX_SEQ_LEN = 300
BATCH_SIZE = 16
SPATIAL_DIM = 64
TEMPORAL_DIM = 32

# -------------- Model Definitions --------------

class SpatialAttention(nn.Module):
    """Spatial Attention"""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_weights=True):
        weights = self.attention(x)
        weights = weights.squeeze(-1)
        if return_weights:
            weighted_x = x * weights.unsqueeze(-1)
            return weighted_x, weights
        return x * weights.unsqueeze(-1)

class TemporalAttention(nn.Module):
    """Temporal Attention"""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_weights=True):
        weights = self.attention(x)
        weights = weights.squeeze(-1)
        if return_weights:
            weighted_x = x * weights.unsqueeze(-1)
            return weighted_x, weights
        return x * weights.unsqueeze(-1)

class CrossAttention(nn.Module):
    """Cross Attention"""
    def __init__(self, spatial_dim, temporal_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
    def forward(self, spatial_feat, temporal_feat, return_weights=True):
        spatial_proj = self.spatial_proj(spatial_feat)
        temporal_proj = self.temporal_proj(temporal_feat)
        
        combined = torch.cat([spatial_proj, temporal_proj], dim=-1)
        
        weights = self.attention(combined)
        weights = weights.squeeze(-1)
        
        attended = self.out_proj(combined)
        
        if return_weights:
            weighted_attended = attended * weights.unsqueeze(-1)
            return weighted_attended, weights
        return attended * weights.unsqueeze(-1)


class AttentionOnlyModel(nn.Module):
    """
    消融模型: 仅注意力机制 (无TCN, 无Space2Vec, 无时间编码)
    """
    def __init__(self, num_classes, spatial_dim=SPATIAL_DIM, temporal_dim=TEMPORAL_DIM, dropout=0.1):
        super().__init__()
        
        print("\n=== Ablation: Attention Only (NO TCN, NO Space2Vec, NO Temporal Encoding) ===")
        
        # 1. 空间特征的简单线性映射 (替代Space2Vec)
        self.spatial_proj = nn.Sequential(
            nn.Linear(2, spatial_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"  Spatial Proj: lat,lon (2) -> {spatial_dim}")
        
        # 2. 运动特征的简单线性映射 (不使用delta_h, day_frac等时间编码)
        self.kinetic_proj = nn.Sequential(
            nn.Linear(2, temporal_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"  Kinetic Proj: sog,cog (2) -> {temporal_dim}")
        
        # 合并后的维度
        combined_dim = spatial_dim + temporal_dim
        print(f"  Combined Feature Dim: {combined_dim}")
        
        self.feature_norm = nn.LayerNorm(combined_dim)
        
        # 3. 注意力机制 (直接处理合并后的特征，替代TCN)
        self.spatial_attention = SpatialAttention(combined_dim, dropout=dropout)
        self.temporal_attention = TemporalAttention(combined_dim, dropout=dropout)
        
        cross_hidden = 64
        self.cross_attention = CrossAttention(
            spatial_dim=combined_dim,
            temporal_dim=combined_dim,
            hidden_dim=cross_hidden,
            dropout=dropout
        )
        cross_output_dim = cross_hidden * 2  # 128
        print(f"  GA: Spatial({combined_dim}) + Temporal({combined_dim}) + Cross -> {cross_output_dim}")
        
        self.cross_norm = nn.LayerNorm(cross_output_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cross_output_dim, cross_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cross_output_dim // 2, num_classes)
        )
        print(f"  Classifier: {cross_output_dim} -> {cross_output_dim // 2} -> {num_classes}")
        
        print(f"\n  数据流: lat,lon(Linear) + sog,cog(Linear) -> GA -> GlobalPool -> FC -> 分类")
        print(f"  注意: delta_h, day_frac 完全不使用!")
        print("="*50 + "\n")
        
    def forward(self, x, lengths=None, return_attention_weights=False):
        # 取出空间坐标，不使用Space2Vec，使用线性层
        spatial_feat_raw = x[:, :, 0:2]  # (batch, seq_len, 2)
        s_enc = self.spatial_proj(spatial_feat_raw)  # (batch, seq_len, 64)
        
        # 取出运动特征，不使用delta_h, day_frac
        kinematic_feat_raw = x[:, :, 2:4] # (batch, seq_len, 2)
        k_enc = self.kinetic_proj(kinematic_feat_raw) # (batch, seq_len, 32)
        
        # 合并特征 (替代TCN的输入，直接传给注意力层)
        combined_features = torch.cat([s_enc, k_enc], dim=-1) # (batch, seq_len, 96)
        combined_features = self.feature_norm(combined_features)
        
        # 三层注意力机制
        if return_attention_weights:
            sa, sw = self.spatial_attention(combined_features, return_weights=True)
            ta, tw = self.temporal_attention(combined_features, return_weights=True)
            ca, cw = self.cross_attention(sa, ta, return_weights=True)
        else:
            sa = self.spatial_attention(combined_features, return_weights=False)
            ta = self.temporal_attention(combined_features, return_weights=False)
            ca = self.cross_attention(sa, ta, return_weights=False)
            sw = tw = cw = None
            
        ca = self.cross_norm(ca)
        
        # 全局池化
        ca = ca.permute(0, 2, 1)  # (batch, channels, seq_len)
        pooled = self.global_pool(ca).squeeze(-1)  # (batch, channels)
        
        # 分类器
        output = self.classifier(pooled)
        
        if return_attention_weights:
            return output, {'spatial': sw, 'temporal': tw, 'cross': cw}
        
        return output

# -------------- Data Processing --------------
def clean_and_extract_features(df, max_len=MAX_SEQ_LEN):
    try:
        required_raw = ['lat', 'lon', 'sog', 'cog', 'postime']
        for col in required_raw:
            if col not in df.columns:
                if col == 'postime': 
                    raise ValueError("Missing 'postime'")
                df[col] = 0.0

        cols_numeric = ['lat', 'lon', 'sog', 'cog']
        for col in cols_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['postime'] = pd.to_datetime(df['postime'], unit='s')
        df = df.sort_values('postime').reset_index(drop=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df['lat'] = df['lat'].interpolate(method='linear', limit_direction='both')
        df['lon'] = df['lon'].interpolate(method='linear', limit_direction='both')
        df['sog'] = df['sog'].fillna(0.0)
        df['cog'] = df['cog'].fillna(0.0)
        df.fillna(0.0, inplace=True)

        df['delta_h'] = df['postime'].diff().dt.total_seconds() / 3600.0
        df.loc[0, 'delta_h'] = 0.0
        df['delta_h'] = df['delta_h'].fillna(0.0)

        df['day_frac'] = (df['postime'].dt.hour * 3600 + 
                          df['postime'].dt.minute * 60 + 
                          df['postime'].dt.second) / 86400.0
        
        # 尽管这里依然提取了6个特征以保持数据处理逻辑不变，在模型 forward 中只使用前4个(lat, lon, sog, cog)
        final_cols = ['lat', 'lon', 'sog', 'cog', 'delta_h', 'day_frac']
        result_df = df[final_cols].astype(float)
        
        if len(result_df) > max_len:
            indices = np.linspace(0, len(result_df)-1, max_len, dtype=int)
            result_df = result_df.iloc[indices]
        
        return result_df

    except Exception as e:
        return None

class ShipTrajectoryDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=MAX_SEQ_LEN):
        self.X = []
        self.y = []
        self.filenames = []
        self.max_seq_len = max_seq_len
        
        print(f"Scanning directory: {data_dir}")
        for ship_type in CLASS_NAMES:
            ship_dir = os.path.join(data_dir, ship_type)
            if not os.path.exists(ship_dir): 
                continue
            files = glob.glob(os.path.join(ship_dir, '*.csv'))
            for csv_file in files:
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = [c.lower().strip() for c in df.columns]
                    cleaned_df = clean_and_extract_features(df, max_len=self.max_seq_len)
                    if cleaned_df is not None and len(cleaned_df) > 5:
                        data_np = cleaned_df.values
                        if np.isnan(data_np).any() or np.isinf(data_np).any():
                            continue
                        self.X.append(data_np)
                        self.y.append(CLASS_MAP[ship_type])
                        self.filenames.append(os.path.basename(csv_file))
                except:
                    pass
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long), self.filenames[idx]

def pad_collate_fn(batch):
    seqs, labels, filenames = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return seqs, torch.tensor(labels), None, lengths, filenames

# -------------- Training Components --------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == 'max':
            improvement = score - self.best_score
        else:
            improvement = self.best_score - score
        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

class TopkSaver:
    def __init__(self, k=3, save_dir=MODEL_DIR):
        self.k = k
        self.heap = []
        self.save_dir = save_dir
        
    def push(self, f1_val, epoch, model):
        path = os.path.join(self.save_dir, f'epoch_{epoch}_f1_{f1_val:.4f}.pth')
        torch.save(model.state_dict(), path)
        heapq.heappush(self.heap, (f1_val, epoch, path))
        if len(self.heap) > self.k:
            _, _, old_path = heapq.heappop(self.heap)
            if os.path.exists(old_path):
                os.remove(old_path)
                
    def best_checkpoints(self):
        return sorted(self.heap, key=lambda x: -x[0])

# -------------- Training Functions --------------
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f'Train Ep {epoch}', bar_format='{l_bar}{bar:10}{r_bar}')
    for batch_X, batch_Y, _, lengths, _ in pbar:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X, lengths)
        if torch.isnan(outputs).any():
            continue
        loss = criterion(outputs, batch_Y)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        acc = (outputs.argmax(1) == batch_Y).float().mean().item()
        running_loss += loss.item() * batch_Y.size(0)
        running_acc += acc * batch_Y.size(0)
        n += batch_Y.size(0)
        pbar.set_postfix({'Loss': running_loss/n, 'Acc': running_acc/n})
    return running_loss / (n + 1e-8), running_acc / (n + 1e-8)

def evaluate(model, loader, criterion, epoch, mode='Val'):
    model.eval()
    running_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_Y, _, lengths, _ in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X, lengths)
            loss = criterion(outputs, batch_Y)
            running_loss += loss.item() * batch_Y.size(0)
            n += batch_Y.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    print(f"[{mode}] Epoch {epoch}: Loss={running_loss/n:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    return running_loss/n, acc, f1, all_preds, all_labels

def ensemble_predict(models_with_weights, test_loader):
    ship_votes = defaultdict(lambda: np.zeros(len(CLASS_NAMES)))
    ship_true_labels = {}
    for (f1_weight, _, model_path) in models_with_weights:
        model = AttentionOnlyModel(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            for batch_X, batch_Y, _, lengths, filenames in tqdm(test_loader, desc="Inference"):
                batch_X = batch_X.to(device)
                outputs = model(batch_X, lengths)
                preds = outputs.argmax(1).cpu().numpy()
                true_labels = batch_Y.numpy()
                for fname, pred_cls, true_lbl in zip(filenames, preds, true_labels):
                    try:
                        ship_id = fname.split('_')[0]
                    except:
                        ship_id = fname
                    ship_votes[ship_id][pred_cls] += f1_weight
                    ship_true_labels[ship_id] = true_lbl
                torch.cuda.empty_cache()
    return ship_votes, ship_true_labels

def calculate_final_metrics(ship_votes, ship_true_labels):
    final_preds, final_true, ship_ids = [], [], []
    for ship_id, votes in ship_votes.items():
        final_preds.append(np.argmax(votes))
        final_true.append(ship_true_labels[ship_id])
        ship_ids.append(ship_id)
    acc = accuracy_score(final_true, final_preds)
    prec = precision_score(final_true, final_preds, average='weighted', zero_division=0)
    rec = recall_score(final_true, final_preds, average='macro', zero_division=0)
    f1 = f1_score(final_true, final_preds, average='weighted', zero_division=0)
    report_dict = classification_report(final_true, final_preds, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    detail_df = pd.DataFrame({
        'ShipID': ship_ids,
        'True_Type': [CLASS_NAMES[i] for i in final_true],
        'Pred_Type': [CLASS_NAMES[i] for i in final_preds],
        'Is_Correct': [p == t for p, t in zip(final_preds, final_true)]
    })
    return acc, prec, rec, f1, report_df, detail_df

def main():
    with open(LOG_FILE, 'w') as f:
        f.write(f"Ablation Study: Attention Only (NO TCN, NO Space2Vec, NO Temporal Encoding)\n")
        f.write(f"Training Start: {datetime.now()}\n")
        f.write(f"Path: {DATA_ROOT}\n")
    
    def log(msg):
        print(msg)
        with open(LOG_FILE, 'a') as f: 
            f.write(msg + "\n")

    log("Loading Data for Ablation (Attention Only)...")
    train_ds = ShipTrajectoryDataset(TRAIN_DIR, max_seq_len=MAX_SEQ_LEN)
    val_ds = ShipTrajectoryDataset(VAL_DIR, max_seq_len=MAX_SEQ_LEN)
    test_ds = ShipTrajectoryDataset(TEST_DIR, max_seq_len=MAX_SEQ_LEN)
    
    log(f"Samples - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    if len(train_ds) == 0: 
        log("Error: No training data found!")
        return

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    model = AttentionOnlyModel(num_classes=len(CLASS_NAMES), dropout=0.1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Total Parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4) # lr adapted from similar ablations
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    saver = TopkSaver(k=3, save_dir=MODEL_DIR)
    stopper = EarlyStopping(patience=7, mode='max')
    
    log("\n=== Training: Attention Only ===")
    for epoch in range(30):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, epoch)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_dl, criterion, epoch)
        scheduler.step()
        log(f"Ep {epoch}: Tr_Loss={tr_loss:.4f}, Val_F1={val_f1:.4f}")
        saver.push(val_f1, epoch, model)
        if stopper(val_f1):
            log("Early Stopping Triggered.")
            break

    log("\n=== Ensemble Evaluation ===")
    best_models = saver.best_checkpoints()
    if not best_models:
        current_model_path = os.path.join(MODEL_DIR, 'current_model.pth')
        torch.save(model.state_dict(), current_model_path)
        best_models = [(0.0, 0, current_model_path)]
    
    ship_votes, ship_true_labels = ensemble_predict(best_models, test_dl)
    acc, prec, rec, f1, report_df, detail_df = calculate_final_metrics(ship_votes, ship_true_labels)
    
    print("\n" + "="*50)
    print("ABLATION: ATTENTION ONLY (NO TCN, NO SPACE2VEC, NO TEMPORAL ENC):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("="*50)
    
    log(f"\nFinal: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    report_df.to_csv(os.path.join(RUN_DIR, "final_classification_report.csv"))
    detail_df.to_csv(os.path.join(RUN_DIR, "final_ship_predictions.csv"), index=False)
    
    import json
    config = {
        'model_type': 'Attention Only (Ablation)',
        'description': 'Removed TCN. Removed Space2Vec, used simple Linear on lat/lon. Removed temporal encoding (ignored delta_h, day_frac). Preserved Spatial+Temporal+Cross Attention.',
        'features_used': ['lat', 'lon', 'sog', 'cog'],
        'features_ignored': ['delta_h', 'day_frac', 'TCN', 'Space2Vec'],
        'spatial_dim': SPATIAL_DIM,
        'temporal_dim': TEMPORAL_DIM,
        'final_metrics': {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1)}
    }
    with open(os.path.join(RUN_DIR, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    log(f"\nAll results saved to: {RUN_DIR}")

if __name__ == '__main__':
    main()
