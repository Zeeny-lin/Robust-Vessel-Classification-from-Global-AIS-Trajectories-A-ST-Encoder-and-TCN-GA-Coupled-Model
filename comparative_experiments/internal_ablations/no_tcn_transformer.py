"""
Function: Ablation Study - Remove TCN, Keep Only Attention Mechanism
Modifications (compared to original model.py):
         1. Removed TCN module (Chomp1d, TemporalBlock, TemporalConvNet)
         2. Replaced TCN with Transformer Encoder for sequence modeling
         3. Kept three-layer attention: Spatial Attention, Temporal Attention, Spatiotemporal Cross Attention
         4. Kept Space2Vec encoding for spatial features
         5. Purpose: Verify whether TCN is effective by comparing with attention-only model
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
import math

# -------------- Device Configuration --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------- Path Configuration --------------
DATA_ROOT = r'D:\日常项目\25.8论文\data\data\process_seg'

TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# 消融实验结果目录（与原模型分开） 
RESULT_DIR = r'D:\日常项目\25.8论文\resultprocess\Ablation_NoTCN'
os.makedirs(RESULT_DIR, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULT_DIR, f"run_{current_time}")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_FILE = os.path.join(RUN_DIR, "training_log.txt")
MODEL_DIR = os.path.join(RUN_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Oil Tanker']
CLASS_MAP = {k: v for v, k in enumerate(CLASS_NAMES)}

# -------------- Hyperparameter Configuration --------------
MAX_SEQ_LEN = 300
BATCH_SIZE = 16
SPATIAL_EMBED_DIM = 64  # Space2Vec输出维度
TEMPORAL_DIM = 32
# TCN参数已移除，使用Transformer Encoder替代
TRANSFORMER_D_MODEL = 128  # Transformer隐藏层维度 (与原TCN输出维度一致)
TRANSFORMER_NHEAD = 4      # 多头注意力头数
TRANSFORMER_NUM_LAYERS = 2 # Transformer编码器层数

# -------------- Model Definition (Three-Layer Attention without TCN, using Transformer) --------------

class Space2VecEncoder(nn.Module):
    """Space2Vec位置编码器"""
    def __init__(self, coord_dim=2, frequency_num=16, max_radius=10000, 
                 min_radius=10, ffn_hidden_dim=256, ffn_dropout_rate=0.5, 
                 output_dim=64):
        super().__init__()
        self.frequency_num = frequency_num
        self.register_buffer('freq_bands', torch.exp(torch.linspace(
            np.log(2*np.pi/min_radius), np.log(2*np.pi/max_radius), frequency_num)))
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim * frequency_num * 2, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(ffn_dropout_rate),
            nn.Linear(ffn_hidden_dim, output_dim)
        )
        
    def forward(self, coords):
        # coords: (batch, seq_len, 2)
        batch_size, seq_len, _ = coords.shape
        arg = coords.unsqueeze(-1) * self.freq_bands  # (batch, seq_len, 2, freq_num)
        sin_enc = torch.sin(arg)  # (batch, seq_len, 2, freq_num)
        cos_enc = torch.cos(arg)  # (batch, seq_len, 2, freq_num)
        # 拼接并展平
        x = torch.cat([sin_enc, cos_enc], dim=-1)  # (batch, seq_len, 2, freq_num*2)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, 2*freq_num*2)
        out = self.mlp(x)  # (batch, seq_len, output_dim)
        return out

class PositionalEncoding(nn.Module):
    """标准的正弦位置编码，用于Transformer"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SpatialAttention(nn.Module):
    """Spatial Attention: Focus on spatial features"""
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
        # x: (batch, seq_len, input_dim)
        weights = self.attention(x)  # (batch, seq_len, 1)
        weights = weights.squeeze(-1)  # (batch, seq_len)
        
        if return_weights:
            weighted_x = x * weights.unsqueeze(-1)
            return weighted_x, weights
        return x * weights.unsqueeze(-1)

class TemporalAttention(nn.Module):
    """Temporal Attention: Focus on time-related features"""
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
        # x: (batch, seq_len, input_dim)
        weights = self.attention(x)  # (batch, seq_len, 1)
        weights = weights.squeeze(-1)  # (batch, seq_len)
        
        if return_weights:
            weighted_x = x * weights.unsqueeze(-1)
            return weighted_x, weights
        return x * weights.unsqueeze(-1)

class CrossAttention(nn.Module):
    """Spatiotemporal Cross Attention: Fuse spatial and temporal attended features"""
    def __init__(self, spatial_dim, temporal_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        
        # Attention scoring
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
    def forward(self, spatial_feat, temporal_feat, return_weights=True):
        batch_size, seq_len, _ = spatial_feat.shape
        
        # Project to same dimension
        spatial_proj = self.spatial_proj(spatial_feat)  # (batch, seq_len, hidden_dim)
        temporal_proj = self.temporal_proj(temporal_feat)  # (batch, seq_len, hidden_dim)
        
        # Concatenate
        combined = torch.cat([spatial_proj, temporal_proj], dim=-1)  # (batch, seq_len, hidden_dim*2)
        
        # Calculate attention weights
        weights = self.attention(combined)  # (batch, seq_len, 1)
        weights = weights.squeeze(-1)  # (batch, seq_len)
        
        # Apply attention
        attended = self.out_proj(combined)  # (batch, seq_len, hidden_dim*2)
        
        if return_weights:
            weighted_attended = attended * weights.unsqueeze(-1)
            return weighted_attended, weights
        return attended * weights.unsqueeze(-1)


class Space2VecAttentionOnly(nn.Module):
    """
    消融实验模型: 移除TCN，使用Transformer Encoder替代。
    保留三层注意力机制 (Spatial + Temporal + Cross Attention)。
    用于验证TCN模块的有效性。
    """
    def __init__(self, num_classes, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD,
                 num_transformer_layers=TRANSFORMER_NUM_LAYERS, dropout=0.1, 
                 spatial_embed_dim=SPATIAL_EMBED_DIM, temporal_dim=TEMPORAL_DIM):
        super().__init__()
        
        print("\n=== Ablation Model: NO TCN, Transformer + Three-Layer Attention ===")
        print(f"Input features: 6 (lat, lon, sog, cog, delta_h, day_frac)")
        print(f"Space2Vec output dim: {spatial_embed_dim}")
        print(f"Temporal feature dim: {temporal_dim}")
        
        # 1. Space2Vec encoder (process spatial coordinates)
        self.space2vec = Space2VecEncoder(
            coord_dim=2,
            frequency_num=16,
            max_radius=10000,
            min_radius=10,
            ffn_hidden_dim=256,
            ffn_dropout_rate=0.5,
            output_dim=spatial_embed_dim
        )
        print(f"\nSpace2Vec Encoder: 2 -> {spatial_embed_dim}")
        
        # 2. Temporal feature projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(4, temporal_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"Temporal Projection: 4 -> {temporal_dim}")
        
        # 3. Feature fusion - 线性投影到Transformer维度 (替代TCN)
        fusion_input_dim = spatial_embed_dim + temporal_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 输入归一化，防止Transformer数值不稳定导致NaN
        self.input_norm = nn.LayerNorm(d_model)
        print(f"\n[ABLATION] TCN removed! Using Transformer Encoder instead.")
        print(f"Feature Projection: {fusion_input_dim} -> {d_model} (with LayerNorm)")
        
        # 4. Positional encoding for Transformer
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_SEQ_LEN + 50, dropout=dropout)
        print(f"Positional Encoding: max_len={MAX_SEQ_LEN + 50}")
        
        # 5. Transformer Encoder (替代TCN进行序列建模)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        print(f"Transformer Encoder:")
        print(f"  d_model={d_model}, nhead={nhead}, num_layers={num_transformer_layers}")
        print(f"  dim_feedforward={d_model * 4}")
        
        transformer_output_dim = d_model
        print(f"Transformer output dimension: {transformer_output_dim}")
        
        # 6. Attention mechanisms (after Transformer, same structure as original)
        print(f"\nAttention Structure (applied after Transformer Encoder):")
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(transformer_output_dim, dropout=dropout)
        print(f"  Spatial Attention: {transformer_output_dim} -> {transformer_output_dim}")
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(transformer_output_dim, dropout=dropout)
        print(f"  Temporal Attention: {transformer_output_dim} -> {transformer_output_dim}")
        
        # Cross attention
        self.cross_attention = CrossAttention(
            spatial_dim=transformer_output_dim,
            temporal_dim=transformer_output_dim,
            hidden_dim=64,
            dropout=dropout
        )
        print(f"  Cross Attention: ({transformer_output_dim}, {transformer_output_dim}) -> {128}")
        
        # 7. Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Cross attention output dimension是128 (hidden_dim*2 = 64*2)
        cross_output_dim = 128
        
        self.fc1 = nn.Linear(cross_output_dim, cross_output_dim // 2)
        self.fc2 = nn.Linear(cross_output_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        print(f"\nClassification Head:")
        print(f"  Global Pool: (batch, {cross_output_dim}, seq_len) -> (batch, {cross_output_dim})")
        print(f"  FC1: {cross_output_dim} -> {cross_output_dim // 2}")
        print(f"  FC2: {cross_output_dim // 2} -> {num_classes}")
        
        # 层归一化
        self.transformer_norm = nn.LayerNorm(transformer_output_dim)
        self.cross_norm = nn.LayerNorm(cross_output_dim)
        
        print(f"\n模型结构总结 (消融实验 - 无TCN):")
        print(f"  1. Space2Vec编码空间特征 (2->{spatial_embed_dim})")
        print(f"  2. 投影时间特征 (4->{temporal_dim})")
        print(f"  3. 线性投影 ({fusion_input_dim}->{d_model})")
        print(f"  4. Transformer Encoder ({d_model}, {num_transformer_layers}层)")
        print(f"  5. 空间注意力 (关注空间模式)")
        print(f"  6. 时间注意力 (关注时间模式)")
        print(f"  7. 交叉注意力 (融合空间和时间特征)")
        print(f"  8. 分类头")
        print("="*50 + "\n")
        
    def forward(self, x, lengths=None, return_attention_weights=False):
        batch_size, seq_len, _ = x.shape
        
        # 分割特征
        spatial_feat_raw = x[:, :, 0:2]  # lat, lon
        temporal_feat_raw = x[:, :, 2:6]  # sog, cog, delta_h, day_frac
        
        # === 1. Space2Vec编码 ===
        spatial_encoded = self.space2vec(spatial_feat_raw)  # (batch, seq_len, spatial_embed_dim)
        
        # === 2. 时间特征投影 ===
        temporal_proj = self.temporal_proj(temporal_feat_raw)  # (batch, seq_len, temporal_dim)
        
        # === 3. 特征拼接 ===
        combined_features = torch.cat([spatial_encoded, temporal_proj], dim=-1)  # (batch, seq_len, spatial_embed_dim+temporal_dim)
        
        # === 4. 线性投影到Transformer维度 (替代TCN) ===
        projected = self.feature_proj(combined_features)  # (batch, seq_len, d_model)
        projected = self.input_norm(projected)  # 归一化，防止NaN
        
        # === 5. 位置编码 + Transformer Encoder ===
        projected = self.pos_encoder(projected)  # 添加位置编码
        transformer_output = self.transformer_encoder(projected)  # (batch, seq_len, d_model)
        transformer_output = self.transformer_norm(transformer_output)
        
        # NaN保护: 将NaN替换为0
        if torch.isnan(transformer_output).any():
            transformer_output = torch.nan_to_num(transformer_output, nan=0.0)
        
        # === 6. 注意力机制 (Transformer之后) ===
        # 空间注意力
        if return_attention_weights:
            spatial_attended, spatial_weights = self.spatial_attention(transformer_output, return_weights=True)
        else:
            spatial_attended = self.spatial_attention(transformer_output, return_weights=False)
            spatial_weights = None
        
        # 时间注意力
        if return_attention_weights:
            temporal_attended, temporal_weights = self.temporal_attention(transformer_output, return_weights=True)
        else:
            temporal_attended = self.temporal_attention(transformer_output, return_weights=False)
            temporal_weights = None
        
        # 交叉注意力
        if return_attention_weights:
            cross_attended, cross_weights = self.cross_attention(
                spatial_attended, temporal_attended, return_weights=True
            )
        else:
            cross_attended = self.cross_attention(
                spatial_attended, temporal_attended, return_weights=False
            )
            cross_weights = None
        
        cross_attended = self.cross_norm(cross_attended)
        
        # === 7. 全局池化 ===
        cross_attended = cross_attended.permute(0, 2, 1)  # (batch, channels, seq_len)
        pooled = self.global_pool(cross_attended).squeeze(-1)  # (batch, channels)
        
        # === 8. 分类头 ===
        pooled = self.dropout(pooled)
        pooled = F.relu(self.fc1(pooled))
        pooled = self.dropout(pooled)
        output = self.fc2(pooled)
        
        if return_attention_weights:
            return output, {
                'spatial': spatial_weights,
                'temporal': temporal_weights,
                'cross': cross_weights
            }
        
        return output

# -------------- Data Processing (与原模型完全一致) --------------
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
                except Exception as e:
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

# -------------- Training Components (与原模型完全一致) --------------
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

# -------------- Training Functions (adapted for ablation model) --------------
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f'Train Ep {epoch}', bar_format='{l_bar}{bar:10}{r_bar}')
    
    for batch_X, batch_Y, _, lengths, _ in pbar:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X, lengths)
        
        # NaN保护: 检查输出
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
        
        if n % 100 == 0:
            torch.cuda.empty_cache()
        
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
    
    print("\nRunning Ensemble Prediction (Ablation: No TCN, Transformer + Attention Only)...")
    
    for (f1_weight, _, model_path) in models_with_weights:
        model = Space2VecAttentionOnly(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print(f"  -> Inferencing with model (Val F1: {f1_weight:.4f})...")
        
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
    final_preds = []
    final_true = []
    ship_ids = []
    
    for ship_id, votes in ship_votes.items():
        pred_label = np.argmax(votes)
        true_label = ship_true_labels[ship_id]
        
        final_preds.append(pred_label)
        final_true.append(true_label)
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

def analyze_attention_weights_per_experiment(model, sample_data, raw_features, filename, true_label, save_dir):
    """Analyze attention weights following the experiment design document"""
    model.eval()
    with torch.no_grad():
        sample_tensor = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Forward pass to get attention weights
        _, attention_weights = model(sample_tensor, return_attention_weights=True)
        
        spatial_weights = attention_weights['spatial'][0].cpu().numpy()  # (seq_len,)
        temporal_weights = attention_weights['temporal'][0].cpu().numpy()  # (seq_len,)
        cross_weights = attention_weights['cross'][0].cpu().numpy()  # (seq_len,)
        
        seq_len = len(spatial_weights)
        timesteps = np.arange(seq_len)
        
        # Create attention weights dataframe
        attention_df = pd.DataFrame({
            'timestep': timesteps,
            'spatial_attention': spatial_weights,
            'temporal_attention': temporal_weights,
            'cross_attention': cross_weights
        })
        
        # Add raw features for behavior analysis
        attention_df['lat'] = raw_features[:seq_len, 0]
        attention_df['lon'] = raw_features[:seq_len, 1]
        attention_df['sog'] = raw_features[:seq_len, 2]
        attention_df['cog'] = raw_features[:seq_len, 3]
        attention_df['delta_h'] = raw_features[:seq_len, 4]
        attention_df['day_frac'] = raw_features[:seq_len, 5]
        
        # Save attention weights to CSV
        csv_path = os.path.join(save_dir, f"attention_weights_{filename.replace('.csv', '')}.csv")
        attention_df.to_csv(csv_path, index=False)
        
        # Segment by attention weights (Top-20%, Mid-40%, Bottom-40%)
        combined_attention = (spatial_weights + temporal_weights + cross_weights) / 3
        sorted_indices = np.argsort(combined_attention)[::-1]  # Descending order
        
        n_timesteps = len(combined_attention)
        top_20_count = max(1, int(n_timesteps * 0.2))
        mid_40_count = max(1, int(n_timesteps * 0.4))
        
        high_weight_indices = sorted_indices[:top_20_count]
        mid_weight_indices = sorted_indices[top_20_count:top_20_count + mid_40_count]
        low_weight_indices = sorted_indices[top_20_count + mid_40_count:]
        
        # Calculate statistics for each segment
        segments = {
            'high': high_weight_indices,
            'mid': mid_weight_indices,
            'low': low_weight_indices
        }
        
        segment_stats = {}
        for seg_name, indices in segments.items():
            segment_stats[seg_name] = {
                'spatial_mean': np.mean(spatial_weights[indices]),
                'spatial_std': np.std(spatial_weights[indices]),
                'temporal_mean': np.mean(temporal_weights[indices]),
                'temporal_std': np.std(temporal_weights[indices]),
                'cross_mean': np.mean(cross_weights[indices]),
                'cross_std': np.std(cross_weights[indices]),
                'sog_mean': np.mean(raw_features[indices, 2]),
                'sog_std': np.std(raw_features[indices, 2]),
                'cog_change_rate': np.mean(np.abs(np.diff(raw_features[indices, 3]))) if len(indices) > 1 else 0
            }
        
        # Visualization
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory with attention overlay (spatial)
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(raw_features[:, 1], raw_features[:, 0], 
                            c=combined_attention, cmap='YlOrRd', s=50, alpha=0.7)
        ax1.plot(raw_features[:, 1], raw_features[:, 0], 'b-', alpha=0.3, linewidth=1)
        ax1.set_xlabel('Longitude', fontsize=11)
        ax1.set_ylabel('Latitude', fontsize=11)
        ax1.set_title('Trajectory with Attention Weights', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Attention Weight')
        ax1.grid(True, alpha=0.3)
        
        # 2. Three attention weights over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(timesteps, spatial_weights, label='Spatial Attention', linewidth=2, alpha=0.8)
        ax2.plot(timesteps, temporal_weights, label='Temporal Attention', linewidth=2, alpha=0.8)
        ax2.plot(timesteps, cross_weights, label='Cross Attention', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Attention Weight', fontsize=11)
        ax2.set_title('Attention Weights by Time Step', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Combined attention with behavior (SOG)
        ax3 = fig.add_subplot(gs[1, :])
        ax3_twin = ax3.twinx()
        
        l1 = ax3.plot(timesteps, combined_attention, 'r-', linewidth=2, label='Combined Attention')
        ax3.fill_between(timesteps, 0, combined_attention, alpha=0.3, color='red')
        ax3.set_xlabel('Time Step', fontsize=11)
        ax3.set_ylabel('Attention Weight', color='r', fontsize=11)
        ax3.tick_params(axis='y', labelcolor='r')
        
        l2 = ax3_twin.plot(timesteps, raw_features[:seq_len, 2], 'b-', linewidth=2, label='SOG (Speed)')
        ax3_twin.set_ylabel('Speed over Ground (knots)', color='b', fontsize=11)
        ax3_twin.tick_params(axis='y', labelcolor='b')
        
        ax3.set_title('Attention Weight vs Speed Behavior', fontsize=12, fontweight='bold')
        
        # Combined legend
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Combined attention with behavior (COG change)
        ax4 = fig.add_subplot(gs[2, :])
        ax4_twin = ax4.twinx()
        
        cog_changes = np.abs(np.diff(raw_features[:seq_len, 3]))
        cog_changes = np.concatenate([[0], cog_changes])  # Add 0 for first timestep
        
        l3 = ax4.plot(timesteps, combined_attention, 'r-', linewidth=2, label='Combined Attention')
        ax4.fill_between(timesteps, 0, combined_attention, alpha=0.3, color='red')
        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Attention Weight', color='r', fontsize=11)
        ax4.tick_params(axis='y', labelcolor='r')
        
        l4 = ax4_twin.plot(timesteps, cog_changes, 'g-', linewidth=2, label='COG Change Rate')
        ax4_twin.set_ylabel('Course over Ground Change (degrees)', color='g', fontsize=11)
        ax4_twin.tick_params(axis='y', labelcolor='g')
        
        ax4.set_title('Attention Weight vs Course Change Behavior', fontsize=12, fontweight='bold')
        
        lines = l3 + l4
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Attention weight distribution by segment
        ax5 = fig.add_subplot(gs[3, 0])
        segments_data = [
            spatial_weights[high_weight_indices],
            spatial_weights[mid_weight_indices],
            spatial_weights[low_weight_indices]
        ]
        bp = ax5.boxplot(segments_data, labels=['Top 20%', 'Mid 40%', 'Bottom 40%'], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax5.set_ylabel('Spatial Attention Weight', fontsize=11)
        ax5.set_title('Spatial Attention by Segment', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Behavior metrics by segment
        ax6 = fig.add_subplot(gs[3, 1])
        seg_names = ['Top 20%', 'Mid 40%', 'Bottom 40%']
        sog_means = [segment_stats['high']['sog_mean'], 
                     segment_stats['mid']['sog_mean'], 
                     segment_stats['low']['sog_mean']]
        cog_rates = [segment_stats['high']['cog_change_rate'], 
                     segment_stats['mid']['cog_change_rate'], 
                     segment_stats['low']['cog_change_rate']]
        
        x = np.arange(len(seg_names))
        width = 0.35
        
        ax6_twin = ax6.twinx()
        bars1 = ax6.bar(x - width/2, sog_means, width, label='Avg SOG', color='skyblue')
        bars2 = ax6_twin.bar(x + width/2, cog_rates, width, label='COG Change Rate', color='lightcoral')
        
        ax6.set_xlabel('Attention Weight Segment', fontsize=11)
        ax6.set_ylabel('Average SOG (knots)', color='blue', fontsize=11)
        ax6_twin.set_ylabel('COG Change Rate (deg/step)', color='red', fontsize=11)
        ax6.set_xticks(x)
        ax6.set_xticklabels(seg_names)
        ax6.set_title('Behavior Metrics by Attention Segment', fontsize=12, fontweight='bold')
        ax6.tick_params(axis='y', labelcolor='blue')
        ax6_twin.tick_params(axis='y', labelcolor='red')
        
        # Combined legend
        lines = [bars1, bars2]
        labels = ['Avg SOG', 'COG Change Rate']
        ax6.legend(lines, labels, loc='upper left', fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Attention Analysis (No TCN) - {filename}\nTrue Class: {CLASS_NAMES[true_label]}', 
                    fontsize=14, fontweight='bold')
        
        # Save figure
        fig_path = os.path.join(save_dir, f"attention_analysis_{filename.replace('.csv', '')}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save segment statistics
        segment_stats_df = pd.DataFrame(segment_stats).T
        segment_stats_df['filename'] = filename
        segment_stats_df['true_class'] = CLASS_NAMES[true_label]
        stats_path = os.path.join(save_dir, f"segment_stats_{filename.replace('.csv', '')}.csv")
        segment_stats_df.to_csv(stats_path)
        
        print(f"Attention analysis saved: {fig_path}")
        
        return attention_df, segment_stats_df

def main():
    # 1. Log setup
    with open(LOG_FILE, 'w') as f:
        f.write(f"Ablation Study: No TCN (Transformer + Attention Only)\n")
        f.write(f"Training Start: {datetime.now()}\n")
        f.write(f"Path: {DATA_ROOT}\n")
        f.write(f"Max Sequence Length: {MAX_SEQ_LEN}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Space2Vec Embed Dim: {SPATIAL_EMBED_DIM}, Temporal Dim: {TEMPORAL_DIM}\n")
        f.write(f"Transformer: d_model={TRANSFORMER_D_MODEL}, nhead={TRANSFORMER_NHEAD}, layers={TRANSFORMER_NUM_LAYERS}\n")
    
    def log(msg):
        print(msg)
        with open(LOG_FILE, 'a') as f: 
            f.write(msg + "\n")

    # 2. Data loading
    log("Loading Data for Ablation Study (No TCN, Transformer + Attention Only)...")
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

    # 3. Initialize ablation model (No TCN)
    model = Space2VecAttentionOnly(
        num_classes=len(CLASS_NAMES),
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_transformer_layers=TRANSFORMER_NUM_LAYERS,
        dropout=0.1,
        spatial_embed_dim=SPATIAL_EMBED_DIM,
        temporal_dim=TEMPORAL_DIM
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total Parameters: {total_params:,}")
    log(f"Trainable Parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    saver = TopkSaver(k=3, save_dir=MODEL_DIR)
    stopper = EarlyStopping(patience=7, mode='max')
    
    # 4. Training
    log("\n=== Starting Ablation Training (No TCN, Transformer + Attention Only) ===")
    for epoch in range(30):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, epoch)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_dl, criterion, epoch)
        
        scheduler.step()
        log(f"Ep {epoch}: Tr_Loss={tr_loss:.4f}, Val_F1={val_f1:.4f}")
        
        saver.push(val_f1, epoch, model)
        if stopper(val_f1):
            log("Early Stopping Triggered.")
            break

    # 5. Ensemble testing
    log("\n=== Starting Ensemble Evaluation (Ship Level) - Ablation: No TCN ===")
    best_models = saver.best_checkpoints()
    
    if not best_models:
        log("Warning: No models saved! Using current model for testing.")
        current_model_path = os.path.join(MODEL_DIR, f'current_model.pth')
        torch.save(model.state_dict(), current_model_path)
        best_models = [(0.0, 0, current_model_path)]
    
    ship_votes, ship_true_labels = ensemble_predict(best_models, test_dl)
    
    # 6. Calculate final metrics
    acc, prec, rec, f1, report_df, detail_df = calculate_final_metrics(ship_votes, ship_true_labels)
    
    # 7. Output results
    print("\n" + "="*50)
    print("ABLATION RESULTS - NO TCN (TRANSFORMER + ATTENTION ONLY):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("="*50)
    
    log(f"\nFinal Ship Metrics (Ablation - No TCN):\nAcc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    
    # 8. Save files
    report_df.to_csv(os.path.join(RUN_DIR, "final_classification_report.csv"))
    detail_df.to_csv(os.path.join(RUN_DIR, "final_ship_predictions.csv"), index=False)
    
    # 9. Attention analysis following experiment design
    log("\n=== Analyzing Attention Weights per Experiment Design (Ablation) ===")
    sample_dir = os.path.join(RUN_DIR, "attention_analysis")
    os.makedirs(sample_dir, exist_ok=True)
    
    attention_model = Space2VecAttentionOnly(
        num_classes=len(CLASS_NAMES),
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_transformer_layers=TRANSFORMER_NUM_LAYERS,
        dropout=0.1,
        spatial_embed_dim=SPATIAL_EMBED_DIM,
        temporal_dim=TEMPORAL_DIM
    ).to(device)
    
    best_model_path = best_models[0][2]
    attention_model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Collect samples from each class
    all_segment_stats = []
    class_samples = {}
    for i, (X, y, fname) in enumerate(test_ds):
        class_label = y.item()
        if class_label not in class_samples or len(class_samples[class_label]) < 3:
            if class_label not in class_samples:
                class_samples[class_label] = []
            class_samples[class_label].append((X.numpy(), y.item(), fname))
    
    # Analyze samples
    for class_idx, samples in class_samples.items():
        log(f"\nAnalyzing {len(samples)} samples from class: {CLASS_NAMES[class_idx]}")
        for sample_data, sample_label, fname in samples:
            try:
                attention_df, segment_stats = analyze_attention_weights_per_experiment(
                    attention_model, sample_data, sample_data, fname, sample_label, sample_dir
                )
                all_segment_stats.append(segment_stats)
            except Exception as e:
                log(f"Failed to analyze {fname}: {e}")
    
    # 10. Summary statistics
    if all_segment_stats:
        all_stats = pd.concat(all_segment_stats, ignore_index=True)
        summary_path = os.path.join(sample_dir, "all_segment_statistics.csv")
        all_stats.to_csv(summary_path, index=False)
        
        log("\n=== Attention Analysis Summary (Ablation - No TCN) ===")
        log(f"Total samples analyzed: {len(all_segment_stats)}")
        
        # Statistics by class
        if 'true_class' in all_stats.columns:
            log("\nSegment Statistics by Class:")
            numeric_cols = all_stats.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                class_summary = all_stats.groupby(['true_class'])[numeric_cols].mean()
                log(class_summary.to_string())
            else:
                log("No numeric columns to aggregate.")
    
    # 11. Save model configuration
    import json
    config = {
        'model_type': 'Space2VecAttentionOnly (Ablation: No TCN)',
        'ablation_description': 'TCN module removed, replaced with Transformer Encoder',
        'space2vec_params': {
            'embed_dim': SPATIAL_EMBED_DIM,
            'frequency_num': 16,
            'max_radius': 10000,
            'min_radius': 10
        },
        'transformer_params': {
            'd_model': TRANSFORMER_D_MODEL,
            'nhead': TRANSFORMER_NHEAD,
            'num_layers': TRANSFORMER_NUM_LAYERS,
            'dim_feedforward': TRANSFORMER_D_MODEL * 4
        },
        'attention_structure': {
            'order': 'Transformer first, then attention',
            'spatial_attention': 'Applied on Transformer output',
            'temporal_attention': 'Applied on Transformer output',
            'cross_attention': 'Fuses spatial and temporal attended features'
        },
        'max_seq_len': MAX_SEQ_LEN,
        'final_metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        }
    }
    
    with open(os.path.join(RUN_DIR, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    log(f"\nAll results saved to: {RUN_DIR}")
    log("Ablation training completed successfully!")

if __name__ == '__main__':
    main()
