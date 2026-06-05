"""
消融实验: Space2Vec + 时间编码 + 三层注意力(GA) (无TCN, 无Transformer)
lat,lon -> Space2Vec(64维)
sog,cog,delta_h,day_frac -> Linear(4->32)
两者分别进入三层注意力(Spatial+Temporal+Cross) -> 分类头
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_ROOT = r'D:\日常项目\25.8论文\data\data\process_seg'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

RESULT_DIR = r'D:\日常项目\25.8论文\resultprocess\Ablation_SpatioTemporal_GA'
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
BATCH_SIZE = 32
SPATIAL_EMBED_DIM = 64
TEMPORAL_DIM = 32

# -------------- Modules --------------

class Space2VecEncoder(nn.Module):
    def __init__(self, coord_dim=2, frequency_num=16, max_radius=10000,
                 min_radius=10, ffn_hidden_dim=256, ffn_dropout_rate=0.5, output_dim=64):
        super().__init__()
        self.register_buffer('freq_bands', torch.exp(torch.linspace(
            np.log(2*np.pi/min_radius), np.log(2*np.pi/max_radius), frequency_num)))
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim * frequency_num * 2, ffn_hidden_dim),
            nn.ReLU(), nn.Dropout(ffn_dropout_rate),
            nn.Linear(ffn_hidden_dim, output_dim)
        )
    def forward(self, coords):
        B, S, _ = coords.shape
        arg = coords.unsqueeze(-1) * self.freq_bands
        x = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return self.mlp(x.view(B, S, -1))

class SpatialAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1), nn.Sigmoid())
    def forward(self, x, return_weights=True):
        w = self.attention(x).squeeze(-1)
        out = x * w.unsqueeze(-1)
        return (out, w) if return_weights else out

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1), nn.Sigmoid())
    def forward(self, x, return_weights=True):
        w = self.attention(x).squeeze(-1)
        out = x * w.unsqueeze(-1)
        return (out, w) if return_weights else out

class CrossAttention(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.out_proj = nn.Linear(hidden_dim*2, hidden_dim*2)
    def forward(self, s_feat, t_feat, return_weights=True):
        sp = self.spatial_proj(s_feat); tp = self.temporal_proj(t_feat)
        combined = torch.cat([sp, tp], dim=-1)
        w = self.attention(combined).squeeze(-1)
        out = self.out_proj(combined) * w.unsqueeze(-1)
        return (out, w) if return_weights else out

# -------------- Model --------------

class SpatioTemporalGAModel(nn.Module):
    """
    时空编码(Space2Vec + Temporal Encoding) + 三层注意力(GA), 无TCN
    数据流完全对齐 model.py:
      空间 -> Space2Vec(64)
      时间 -> Linear(4->32)
      拼接(96) -> Linear(96->64) [代替TCN]
      -> 分别通过 Spatial Attention(64) 和 Temporal Attention(64)
      -> 拼接(128) -> Cross Attention (融合为128维) -> GlobalPool -> FC分类
    """
    def __init__(self, num_classes, spatial_dim=SPATIAL_EMBED_DIM, temporal_dim=TEMPORAL_DIM, dropout=0.1):
        super().__init__()
        print("\n=== Ablation: Space2Vec + Temporal Encoding + GA (NO TCN) ===")
        
        self.space2vec = Space2VecEncoder(coord_dim=2, output_dim=spatial_dim)
        self.temporal_proj = nn.Sequential(nn.Linear(4, temporal_dim), nn.ReLU(), nn.Dropout(dropout))
        
        feat_dim = spatial_dim + temporal_dim
        print(f"  Space2Vec(lat,lon) -> {spatial_dim}")
        print(f"  Temporal Proj(4) -> {temporal_dim}")
        print(f"  Combined Features (Concat) -> {feat_dim}")
        
        # 替代原本的 TCN 结构
        tcn_replacement_dim = 64
        self.tcn_replacement = nn.Sequential(
            nn.Linear(feat_dim, tcn_replacement_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tcn_norm = nn.LayerNorm(tcn_replacement_dim)
        print(f"  TCN Replacement (Linear): {feat_dim} -> {tcn_replacement_dim}")
        
        # 注意力机制接收 TCN 替代层的输出 (64维)
        self.spatial_attention = SpatialAttention(tcn_replacement_dim, dropout=dropout)
        self.temporal_attention = TemporalAttention(tcn_replacement_dim, dropout=dropout)
        
        # Cross attention 接收 spatial 和 temporal 的拼接
        cross_hidden = 64
        self.cross_attention = CrossAttention(
            spatial_dim=tcn_replacement_dim, 
            temporal_dim=tcn_replacement_dim, 
            hidden_dim=cross_hidden, 
            dropout=dropout
        )
        cross_out = cross_hidden * 2 # 128
        print(f"  GA: Spatial({tcn_replacement_dim}) || Temporal({tcn_replacement_dim}) -> Cross({cross_out})")
        
        self.cross_norm = nn.LayerNorm(cross_out)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(cross_out, cross_out // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(cross_out // 2, num_classes))
        print(f"  Classifier: {cross_out} -> {cross_out//2} -> {num_classes}")
        print("="*50 + "\n")

    def forward(self, x, lengths=None, return_attention_weights=False):
        s_enc = self.space2vec(x[:, :, 0:2])
        t_enc = self.temporal_proj(x[:, :, 2:6])
        
        # 严格按照 model.py 的逻辑：
        # 拼接 -> TCN(这里用Linear替代) -> Spatial&Temporal -> Cross -> Pool -> FC
        combined = torch.cat([s_enc, t_enc], dim=-1)
        hidden = self.tcn_replacement(combined)
        hidden = self.tcn_norm(hidden)
        
        if return_attention_weights:
            sa, sw = self.spatial_attention(hidden, True)
            ta, tw = self.temporal_attention(hidden, True)
            ca, cw = self.cross_attention(sa, ta, True)
        else:
            sa = self.spatial_attention(hidden, False)
            ta = self.temporal_attention(hidden, False)
            ca = self.cross_attention(sa, ta, False)
            sw = tw = cw = None
        
        ca = self.cross_norm(ca)
        pooled = self.global_pool(ca.permute(0,2,1)).squeeze(-1)
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
                if col == 'postime': raise ValueError("Missing 'postime'")
                df[col] = 0.0
        for col in ['lat', 'lon', 'sog', 'cog']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['postime'] = pd.to_datetime(df['postime'], unit='s')
        df = df.sort_values('postime').reset_index(drop=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['lat'] = df['lat'].interpolate(method='linear', limit_direction='both')
        df['lon'] = df['lon'].interpolate(method='linear', limit_direction='both')
        df['sog'] = df['sog'].fillna(0.0); df['cog'] = df['cog'].fillna(0.0)
        df.fillna(0.0, inplace=True)
        df['delta_h'] = df['postime'].diff().dt.total_seconds() / 3600.0
        df.loc[0, 'delta_h'] = 0.0; df['delta_h'] = df['delta_h'].fillna(0.0)
        df['day_frac'] = (df['postime'].dt.hour*3600+df['postime'].dt.minute*60+df['postime'].dt.second)/86400.0
        result_df = df[['lat','lon','sog','cog','delta_h','day_frac']].astype(float)
        if len(result_df) > max_len:
            result_df = result_df.iloc[np.linspace(0, len(result_df)-1, max_len, dtype=int)]
        return result_df
    except: return None

class ShipTrajectoryDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=MAX_SEQ_LEN):
        self.X, self.y, self.filenames = [], [], []
        self.max_seq_len = max_seq_len
        print(f"Scanning directory: {data_dir}")
        for ship_type in CLASS_NAMES:
            ship_dir = os.path.join(data_dir, ship_type)
            if not os.path.exists(ship_dir): continue
            for csv_file in glob.glob(os.path.join(ship_dir, '*.csv')):
                try:
                    df = pd.read_csv(csv_file); df.columns = [c.lower().strip() for c in df.columns]
                    cleaned = clean_and_extract_features(df, max_len=self.max_seq_len)
                    if cleaned is not None and len(cleaned) > 5:
                        data_np = cleaned.values
                        if not (np.isnan(data_np).any() or np.isinf(data_np).any()):
                            self.X.append(data_np); self.y.append(CLASS_MAP[ship_type])
                            self.filenames.append(os.path.basename(csv_file))
                except: pass
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long), self.filenames[idx]

def pad_collate_fn(batch):
    seqs, labels, fns = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return seqs, torch.tensor(labels), None, lengths, fns

class EarlyStopping:
    def __init__(self, patience=5, mode='max'):
        self.patience, self.mode, self.counter, self.best_score, self.early_stop = patience, mode, 0, None, False
    def __call__(self, score):
        if self.best_score is None: self.best_score = score; return False
        imp = (score - self.best_score) if self.mode == 'max' else (self.best_score - score)
        if imp > 0: self.best_score = score; self.counter = 0; return False
        self.counter += 1
        if self.counter >= self.patience: self.early_stop = True
        return self.early_stop

class TopkSaver:
    def __init__(self, k=3, save_dir=MODEL_DIR):
        self.k, self.heap, self.save_dir = k, [], save_dir
    def push(self, f1_val, epoch, model):
        path = os.path.join(self.save_dir, f'epoch_{epoch}_f1_{f1_val:.4f}.pth')
        torch.save(model.state_dict(), path)
        heapq.heappush(self.heap, (f1_val, epoch, path))
        if len(self.heap) > self.k:
            _, _, old = heapq.heappop(self.heap)
            if os.path.exists(old): os.remove(old)
    def best_checkpoints(self): return sorted(self.heap, key=lambda x: -x[0])

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train(); rl, ra, n = 0., 0., 0
    pbar = tqdm(loader, desc=f'Train Ep {epoch}', bar_format='{l_bar}{bar:10}{r_bar}')
    for bX, bY, _, lengths, _ in pbar:
        bX, bY = bX.to(device), bY.to(device); optimizer.zero_grad()
        out = model(bX, lengths)
        if torch.isnan(out).any(): continue
        loss = criterion(out, bY)
        if torch.isnan(loss): continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        acc = (out.argmax(1)==bY).float().mean().item()
        rl += loss.item()*bY.size(0); ra += acc*bY.size(0); n += bY.size(0)
        pbar.set_postfix({'Loss': rl/n, 'Acc': ra/n})
    return rl/(n+1e-8), ra/(n+1e-8)

def evaluate(model, loader, criterion, epoch, mode='Val'):
    model.eval(); rl, n, preds, labels = 0., 0, [], []
    with torch.no_grad():
        for bX, bY, _, lengths, _ in loader:
            bX, bY = bX.to(device), bY.to(device)
            out = model(bX, lengths); loss = criterion(out, bY)
            rl += loss.item()*bY.size(0); n += bY.size(0)
            preds.extend(out.argmax(1).cpu().numpy()); labels.extend(bY.cpu().numpy())
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    print(f"[{mode}] Epoch {epoch}: Loss={rl/n:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
    return rl/n, acc, f1, preds, labels

def ensemble_predict(models_with_weights, test_loader):
    ship_votes = defaultdict(lambda: np.zeros(len(CLASS_NAMES))); ship_true = {}
    for (w, _, path) in models_with_weights:
        m = SpatioTemporalGAModel(num_classes=len(CLASS_NAMES)).to(device)
        m.load_state_dict(torch.load(path, map_location=device)); m.eval()
        with torch.no_grad():
            for bX, bY, _, lengths, fns in tqdm(test_loader, desc="Inference"):
                out = m(bX.to(device), lengths)
                for fn, p, t in zip(fns, out.argmax(1).cpu().numpy(), bY.numpy()):
                    sid = fn.split('_')[0]; ship_votes[sid][p] += w; ship_true[sid] = t
                torch.cuda.empty_cache()
    return ship_votes, ship_true

def calculate_final_metrics(ship_votes, ship_true):
    fp, ft, ids = [], [], []
    for sid, v in ship_votes.items():
        fp.append(np.argmax(v)); ft.append(ship_true[sid]); ids.append(sid)
    acc = accuracy_score(ft, fp)
    prec = precision_score(ft, fp, average='weighted', zero_division=0)
    rec = recall_score(ft, fp, average='macro', zero_division=0)
    f1 = f1_score(ft, fp, average='weighted', zero_division=0)
    report_df = pd.DataFrame(classification_report(ft, fp, target_names=CLASS_NAMES, output_dict=True)).T
    detail_df = pd.DataFrame({'ShipID': ids, 'True_Type': [CLASS_NAMES[i] for i in ft],
        'Pred_Type': [CLASS_NAMES[i] for i in fp], 'Is_Correct': [p==t for p,t in zip(fp,ft)]})
    return acc, prec, rec, f1, report_df, detail_df

def main():
    with open(LOG_FILE,'w') as f: f.write(f"Ablation: Space2Vec + Temporal + GA\nStart: {datetime.now()}\n")
    def log(msg): print(msg); open(LOG_FILE,'a').write(msg+"\n")
    
    log("Loading Data...")
    train_ds = ShipTrajectoryDataset(TRAIN_DIR); val_ds = ShipTrajectoryDataset(VAL_DIR); test_ds = ShipTrajectoryDataset(TEST_DIR)
    log(f"Samples - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    if len(train_ds)==0: log("No data!"); return
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    
    model = SpatioTemporalGAModel(num_classes=len(CLASS_NAMES), dropout=0.1).to(device)
    log(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    saver = TopkSaver(k=3); stopper = EarlyStopping(patience=7)
    
    log("\n=== Training: Space2Vec + Temporal Encoding + GA ===")
    for epoch in range(30):
        tl, ta = train_one_epoch(model, train_dl, criterion, optimizer, epoch)
        vl, va, vf, _, _ = evaluate(model, val_dl, criterion, epoch)
        scheduler.step(); log(f"Ep {epoch}: Tr_Loss={tl:.4f}, Val_F1={vf:.4f}")
        saver.push(vf, epoch, model)
        if stopper(vf): log("Early Stopping."); break
    
    best = saver.best_checkpoints()
    if not best: torch.save(model.state_dict(), os.path.join(MODEL_DIR,'cur.pth')); best=[(0,0,os.path.join(MODEL_DIR,'cur.pth'))]
    sv, st = ensemble_predict(best, test_dl)
    acc, prec, rec, f1, rdf, ddf = calculate_final_metrics(sv, st)
    
    print(f"\n{'='*50}\nABLATION: Space2Vec + Temporal + GA (NO TCN):\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n{'='*50}")
    log(f"\nFinal: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    rdf.to_csv(os.path.join(RUN_DIR,"final_classification_report.csv"))
    ddf.to_csv(os.path.join(RUN_DIR,"final_ship_predictions.csv"), index=False)
    
    import json
    json.dump({'model_type':'Space2Vec+Temporal+GA','features_used':['lat','lon','sog','cog','delta_h','day_frac'],
        'final_metrics':{'accuracy':float(acc),'precision':float(prec),'recall':float(rec),'f1':float(f1)}},
        open(os.path.join(RUN_DIR,"model_config.json"),'w'), indent=2)
    log(f"Results saved to: {RUN_DIR}")

if __name__ == '__main__': main()
