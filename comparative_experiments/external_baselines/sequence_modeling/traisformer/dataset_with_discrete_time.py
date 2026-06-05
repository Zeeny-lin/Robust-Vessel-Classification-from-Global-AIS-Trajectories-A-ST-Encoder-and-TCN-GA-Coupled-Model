import torch
import numpy as np
import pandas as pd
import os
import glob
from torch.utils.data import Dataset

class ShipDatasetWithDiscreteTime(Dataset):
    """支持离散化时间嵌入的船舶数据集"""
    
    def __init__(self, data_root, max_seqlen=650, min_seqlen=50, 
                 max_time_interval=15658, max_time_window=744):
        self.data_root = data_root
        self.max_seqlen = max_seqlen
        self.min_seqlen = min_seqlen
        self.max_time_interval = max_time_interval
        self.max_time_window = max_time_window
        self.data = []
        
        # 船舶类型映射
        self.ship_types = {
            '集装箱船': 0,
            '干散货船': 1,
            '渔船': 2,
            '油船': 3
        }
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        print("🔄 加载船舶轨迹数据 (离散化时间嵌入版本)...")
        
        train_dir = os.path.join(self.data_root, 'train')
        if not os.path.exists(train_dir):
            print(f"❌ 训练目录不存在: {train_dir}")
            return
        
        for ship_type, label in self.ship_types.items():
            type_dir = os.path.join(train_dir, ship_type)
            if not os.path.exists(type_dir):
                print(f"⚠️  跳过不存在的目录: {type_dir}")
                continue
            
            csv_files = glob.glob(os.path.join(type_dir, '*.csv'))
            print(f"📁 {ship_type}: {len(csv_files)} 个文件")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # 检查必要列
                    required_cols = ['shipno', 'lat', 'lon', 'sog', 'cog', 'time_interval', 'time_window']
                    if not all(col in df.columns for col in required_cols):
                        continue
                    
                    # 按船舶分组
                    for shipno, group in df.groupby('shipno'):
                        if len(group) < self.min_seqlen:
                            continue
                        
                        # 按时间排序
                        group = group.sort_values('postime').reset_index(drop=True)
                        
                        # 截断序列
                        if len(group) > self.max_seqlen:
                            start_idx = (len(group) - self.max_seqlen) // 2
                            group = group.iloc[start_idx:start_idx + self.max_seqlen]
                        
                        # 提取特征
                        trajectory = self._extract_features(group)
                        
                        self.data.append({
                            'trajectory': trajectory.astype(np.float32),
                            'label': label,
                            'ship_type': ship_type,
                            'shipno': shipno
                        })
                        
                except Exception as e:
                    print(f"❌ 处理文件 {csv_file} 时出错: {e}")
        
        print(f"✅ 加载完成: {len(self.data)} 个样本")
    
    def _extract_features(self, group):
        """提取特征 - 为离散化嵌入准备"""
        # 提取原始特征
        lat = group['lat'].values
        lon = group['lon'].values  
        sog = group['sog'].values
        cog = group['cog'].values
        time_interval = group['time_interval'].values
        time_window = group['time_window'].values
        
        # 时间特征标准化 (模型内部会反归一化后离散化)
        time_interval_norm = np.clip(time_interval / self.max_time_interval, 0, 0.999)
        time_window_norm = np.clip(time_window / self.max_time_window, 0, 0.999)
        
        # 组合特征: [lat, lon, sog, cog, time_interval_norm, time_window_norm]
        trajectory = np.column_stack([
            lat, lon, sog, cog, time_interval_norm, time_window_norm
        ])
        
        return trajectory
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        trajectory = item['trajectory']  # 6维特征
        
        # 填充到最大长度
        seq_len = len(trajectory)
        if seq_len < self.max_seqlen:
            padding = np.zeros((self.max_seqlen - seq_len, 6), dtype=np.float32)
            trajectory = np.vstack([trajectory, padding])
        
        # 创建mask
        mask = np.zeros(self.max_seqlen, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        return {
            'sequence': torch.tensor(trajectory, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'label': item['label'],
            'shipno': item['shipno']
        }