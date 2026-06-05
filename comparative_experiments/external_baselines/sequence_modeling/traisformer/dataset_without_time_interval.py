#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
不包含时间间隔特征的船舶数据集
特征：[lat, lon, sog, cog, time_window] - 5维
"""

import torch
import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import Dataset

class ShipDatasetWithoutTimeInterval(Dataset):
    def __init__(self, data_root, max_seqlen=650, min_seqlen=50, max_time_window=744):
        self.data_root = data_root
        self.max_seqlen = max_seqlen
        self.min_seqlen = min_seqlen
        self.max_time_window = max_time_window
        self.ship_types = ['集装箱船', '干散货船', '渔船', '油船']
        self.data = []
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        print("🔄 加载船舶数据（不含时间间隔特征）...")
        
        train_dir = os.path.join(self.data_root, 'train')
        if not os.path.exists(train_dir):
            print(f"❌ 训练目录不存在: {train_dir}")
            return
        
        for label, ship_type in enumerate(self.ship_types):
            ship_dir = os.path.join(train_dir, ship_type)
            if not os.path.exists(ship_dir):
                print(f"⚠️ 船型目录不存在: {ship_type}")
                continue
            
            csv_files = glob.glob(os.path.join(ship_dir, '*.csv'))
            print(f"📂 {ship_type}: {len(csv_files)} 个文件")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # 检查必要列
                    required_cols = ['shipno', 'lat', 'lon', 'sog', 'cog', 'time_window']
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
                        
                        # 提取特征（5维：lat, lon, sog, cog, time_window）
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
        """提取5维特征：[lat, lon, sog, cog, time_window_norm]"""
        # 提取原始特征
        lat = group['lat'].values
        lon = group['lon'].values  
        sog = group['sog'].values
        cog = group['cog'].values
        time_window = group['time_window'].values
        
        # 时间窗口特征标准化
        time_window_norm = np.clip(time_window / self.max_time_window, 0, 0.999)
        
        # 组合特征: [lat, lon, sog, cog, time_window_norm]
        trajectory = np.column_stack([
            lat, lon, sog, cog, time_window_norm
        ])
        
        return trajectory
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        trajectory = sample['trajectory']
        seq_len = len(trajectory)
        
        # 创建固定长度的序列和掩码
        sequence = np.zeros((self.max_seqlen, 5))  # 5维特征
        mask = np.zeros(self.max_seqlen, dtype=bool)
        
        sequence[:seq_len] = trajectory
        mask[:seq_len] = True
        
        return {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': sample['label'],
            'shipno': sample['shipno']
        }

class TestDatasetWithoutTimeInterval(Dataset):
    def __init__(self, data_root, max_seqlen=650, min_seqlen=50, max_time_window=744):
        self.data_root = data_root
        self.max_seqlen = max_seqlen
        self.min_seqlen = min_seqlen
        self.max_time_window = max_time_window
        self.data = []
        
        self._load_test_data()
    
    def _load_test_data(self):
        """加载测试数据"""
        print("🔄 加载测试数据（不含时间间隔特征）...")
        
        test_dir = os.path.join(self.data_root, 'test')
        if not os.path.exists(test_dir):
            print(f"❌ 测试目录不存在: {test_dir}")
            return
        
        csv_files = glob.glob(os.path.join(test_dir, '*.csv'))
        print(f"📂 找到 {len(csv_files)} 个测试文件")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # 检查必要列
                required_cols = ['shipno', 'lat', 'lon', 'sog', 'cog']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                # 如果没有time_window列，计算它
                if 'time_window' not in df.columns:
                    df = self._add_time_window_feature(df)
                
                # 按船舶分组
                for shipno, group in df.groupby('shipno'):
                    if len(group) < self.min_seqlen:
                        continue
                    
                    # 按时间排序
                    if 'postime' in group.columns:
                        group = group.sort_values('postime').reset_index(drop=True)
                    
                    # 截断序列
                    if len(group) > self.max_seqlen:
                        start_idx = (len(group) - self.max_seqlen) // 2
                        group = group.iloc[start_idx:start_idx + self.max_seqlen]
                    
                    # 提取特征
                    trajectory = self._extract_features(group)
                    
                    self.data.append({
                        'trajectory': trajectory.astype(np.float32),
                        'shipno': shipno,
                        'filename': os.path.basename(csv_file)
                    })
                    
            except Exception as e:
                print(f"❌ 处理测试文件 {csv_file} 时出错: {e}")
        
        print(f"✅ 测试数据加载完成: {len(self.data)} 个样本")
    
    def _add_time_window_feature(self, df):
        """为测试数据添加时间窗口特征"""
        df['time_window'] = 1  # 默认值
        
        if 'postime' not in df.columns:
            return df
        
        try:
            timestamps = pd.to_datetime(df['postime'], errors='coerce')
            if not timestamps.isna().all():
                time_windows = (timestamps.dt.day - 1) * 24 + timestamps.dt.hour + 1
                df['time_window'] = time_windows.fillna(1).astype(int)
        except:
            pass
        
        return df
    
    def _extract_features(self, group):
        """提取5维特征"""
        lat = group['lat'].values
        lon = group['lon'].values  
        sog = group['sog'].values
        cog = group['cog'].values
        time_window = group['time_window'].values
        
        # 时间窗口特征标准化
        time_window_norm = np.clip(time_window / self.max_time_window, 0, 0.999)
        
        # 组合特征: [lat, lon, sog, cog, time_window_norm]
        trajectory = np.column_stack([
            lat, lon, sog, cog, time_window_norm
        ])
        
        return trajectory
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        trajectory = sample['trajectory']
        seq_len = len(trajectory)
        
        # 创建固定长度的序列和掩码
        sequence = np.zeros((self.max_seqlen, 5))  # 5维特征
        mask = np.zeros(self.max_seqlen, dtype=bool)
        
        sequence[:seq_len] = trajectory
        mask[:seq_len] = True
        
        return {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'shipno': sample['shipno'],
            'filename': sample['filename']
        }