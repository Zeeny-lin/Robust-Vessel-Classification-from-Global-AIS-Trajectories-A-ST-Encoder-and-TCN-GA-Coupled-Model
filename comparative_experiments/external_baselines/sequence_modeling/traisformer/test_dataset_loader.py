#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门用于测试集预测的数据加载器
"""

import torch
import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import Dataset

class TestDatasetForPrediction(Dataset):
    """专门用于测试集预测的数据集类"""
    
    def __init__(self, data_root, max_seqlen=650, min_seqlen=50):
        self.data_root = data_root
        self.max_seqlen = max_seqlen
        self.min_seqlen = min_seqlen
        self.data = []
        
        self._load_test_data()
        print(f"📊 测试数据加载完成: {len(self.data)} 个样本")
    
    def _load_test_data(self):
        """加载测试数据"""
        print(f"🔍 在目录中查找测试数据: {self.data_root}")
        
        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_root, "*.csv"))
        
        if not csv_files:
            print(f"⚠️  在 {self.data_root} 中没有找到CSV文件")
            return
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件")
        
        # 先检查第一个文件的格式
        if csv_files:
            sample_file = csv_files[0]
            try:
                sample_df = pd.read_csv(sample_file)
                print(f"📋 样本文件格式: {sample_file}")
                print(f"   列名: {list(sample_df.columns)}")
                print(f"   数据形状: {sample_df.shape}")
                print(f"   前几行数据类型:")
                for col in sample_df.columns:
                    print(f"     {col}: {sample_df[col].dtype} - 样本值: {sample_df[col].iloc[0] if len(sample_df) > 0 else 'N/A'}")
            except Exception as e:
                print(f"❌ 无法读取样本文件: {e}")
        
        successful_files = 0
        for csv_file in csv_files:
            try:
                # 从文件名提取shipno
                shipno = os.path.basename(csv_file).replace('.csv', '')
                
                # 读取数据
                df = pd.read_csv(csv_file)
                
                # 检查数据长度
                if len(df) < self.min_seqlen:
                    continue
                
                # 确保有必要的列
                required_cols = ['lat', 'lon', 'sog', 'cog']
                if not all(col in df.columns for col in required_cols):
                    print(f"⚠️  文件 {csv_file} 缺少必要的列，有的列: {list(df.columns)}")
                    continue
                
                # 处理数据
                sequence_data = self._process_sequence(df)
                
                if sequence_data is not None:
                    self.data.append({
                        'sequence': sequence_data['sequence'],
                        'mask': sequence_data['mask'],
                        'shipno': shipno
                    })
                    successful_files += 1
                    
                    # 只显示前几个成功的文件信息
                    if successful_files <= 3:
                        print(f"✅ 成功处理文件: {shipno}")
                        
            except Exception as e:
                print(f"❌ 处理文件 {csv_file} 时出错: {e}")
                continue
        
        print(f"📊 成功处理 {successful_files}/{len(csv_files)} 个文件")
    
    def _process_sequence(self, df):
        """处理单个序列数据"""
        try:
            # 截取或填充到指定长度
            if len(df) > self.max_seqlen:
                df = df.iloc[:self.max_seqlen].copy()
            
            seq_len = len(df)
            
            # 创建特征矩阵 (6维: lat, lon, sog, cog, time_interval, time_window)
            features = np.zeros((self.max_seqlen, 6), dtype=np.float32)
            mask = np.zeros(self.max_seqlen, dtype=np.float32)
            
            # 确保数据类型为数值型
            try:
                lat_values = pd.to_numeric(df['lat'], errors='coerce').fillna(0).values
                lon_values = pd.to_numeric(df['lon'], errors='coerce').fillna(0).values
                sog_values = pd.to_numeric(df['sog'], errors='coerce').fillna(0).values
                cog_values = pd.to_numeric(df['cog'], errors='coerce').fillna(0).values
            except Exception as e:
                print(f"⚠️  数据类型转换失败: {e}")
                return None
            
            # 检查是否有有效数据
            if len(lat_values) == 0 or np.all(np.isnan(lat_values)):
                print("⚠️  没有有效的位置数据")
                return None
            
            # 基本特征归一化 - 安全处理
            lat_min, lat_max = np.nanmin(lat_values), np.nanmax(lat_values)
            lon_min, lon_max = np.nanmin(lon_values), np.nanmax(lon_values)
            sog_max = np.nanmax(sog_values)
            
            # 避免除零错误
            lat_range = lat_max - lat_min if lat_max != lat_min else 1.0
            lon_range = lon_max - lon_min if lon_max != lon_min else 1.0
            sog_max = sog_max if sog_max > 0 else 1.0
            
            lat_norm = (lat_values - lat_min) / lat_range
            lon_norm = (lon_values - lon_min) / lon_range
            sog_norm = sog_values / sog_max
            cog_norm = cog_values / 360.0
            
            # 处理时间特征
            if 'postime' in df.columns:
                try:
                    # 尝试转换时间戳
                    time_values = pd.to_numeric(df['postime'], errors='coerce').fillna(0).values
                    if len(time_values) > 1:
                        time_diffs = np.diff(time_values, prepend=time_values[0])
                        time_intervals = np.clip(time_diffs, 0, 15658) / 15658.0
                        
                        # 时间窗口特征
                        hours = (time_values - time_values[0]) / 3600.0
                        time_windows = np.clip(hours, 0, 744) / 744.0
                    else:
                        time_intervals = np.zeros(seq_len)
                        time_windows = np.zeros(seq_len)
                except:
                    time_intervals = np.zeros(seq_len)
                    time_windows = np.linspace(0, 1, seq_len)
            else:
                # 如果没有时间信息，使用序列位置作为时间特征
                time_intervals = np.zeros(seq_len)
                time_windows = np.linspace(0, 1, seq_len)
            
            # 填充特征 - 确保没有NaN值
            features[:seq_len, 0] = np.nan_to_num(lat_norm, 0)
            features[:seq_len, 1] = np.nan_to_num(lon_norm, 0)
            features[:seq_len, 2] = np.nan_to_num(sog_norm, 0)
            features[:seq_len, 3] = np.nan_to_num(cog_norm, 0)
            features[:seq_len, 4] = np.nan_to_num(time_intervals, 0)
            features[:seq_len, 5] = np.nan_to_num(time_windows, 0)
            
            # 设置mask
            mask[:seq_len] = 1.0
            
            return {
                'sequence': torch.FloatTensor(features),
                'mask': torch.FloatTensor(mask)
            }
            
        except Exception as e:
            print(f"❌ 处理序列数据时出错: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

