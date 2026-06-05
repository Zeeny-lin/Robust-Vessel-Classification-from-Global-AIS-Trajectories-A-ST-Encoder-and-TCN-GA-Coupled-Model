#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的时间特征添加脚本：只在原始数据后添加两列时间特征
不做任何数据过滤，保持所有原始数据

时间特征：
1. time_interval: 第一个点为0，后续为与前一点的时间差(秒)
2. time_window: 月内24小时位置编号 (1-744)
"""

import pandas as pd
import numpy as np
import os
import glob

class SimpleTimeFeatureAdder:
    def __init__(self):
        self.data_dir = r'F:/船舶轨迹分类/轨迹识别船型/轨迹识别船型'  # 根目录
        self.output_dir = r'F:/船舶轨迹分类/轨迹识别船型/轨迹识别船型2'  # 输出目录
        self.ship_types = ['集装箱船', '干散货船', '渔船', '油船']
        
    def compute_time_interval_feature(self, timestamps):
        """计算时间间隔特征 - 第一个点为0，后续为时间差"""
        if len(timestamps) < 2:
            return np.array([0])
        
        # 计算相邻点时间差
        time_diffs = np.diff(timestamps)
        
        # 第一个点设为0，后续为时间差
        time_intervals = np.concatenate([[0], time_diffs])
        
        return time_intervals
    
    def compute_time_window_feature(self, timestamps):
        """计算时间窗口特征 - 基于月内小时位置"""
        # 使用pandas处理时间戳
        dt_series = pd.to_datetime(timestamps, unit='s', errors='coerce')
        
        # 计算月内位置：(日期-1) * 24 + 小时 + 1
        time_windows = (dt_series.day - 1) * 24 + dt_series.hour + 1
        
        # 处理无效值
        time_windows = pd.Series(time_windows).fillna(1).astype(int)
        
        return time_windows.values
    
    def add_time_features_to_csv(self, csv_file):
        """为单个CSV文件添加时间特征"""
        print(f"Processing: {csv_file}")
        
        # 读取CSV
        df = pd.read_csv(csv_file)
        
        # 检查必要列
        if 'postime' not in df.columns or 'shipno' not in df.columns:
            print(f"Warning: Missing required columns in {csv_file}")
            return df
        
        # 按船舶分组处理
        ship_groups = df.groupby('shipno')
        
        # 初始化新列 - 使用NaN，稍后填充计算出的值
        df['time_interval'] = np.nan
        df['time_window'] = np.nan
        
        for shipno, group in ship_groups:
            # 按时间排序
            group_sorted = group.sort_values('postime').reset_index()
            
            try:
                # 更灵活的时间戳解析
                postime_series = group_sorted['postime']
                
                # 尝试多种格式解析
                if postime_series.dtype == 'object':
                    # 字符串格式，尝试解析
                    datetime_series = pd.to_datetime(postime_series, format='%Y/%m/%d %H:%M:%S', errors='coerce')
                    if datetime_series.isna().all():
                        # 如果第一种格式失败，尝试其他格式
                        datetime_series = pd.to_datetime(postime_series, errors='coerce')
                    
                    if datetime_series.isna().any():
                        print(f"Warning: Some invalid timestamps for ship {shipno}")
                        print(f"Sample postime values: {postime_series.head().tolist()}")
                    
                    timestamps = datetime_series.astype('int64') // 10**9
                else:
                    # 数值格式，假设已经是Unix时间戳
                    timestamps = pd.to_numeric(postime_series, errors='coerce')
                
                timestamps = timestamps.values
                
                # 检查有效时间戳
                valid_mask = ~pd.isna(timestamps)
                if not valid_mask.any():
                    print(f"Warning: No valid timestamps for ship {shipno}, skipping")
                    continue
                
                if not valid_mask.all():
                    print(f"Warning: {(~valid_mask).sum()} invalid timestamps for ship {shipno}")
                    # 只使用有效的时间戳
                    timestamps = timestamps[valid_mask]
                    group_sorted = group_sorted[valid_mask].reset_index(drop=True)
                
                # 计算时间特征
                time_intervals = self.compute_time_interval_feature(timestamps)
                time_windows = self.compute_time_window_feature(timestamps)
                
                # 更新DataFrame
                df.loc[group_sorted['index'], 'time_interval'] = time_intervals
                df.loc[group_sorted['index'], 'time_window'] = time_windows
                
            except Exception as e:
                print(f"Error processing ship {shipno}: {e}")
                continue
        
        # 填充剩余的NaN值
        df['time_interval'] = df['time_interval'].fillna(0)
        df['time_window'] = df['time_window'].fillna(1)
        
        # 验证结果
        print(f"Time interval stats: min={df['time_interval'].min()}, max={df['time_interval'].max()}, unique_count={df['time_interval'].nunique()}")
        print(f"Time window stats: min={df['time_window'].min()}, max={df['time_window'].max()}, unique_count={df['time_window'].nunique()}")
        
        return df
    
    def process_all_data(self):
        """处理所有数据：训练数据（按船型分类）+ 测试数据"""
        print("🚀 开始处理所有船舶数据...")
        print(f"输入目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ 输入目录不存在: {self.data_dir}")
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        total_success = 0
        total_error = 0
        
        # 1. 处理训练数据（按船型分类）
        train_dir = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_dir):
            print("\n� 处理训练数据...")
            
            # 创建输出训练目录
            output_train_dir = os.path.join(self.output_dir, 'train')
            os.makedirs(output_train_dir, exist_ok=True)
            
            for ship_type in self.ship_types:
                ship_type_dir = os.path.join(train_dir, ship_type)
                
                if not os.path.exists(ship_type_dir):
                    print(f"  ⚠️  船型目录不存在: {ship_type}")
                    continue
                
                # 创建输出船型目录
                output_ship_dir = os.path.join(output_train_dir, ship_type)
                os.makedirs(output_ship_dir, exist_ok=True)
                
                csv_files = glob.glob(os.path.join(ship_type_dir, '*.csv'))
                print(f"\n🚢 处理 {ship_type}: {len(csv_files)} 个文件")
                
                success_count = 0
                for csv_file in csv_files:
                    try:
                        enhanced_df = self.add_time_features_to_csv(csv_file)
                        
                        output_file = os.path.join(output_ship_dir, os.path.basename(csv_file))
                        enhanced_df.to_csv(output_file, index=False)
                        
                        success_count += 1
                        total_success += 1
                        
                        if success_count <= 3:  # 只显示前3个成功的文件
                            print(f"    ✅ {os.path.basename(csv_file)}")
                        
                    except Exception as e:
                        print(f"    ❌ 失败: {os.path.basename(csv_file)} - {e}")
                        total_error += 1
                
                print(f"  ✅ {ship_type}: 成功 {success_count}/{len(csv_files)} 个文件")
        
        # 2. 处理测试数据
        test_dir = os.path.join(self.data_dir, 'test')
        if os.path.exists(test_dir):
            print("\n🧪 处理测试数据...")
            
            # 创建输出测试目录
            output_test_dir = os.path.join(self.output_dir, 'test')
            os.makedirs(output_test_dir, exist_ok=True)
            
            csv_files = glob.glob(os.path.join(test_dir, '*.csv'))
            print(f"找到 {len(csv_files)} 个测试文件")
            
            success_count = 0
            for csv_file in csv_files:
                try:
                    enhanced_df = self.add_time_features_to_csv(csv_file)
                    
                    output_file = os.path.join(output_test_dir, os.path.basename(csv_file))
                    enhanced_df.to_csv(output_file, index=False)
                    
                    success_count += 1
                    total_success += 1
                    
                    if success_count <= 3:  # 只显示前3个成功的文件
                        print(f"  ✅ {os.path.basename(csv_file)}")
                    
                except Exception as e:
                    print(f"  ❌ 失败: {os.path.basename(csv_file)} - {e}")
                    total_error += 1
            
            print(f"✅ 测试数据: 成功 {success_count}/{len(csv_files)} 个文件")
        
        # 3. 最终统计
        print(f"\n📊 处理完成统计:")
        print(f"✅ 总成功: {total_success} 个文件")
        print(f"❌ 总失败: {total_error} 个文件")
        print(f"📁 输出目录: {self.output_dir}")
        
        # 验证输出结构
        if os.path.exists(self.output_dir):
            train_output = os.path.join(self.output_dir, 'train')
            test_output = os.path.join(self.output_dir, 'test')
            
            if os.path.exists(train_output):
                for ship_type in self.ship_types:
                    ship_dir = os.path.join(train_output, ship_type)
                    if os.path.exists(ship_dir):
                        count = len(glob.glob(os.path.join(ship_dir, '*.csv')))
                        print(f"  📂 train/{ship_type}: {count} 个文件")
            
            if os.path.exists(test_output):
                test_count = len(glob.glob(os.path.join(test_output, '*.csv')))
                print(f"  📂 test: {test_count} 个文件")

if __name__ == "__main__":
    adder = SimpleTimeFeatureAdder()
    adder.process_all_data()  # 改为处理所有数据

