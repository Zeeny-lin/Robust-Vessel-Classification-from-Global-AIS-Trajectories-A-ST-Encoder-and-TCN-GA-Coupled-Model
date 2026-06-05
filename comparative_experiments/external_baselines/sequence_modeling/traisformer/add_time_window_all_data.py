#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为所有船舶数据添加时间窗口特征
支持完整的数据集结构：train/四个船型文件夹 + test/直接文件

时间特征：
time_window: 月内24小时位置编号 (1-744)
"""

import pandas as pd
import numpy as np
import os
import glob

class TimeWindowFeatureAdder:
    def __init__(self):
        self.data_dir = r'G:/smoothed'  # 原始数据目录
        self.output_dir = r'G:/smoothed_time_features'  # 输出目录
        self.ship_types = ['集装箱船', '干散货船', '渔船', '油船']
        
    def compute_time_window_feature(self, timestamps):
        """计算时间窗口特征 - 基于月内小时位置"""
        dt_series = pd.to_datetime(timestamps, unit='s', errors='coerce')
        time_windows = (dt_series.day - 1) * 24 + dt_series.hour + 1
        time_windows = pd.Series(time_windows).fillna(1).astype(int)
        return time_windows.values
    
    def add_time_window_to_csv(self, csv_file):
        """为单个CSV文件添加时间窗口特征"""
        print(f"  处理: {os.path.basename(csv_file)}")
        
        df = pd.read_csv(csv_file)
        
        if 'postime' not in df.columns or 'shipno' not in df.columns:
            print(f"    ⚠️  缺少必要列: {csv_file}")
            return df
        
        df['time_window'] = np.nan
        
        for shipno, group in df.groupby('shipno'):
            group_sorted = group.sort_values('postime').reset_index()
            
            try:
                postime_series = group_sorted['postime']
                
                if postime_series.dtype == 'object':
                    datetime_series = pd.to_datetime(postime_series, format='%Y/%m/%d %H:%M:%S', errors='coerce')
                    if datetime_series.isna().all():
                        datetime_series = pd.to_datetime(postime_series, errors='coerce')
                    timestamps = datetime_series.astype('int64') // 10**9
                else:
                    timestamps = pd.to_numeric(postime_series, errors='coerce')
                
                timestamps = timestamps.values
                valid_mask = ~pd.isna(timestamps)
                
                if not valid_mask.any():
                    continue
                
                if not valid_mask.all():
                    timestamps = timestamps[valid_mask]
                    group_sorted = group_sorted[valid_mask].reset_index(drop=True)
                
                time_windows = self.compute_time_window_feature(timestamps)
                df.loc[group_sorted['index'], 'time_window'] = time_windows
                
            except Exception as e:
                print(f"    ❌ 处理船舶 {shipno} 失败: {e}")
                continue
        
        df['time_window'] = df['time_window'].fillna(1)
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
            print("\n📚 处理训练数据...")
            
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
                        enhanced_df = self.add_time_window_to_csv(csv_file)
                        
                        output_file = os.path.join(output_ship_dir, os.path.basename(csv_file))
                        enhanced_df.to_csv(output_file, index=False)
                        
                        success_count += 1
                        total_success += 1
                        
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
                    enhanced_df = self.add_time_window_to_csv(csv_file)
                    
                    output_file = os.path.join(output_test_dir, os.path.basename(csv_file))
                    enhanced_df.to_csv(output_file, index=False)
                    
                    success_count += 1
                    total_success += 1
                    
                except Exception as e:
                    print(f"  ❌ 失败: {os.path.basename(csv_file)} - {e}")
                    total_error += 1
            
            print(f"✅ 测试数据: 成功 {success_count}/{len(csv_files)} 个文件")
        
        # 3. 总结
        print(f"\n🎉 处理完成!")
        print(f"✅ 总成功: {total_success} 个文件")
        print(f"❌ 总失败: {total_error} 个文件")
        print(f"📁 输出目录: {self.output_dir}")

if __name__ == "__main__":
    processor = TimeWindowFeatureAdder()
    processor.process_all_data()