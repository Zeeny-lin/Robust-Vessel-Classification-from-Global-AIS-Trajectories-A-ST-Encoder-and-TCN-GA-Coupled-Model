# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customized Pytorch Dataset with Time Features.
"""

import numpy as np
import os
import pickle
import datetime
from calendar import monthrange

import torch
from torch.utils.data import Dataset, DataLoader


class AISDatasetWithTimeFeatures(Dataset):
    """Customized Pytorch dataset with time interval and time window features.
    """
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 max_time_interval=3600):  # 最大时间间隔（秒），用于标准化
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length.
            max_time_interval: 最大时间间隔，用于时间间隔特征的标准化
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        self.max_time_interval = max_time_interval
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
    
    def compute_time_interval_feature(self, timestamps):
        """计算时间间隔特征
        
        Args:
            timestamps: numpy array of timestamps
            
        Returns:
            time_intervals: numpy array of normalized time intervals [0,1)
        """
        if len(timestamps) < 2:
            return np.array([0.0])
        
        # 计算相邻点的时间差
        time_diffs = np.diff(timestamps)
        
        # 第一个点的时间间隔设为0
        time_intervals = np.concatenate([[0.0], time_diffs])
        
        # 标准化到[0,1)范围
        time_intervals_norm = np.clip(time_intervals / self.max_time_interval, 0, 0.9999)
        
        return time_intervals_norm
    
    def compute_time_window_feature(self, timestamps):
        """计算时间窗口特征
        
        Args:
            timestamps: numpy array of timestamps
            
        Returns:
            time_windows: numpy array of time window indices (1-based)
        """
        time_windows = []
        
        for timestamp in timestamps:
            # 将时间戳转换为datetime对象
            dt = datetime.datetime.fromtimestamp(timestamp)
            
            # 获取该月的天数
            days_in_month = monthrange(dt.year, dt.month)[1]
            
            # 计算时间窗口编号：(日期-1) * 24 + 小时 + 1
            # 日期从1开始，小时从0开始，所以要+1使编号从1开始
            time_window = (dt.day - 1) * 24 + dt.hour + 1
            
            time_windows.append(time_window)
        
        time_windows = np.array(time_windows)
        
        # 标准化到[0,1)范围（假设最大744个时间窗口：31天*24小时）
        max_time_windows = 744
        time_windows_norm = np.clip((time_windows - 1) / (max_time_windows - 1), 0, 0.9999)
        
        return time_windows_norm
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, 6). Features: [lat,lon,sog,cog,time_interval,time_window]
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"]  # lat, lon, sog, cog, timestamp
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        
        # 计算时间特征
        timestamps = m_v[:, 4]  # 提取时间戳
        time_intervals = self.compute_time_interval_feature(timestamps)
        time_windows = self.compute_time_window_feature(timestamps)
        
        # 组合所有特征：[lat, lon, sog, cog, time_interval, time_window]
        seq = np.zeros((self.max_seqlen, 6))
        seq[:seqlen, :4] = m_v[:seqlen, :4]  # lat, lon, sog, cog
        seq[:seqlen, 4] = time_intervals[:seqlen]    # time_interval
        seq[:seqlen, 5] = time_windows[:seqlen]      # time_window
        
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        
        return seq, mask, seqlen, mmsi, time_start


class AISDataset_grad_with_time(Dataset):
    """Customized Pytorch dataset with gradient and time features.
    Return the positions, gradient of positions, and time features.
    """
    def __init__(self, 
                 l_data, 
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 max_time_interval=3600):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
            max_seqlen: (optional) max sequence length.
            max_time_interval: 最大时间间隔，用于时间间隔特征的标准化
        """    
            
        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device
        self.max_time_interval = max_time_interval
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
    
    def compute_time_interval_feature(self, timestamps):
        """计算时间间隔特征"""
        if len(timestamps) < 2:
            return np.array([0.0])
        
        time_diffs = np.diff(timestamps)
        time_intervals = np.concatenate([[0.0], time_diffs])
        time_intervals_norm = np.clip(time_intervals / self.max_time_interval, 0, 0.9999)
        
        return time_intervals_norm
    
    def compute_time_window_feature(self, timestamps):
        """计算时间窗口特征"""
        time_windows = []
        
        for timestamp in timestamps:
            dt = datetime.datetime.fromtimestamp(timestamp)
            time_window = (dt.day - 1) * 24 + dt.hour + 1
            time_windows.append(time_window)
        
        time_windows = np.array(time_windows)
        max_time_windows = 744
        time_windows_norm = np.clip((time_windows - 1) / (max_time_windows - 1), 0, 0.9999)
        
        return time_windows_norm
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, 6). Features: [lat,lon,dlat,dlon,time_interval,time_window]
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"]  # lat, lon, sog, cog, timestamp
        m_v[m_v==1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        
        # 计算位置梯度
        seq = np.zeros((self.max_seqlen, 6))
        # lat and lon
        seq[:seqlen, :2] = m_v[:seqlen, :2] 
        # dlat and dlon
        dpos = (m_v[1:, :2] - m_v[:-1, :2] + self.dpos_max) / (2 * self.dpos_max)
        dpos = np.concatenate((dpos[:1, :], dpos), axis=0)
        dpos[dpos >= 1] = 0.9999
        dpos[dpos <= 0] = 0.0
        seq[:seqlen, 2:4] = dpos[:seqlen, :2] 
        
        # 计算时间特征
        timestamps = m_v[:, 4]
        time_intervals = self.compute_time_interval_feature(timestamps)
        time_windows = self.compute_time_window_feature(timestamps)
        
        seq[:seqlen, 4] = time_intervals[:seqlen]    # time_interval
        seq[:seqlen, 5] = time_windows[:seqlen]      # time_window
        
        # convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        
        return seq, mask, seqlen, mmsi, time_start 