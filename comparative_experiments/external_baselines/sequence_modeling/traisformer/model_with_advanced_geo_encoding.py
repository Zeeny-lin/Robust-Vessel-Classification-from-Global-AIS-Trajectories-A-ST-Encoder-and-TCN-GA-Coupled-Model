#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强化版地理编码 + 精度提升策略的完整模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TheorySpace2VecEncoder(nn.Module):
    """基于Space2Vec-theory的地理编码器"""
    
    def __init__(self, frequency_num=32, max_radius=5221082, min_radius=10, coord_dim=2):
        super().__init__()
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.coord_dim = coord_dim
        
        # 几何级数初始化频率
        self.frequencies = nn.Parameter(
            torch.logspace(
                math.log10(min_radius), 
                math.log10(max_radius), 
                frequency_num
            )
        )
        
        # 三个单位向量：0°, 120°, 240°
        angles = [0, 2*math.pi/3, 4*math.pi/3]
        self.unit_vectors = nn.Parameter(
            torch.tensor([[math.cos(angle), math.sin(angle)] for angle in angles], 
                        dtype=torch.float32),
            requires_grad=False
        )
        
    def forward(self, coords):
        """
        coords: (batch, seq_len, 2) [lat, lon] 
        return: (batch, seq_len, frequency_num * 3 * 2)
        """
        batch_size, seq_len, _ = coords.shape
        
        # 归一化坐标
        lat_norm = coords[:, :, 0:1] / 90.0
        lon_norm = coords[:, :, 1:2] / 180.0
        coords_norm = torch.cat([lat_norm, lon_norm], dim=-1)
        
        embeddings = []
        
        for freq_idx in range(self.frequency_num):
            freq = self.frequencies[freq_idx]
            
            for unit_vec in self.unit_vectors:
                dot_product = torch.sum(coords_norm * unit_vec.unsqueeze(0).unsqueeze(0), dim=-1, keepdim=True)
                angle = dot_product * freq
                
                embeddings.append(torch.sin(angle))
                embeddings.append(torch.cos(angle))
        
        space_embedding = torch.cat(embeddings, dim=-1)
        return space_embedding

class AdvancedGeoEncoder(nn.Module):
    """高级地理编码器 - 卷积 + 注意力 + 残差"""
    
    def __init__(self, space2vec_dim, d_model, freq_num):
        super().__init__()
        self.freq_num = freq_num
        self.space2vec_dim = space2vec_dim
        
        # 多尺度卷积分支
        self.conv_branch1 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.conv_branch2 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.conv_branch3 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # 特征融合
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # 频率维度自注意力
        self.freq_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 位置编码（频率维度）
        self.freq_pos_encoding = nn.Parameter(torch.randn(freq_num, 128))
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 最终投影
        self.projection = nn.Sequential(
            nn.Linear(256, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, space2vec_features):
        batch_size, seq_len, _ = space2vec_features.shape
        
        # 重塑为 (batch*seq_len, 6, freq_num)
        x = space2vec_features.view(batch_size * seq_len, self.freq_num, 6)
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        conv1 = self.conv_branch1(x)
        conv2 = self.conv_branch2(x)
        conv3 = self.conv_branch3(x)
        
        # 拼接多尺度特征
        x = torch.cat([conv1, conv2, conv3], dim=1)
        x = self.conv_fusion(x)
        
        # 转换为注意力格式
        x = x.transpose(1, 2)  # (batch*seq_len, freq_num, 128)
        
        # 添加位置编码
        x = x + self.freq_pos_encoding.unsqueeze(0)
        
        # 自注意力
        attended_x, _ = self.freq_attention(x, x, x)
        x = x + attended_x  # 残差连接
        
        # 全局池化
        x = x.transpose(1, 2)
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=-1)
        
        # 投影
        x = self.projection(x)
        
        # 重塑回原始形状
        x = x.view(batch_size, seq_len, -1)
        
        return x

class SimplifiedCrossModalFusion(nn.Module):
    """简化的跨模态特征融合模块"""
    
    def __init__(self, d_model):
        super().__init__()
        
        # 简单的特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, geo_emb, time_emb, other_emb):
        # 直接拼接并融合
        all_features = torch.cat([geo_emb, time_emb, other_emb], dim=-1)
        return self.fusion(all_features)

class SimplifiedTransformerLayer(nn.Module):
    """简化的Transformer层"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, src_key_padding_mask=None):
        # 自注意力 + 残差
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + attn_out)
        
        # 前馈网络 + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class TransformerWithAdvancedGeoEncoding(nn.Module):
    """内存优化版Transformer - 高级地理编码"""
    
    def __init__(self, 
                 d_model=512,  # 从768减少到512
                 nhead=8,     # 从12减少到8
                 num_layers=4, # 从8减少到4
                 num_classes=4, 
                 max_seqlen=650,
                 max_time_interval=15658,
                 max_time_window=744,
                 time_interval_bins=1000,  # 从2000减少到1000
                 time_window_bins=744,     # 从1488减少到744
                 space2vec_freq_num=16,    # 从32减少到16
                 space2vec_max_radius=5221082,
                 space2vec_min_radius=10,
                 dropout=0.15):
        
        super().__init__()
        
        self.max_time_interval = max_time_interval
        self.max_time_window = max_time_window
        self.time_interval_bins = time_interval_bins
        self.time_window_bins = time_window_bins
        self.d_model = d_model
        
        # Theory Space2Vec地理编码器 - 减少频率数量
        self.space2vec_encoder = TheorySpace2VecEncoder(
            frequency_num=space2vec_freq_num,
            max_radius=space2vec_max_radius,
            min_radius=space2vec_min_radius,
            coord_dim=2
        )
        space2vec_dim = space2vec_freq_num * 3 * 2
        
        # 简化的地理编码器
        self.geo_encoder = nn.Sequential(
            nn.Linear(space2vec_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 简化的时间特征编码
        self.time_interval_embedding = nn.Embedding(time_interval_bins, 64)  # 从128减少到64
        self.time_window_embedding = nn.Embedding(time_window_bins, 64)
        self.time_fusion = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # 简化的其他特征编码
        self.other_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 简化的跨模态融合
        self.cross_modal_fusion = SimplifiedCrossModalFusion(d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(max_seqlen, d_model))
        
        # 简化的Transformer编码器
        self.transformer_layers = nn.ModuleList([
            SimplifiedTransformerLayer(d_model, nhead, d_model * 2, dropout)  # FFN维度减半
            for _ in range(num_layers)
        ])
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _discretize_time_features(self, time_intervals, time_windows):
        """时间特征离散化"""
        time_interval_indices = torch.clamp(
            (time_intervals * self.time_interval_bins / self.max_time_interval).long(),
            0, self.time_interval_bins - 1
        )
        
        time_window_indices = torch.clamp(
            (time_windows * self.time_window_bins / self.max_time_window).long(),
            0, self.time_window_bins - 1
        )
        
        return time_interval_indices, time_window_indices
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[:2]
        
        # 分离特征
        coords = x[:, :, :2]
        other_continuous = x[:, :, 2:4]
        time_interval_norm = x[:, :, 4]
        time_window_norm = x[:, :, 5]
        
        # 1. 高级地理编码
        space2vec_features = self.space2vec_encoder(coords)
        geo_emb = self.geo_encoder(space2vec_features)  # 修改：使用geo_encoder而不是advanced_geo_encoder
        
        # 2. 深度时间编码
        time_intervals = time_interval_norm * self.max_time_interval
        time_windows = time_window_norm * self.max_time_window
        
        time_interval_indices, time_window_indices = self._discretize_time_features(
            time_intervals, time_windows
        )
        
        time_int_emb = self.time_interval_embedding(time_interval_indices)
        time_win_emb = self.time_window_embedding(time_window_indices)
        time_combined = torch.cat([time_int_emb, time_win_emb], dim=-1)
        time_emb = self.time_fusion(time_combined)
        
        # 3. 深度其他特征编码
        other_emb = self.other_encoder(other_continuous)
        
        # 4. 跨模态融合
        x = self.cross_modal_fusion(geo_emb, time_emb, other_emb)
        
        # 5. 位置编码
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 6. 增强Transformer编码
        attn_mask = (mask == 0) if mask is not None else None
        
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=attn_mask)
        
        # 7. 全局池化
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        # 8. 分类
        logits = self.classifier(x)
        
        return logits




