import torch
import torch.nn as nn
import numpy as np
import math

class GridCellSpatialRelationLocationEncoder(nn.Module):
    """GridCell空间关系位置编码器"""
    
    def __init__(self, 
                 spa_embed_dim=64,
                 coord_dim=2,
                 frequency_num=16,
                 max_radius=10000,
                 min_radius=10,
                 freq_init="geometric",
                 device="cuda",
                 ffn_act="relu",
                 ffn_num_hidden_layers=1,
                 ffn_dropout_rate=0.5,
                 ffn_hidden_dim=256,
                 ffn_use_layernormalize=True,
                 ffn_skip_connection=True,
                 ffn_context_str="GridCellSpatialRelationEncoder"):
        
        super().__init__()
        
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.device = device
        
        # 初始化频率
        if freq_init == "geometric":
            self.frequencies = nn.Parameter(
                torch.logspace(
                    math.log10(min_radius),
                    math.log10(max_radius),
                    frequency_num,
                    device=device
                )
            )
        else:
            self.frequencies = nn.Parameter(
                torch.linspace(min_radius, max_radius, frequency_num, device=device)
            )
        
        # GridCell编码维度：frequency_num * coord_dim * 2 (sin + cos)
        gridcell_dim = frequency_num * coord_dim * 2
        
        # FFN层
        ffn_layers = []
        input_dim = gridcell_dim
        
        for i in range(ffn_num_hidden_layers):
            ffn_layers.append(nn.Linear(input_dim, ffn_hidden_dim))
            
            if ffn_use_layernormalize:
                ffn_layers.append(nn.LayerNorm(ffn_hidden_dim))
            
            if ffn_act == "relu":
                ffn_layers.append(nn.ReLU())
            elif ffn_act == "gelu":
                ffn_layers.append(nn.GELU())
            
            if ffn_dropout_rate > 0:
                ffn_layers.append(nn.Dropout(ffn_dropout_rate))
            
            input_dim = ffn_hidden_dim
        
        # 输出层
        ffn_layers.append(nn.Linear(input_dim, spa_embed_dim))
        
        self.ffn = nn.Sequential(*ffn_layers)
        self.ffn_skip_connection = ffn_skip_connection
        
        # 如果使用跳跃连接，需要投影层
        if ffn_skip_connection and gridcell_dim != spa_embed_dim:
            self.skip_projection = nn.Linear(gridcell_dim, spa_embed_dim)
        else:
            self.skip_projection = None
    
    def forward(self, coords):
        """
        coords: (batch, seq_len, 2) [lat, lon] 或 (batch, 2) 或 numpy array
        return: (batch, seq_len, spa_embed_dim) 或 (batch, spa_embed_dim)
        """
        # 处理输入格式
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float().to(self.device)
        
        if coords.device != self.device:
            coords = coords.to(self.device)
        
        original_shape = coords.shape
        
        # 确保输入是3D: (batch, seq_len, coord_dim)
        if len(coords.shape) == 2:
            coords = coords.unsqueeze(0)  # (1, seq_len, coord_dim)
        
        batch_size, seq_len, coord_dim = coords.shape
        
        # 归一化坐标到[-1, 1]
        lat_norm = coords[:, :, 0:1] / 90.0   # [-90, 90] -> [-1, 1]
        lon_norm = coords[:, :, 1:2] / 180.0  # [-180, 180] -> [-1, 1]
        coords_norm = torch.cat([lat_norm, lon_norm], dim=-1)
        
        # GridCell编码
        embeddings = []
        
        for freq in self.frequencies:
            for coord_idx in range(self.coord_dim):
                coord_values = coords_norm[:, :, coord_idx:coord_idx+1]  # (batch, seq_len, 1)
                
                # 应用频率
                scaled_coords = coord_values * freq
                
                # sin和cos编码
                embeddings.append(torch.sin(scaled_coords))
                embeddings.append(torch.cos(scaled_coords))
        
        # 拼接所有编码
        gridcell_encoding = torch.cat(embeddings, dim=-1)  # (batch, seq_len, freq_num*coord_dim*2)
        
        # 通过FFN
        output = self.ffn(gridcell_encoding)
        
        # 跳跃连接
        if self.ffn_skip_connection:
            if self.skip_projection is not None:
                skip_input = self.skip_projection(gridcell_encoding)
            else:
                skip_input = gridcell_encoding
            output = output + skip_input
        
        # 恢复原始形状
        if len(original_shape) == 2:
            output = output.squeeze(0)
        
        return output

class TransformerWithGridCellEncoding(nn.Module):
    """使用GridCell地理编码的Transformer分类器"""
    
    def __init__(self, 
                 d_model=256, 
                 nhead=8, 
                 num_layers=6, 
                 num_classes=4, 
                 max_seqlen=650,
                 # 时间特征离散化参数
                 max_time_interval=15658,
                 max_time_window=744,
                 time_interval_bins=1000,
                 time_window_bins=744,
                 # GridCell地理编码参数
                 gridcell_spa_embed_dim=64,
                 gridcell_frequency_num=16,
                 gridcell_max_radius=10000,
                 gridcell_min_radius=10,
                 gridcell_ffn_hidden_dim=256):
        
        super().__init__()
        
        self.max_time_interval = max_time_interval
        self.max_time_window = max_time_window
        self.time_interval_bins = time_interval_bins
        self.time_window_bins = time_window_bins
        self.d_model = d_model
        
        # GridCell地理编码器
        self.gridcell_encoder = GridCellSpatialRelationLocationEncoder(
            spa_embed_dim=gridcell_spa_embed_dim,
            coord_dim=2,
            frequency_num=gridcell_frequency_num,
            max_radius=gridcell_max_radius,
            min_radius=gridcell_min_radius,
            freq_init="geometric",
            device="cuda" if torch.cuda.is_available() else "cpu",
            ffn_act="relu",
            ffn_num_hidden_layers=1,
            ffn_dropout_rate=0.1,
            ffn_hidden_dim=gridcell_ffn_hidden_dim,
            ffn_use_layernormalize=True,
            ffn_skip_connection=True,
            ffn_context_str="GridCellSpatialRelationEncoder"
        )
        
        # 其他连续特征投影 (sog, cog)
        self.other_continuous_projection = nn.Linear(2, d_model // 4)
        
        # 时间特征的离散化嵌入
        self.time_interval_embedding = nn.Embedding(time_interval_bins, d_model // 4)
        self.time_window_embedding = nn.Embedding(time_window_bins, d_model // 4)
        
        # GridCell特征投影到模型维度
        self.gridcell_projection = nn.Linear(gridcell_spa_embed_dim, d_model // 4)
        
        # 特征融合层
        self.feature_fusion = nn.Linear(d_model, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(max_seqlen, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        
        # 权重初始化
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
        """将时间特征离散化为索引"""
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
        """
        前向传播
        x: (batch, seq_len, 6) - [lat, lon, sog, cog, time_interval_norm, time_window_norm]
        """
        batch_size, seq_len = x.shape[:2]
        
        # 分离特征
        coords = x[:, :, :2]               # lat, lon
        other_continuous = x[:, :, 2:4]    # sog, cog
        time_interval_norm = x[:, :, 4]    # 归一化的时间间隔
        time_window_norm = x[:, :, 5]      # 归一化的时间窗口
        
        # 1. GridCell地理编码
        gridcell_features = self.gridcell_encoder(coords)  # (batch, seq, gridcell_spa_embed_dim)
        gridcell_emb = self.gridcell_projection(gridcell_features)  # (batch, seq, d_model//4)
        
        # 2. 其他连续特征编码
        other_continuous_emb = self.other_continuous_projection(other_continuous)  # (batch, seq, d_model//4)
        
        # 3. 时间特征离散化和嵌入
        time_intervals = time_interval_norm * self.max_time_interval
        time_windows = time_window_norm * self.max_time_window
        
        time_interval_indices, time_window_indices = self._discretize_time_features(
            time_intervals, time_windows
        )
        
        time_interval_emb = self.time_interval_embedding(time_interval_indices)  # (batch, seq, d_model//4)
        time_window_emb = self.time_window_embedding(time_window_indices)  # (batch, seq, d_model//4)
        
        # 4. 拼接所有特征嵌入
        combined_features = torch.cat([
            gridcell_emb,            # d_model//4 - GridCell地理编码
            other_continuous_emb,    # d_model//4 - sog, cog
            time_interval_emb,       # d_model//4 - 时间间隔
            time_window_emb          # d_model//4 - 时间窗口
        ], dim=-1)  # 总共 d_model 维
        
        # 5. 特征融合
        x = self.feature_fusion(combined_features)
        
        # 6. 添加位置编码
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 7. 创建attention mask
        if mask is not None:
            attn_mask = (mask == 0)
        else:
            attn_mask = None
        
        # 8. Transformer编码
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # 9. 全局平均池化
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        x = self.dropout(x)
        return self.classifier(x)
    
    def get_embedding_info(self):
        """获取嵌入层信息"""
        return {
            'gridcell_spa_embed_dim': self.gridcell_encoder.spa_embed_dim,
            'gridcell_frequency_num': self.gridcell_encoder.frequency_num,
            'gridcell_coord_dim': self.gridcell_encoder.coord_dim,
            'other_continuous_features': 2,
            'other_continuous_emb_dim': self.other_continuous_projection.out_features,
            'time_interval_vocab': self.time_interval_bins,
            'time_interval_emb_dim': self.time_interval_embedding.embedding_dim,
            'time_window_vocab': self.time_window_bins,
            'time_window_emb_dim': self.time_window_embedding.embedding_dim,
            'total_model_dim': self.feature_fusion.out_features
        }