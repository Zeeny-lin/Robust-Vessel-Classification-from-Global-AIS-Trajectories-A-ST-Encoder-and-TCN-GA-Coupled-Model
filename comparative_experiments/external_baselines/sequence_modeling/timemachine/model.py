import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiScaleFeatureProjection(nn.Module):
    """多尺度特征投影层 - 可学习的升维"""

    def __init__(self, input_dim, d_model, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.d_model = d_model

        # 多尺度卷积投影
        self.conv_projections = nn.ModuleList([
            nn.Conv1d(input_dim, d_model // num_scales,
                      kernel_size=2 * i + 1, padding=i)
            for i in range(num_scales)
        ])

        # 全连接投影作为补充
        self.linear_projection = nn.Linear(input_dim, d_model // num_scales)

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // num_scales, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape

        # 多尺度卷积特征
        x_conv = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        conv_features = []

        for conv_proj in self.conv_projections:
            conv_out = conv_proj(x_conv)  # [batch_size, d_model//num_scales, seq_len]
            conv_features.append(conv_out.transpose(1, 2))  # [batch_size, seq_len, d_model//num_scales]

        # 线性投影特征
        linear_features = self.linear_projection(x)  # [batch_size, seq_len, d_model//num_scales]

        # 拼接所有特征
        all_features = torch.cat(conv_features + [linear_features], dim=-1)

        # 融合
        output = self.feature_fusion(all_features)

        return output


class CrossModalAttentionFusion(nn.Module):
    """跨模态注意力融合 - 改进的特征融合"""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # 门控融合网络
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, motion_features, geo_features):
        # motion_features, geo_features: [batch_size, seq_len, d_model]

        # 1. 自注意力增强各自特征
        motion_enhanced, _ = self.self_attention(
            motion_features, motion_features, motion_features
        )
        geo_enhanced, _ = self.self_attention(
            geo_features, geo_features, geo_features
        )

        # 2. 交叉注意力学习跨模态关系
        # 让运动特征关注地理特征
        motion_cross, motion_attn_weights = self.cross_attention(
            motion_enhanced, geo_enhanced, geo_enhanced
        )

        # 让地理特征关注运动特征
        geo_cross, geo_attn_weights = self.cross_attention(
            geo_enhanced, motion_enhanced, motion_enhanced
        )

        # 3. 门控融合机制
        # 计算每个模态的重要性门控
        combined_features = torch.cat([motion_cross, geo_cross], dim=-1)
        gate = self.gate_network(combined_features)

        # 加权融合
        fused_features = gate * motion_cross + (1 - gate) * geo_cross

        # 4. 特征变换和残差连接
        enhanced_features = self.feature_transform(
            torch.cat([motion_cross, geo_cross], dim=-1)
        )

        # 残差连接
        output = fused_features + enhanced_features
        output = self.dropout(output)

        return output, (motion_attn_weights, geo_attn_weights)


class AdaptivePooling(nn.Module):
    """自适应池化层"""

    def __init__(self, d_model, pooling_types=['attention', 'avg', 'max']):
        super().__init__()
        self.pooling_types = pooling_types
        self.d_model = d_model

        # 注意力池化
        if 'attention' in pooling_types:
            self.attention_pool = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=0.1, batch_first=True
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 池化权重学习
        self.pool_weights = nn.Parameter(torch.ones(len(pooling_types)) / len(pooling_types))

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * len(pooling_types), d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size = x.size(0)
        pooled_features = []

        for pool_type in self.pooling_types:
            if pool_type == 'attention':
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x_with_cls = torch.cat([cls_tokens, x], dim=1)
                attn_out, _ = self.attention_pool(cls_tokens, x_with_cls, x_with_cls)
                pooled = attn_out.squeeze(1)

            elif pool_type == 'avg':
                pooled = x.mean(dim=1)

            elif pool_type == 'max':
                pooled, _ = x.max(dim=1)

            elif pool_type == 'last':
                pooled = x[:, -1, :]

            pooled_features.append(pooled)

        # 加权组合不同池化结果
        weights = F.softmax(self.pool_weights, dim=0)
        weighted_features = []
        for i, feature in enumerate(pooled_features):
            weighted_features.append(weights[i] * feature)

        # 拼接所有池化特征
        combined = torch.cat(pooled_features, dim=-1)
        output = self.output_projection(combined)

        return output


class ShipClassificationModel(nn.Module):
    """改进的基于Mamba的船舶轨迹分类模型"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.use_geo_encoding = getattr(configs, 'use_geo_encoding', True)

        # RevIN标准化层
        if self.configs.revin == 1:
            self.revin_layer = RevIN(self.configs.n_features)

        # 改进的多尺度特征投影
        self.motion_projection = MultiScaleFeatureProjection(
            self.configs.n_features, self.configs.d_model
        )

        # 位置编码
        self.pos_encoding = LearnablePositionalEncoding(
            self.configs.d_model, max_len=getattr(configs, 'seq_len', 640)
        )

        # 地理编码投影层
        if self.use_geo_encoding:
            self.geo_embed_dim = getattr(configs, 'spa_embed_dim', 64)
            self.geo_projection = MultiScaleFeatureProjection(
                self.geo_embed_dim, self.configs.d_model
            )

            # 跨模态注意力融合
            self.cross_modal_fusion = CrossModalAttentionFusion(
                self.configs.d_model,
                num_heads=getattr(configs, 'fusion_heads', 8),
                dropout=self.configs.dropout
            )

        # Mamba层（保持原有结构）
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.configs.d_model,
                d_state=self.configs.d_state,
                d_conv=self.configs.dconv,
                expand=self.configs.e_fact
            ) for _ in range(self.configs.n_layers)
        ])

        # 层标准化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.configs.d_model)
            for _ in range(self.configs.n_layers)
        ])

        # 改进的自适应池化
        self.adaptive_pooling = AdaptivePooling(
            self.configs.d_model,
            pooling_types=['attention', 'avg', 'max']
        )

        # 改进的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(self.configs.dropout),
            nn.Linear(self.configs.d_model, self.configs.d_model),
            nn.LayerNorm(self.configs.d_model),
            nn.ReLU(),
            nn.Dropout(self.configs.dropout),
            nn.Linear(self.configs.d_model, self.configs.d_model // 2),
            nn.LayerNorm(self.configs.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.configs.dropout),
            nn.Linear(self.configs.d_model // 2, self.configs.num_classes)
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, nn.Linear):
            # Xavier初始化
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, geo_embedding=None):
        """
        改进的前向传播
        Args:
            x: [batch_size, seq_len, n_features] 输入序列
            geo_embedding: [batch_size, seq_len, geo_embed_dim] 地理编码
        Returns:
            logits: [batch_size, num_classes] 分类logits
            attention_weights: 注意力权重（可选）
        """
        batch_size, seq_len, n_features = x.shape

        # RevIN标准化
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(dim=1, keepdim=True)
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # 多尺度特征投影
        motion_features = self.motion_projection(x)

        # 添加位置编码
        motion_features = self.pos_encoding(motion_features)

        attention_weights = None
        if self.use_geo_encoding and geo_embedding is not None:
            # 地理特征投影
            geo_features = self.geo_projection(geo_embedding)
            geo_features = self.pos_encoding(geo_features)

            # 跨模态注意力融合
            fused_features, attention_weights = self.cross_modal_fusion(
                motion_features, geo_features
            )
            x = fused_features
        else:
            x = motion_features

        # Mamba层处理
        for i, (mamba_layer, layer_norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            residual = x
            x = mamba_layer(x)

            if self.configs.residual == 1:
                x = layer_norm(x + residual)
            else:
                x = layer_norm(x)

        # 自适应池化
        pooled = self.adaptive_pooling(x)

        # 分类
        logits = self.classifier(pooled)

        if attention_weights is not None:
            return logits
            # return logits, attention_weights
        else:
            return logits