from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN residual block with dilated temporal convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + residual)


class TemporalConvNet(nn.Module):
    """Paper step 3: long-sequence dependency mining through dilated TCN."""

    def __init__(self, input_dim: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(channels):
            in_channels = input_dim if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Space2VecEncoder(nn.Module):
    """Paper step 2: multi-scale geographic encoding for latitude/longitude."""

    def __init__(
        self,
        coord_dim: int = 2,
        frequency_num: int = 32,
        max_radius: float = 10000.0,
        min_radius: float = 10.0,
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        freq = torch.exp(torch.linspace(np.log(2 * np.pi / min_radius), np.log(2 * np.pi / max_radius), frequency_num))
        self.register_buffer("freq_bands", freq)
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim * frequency_num * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        coords = coords.clamp(min=-180.0, max=180.0)
        arg = coords.unsqueeze(-1) * self.freq_bands
        enc = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return self.mlp(enc.flatten(start_dim=2))


class MultiHeadSelfAttention(nn.Module):
    """True multi-head self-attention with residual connection and layer norm."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, return_weights: bool = False):
        out, weights = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=False,
        )
        out = self.norm(x + self.dropout(out))
        if not return_weights:
            return out, None
        token_weights = weights.mean(dim=1).mean(dim=1)
        if key_padding_mask is not None:
            token_weights = token_weights.masked_fill(key_padding_mask, 0.0)
        return out, token_weights


class CrossAttention(nn.Module):
    """Spatial queries attend to temporal keys/values for spatiotemporal fusion."""

    def __init__(self, spatial_dim: int, temporal_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            spatial_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=temporal_dim,
            vdim=temporal_dim,
        )
        self.norm = nn.LayerNorm(spatial_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ):
        out, weights = self.attn(
            spatial_feat,
            temporal_feat,
            temporal_feat,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=False,
        )
        out = self.norm(spatial_feat + self.dropout(out))
        if not return_weights:
            return out, None
        token_weights = weights.mean(dim=1).mean(dim=1)
        if key_padding_mask is not None:
            token_weights = token_weights.masked_fill(key_padding_mask, 0.0)
        return out, token_weights


class Space2VecTcnMhaClassifier(nn.Module):
    """Main model: Space2Vec + temporal encoder + separate TCNs + MHA + classifier."""

    def __init__(
        self,
        num_classes: int,
        spatial_embed_dim: int = 64,
        temporal_embed_dim: int = 32,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [128, 128]

        hidden_dim = tcn_channels[-1]
        self.space2vec = Space2VecEncoder(output_dim=spatial_embed_dim)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(4, temporal_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_tcn = TemporalConvNet(spatial_embed_dim, tcn_channels, kernel_size, dropout)
        self.temporal_tcn = TemporalConvNet(temporal_embed_dim, tcn_channels, kernel_size, dropout)

        self.spatial_attention = MultiHeadSelfAttention(hidden_dim, attention_heads, dropout)
        self.temporal_attention = MultiHeadSelfAttention(hidden_dim, attention_heads, dropout)
        self.cross_attention = CrossAttention(hidden_dim, hidden_dim, attention_heads, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    @staticmethod
    def _padding_mask(lengths: torch.Tensor | None, seq_len: int, device: torch.device):
        if lengths is None:
            return None
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return positions >= lengths.to(device).unsqueeze(1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_attention_weights: bool = False):
        spatial_raw = x[:, :, 0:2]
        temporal_raw = x[:, :, 2:6]
        mask = self._padding_mask(lengths, x.size(1), x.device)

        spatial = self.space2vec(spatial_raw)
        temporal = self.temporal_encoder(temporal_raw)

        spatial = self.spatial_tcn(spatial.transpose(1, 2)).transpose(1, 2)
        temporal = self.temporal_tcn(temporal.transpose(1, 2)).transpose(1, 2)

        spatial, spatial_weights = self.spatial_attention(spatial, mask, return_attention_weights)
        temporal, temporal_weights = self.temporal_attention(temporal, mask, return_attention_weights)
        fused, cross_weights = self.cross_attention(spatial, temporal, mask, return_attention_weights)

        pooled = self._masked_mean(fused, mask)
        logits = self.classifier(pooled)

        if not return_attention_weights:
            return logits
        attention = {
            "spatial": spatial_weights,
            "temporal": temporal_weights,
            "cross": cross_weights,
        }
        attention["combined"] = torch.stack([v for v in attention.values() if v is not None]).mean(dim=0)
        return logits, attention

