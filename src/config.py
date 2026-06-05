from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


CLASS_NAMES = ["Bulk Carrier", "Container Ship", "Fishing", "Oil Tanker"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class ExperimentConfig:
    """Configuration shared by data, model, training, and evaluation steps."""

    data_root: Path = Path("data/data/process_seg")
    output_dir: Path = Path("runs/tcn_mha")

    max_seq_len: int = 300
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    l2_lambda: float = 1e-5
    patience: int = 8
    top_k_checkpoints: int = 3
    seed: int = 42

    spatial_embed_dim: int = 64
    temporal_embed_dim: int = 32
    tcn_channels: list[int] = field(default_factory=lambda: [128, 128])
    tcn_kernel_size: int = 3
    attention_heads: int = 4
    dropout: float = 0.1

    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    @property
    def train_dir(self) -> Path:
        return self.data_root / self.train_split

    @property
    def val_dir(self) -> Path:
        return self.data_root / self.val_split

    @property
    def test_dir(self) -> Path:
        return self.data_root / self.test_split

