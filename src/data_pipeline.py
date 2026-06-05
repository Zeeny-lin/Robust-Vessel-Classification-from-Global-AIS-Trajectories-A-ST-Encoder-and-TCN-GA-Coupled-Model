from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CLASS_MAP


FEATURE_COLUMNS = ["lat", "lon", "sog", "cog", "delta_h", "day_frac"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lower().strip() for col in df.columns]
    aliases = {
        "latitude": "lat",
        "longitude": "lon",
        "speed": "sog",
        "course": "cog",
        "timestamp": "postime",
        "time": "postime",
    }
    return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})


def _build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "postime" not in df.columns:
        return df

    numeric_time = pd.to_numeric(df["postime"], errors="coerce")
    if numeric_time.notna().mean() > 0.8:
        postime = pd.to_datetime(numeric_time, unit="s", errors="coerce")
    else:
        postime = pd.to_datetime(df["postime"], errors="coerce")

    df = df.assign(postime=postime).sort_values("postime").reset_index(drop=True)
    delta_h = df["postime"].diff().dt.total_seconds().div(3600.0)
    day_frac = (
        df["postime"].dt.hour * 3600
        + df["postime"].dt.minute * 60
        + df["postime"].dt.second
    ) / 86400.0

    if "delta_h" not in df.columns:
        df["delta_h"] = delta_h.fillna(0.0)
    if "day_frac" not in df.columns:
        df["day_frac"] = day_frac.fillna(0.0)
    return df


def clean_and_extract_features(csv_path: Path, max_seq_len: int) -> pd.DataFrame | None:
    """Paper step 1: trajectory cleaning and standardized feature extraction."""

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    df = _normalize_columns(df)
    df = _build_temporal_features(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", limit_direction="both").fillna(0.0)

    df["lat"] = df["lat"].clip(-90, 90)
    df["lon"] = df["lon"].clip(-180, 180)
    df["sog"] = df["sog"].clip(0, 60)
    df["cog"] = df["cog"].mod(360)
    df["delta_h"] = df["delta_h"].clip(0, 72)
    df["day_frac"] = df["day_frac"].clip(0, 1)

    if len(df) == 0:
        return None
    return df.iloc[:max_seq_len].reset_index(drop=True)


class ShipTrajectoryDataset(Dataset):
    """Loads class-folder CSV trajectories for segment-level training."""

    def __init__(self, data_dir: Path | str, max_seq_len: int):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[Path, int, str]] = []

        for class_name, label in CLASS_MAP.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for csv_path in sorted(class_dir.glob("*.csv")):
                ship_id = csv_path.stem.rsplit("_", 1)[0]
                self.samples.append((csv_path, label, ship_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        csv_path, label, ship_id = self.samples[index]
        features = clean_and_extract_features(csv_path, self.max_seq_len)
        if features is None:
            features = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)

        values = torch.tensor(features.values, dtype=torch.float32)
        length = torch.tensor(min(len(values), self.max_seq_len), dtype=torch.long)
        return values, torch.tensor(label, dtype=torch.long), length, ship_id, csv_path.name


def pad_collate_fn(batch):
    sequences, labels, lengths, ship_ids, file_names = zip(*batch)
    max_len = max(seq.size(0) for seq in sequences)
    feature_dim = sequences[0].size(1)

    padded = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq

    return (
        padded,
        torch.stack(labels),
        torch.stack(lengths),
        list(ship_ids),
        list(file_names),
    )

