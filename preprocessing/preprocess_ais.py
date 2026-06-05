from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from scipy.interpolate import pchip_interpolate
except ImportError:
    def pchip_interpolate(xi, yi, new_x):
        """Fallback interpolation used when SciPy is unavailable."""
        return np.interp(new_x, xi, yi)


CORE_COLUMNS = ["shipno", "lat", "lon", "postime", "sog", "cog"]


@dataclass
class PreprocessingConfig:
    """Parameters for the six-stage AIS preprocessing pipeline."""

    time_gap_seconds: int = 24 * 3600
    speed_threshold_kn: float = 30.0
    sliding_window: int = 10
    interpolation_interval_seconds: int = 4800
    max_segment_points: int = 300
    kalman_accuracy_m: float = 1.5
    sbc_distance_threshold_m: float = 1000.0
    sbc_speed_threshold_kn: float = 7.0
    min_points: int = 2


def normalize_columns(df: pd.DataFrame, fallback_shipno: str | None = None) -> pd.DataFrame:
    """Normalize common AIS field aliases into the paper's core fields."""

    df = df.copy()
    df.columns = [str(col).lower().strip() for col in df.columns]
    aliases = {
        "drmmsi": "shipno",
        "mmsi": "shipno",
        "ship_id": "shipno",
        "latitude": "lat",
        "longitude": "lon",
        "speed": "sog",
        "course": "cog",
        "timestamp": "postime",
        "time": "postime",
    }
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns})
    if "shipno" not in df.columns:
        df["shipno"] = fallback_shipno or "unknown"
    return df


def to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Convert numeric or datetime-like timestamps to Unix seconds."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric.astype("float64")
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.astype("int64").astype("float64") / 1_000_000_000


def extract_ship_id(path: Path | str) -> str:
    return Path(path).stem.split("_", 1)[0]


def prepare_dataframe(df: pd.DataFrame, fallback_shipno: str | None = None) -> pd.DataFrame:
    df = normalize_columns(df, fallback_shipno=fallback_shipno)
    for col in ["lat", "lon", "sog", "cog", "postime"]:
        if col not in df.columns:
            df[col] = np.nan
    df["postime"] = to_epoch_seconds(df["postime"])
    for col in ["lat", "lon", "sog", "cog", "postime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["lat", "lon", "postime"])
    df = df.sort_values("postime").drop_duplicates(subset=["postime"]).reset_index(drop=True)
    if len(df) == 0:
        return df

    df["lat"] = df["lat"].clip(-90, 90)
    df["lon"] = df["lon"].clip(-180, 180)
    df["sog"] = df["sog"].clip(0, 80)
    df["cog"] = df["cog"].mod(360)
    return df


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    radius_m = 6_371_000.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dlambda = math.radians(float(lon2) - float(lon1))
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def segment_by_time(df: pd.DataFrame, gap_seconds: int) -> list[pd.DataFrame]:
    """Stage 1: split trajectories when adjacent AIS timestamps are too far apart."""

    if len(df) < 2:
        return []
    df = df.sort_values("postime").reset_index(drop=True)
    gaps = df["postime"].diff().fillna(0)
    break_indices = np.where(gaps > gap_seconds)[0]

    segments = []
    start = 0
    for idx in break_indices:
        segments.append(df.iloc[start:idx].copy())
        start = idx
    segments.append(df.iloc[start:].copy())
    return [seg.reset_index(drop=True) for seg in segments if len(seg) >= 2]


def statistical_clean(df: pd.DataFrame, speed_threshold_kn: float = 30.0, sliding_window: int = 10) -> pd.DataFrame:
    """Stage 2: remove physically implausible jumps and local coordinate outliers."""

    if len(df) < 3:
        return df

    df = df.sort_values("postime").reset_index(drop=True)
    noise = np.zeros(len(df), dtype=bool)

    for idx in range(len(df) - 1):
        dt_h = (df.at[idx + 1, "postime"] - df.at[idx, "postime"]) / 3600.0
        if dt_h <= 0:
            noise[idx + 1] = True
            continue
        dist_nm = haversine_m(df.at[idx, "lat"], df.at[idx, "lon"], df.at[idx + 1, "lat"], df.at[idx + 1, "lon"]) / 1852.25
        implied_speed = dist_nm / dt_h
        if implied_speed > speed_threshold_kn:
            noise[idx + 1] = True

    df = df.loc[~noise].reset_index(drop=True)
    if len(df) < 3 or sliding_window <= 2:
        return df

    noise = np.zeros(len(df), dtype=bool)
    for start in range(0, len(df), sliding_window):
        end = min(start + sliding_window, len(df))
        window = df.iloc[start:end]
        if len(window) < 3:
            continue
        for col in ["lat", "lon"]:
            std = window[col].std()
            if pd.isna(std) or std == 0:
                continue
            mean = window[col].mean()
            noise[start:end] |= (window[col] - mean).abs().to_numpy() > 1.5 * std
    return df.loc[~noise].reset_index(drop=True)


def fill_missing_motion(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing SOG/COG from adjacent coordinates and timestamps."""

    df = df.sort_values("postime").reset_index(drop=True)
    if len(df) > 1:
        for col in ["sog", "cog"]:
            if pd.isna(df.at[0, col]):
                df.at[0, col] = df.at[1, col]

    for idx in range(1, len(df)):
        dt_h = (df.at[idx, "postime"] - df.at[idx - 1, "postime"]) / 3600.0
        if dt_h <= 0:
            continue
        if pd.isna(df.at[idx, "sog"]):
            dist_nm = haversine_m(df.at[idx - 1, "lat"], df.at[idx - 1, "lon"], df.at[idx, "lat"], df.at[idx, "lon"]) / 1852.25
            df.at[idx, "sog"] = round(dist_nm / dt_h, 1)
        if pd.isna(df.at[idx, "cog"]):
            df.at[idx, "cog"] = round(
                bearing_deg(df.at[idx - 1, "lat"], df.at[idx - 1, "lon"], df.at[idx, "lat"], df.at[idx, "lon"]),
                1,
            )
    return df


def pchip_interpolate_trajectory(df: pd.DataFrame, interval_seconds: int) -> pd.DataFrame | None:
    """Stage 3: resample AIS trajectories with PCHIP and circular COG interpolation."""

    df = fill_missing_motion(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", "sog", "cog", "postime"])
    if len(df) < 2:
        return None

    timestamps = df["postime"].to_numpy(dtype=float)
    xi = timestamps - timestamps[0]
    if xi[-1] < interval_seconds:
        return None
    new_x = np.arange(0, xi[-1] + 1, interval_seconds)

    cog_rad = np.radians(df["cog"].to_numpy(dtype=float))
    cog_x = np.cos(cog_rad)
    cog_y = np.sin(cog_rad)
    new_cog = np.degrees(
        np.arctan2(
            pchip_interpolate(xi, cog_y, new_x),
            pchip_interpolate(xi, cog_x, new_x),
        )
    ) % 360.0

    return pd.DataFrame(
        {
            "shipno": str(df["shipno"].iloc[0]),
            "lat": pchip_interpolate(xi, df["lat"].to_numpy(dtype=float), new_x),
            "lon": pchip_interpolate(xi, df["lon"].to_numpy(dtype=float), new_x),
            "postime": (timestamps[0] + new_x).astype(int),
            "sog": np.round(pchip_interpolate(xi, df["sog"].to_numpy(dtype=float), new_x), 1),
            "cog": np.round(new_cog, 1),
        }
    )


def split_by_length(df: pd.DataFrame, max_points: int) -> list[pd.DataFrame]:
    """Stage 4: split long trajectories into fixed-length modeling segments."""

    if len(df) <= max_points:
        return [df.reset_index(drop=True)]
    pieces = []
    base_shipno = str(df["shipno"].iloc[0])
    for idx, start in enumerate(range(0, len(df), max_points), start=1):
        piece = df.iloc[start : start + max_points].copy().reset_index(drop=True)
        if len(piece) >= 2:
            piece["shipno"] = f"{base_shipno}_{idx}"
            pieces.append(piece)
    return pieces


class KalmanSmoother:
    """Stage 5 helper: lightweight position smoother for latitude and longitude."""

    def __init__(self, accuracy_m: float = 1.5):
        self.accuracy_m = accuracy_m
        self.variance = -1.0
        self.lon = 0.0
        self.lat = 0.0
        self.timestamp = 0.0

    def process(self, lat: float, lon: float, timestamp: float, speed_kn: float) -> tuple[float, float]:
        if self.variance < 0:
            self.lat = float(lat)
            self.lon = float(lon)
            self.timestamp = float(timestamp)
            self.variance = self.accuracy_m**2
            return self.lat, self.lon

        duration = max(float(timestamp) - self.timestamp, 0.0)
        speed_mps = max(float(speed_kn), 0.0) * 1852.25 / 3600.0
        self.variance += duration * speed_mps * speed_mps / 1000.0
        self.timestamp = float(timestamp)
        k_gain = self.variance / (self.variance + self.accuracy_m**2)
        self.lat += k_gain * (float(lat) - self.lat)
        self.lon += k_gain * (float(lon) - self.lon)
        self.variance = (1.0 - k_gain) * self.variance
        return self.lat, self.lon


def kalman_smooth(df: pd.DataFrame, accuracy_m: float = 1.5) -> pd.DataFrame:
    smoother = KalmanSmoother(accuracy_m=accuracy_m)
    smoothed = df.copy()
    coords = [
        smoother.process(row.lat, row.lon, row.postime, row.sog if not pd.isna(row.sog) else 0.0)
        for row in df.itertuples(index=False)
    ]
    smoothed["lat"] = [lat for lat, _ in coords]
    smoothed["lon"] = [lon for _, lon in coords]
    return smoothed


def speed_based_compress(df: pd.DataFrame, distance_threshold_m: float = 1000.0, speed_threshold_kn: float = 7.0) -> pd.DataFrame:
    """Stage 6: speed and distance based compression while preserving behavior-critical points."""

    if len(df) <= 2:
        return df

    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    ts = df["postime"].to_numpy(dtype=float)
    speed = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        dist_m = haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
        speed[i] = dist_m / max(ts[i] - ts[i - 1], 1.0) * 3600.0 / 1852.25

    keep = [0]
    start = 0
    end = 1
    while end < len(df):
        for i in range(start + 1, end):
            delta_end = max(ts[end] - ts[start], 1.0)
            delta_i = max(ts[i] - ts[start], 1.0)
            ratio = delta_i / delta_end
            pred_lat = lat[start] + (lat[end] - lat[start]) * ratio
            pred_lon = lon[start] + (lon[end] - lon[start]) * ratio
            dist_error = haversine_m(lat[i], lon[i], pred_lat, pred_lon)
            v_prev = speed[i]
            v_next = speed[i + 1] if i + 1 < len(df) else v_prev
            if dist_error > distance_threshold_m or abs(v_prev - v_next) > speed_threshold_kn:
                keep.append(i)
                start = i
                end = i + 1
                break
        else:
            end += 1

    keep.append(len(df) - 1)
    keep = sorted(set(keep))
    return df.iloc[keep].reset_index(drop=True)


def iter_csv_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def write_outputs(outputs: list[pd.DataFrame], src: Path, src_root: Path, dst_root: Path) -> int:
    rel = src.relative_to(src_root)
    out_dir = dst_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = src.stem
    written = 0
    for idx, df in enumerate(outputs, start=1):
        if len(df) < 2:
            continue
        out_name = f"{base}_{idx}.csv" if len(outputs) > 1 else f"{base}.csv"
        df.to_csv(out_dir / out_name, index=False)
        written += 1
    return written


def process_tree(
    src_root: Path | str,
    dst_root: Path | str,
    transform: Callable[[pd.DataFrame, Path], list[pd.DataFrame]],
    stage_name: str,
) -> None:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    files = iter_csv_files(src_root)
    for src in tqdm(files, desc=stage_name):
        try:
            raw = pd.read_csv(src)
            df = prepare_dataframe(raw, fallback_shipno=src.stem)
            outputs = transform(df, src)
            write_outputs(outputs, src, src_root, dst_root)
        except Exception as exc:
            print(f"[WARN] Skipped {src}: {exc}")


def collect_stats(root: Path | str) -> dict[str, int]:
    root = Path(root)
    files = iter_csv_files(root)
    points = 0
    ships = set()
    for path in files:
        try:
            points += len(pd.read_csv(path))
            ships.add(extract_ship_id(path))
        except Exception:
            continue
    return {"ship_count": len(ships), "trajectory_count": len(files), "point_count": points}


def run_full_preprocessing(raw_root: Path | str, output_root: Path | str, config: PreprocessingConfig) -> Path:
    """Run the paper-aligned six-stage pipeline and return the final data root."""

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    stage1 = output_root / "01_segmented"
    stage2 = output_root / "02_cleaned"
    stage3 = output_root / "03_interpolated"
    stage4 = output_root / "04_split"
    stage5 = output_root / "05_smoothed"
    stage6 = output_root / "06_compressed"

    process_tree(
        raw_root,
        stage1,
        lambda df, src: segment_by_time(df, config.time_gap_seconds),
        "Stage 1/6 segmentation",
    )
    process_tree(
        stage1,
        stage2,
        lambda df, src: [statistical_clean(df, config.speed_threshold_kn, config.sliding_window)],
        "Stage 2/6 cleaning",
    )
    process_tree(
        stage2,
        stage3,
        lambda df, src: [out] if (out := pchip_interpolate_trajectory(df, config.interpolation_interval_seconds)) is not None else [],
        "Stage 3/6 interpolation",
    )
    process_tree(
        stage3,
        stage4,
        lambda df, src: split_by_length(df, config.max_segment_points),
        "Stage 4/6 length splitting",
    )
    process_tree(
        stage4,
        stage5,
        lambda df, src: [kalman_smooth(df, config.kalman_accuracy_m)],
        "Stage 5/6 Kalman smoothing",
    )
    process_tree(
        stage5,
        stage6,
        lambda df, src: [speed_based_compress(df, config.sbc_distance_threshold_m, config.sbc_speed_threshold_kn)],
        "Stage 6/6 SBC compression",
    )

    print("Final preprocessing statistics:", collect_stats(stage6))
    return stage6
