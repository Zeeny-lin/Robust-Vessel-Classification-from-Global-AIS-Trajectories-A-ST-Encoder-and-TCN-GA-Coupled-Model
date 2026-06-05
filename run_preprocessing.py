from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing.preprocess_ais import PreprocessingConfig, run_full_preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Run the six-stage AIS preprocessing pipeline.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw AIS CSV root with optional class subfolders.")
    parser.add_argument("--output-root", type=Path, default=Path("data/preprocessed"), help="Preprocessing output root.")
    parser.add_argument("--time-gap-hours", type=float, default=24.0)
    parser.add_argument("--speed-threshold-kn", type=float, default=30.0)
    parser.add_argument("--interpolation-seconds", type=int, default=4800)
    parser.add_argument("--max-segment-points", type=int, default=300)
    parser.add_argument("--kalman-accuracy-m", type=float, default=1.5)
    parser.add_argument("--sbc-distance-m", type=float, default=1000.0)
    parser.add_argument("--sbc-speed-kn", type=float, default=7.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = PreprocessingConfig(
        time_gap_seconds=int(args.time_gap_hours * 3600),
        speed_threshold_kn=args.speed_threshold_kn,
        interpolation_interval_seconds=args.interpolation_seconds,
        max_segment_points=args.max_segment_points,
        kalman_accuracy_m=args.kalman_accuracy_m,
        sbc_distance_threshold_m=args.sbc_distance_m,
        sbc_speed_threshold_kn=args.sbc_speed_kn,
    )
    final_root = run_full_preprocessing(args.raw_root, args.output_root, config)
    print(f"Final processed data root: {final_root}")


if __name__ == "__main__":
    main()
