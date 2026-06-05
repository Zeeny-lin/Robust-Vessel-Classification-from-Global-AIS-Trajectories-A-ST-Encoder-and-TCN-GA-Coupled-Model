from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ExperimentConfig
from src.evaluate import evaluate_checkpoint
from src.train import build_model, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train Space2Vec-TCN-MHA vessel classifier.")
    parser.add_argument("--data-root", type=Path, default=Path("data/data/process_seg"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tcn_mha"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    _, checkpoints, test_loader, device = train(config)
    if checkpoints:
        model = build_model(config, device)
        metrics = evaluate_checkpoint(checkpoints[0], model, test_loader, device, config.output_dir / "test")
        print(metrics["report"])


if __name__ == "__main__":
    main()

