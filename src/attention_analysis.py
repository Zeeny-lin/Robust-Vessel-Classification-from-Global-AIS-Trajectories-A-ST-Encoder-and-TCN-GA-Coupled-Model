from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data_pipeline import FEATURE_COLUMNS, clean_and_extract_features
from .model import Space2VecTcnMhaClassifier


def export_attention_for_csv(
    model: Space2VecTcnMhaClassifier,
    csv_path: Path | str,
    max_seq_len: int,
    device: torch.device,
    output_dir: Path | str,
) -> pd.DataFrame:
    """Paper step 6: export attention weights for interpretability analysis."""

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = clean_and_extract_features(csv_path, max_seq_len)
    if features is None:
        raise ValueError(f"No valid AIS features in {csv_path}")

    x = torch.tensor(features[FEATURE_COLUMNS].values, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([len(features)], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits, attention = model(x, lengths, return_attention_weights=True)
        pred = int(torch.argmax(logits, dim=1).item())

    result = features.copy()
    result["spatial_attention"] = attention["spatial"].squeeze(0).cpu().numpy()[: len(result)]
    result["temporal_attention"] = attention["temporal"].squeeze(0).cpu().numpy()[: len(result)]
    result["cross_attention"] = attention["cross"].squeeze(0).cpu().numpy()[: len(result)]
    result["combined_attention"] = attention["combined"].squeeze(0).cpu().numpy()[: len(result)]
    result["pred_label"] = pred

    stem = csv_path.stem
    result.to_csv(output_dir / f"{stem}_attention.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.index, result["combined_attention"], linewidth=2, label="combined")
    ax.plot(result.index, result["sog"], linewidth=1, alpha=0.6, label="sog")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Attention / SOG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_attention.png", dpi=200)
    plt.close(fig)
    return result

