from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from .config import CLASS_NAMES, ExperimentConfig
from .model import Space2VecTcnMhaClassifier


def segment_predict(model: Space2VecTcnMhaClassifier, loader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_x, batch_y, lengths, ship_ids, file_names in loader:
            batch_x = batch_x.to(device)
            lengths = lengths.to(device)
            logits = model(batch_x, lengths)
            probs = F.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1).tolist()
            for i, file_name in enumerate(file_names):
                rows.append(
                    {
                        "file": file_name,
                        "ship_id": ship_ids[i],
                        "true_label": int(batch_y[i]),
                        "pred_label": int(preds[i]),
                        "confidence": float(probs[i, preds[i]]),
                    }
                )
    return pd.DataFrame(rows)


def ship_level_vote(segment_df: pd.DataFrame) -> tuple[list[int], list[int]]:
    votes = defaultdict(lambda: defaultdict(float))
    labels = {}
    for row in segment_df.itertuples(index=False):
        votes[row.ship_id][row.pred_label] += row.confidence
        labels[row.ship_id] = row.true_label

    y_true, y_pred = [], []
    for ship_id, class_scores in votes.items():
        y_true.append(labels[ship_id])
        y_pred.append(max(class_scores.items(), key=lambda item: item[1])[0])
    return y_true, y_pred


def summarize_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
    }


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    model: Space2VecTcnMhaClassifier,
    loader,
    device: torch.device,
    output_dir: Path | str,
):
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    segment_df = segment_predict(model, loader, device)
    y_true, y_pred = ship_level_vote(segment_df)
    metrics = summarize_metrics(y_true, y_pred)

    segment_df.to_csv(output_dir / "segment_predictions.csv", index=False)
    with open(output_dir / "ship_level_metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            if key != "report":
                f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        f.write(metrics["report"])
    return metrics

