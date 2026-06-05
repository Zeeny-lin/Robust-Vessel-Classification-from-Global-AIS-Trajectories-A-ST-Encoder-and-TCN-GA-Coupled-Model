from __future__ import annotations

import heapq
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASS_NAMES, ExperimentConfig
from .data_pipeline import ShipTrajectoryDataset, pad_collate_fn
from .model import Space2VecTcnMhaClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(config: ExperimentConfig):
    train_ds = ShipTrajectoryDataset(config.train_dir, config.max_seq_len)
    val_ds = ShipTrajectoryDataset(config.val_dir, config.max_seq_len)
    test_ds = ShipTrajectoryDataset(config.test_dir, config.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    return train_loader, val_loader, test_loader


def l2_regularized_loss(model: nn.Module, logits: torch.Tensor, targets: torch.Tensor, l2_lambda: float):
    loss = F.cross_entropy(logits, targets)
    if l2_lambda <= 0:
        return loss
    l2 = torch.zeros((), device=logits.device)
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            l2 = l2 + torch.norm(param, p=2)
    return loss + l2_lambda * l2


def run_epoch(model, loader, optimizer, device, config: ExperimentConfig, training: bool):
    model.train(training)
    losses, preds, labels = [], [], []

    iterator = tqdm(loader, desc="train" if training else "eval", leave=False)
    for batch_x, batch_y, lengths, _, _ in iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        lengths = lengths.to(device)

        with torch.set_grad_enabled(training):
            logits = model(batch_x, lengths)
            loss = l2_regularized_loss(model, logits, batch_y, config.l2_lambda)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        labels.extend(batch_y.detach().cpu().tolist())

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(labels, preds) if labels else 0.0,
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0) if labels else 0.0,
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0) if labels else 0.0,
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0) if labels else 0.0,
    }
    return metrics


class TopKCheckpoints:
    def __init__(self, save_dir: Path, k: int):
        self.save_dir = save_dir
        self.k = k
        self.heap: list[tuple[float, Path]] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def push(self, model: nn.Module, score: float, epoch: int):
        path = self.save_dir / f"epoch_{epoch:03d}_f1_{score:.4f}.pth"
        torch.save(model.state_dict(), path)
        heapq.heappush(self.heap, (score, path))
        if len(self.heap) > self.k:
            _, old_path = heapq.heappop(self.heap)
            old_path.unlink(missing_ok=True)

    def paths(self) -> list[Path]:
        return [path for _, path in sorted(self.heap, reverse=True)]


def build_model(config: ExperimentConfig, device: torch.device) -> Space2VecTcnMhaClassifier:
    return Space2VecTcnMhaClassifier(
        num_classes=len(CLASS_NAMES),
        spatial_embed_dim=config.spatial_embed_dim,
        temporal_embed_dim=config.temporal_embed_dim,
        tcn_channels=config.tcn_channels,
        kernel_size=config.tcn_kernel_size,
        attention_heads=config.attention_heads,
        dropout=config.dropout,
    ).to(device)


def train(config: ExperimentConfig):
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_loaders(config)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    saver = TopKCheckpoints(config.output_dir / "checkpoints", config.top_k_checkpoints)

    best_f1 = -1.0
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, config, training=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, config, training=False)
        saver.push(model, val_metrics["f1_macro"], epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    return model, saver.paths(), test_loader, device

