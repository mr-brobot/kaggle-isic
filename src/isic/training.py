"""
Training utilities for ISIC 2024 models
"""

from typing import Dict

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.train()
    total_loss = 0

    # init metrics
    accuracy_metric = BinaryAccuracy(threshold=threshold, device=device)
    precision_metric = BinaryPrecision(threshold=threshold, device=device)
    recall_metric = BinaryRecall(threshold=threshold, device=device)
    f1_metric = BinaryF1Score(threshold=threshold, device=device)

    for batch_idx, (x_img, x_md, targets) in enumerate(dataloader):
        x_img, x_md, targets = x_img.to(device), x_md.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(x_img, x_md).squeeze()
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        # update metrics
        accuracy_metric.update(probs, targets)
        precision_metric.update(probs, targets)
        recall_metric.update(probs, targets.int())  # https://github.com/pytorch/torcheval/issues/209 # fmt: skip
        f1_metric.update(probs, targets)

        if batch_idx % 100 == 0:
            current_precision = precision_metric.compute().item()
            current_recall = recall_metric.compute().item()
            print(
                f"Batch {batch_idx:3d}/{len(dataloader)}: Loss: {loss.item():.4f} | Precision: {current_precision:.3f} | Recall: {current_recall:.3f}"
            )

        mlflow.log_metric("batch_loss", loss.item())

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_metric.compute().item(),
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "f1": f1_metric.compute().item(),
        "specificity": 0.0,  # TODO: implement
    }

    mlflow.log_metrics(
        {
            "train_loss": metrics["loss"],
            "train_accuracy": metrics["accuracy"],
            "train_precision": metrics["precision"],
            "train_recall": metrics["recall"],
            "train_f1": metrics["f1"],
        }
    )

    return metrics


@torch.no_grad
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    accuracy_metric = BinaryAccuracy(threshold=threshold, device=device)
    precision_metric = BinaryPrecision(threshold=threshold, device=device)
    recall_metric = BinaryRecall(threshold=threshold, device=device)
    f1_metric = BinaryF1Score(threshold=threshold, device=device)
    auroc_metric = BinaryAUROC(device=device)

    for x_img, x_md, targets in dataloader:
        x_img, x_md, targets = x_img.to(device), x_md.to(device), targets.to(device)
        logits = model(x_img, x_md).squeeze()
        loss = criterion(logits, targets)

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).float()

        # update metrics
        accuracy_metric.update(probs, targets)
        precision_metric.update(probs, targets)
        recall_metric.update(probs, targets.int())  # https://github.com/pytorch/torcheval/issues/209 # fmt: skip
        f1_metric.update(probs, targets)
        auroc_metric.update(probs, targets)

        # TODO: remove these arrays arrays
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_metric.compute().item(),
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "f1": f1_metric.compute().item(),
        "specificity": 0.0,  # TODO: implement
        "roc_auc": auroc_metric.compute().item(),
    }

    mlflow.log_metrics(
        {
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_roc_auc": metrics["roc_auc"],
        }
    )

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    return metrics, all_targets, all_predictions


def calculate_pos_weight(train_loader: DataLoader) -> float:
    """
    Calculate positive class weight for handling class imbalance.

    Args:
        train_loader: Training data loader

    Returns:
        Positive class weight
    """
    pos_count = 0
    neg_count = 0

    for _, _, targets in train_loader:
        pos_count += targets.sum().item()
        neg_count += (1 - targets).sum().item()

    pos_weight = neg_count / pos_count
    print(f"Class counts - Negative: {neg_count}, Positive: {pos_count}")
    print(f"Calculated positive weight: {pos_weight:.2f}")

    return pos_weight
