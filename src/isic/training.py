"""
Training utilities for ISIC 2024 models
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import trackio
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
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
    total_loss = torch.tensor(0.0, device=device)

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

        total_loss += loss.detach()

        probs = torch.sigmoid(logits)

        # update metrics
        accuracy_metric.update(probs, targets)
        precision_metric.update(probs, targets)
        recall_metric.update(probs, targets.int())  # https://github.com/pytorch/torcheval/issues/209 # fmt: skip
        f1_metric.update(probs, targets)

        if batch_idx % 100 == 0:
            current_precision = precision_metric.compute().item()
            current_recall = recall_metric.compute().item()
            batch_loss = loss.item()
            print(
                f"Batch {batch_idx:3d}/{len(dataloader)}: Loss: {batch_loss:.4f} | Precision: {current_precision:.3f} | Recall: {current_recall:.3f}"
            )
            trackio.log({"batch_loss": batch_loss})

    metric_tensors = torch.stack(
        [
            total_loss / len(dataloader),
            accuracy_metric.compute(),
            precision_metric.compute(),
            recall_metric.compute(),
            f1_metric.compute(),
        ]
    )
    metric_values = metric_tensors.cpu().numpy()

    metrics = {
        "loss": float(metric_values[0]),
        "accuracy": float(metric_values[1]),
        "precision": float(metric_values[2]),
        "recall": float(metric_values[3]),
        "f1": float(metric_values[4]),
        "specificity": 0.0,  # TODO: implement
    }

    trackio.log(
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
) -> tuple[Dict[str, float], np.ndarray]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)

    accuracy_metric = BinaryAccuracy(threshold=threshold, device=device)
    precision_metric = BinaryPrecision(threshold=threshold, device=device)
    recall_metric = BinaryRecall(threshold=threshold, device=device)
    f1_metric = BinaryF1Score(threshold=threshold, device=device)
    auroc_metric = BinaryAUROC(device=device)
    confusion_matrix_metric = BinaryConfusionMatrix(threshold=threshold, device=device)

    for x_img, x_md, targets in dataloader:
        x_img, x_md, targets = x_img.to(device), x_md.to(device), targets.to(device)
        logits = model(x_img, x_md).squeeze()
        loss = criterion(logits, targets)

        total_loss += loss.detach()

        probs = torch.sigmoid(logits)

        # update metrics
        accuracy_metric.update(probs, targets)
        precision_metric.update(probs, targets)
        recall_metric.update(probs, targets.int())  # https://github.com/pytorch/torcheval/issues/209 # fmt: skip
        f1_metric.update(probs, targets)
        auroc_metric.update(probs, targets)
        confusion_matrix_metric.update(probs, targets.int())

    metric_tensors = torch.stack(
        [
            total_loss / len(dataloader),
            accuracy_metric.compute(),
            precision_metric.compute(),
            recall_metric.compute(),
            f1_metric.compute(),
            auroc_metric.compute(),
        ]
    )
    metric_values = metric_tensors.cpu().numpy()

    metrics = {
        "loss": float(metric_values[0]),
        "accuracy": float(metric_values[1]),
        "precision": float(metric_values[2]),
        "recall": float(metric_values[3]),
        "f1": float(metric_values[4]),
        "specificity": 0.0,  # TODO: implement
        "roc_auc": float(metric_values[5]),
    }

    trackio.log(
        {
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_roc_auc": metrics["roc_auc"],
        }
    )

    confusion_mat = confusion_matrix_metric.compute().cpu().numpy()
    console = Console()
    console.print(render_confusion_matrix(confusion_mat))

    return metrics, confusion_mat


def render_confusion_matrix(confusion_matrix: np.ndarray) -> Table:
    """
    Display formatted confusion matrix for validation results.

    Args:
        confusion_matrix: 2x2 confusion matrix from validation
    """
    table = Table(
        title="Validation Confusion Matrix",
        caption="Rows: Actual | Columns: Predicted",
        show_header=True,
    )

    table.add_column("", style="dim")
    table.add_column("Benign", justify="right")
    table.add_column("Malignant", justify="right")

    table.add_row(
        "Benign", f"{int(confusion_matrix[0, 0]):,}", f"{int(confusion_matrix[0, 1]):,}"
    )
    table.add_row(
        "Malignant",
        f"{int(confusion_matrix[1, 0]):,}",
        f"{int(confusion_matrix[1, 1]):,}",
    )
    return table
