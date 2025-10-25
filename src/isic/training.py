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
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
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
    console: Console | None = None,
) -> Dict[str, float]:
    model.train()
    total_loss = torch.tensor(0.0, device=device)

    if console is None:
        console = Console()

    # init metrics
    accuracy_metric = BinaryAccuracy(threshold=threshold, device=device)
    precision_metric = BinaryPrecision(threshold=threshold, device=device)
    recall_metric = BinaryRecall(threshold=threshold, device=device)
    f1_metric = BinaryF1Score(threshold=threshold, device=device)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TextColumn("Loss: [cyan]{task.fields[loss]:.4f}"),
        TextColumn("Prec: [green]{task.fields[precision]:.3f}"),
        TextColumn("Rec: [yellow]{task.fields[recall]:.3f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Training", total=len(dataloader), loss=0.0, precision=0.0, recall=0.0
        )

        for batch_idx, (x_img, x_md, targets) in enumerate(dataloader):
            x_img, x_md, targets = x_img.to(device), x_md.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(x_img, x_md).squeeze()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()

            probs = torch.sigmoid(logits)

            accuracy_metric.update(probs, targets)
            precision_metric.update(probs, targets)
            recall_metric.update(probs, targets.int())  # https://github.com/pytorch/torcheval/issues/209 # fmt: skip
            f1_metric.update(probs, targets)

            # only compute recall if we've seen positive samples (avoids NaN warnings)
            recall_value = (
                recall_metric.compute().item()
                if recall_metric.num_true_labels > 0  # type: ignore[attr-defined]
                else 0.0
            )

            progress.update(
                task,
                advance=1,
                loss=(total_loss / (batch_idx + 1)).item(),
                precision=precision_metric.compute().item(),
                recall=recall_value,
            )

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
    console: Console | None = None,
) -> tuple[Dict[str, float], np.ndarray]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)

    if console is None:
        console = Console()

    accuracy_metric = BinaryAccuracy(threshold=threshold, device=device)
    precision_metric = BinaryPrecision(threshold=threshold, device=device)
    recall_metric = BinaryRecall(threshold=threshold, device=device)
    f1_metric = BinaryF1Score(threshold=threshold, device=device)
    auroc_metric = BinaryAUROC(device=device)
    confusion_matrix_metric = BinaryConfusionMatrix(threshold=threshold, device=device)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TextColumn("Loss: [cyan]{task.fields[loss]:.4f}"),
        TextColumn("Prec: [green]{task.fields[precision]:.3f}"),
        TextColumn("Rec: [yellow]{task.fields[recall]:.3f}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Validation", total=len(dataloader), loss=0.0, precision=0.0, recall=0.0
        )

        for batch_idx, (x_img, x_md, targets) in enumerate(dataloader):
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

            # only compute recall if we've seen positive samples (avoids NaN warnings)
            recall_value = (
                recall_metric.compute().item()
                if recall_metric.num_true_labels > 0  # type: ignore[attr-defined]
                else 0.0
            )

            # update progress bar
            progress.update(
                task,
                advance=1,
                loss=(total_loss / (batch_idx + 1)).item(),
                precision=precision_metric.compute().item(),
                recall=recall_value,
            )

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
