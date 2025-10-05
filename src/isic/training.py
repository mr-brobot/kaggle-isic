"""
Training utilities for ISIC 2024 models
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import trackio
from sklearn.metrics import classification_report, confusion_matrix
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

        trackio.log({"batch_loss": loss.item()})

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_metric.compute().item(),
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "f1": f1_metric.compute().item(),
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

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    return metrics, all_targets, all_predictions


def training_summary(
    val_targets: np.ndarray,
    val_predictions: np.ndarray,
) -> str:
    """
    Generate formatted summary for completed training session.

    Args:
        val_targets: True validation targets
        val_predictions: Predicted validation labels

    Returns:
        Formatted training summary string
    """
    lines = []

    lines.append("\nTraining Summary:")

    lines.append("\nValidation Results:")
    cm = confusion_matrix(val_targets, val_predictions)
    lines.append("Confusion Matrix:")
    lines.append("                    Predicted")
    lines.append("                Benign  Malignant  Total")
    lines.append(
        f"Actual Benign     {cm[0, 0]:4d}      {cm[0, 1]:4d}   {cm[0, 0] + cm[0, 1]:4d}"
    )
    lines.append(
        f"    Malignant     {cm[1, 0]:4d}      {cm[1, 1]:4d}     {cm[1, 0] + cm[1, 1]:3d}"
    )

    # Calculate key medical metrics
    if cm[1, 0] + cm[1, 1] > 0:  # If there are malignant cases
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        missed_malignant = cm[1, 0]
        detected_malignant = cm[1, 1]

        lines.append("\n🏥 CRITICAL MEDICAL METRICS:")
        lines.append(f"• Malignant cases in validation: {cm[1, 0] + cm[1, 1]}")
        lines.append(f"• Malignant cases DETECTED: {detected_malignant}")
        lines.append(f"• Malignant cases MISSED: {missed_malignant}")
        lines.append(f"• Sensitivity (Detection Rate): {sensitivity:.1%}")

        if missed_malignant > 0:
            lines.append(
                f"⚠️  WARNING: {missed_malignant} malignant cases would go undiagnosed!"
            )

    lines.append("\nDetailed Classification Report:")
    lines.append(
        classification_report(
            val_targets, val_predictions, target_names=["Benign", "Malignant"], digits=4
        )
    )

    return "\n".join(lines)
