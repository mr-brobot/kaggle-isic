"""
Binary classification metrics computation for PyTorch and HuggingFace Trainer
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from rich.console import Console
from rich.table import Table
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from transformers import EvalPrediction


@dataclass
class BinaryMetricsComputer:
    """Compute binary classification metrics for both PyTorch loops and HuggingFace Trainer."""

    threshold: float = 0.5
    device: Optional[torch.device] = None

    # Internal state - initialized in __post_init__
    accuracy: BinaryAccuracy = field(init=False, repr=False)
    precision: BinaryPrecision = field(init=False, repr=False)
    recall: BinaryRecall = field(init=False, repr=False)
    f1: BinaryF1Score = field(init=False, repr=False)
    auroc: BinaryAUROC = field(init=False, repr=False)
    confusion_matrix: BinaryConfusionMatrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize torcheval metrics with device and threshold."""
        if self.device is None:
            self.device = torch.device("cpu")

        self.accuracy = BinaryAccuracy(threshold=self.threshold, device=self.device)
        self.precision = BinaryPrecision(threshold=self.threshold, device=self.device)
        self.recall = BinaryRecall(threshold=self.threshold, device=self.device)
        self.f1 = BinaryF1Score(threshold=self.threshold, device=self.device)
        self.auroc = BinaryAUROC(device=self.device)
        self.confusion_matrix = BinaryConfusionMatrix(
            threshold=self.threshold, device=self.device
        )

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with batch predictions and targets."""
        probs = probs.to(self.device) if self.device else probs
        targets = targets.to(self.device) if self.device else targets

        self.accuracy.update(probs, targets)
        self.precision.update(probs, targets)
        self.recall.update(probs, targets.int())  # BinaryRecall requires int targets
        self.f1.update(probs, targets)
        self.auroc.update(probs, targets)
        self.confusion_matrix.update(probs, targets.int())

    def compute(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated state and return as float dict.

        Handles recall and F1 safely when no positive samples have been seen yet.
        """
        # Check if we have positive samples for recall-dependent metrics
        has_positives = self.recall.num_true_labels > 0  # type: ignore[attr-defined]

        metrics = {
            "accuracy": self.accuracy.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item() if has_positives else 0.0,
            "f1": self.f1.compute().item() if has_positives else 0.0,
            "roc_auc": self.auroc.compute().item(),
        }

        return metrics

    def reset(self) -> None:
        """Reset all metric states."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()
        self.confusion_matrix.reset()

    def to(self, device: torch.device) -> "BinaryMetricsComputer":
        """Move all metrics to specified device."""
        self.device = device

        self.accuracy.to(device)
        self.precision.to(device)
        self.recall.to(device)
        self.f1.to(device)
        self.auroc.to(device)
        self.confusion_matrix.to(device)

        return self

    def render(self, console: Console | None = None) -> None:
        """
        Render confusion matrix as a formatted Rich table and print to console.

        Args:
            console: Optional Rich Console instance. If None, creates a new one.
        """
        if console is None:
            console = Console()

        confusion_mat = self.confusion_matrix.compute().cpu().numpy()

        table = Table(
            title="Confusion Matrix",
            caption="Rows: Actual | Columns: Predicted",
            show_header=True,
        )

        table.add_column("", style="dim")
        table.add_column("Benign", justify="right")
        table.add_column("Malignant", justify="right")

        table.add_row(
            "Benign",
            f"{int(confusion_mat[0, 0]):,}",
            f"{int(confusion_mat[0, 1]):,}",
        )
        table.add_row(
            "Malignant",
            f"{int(confusion_mat[1, 0]):,}",
            f"{int(confusion_mat[1, 1]):,}",
        )

        console.print(table)

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics from EvalPrediction for HuggingFace Trainer.

        This is the primary interface for HuggingFace Trainer - pass the
        instance directly to compute_metrics parameter.

        Args:
            eval_pred: EvalPrediction with predictions (logits) and label_ids as numpy arrays

        Returns:
            Dictionary of metric names to scalar values
        """
        self.reset()

        logits = torch.from_numpy(eval_pred.predictions).to(self.device)
        targets = torch.from_numpy(eval_pred.label_ids).to(self.device).float()

        probs = torch.sigmoid(logits)

        self.update(probs, targets)
        metrics = self.compute()

        self.reset()

        return metrics
