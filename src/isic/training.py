"""
Training utilities for ISIC 2024 models
"""

import time
from typing import Dict, List, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from tqdm import tqdm


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


class Trainer:
    """
    Training class for ISIC models with comprehensive logging and validation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: Optional[float] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for regularization
            pos_weight: Positive class weight for BCEWithLogitsLoss
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Setup loss function
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCELoss()

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",  # Maximize validation AUC
            factor=0.5,
            patience=3,
            verbose=True,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_auc": [],
            "val_loss": [],
            "val_auc": [],
            "val_ap": [],
            "lr": [],
        }

        self.best_val_auc = 0.0
        self.best_model_state = None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        all_predictions = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, metadata, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, metadata).squeeze()

            # Compute loss
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                loss = self.criterion(outputs, targets)
                predictions = torch.sigmoid(outputs)
            else:
                loss = self.criterion(outputs, targets)
                predictions = outputs

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            all_predictions.extend(predictions.cpu().detach().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(all_targets, all_predictions)

        return {"loss": epoch_loss, "auc": epoch_auc}

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, metadata, targets in pbar:
                # Move to device
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(images, metadata).squeeze()

                # Compute loss
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    loss = self.criterion(outputs, targets)
                    predictions = torch.sigmoid(outputs)
                else:
                    loss = self.criterion(outputs, targets)
                    predictions = outputs

                # Statistics
                running_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_auc = roc_auc_score(all_targets, all_predictions)
        epoch_ap = average_precision_score(all_targets, all_predictions)

        return {"loss": epoch_loss, "auc": epoch_auc, "ap": epoch_ap}

    def train(
        self, epochs: int, early_stopping_patience: int = 7
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified epochs.

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Early stopping patience

        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs...")
        print(
            f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

        early_stopping_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate_epoch()

            # Update learning rate scheduler
            self.scheduler.step(val_metrics["auc"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_auc"].append(train_metrics["auc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auc"].append(val_metrics["auc"])
            self.history["val_ap"].append(val_metrics["ap"])
            self.history["lr"].append(current_lr)

            # Check for best model
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self.best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0
                print(f"🎯 New best validation AUC: {val_metrics['auc']:.4f}")
            else:
                early_stopping_counter += 1

            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1:3d}/{epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train AUC: {train_metrics['auc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
                f"Val AP: {val_metrics['ap']:.4f} | LR: {current_lr:.6f}"
            )

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {early_stopping_patience} epochs without improvement"
                )
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights")

        return self.history

    def save_model(self, path: str) -> None:
        """Save model state dict."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "best_val_auc": self.best_val_auc,
                "history": self.history,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        self.history = checkpoint.get("history", {})
        print(f"Model loaded from {path}")


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


def plot_training_history(history: Dict[str, List[float]]) -> None:
    """Plot training history."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0, 0].plot(history["val_loss"], label="Val Loss", color="orange")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # AUC plot
    axes[0, 1].plot(history["train_auc"], label="Train AUC", color="blue")
    axes[0, 1].plot(history["val_auc"], label="Val AUC", color="orange")
    axes[0, 1].set_title("Training and Validation AUC")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AUC")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Validation AP plot
    axes[1, 0].plot(history["val_ap"], label="Val AP", color="green")
    axes[1, 0].set_title("Validation Average Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Average Precision")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate plot
    axes[1, 1].plot(history["lr"], label="Learning Rate", color="red")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
