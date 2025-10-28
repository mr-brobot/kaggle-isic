"""
Training utilities for ISIC 2024 models
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import trackio
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from torch.utils.data import DataLoader

from isic.metrics import BinaryMetricsComputer


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

    metrics_computer = BinaryMetricsComputer(
        threshold=threshold,
        device=device,
    )

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

            metrics_computer.update(probs, targets)

            metrics = metrics_computer.compute()

            progress.update(
                task,
                advance=1,
                loss=(total_loss / (batch_idx + 1)).item(),
                precision=metrics["precision"],
                recall=metrics["recall"],
            )

    metrics = metrics_computer.compute()
    metrics["loss"] = (total_loss / len(dataloader)).item()

    trackio.log({f"train/{k}": v for k, v in metrics.items()})

    return metrics


@torch.no_grad
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    console: Console | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)

    if console is None:
        console = Console()

    metrics_computer = BinaryMetricsComputer(
        threshold=threshold,
        device=device,
    )

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

            metrics_computer.update(probs, targets)

            metrics = metrics_computer.compute()

            progress.update(
                task,
                advance=1,
                loss=(total_loss / (batch_idx + 1)).item(),
                precision=metrics["precision"],
                recall=metrics["recall"],
            )

    metrics = metrics_computer.compute()
    metrics["loss"] = (total_loss / len(dataloader)).item()

    trackio.log({f"val/{k}": v for k, v in metrics.items()})

    metrics_computer.render(console)

    return metrics
