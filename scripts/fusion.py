"""
SageMaker training script for CNN-MLP fusion model.

This script adapts the fusion notebook training code for SageMaker Training Jobs.
It can run locally with SageMaker Local Mode or on remote GPU instances.
"""

import json
import os
from pathlib import Path

import torch
import typer
from datasets import Dataset, load_dataset, load_from_disk
from opentelemetry import trace
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader

from isic.dataset import ImageEncoder, MetadataEncoder, collate_batch
from isic.loss import WeightedFocalLoss
from isic.models import FusionModel
from isic.training import train, validate

console = Console()
tracer = trace.get_tracer(__name__)


def parse_cnn_layers(layers_str: str) -> list[tuple[int, int, bool]]:
    """Parse CNN layers from string format."""
    layers = []
    for layer_spec in layers_str.split(";"):
        out_ch, kernel, pool = layer_spec.split(",")
        layers.append((int(out_ch), int(kernel), bool(int(pool))))
    return layers


def parse_layer_dims(dims_str: str) -> list[int]:
    """Parse layer dimensions from comma-separated string."""
    return [int(d) for d in dims_str.split(",")]


@tracer.start_as_current_span("setup_environment")
def setup_environment(seed: int) -> torch.device:
    """Set up device, seed, and PyTorch settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    console.print(f"[green]Device:[/green] {device}")
    console.print(f"[green]Seed:[/green] {seed}")

    return device


@tracer.start_as_current_span("load_and_prepare_dataset")
def load_and_prepare_dataset(
    train_dir: str | None,
    dataset_name: str,
    image_size: tuple[int, int],
    val_split: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, Dataset]:
    """Load dataset, encode features, and create data loaders."""

    # Load dataset
    if train_dir and Path(train_dir).exists():
        console.print(f"[cyan]Loading preprocessed dataset from[/cyan] {train_dir}")
        ds = load_from_disk(train_dir)
    else:
        console.print(f"[cyan]Loading dataset from HuggingFace:[/cyan] {dataset_name}")
        ds = load_dataset(dataset_name, split="train")
        ds = ds.select_columns(
            ["image", "age_approx", "sex", "anatom_site_general", "target"]
        )

        # Encode metadata
        console.print("[cyan]Encoding metadata...[/cyan]")
        metadata_encoder = MetadataEncoder().fit(ds)
        ds = ds.with_format("arrow")
        ds = ds.map(
            metadata_encoder,
            batched=True,
            batch_size=1000,
            desc="Encoding metadata columns",
        )

        # Encode images
        console.print("[cyan]Setting up image encoder...[/cyan]")
        image_encoder = ImageEncoder(image_size=image_size)
        ds = ds.with_format("torch")
        ds = ds.with_transform(
            image_encoder, columns=["image"], output_all_columns=True
        )

    # Split dataset
    console.print(
        f"[cyan]Splitting dataset with {val_split:.1%} validation split[/cyan]"
    )
    split = ds.train_test_split(test_size=val_split, seed=seed)
    train_ds, val_ds = split["train"], split["test"]

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    console.print(
        f"[green]Batches per epoch -[/green] Train: {len(train_loader):,}, Val: {len(val_loader):,}"
    )

    return train_loader, val_loader, ds


@tracer.start_as_current_span("calculate_class_weights")
def calculate_class_weights(
    dataset: Dataset, class_power: float, device: torch.device
) -> torch.Tensor:
    """Calculate weighted class weights for imbalanced dataset."""
    df = dataset.to_pandas()
    neg_count = (df["target"] == 0).sum()
    pos_count = (df["target"] == 1).sum()
    pos_weight = neg_count / pos_count
    scaled_pos_weight = torch.tensor([pos_weight**class_power], device=device)

    # Create table for class distribution
    table = Table(title="Class Distribution")
    table.add_column("Class", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Weight", style="yellow", justify="right")

    table.add_row("Benign", f"{neg_count:,}", "1.0")
    table.add_row("Malignant", f"{pos_count:,}", f"{pos_weight:.1f}")
    table.add_row("Scaled Weight", "", f"{scaled_pos_weight.item():.1f}")

    console.print(table)

    return scaled_pos_weight


@tracer.start_as_current_span("create_model")
def create_model(
    image_shape: tuple[int, int, int],
    cnn_layers: list[tuple[int, int, bool]],
    metadata_layer_dims: list[int],
    fusion_layer_dims: list[int],
    device: torch.device,
) -> FusionModel:
    """Create and initialize fusion model."""
    model = FusionModel(
        image_shape=image_shape,
        cnn_layers=cnn_layers,
        metadata_layer_dims=metadata_layer_dims,
        fusion_layer_dims=fusion_layer_dims,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]Total trainable parameters:[/green] {total_params:,}")

    return model


@tracer.start_as_current_span("setup_training")
def setup_training(
    model: FusionModel,
    learning_rate: float,
    pos_weight: torch.Tensor,
    focal_power: int,
) -> tuple[WeightedFocalLoss, torch.optim.Adam]:
    """Set up loss function and optimizer."""
    criterion = WeightedFocalLoss(pos_weight=pos_weight, gamma=focal_power)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    console.print(f"[green]Loss:[/green] WeightedFocalLoss (gamma={focal_power})")
    console.print(f"[green]Optimizer:[/green] Adam (lr={learning_rate})")

    return criterion, optimizer


@tracer.start_as_current_span("save_checkpoint")
def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: FusionModel,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    config: dict,
) -> None:
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        },
        checkpoint_path,
    )
    console.print(f"[green]✓[/green] Saved checkpoint (val_loss={val_loss:.4f})")


@tracer.start_as_current_span("save_final_artifacts")
def save_final_artifacts(
    model_dir: str,
    output_data_dir: str,
    model: FusionModel,
    config: dict,
    train_metrics: dict,
    val_metrics: dict,
    best_val_loss: float,
) -> None:
    """Save final model and metrics."""
    # Save model
    model_path = Path(model_dir) / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        model_path,
    )
    console.print(f"[green]✓[/green] Saved model to {model_path}")

    # Save metrics
    metrics_path = Path(output_data_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "train": train_metrics,
                "val": val_metrics,
                "best_val_loss": best_val_loss,
            },
            f,
            indent=2,
        )
    console.print(f"[green]✓[/green] Saved metrics to {metrics_path}")


@tracer.start_as_current_span("train_epoch")
def train_epoch(
    epoch: int,
    total_epochs: int,
    model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: WeightedFocalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
) -> tuple[dict, dict]:
    """Train and validate for one epoch."""
    console.rule(f"[bold]Epoch {epoch + 1}/{total_epochs}")

    train_metrics = train(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        threshold,
        console=console,
    )

    val_metrics = validate(
        model, val_loader, criterion, device, threshold, console=console
    )

    return train_metrics, val_metrics


def main(
    # Hyperparameters
    epochs: int = typer.Option(8, help="Number of training epochs"),
    batch_size: int = typer.Option(128, help="Batch size for training"),
    learning_rate: float = typer.Option(0.001, help="Learning rate for optimizer"),
    class_power: float = typer.Option(
        0.9, help="Class weight scaling power (0-1, scales positive class weight)"
    ),
    focal_power: int = typer.Option(2, help="Focal loss gamma parameter"),
    image_size: int = typer.Option(128, help="Target image size (square)"),
    threshold: float = typer.Option(
        0.5, help="Probability threshold for classification"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    # Model architecture
    cnn_layers: str = typer.Option(
        "16,5,1;32,3,1;64,3,1;64,3,1;32,3,1",
        help="CNN layers as 'out_ch,kernel,pool' separated by semicolons",
    ),
    metadata_layer_dims: str = typer.Option(
        "8,16,32", help="Metadata MLP layer dimensions (comma-separated)"
    ),
    fusion_layer_dims: str = typer.Option(
        "256,128,64,8", help="Fusion MLP layer dimensions (comma-separated)"
    ),
    # SageMaker paths
    model_dir: str = typer.Option(
        os.environ.get("SM_MODEL_DIR", "./model"),
        help="Output directory for final model",
    ),
    output_data_dir: str = typer.Option(
        os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
        help="Output directory for metrics and artifacts",
    ),
    checkpoint_dir: str = typer.Option(
        os.environ.get("SM_CHECKPOINT_DIR", "./checkpoints"),
        help="Output directory for training checkpoints",
    ),
    train_dir: str | None = typer.Option(
        os.environ.get("SM_CHANNEL_TRAINING", None),
        help="Input directory with preprocessed dataset (or None to load from HuggingFace)",
    ),
    # Dataset options
    dataset_name: str = typer.Option(
        "mrbrobot/isic-2024", help="HuggingFace dataset name"
    ),
    val_split: float = typer.Option(0.2, help="Validation split ratio"),
) -> None:
    """Train CNN-MLP fusion model for skin cancer classification."""

    # Display configuration
    console.print(
        Panel.fit(
            "[bold blue]ISIC 2024 - SageMaker Training[/bold blue]\n"
            "[dim]CNN-MLP Fusion Model for Skin Cancer Detection[/dim]",
            border_style="blue",
        )
    )

    # Setup environment
    device = setup_environment(seed)

    # Parse architecture parameters
    cnn_layers_parsed = parse_cnn_layers(cnn_layers)
    metadata_layer_dims_parsed = parse_layer_dims(metadata_layer_dims)
    fusion_layer_dims_parsed = parse_layer_dims(fusion_layer_dims)
    image_shape = (image_size, image_size, 3)
    img_size = (image_size, image_size)

    # Display hyperparameters
    config_table = Table(title="Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Epochs", str(epochs))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Learning Rate", str(learning_rate))
    config_table.add_row("Image Size", f"{image_size}×{image_size}")
    config_table.add_row("Threshold", str(threshold))
    config_table.add_row("CNN Layers", str(len(cnn_layers_parsed)))
    config_table.add_row("Validation Split", f"{val_split:.1%}")

    console.print(config_table)

    # Load and prepare dataset
    train_loader, val_loader, ds = load_and_prepare_dataset(
        train_dir, dataset_name, img_size, val_split, batch_size, seed
    )

    # Calculate class weights
    scaled_pos_weight = calculate_class_weights(ds, class_power, device)

    # Create model
    model = create_model(
        image_shape,
        cnn_layers_parsed,
        metadata_layer_dims_parsed,
        fusion_layer_dims_parsed,
        device,
    )

    # Setup training
    criterion, optimizer = setup_training(
        model, learning_rate, scaled_pos_weight, focal_power
    )

    # Model config for saving
    model_config = {
        "image_shape": image_shape,
        "cnn_layers": cnn_layers_parsed,
        "metadata_layer_dims": metadata_layer_dims_parsed,
        "fusion_layer_dims": fusion_layer_dims_parsed,
    }

    # Training loop
    console.print(
        Panel(
            f"[bold]Starting training for {epochs} epochs[/bold]",
            border_style="yellow",
        )
    )

    best_val_loss = float("inf")
    train_metrics = {}
    val_metrics = {}

    for epoch in range(epochs):
        train_metrics, val_metrics = train_epoch(
            epoch,
            epochs,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            threshold,
        )

        # Save checkpoint if best model so far
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                checkpoint_dir, epoch, model, optimizer, best_val_loss, model_config
            )

    # Save final artifacts
    save_final_artifacts(
        model_dir,
        output_data_dir,
        model,
        model_config,
        train_metrics,
        val_metrics,
        best_val_loss,
    )

    # Summary
    console.print(
        Panel.fit(
            f"[bold green]✓ Training Complete![/bold green]\n\n"
            f"Final Validation Loss: [cyan]{val_metrics['loss']:.4f}[/cyan]\n"
            f"Best Validation Loss: [cyan]{best_val_loss:.4f}[/cyan]\n"
            f"Final Accuracy: [cyan]{val_metrics.get('accuracy', 0):.3f}[/cyan]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    typer.run(main)
