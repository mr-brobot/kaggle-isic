#!/usr/bin/env python3

import time
from functools import partial
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table
from torch.utils.data import DataLoader, random_split

from isic.dataset import ISICDataset, collate_batch
from isic.tracing import dataframe_filter, image_filter, tensor_filter

console = Console()


def setup_environment(data_dir: Path, enable_tracing: bool = True) -> Tuple[Path, Path]:
    train_images_file = data_dir / "train-image.hdf5"
    train_metadata_file = data_dir / "train-metadata.csv"

    if not train_images_file.exists():
        console.print(f"[red]Error: {train_images_file} not found[/red]")
        raise typer.Exit(1)

    if not train_metadata_file.exists():
        console.print(f"[red]Error: {train_metadata_file} not found[/red]")
        raise typer.Exit(1)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("benchmarking")
    
    if enable_tracing:
        mlflow.tracing.configure(
            span_processors=[tensor_filter, dataframe_filter, image_filter]
        )
    else:
        mlflow.tracing.disable()

    return train_images_file, train_metadata_file


def create_dataloader(
    dataset: ISICDataset,
    batch_size: int,
    image_size: Tuple[int, int],
    sample_size: int = 10000,
) -> DataLoader:
    from sklearn.preprocessing import OneHotEncoder

    sex_enc = OneHotEncoder(categories=[["male", "female"]], handle_unknown="ignore")
    anatom_site_general_enc = OneHotEncoder(
        categories=[
            [
                "head/neck",
                "upper extremity",
                "lower extremity",
                "posterior torso",
                "anterior torso",
            ]
        ],
        handle_unknown="ignore",
    )

    sex_enc.fit(dataset.metadata[["sex"]])
    anatom_site_general_enc.fit(dataset.metadata[["anatom_site_general"]])

    md_encoders = {"sex": sex_enc, "anatom_site_general": anatom_site_general_enc}
    collate = partial(collate_batch, img_size=image_size, md_encoders=md_encoders)

    # Use a subset for faster benchmarking
    generator = torch.Generator().manual_seed(42)
    subset, _ = random_split(
        dataset, [sample_size, len(dataset) - sample_size], generator=generator
    )

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )


def benchmark_batches(
    batches: int = typer.Option(100, help="Number of batches to benchmark"),
    batch_size: int = typer.Option(128, help="Batch size for data loading"),
    image_size: int = typer.Option(128, help="Image size (square)"),
    data_dir: Path = typer.Option(Path("data"), help="Directory containing data files"),
    tracing: bool = typer.Option(False, "--tracing", help="Enable MLflow tracing (slower but provides detailed spans)"),
) -> None:
    """Benchmark data loading performance."""

    console.print("[bold blue]ISIC Data Loading Benchmark[/bold blue]")
    console.print(
        f"Batches: {batches}, Batch size: {batch_size}, Image size: {image_size}×{image_size}"
    )
    console.print(f"Data directory: {data_dir}")

    train_images_file, train_metadata_file = setup_environment(data_dir, tracing)

    console.print("[yellow]Setting up dataset and dataloader...[/yellow]")
    dataset = ISICDataset(train_images_file, train_metadata_file)

    img_size = (image_size, image_size)
    dataloader = create_dataloader(dataset, batch_size, img_size)

    console.print(f"[green]Dataset ready[/green] - Total samples: {len(dataset):,}")
    console.print(
        f"[green]Dataloader ready[/green] - Batches available: {len(dataloader):,}"
    )

    with dataset:
        with mlflow.start_run(log_system_metrics=True):
            mlflow.log_params(
                {
                    "batches": batches,
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "total_samples": len(dataset),
                }
            )

            console.print("\n[bold yellow]Running benchmark...[/bold yellow]")

            data_iter = iter(dataloader)
            times = []

            for i in track(range(batches), description="Loading batches"):
                try:
                    start_time = time.time()
                    _ = next(data_iter)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                except StopIteration:
                    console.print(
                        f"[yellow]Warning: Only {i} batches available, stopping early[/yellow]"
                    )
                    break

            # Calculate statistics
            times_array = np.array(times)
            mean_time = np.mean(times_array)
            std_time = np.std(times_array)
            min_time = np.min(times_array)
            max_time = np.max(times_array)
            
            # Calculate estimated epoch time (total batches needed for full dataset)
            total_samples = len(dataset)
            total_batches_per_epoch = (total_samples + batch_size - 1) // batch_size  # Ceiling division
            estimated_epoch_time = mean_time * total_batches_per_epoch

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "mean_batch_time": float(mean_time),
                    "std_batch_time": float(std_time),
                    "min_batch_time": min_time,
                    "max_batch_time": max_time,
                    "batches_processed": len(times),
                    "estimated_epoch_time": float(estimated_epoch_time),
                    "total_batches_per_epoch": total_batches_per_epoch,
                }
            )

            # Create results table
            table = Table(title="Benchmarking Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Batches processed", str(len(times)))
            table.add_row("Mean time per batch", f"{mean_time:.3f}s")
            table.add_row("Std deviation", f"{std_time:.3f}s")
            table.add_row("Min time", f"{min_time:.3f}s")
            table.add_row("Max time", f"{max_time:.3f}s")
            table.add_row("Samples per second", f"{(batch_size / mean_time):.1f}")
            
            # Format epoch time as minutes if > 60s, otherwise seconds
            if estimated_epoch_time >= 60:
                epoch_time_str = f"{estimated_epoch_time / 60:.1f}min"
            else:
                epoch_time_str = f"{estimated_epoch_time:.1f}s"
            
            table.add_row("Estimated epoch time", epoch_time_str)
            table.add_row("Total batches per epoch", str(total_batches_per_epoch))

            console.print("\n")
            console.print(table)

            console.print("\n[green]✓ Benchmark complete![/green]")
            console.print(f"[dim]MLflow run: {mlflow.active_run().info.run_id}[/dim]")


if __name__ == "__main__":
    typer.run(benchmark_batches)
