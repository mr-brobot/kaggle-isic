import time
from typing import Any, Iterator

import numpy as np
import typer
from datasets import load_dataset
from opentelemetry import trace
from rich.console import Console
from rich.progress import track
from rich.table import Table
from torch.utils.data import DataLoader

from isic.dataset import ImageEncoder, MetadataEncoder, collate_batch

console = Console()
tracer = trace.get_tracer(__name__)


@tracer.start_as_current_span("load_batch")
def load_batch(data_iter: Iterator) -> Any:
    """Load a single batch from the data iterator."""
    return next(data_iter)


def benchmark_batches(
    batches: int = typer.Option(100, help="Number of batches to benchmark"),
    batch_size: int = typer.Option(128, help="Batch size for data loading"),
    image_size: int = typer.Option(128, help="Image size (square)"),
) -> None:
    """Benchmark data loading performance."""

    console.print("[bold blue]ISIC Data Loading Benchmark[/bold blue]")
    console.print(
        f"Batches: {batches}, Batch size: {batch_size}, Image size: {image_size}×{image_size}"
    )

    dataset = load_dataset("mrbrobot/isic-2024", split="train")
    dataset = dataset.select_columns(
        ["image", "age_approx", "sex", "anatom_site_general", "target"]
    )

    metadata_encoder = MetadataEncoder().fit(dataset)
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(
        metadata_encoder,
        batched=True,
        batch_size=batch_size,
        desc="Encoding metadata columns",
        load_from_cache_file=False,
    )

    dataset = dataset.with_format("torch")
    image_encoder = ImageEncoder(image_size=(image_size, image_size))
    dataset = dataset.with_transform(
        image_encoder, columns=["image"], output_all_columns=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    console.print(f"[green]Dataset ready[/green] - Total samples: {len(dataset):,}")
    console.print(
        f"[green]Dataloader ready[/green] - Batches available: {len(dataloader):,}"
    )

    console.print("\n[bold yellow]Running benchmark...[/bold yellow]")

    data_iter = iter(dataloader)
    times = []

    for i in track(range(batches), description="Loading batches"):
        try:
            start_time = time.time()
            _ = load_batch(data_iter)
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
    total_batches_per_epoch = (
        total_samples + batch_size - 1
    ) // batch_size  # Ceiling division
    estimated_epoch_time = mean_time * total_batches_per_epoch

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


if __name__ == "__main__":
    typer.run(benchmark_batches)
