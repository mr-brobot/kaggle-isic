from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import torch
import torchvision.transforms.v2 as T
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class ImageEncoder:
    """Transform images with torchvision."""

    def __init__(self, image_size: Tuple[int, int] = (128, 128)):
        self.transform = T.Compose(
            [
                T.ToImage(),  # convert PIL images to tensors
                T.Resize(size=image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.ToDtype(dtype=torch.float32, scale=True),
            ]
        )

    def __call__(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Transform dataset, to be used with `Dataset.with_transform."""
        dataset["image"] = self.transform(dataset["image"])
        return dataset


@dataclass
class MetadataEncoder:
    """Encode metadata features with sklearn transformers."""

    age_scaler: Optional[MinMaxScaler] = None
    sex_encoder: Optional[OneHotEncoder] = None
    anatom_site_general_encoder: Optional[OneHotEncoder] = None

    def fit(self, dataset: Dataset) -> "MetadataEncoder":
        """Fit sklearn encoders on dataset."""
        meta = dataset.to_pandas()[["age_approx", "sex", "anatom_site_general"]]
        meta["age_approx"] = meta["age_approx"].fillna(0)

        self.age_scaler = MinMaxScaler().fit(meta["age_approx"].values.reshape(-1, 1))
        self.sex_encoder = OneHotEncoder(
            categories=[["male", "female"]], handle_unknown="ignore"
        ).fit(meta["sex"].values.reshape(-1, 1))
        self.site_encoder = OneHotEncoder(
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
        ).fit(meta["anatom_site_general"].values.reshape(-1, 1))

        return self

    def __call__(self, batch: pa.Table) -> pa.Table:
        """
        Encode batched metadata examples for use with .map(batched=True).

        Args:
            batch: PyArrow Table with all columns including metadata columns to encode

        Returns:
            PyArrow Table with encoded metadata columns replacing the original columns
        """
        if not self.age_scaler or not self.sex_encoder or not self.site_encoder:
            raise ValueError("Column encoders missing, must call fit() before encoding")

        ages = batch["age_approx"].to_numpy().reshape(-1, 1)
        ages = np.nan_to_num(ages, nan=0.0)
        sexes = batch["sex"].to_numpy().reshape(-1, 1)
        sites = batch["anatom_site_general"].to_numpy().reshape(-1, 1)

        age_scaled = self.age_scaler.transform(ages)  # (n, 1)
        sex_encoded = self.sex_encoder.transform(sexes).toarray()  # (n, 2)
        site_encoded = self.site_encoder.transform(sites).toarray()  # (n, 5)

        batch = batch.drop(["age_approx", "sex", "anatom_site_general"])
        batch = batch.append_column("age_approx", pa.array(age_scaled.flatten()))
        batch = batch.append_column("sex", pa.array(sex_encoded.tolist()))
        batch = batch.append_column(
            "anatom_site_general", pa.array(site_encoded.tolist())
        )

        return batch


def collate_batch(
    batch: Sequence[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple collate function - just stack pre-processed tensors.

    Args:
        batch: List of dicts with keys ["image", "age_approx", "sex", "anatom_site_general", "target"]

    Returns:
        Tuple of (images, metadata, targets)
    """
    images = torch.stack([ex["image"] for ex in batch])

    metadata = torch.stack(
        [
            torch.cat(
                [
                    torch.tensor([ex["age_approx"]], dtype=torch.float32),
                    torch.tensor(ex["sex"], dtype=torch.float32),
                    torch.tensor(ex["anatom_site_general"], dtype=torch.float32),
                ]
            )
            for ex in batch
        ]
    )

    targets = torch.stack([torch.tensor(ex["target"]) for ex in batch]).float()

    return images, metadata, targets
