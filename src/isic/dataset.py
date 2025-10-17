from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import torch
import torchvision.transforms.v2 as T
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from transformers import ProcessorMixin


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


class MessagesFormatter:
    """Format image and text into VLM messages structure."""

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add 'messages' field with VLM chat format.

        Args:
            batch: Batch dict where each value is a list

        Returns:
            Batch dict with added 'messages' field
        """
        result = []

        for image, text in zip(batch["image"], batch["text"]):
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Classify provided example as benign or malignant.",
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                },
            ]
            result.append(messages)

        batch["messages"] = result
        return batch


class MetadataTextFormatter:
    """Format metadata fields as markdown table for VLM."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        """
        Add 'text' column with formatted metadata as markdown tables.

        Args:
            batch: PyArrow Table with metadata columns

        Returns:
            PyArrow Table with added 'text' column containing markdown tables
        """
        ages = batch["age_approx"].to_pylist()
        sexes = batch["sex"].to_pylist()
        sites = batch["anatom_site_general"].to_pylist()

        texts = []
        for age, sex, site in zip(ages, sexes, sites):
            # format values, explicitly stating when unknown
            age_str = f"{int(age)} years" if age else "unknown"
            sex_str = sex if sex else "unknown"
            site_str = site if site else "unknown"

            # build markdown table
            text = dedent(f"""
            | Field | Value |
            |-------|-------|
            | Age | {age_str} |
            | Sex | {sex_str} |
            | Lesion Site | {site_str} |
            """).strip()

            texts.append(text)

        return batch.append_column("text", pa.array(texts))


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


class VLMCollator:
    """Collate function that applies VLM processor to batch for HuggingFace Trainer."""

    processor: ProcessorMixin

    def __init__(self, processor: ProcessorMixin) -> None:
        """
        Initialize collator with VLM processor.

        Args:
            processor: VLM processor instance
        """
        self.processor = processor

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Apply processor to batch and format for HuggingFace Trainer.

        Args:
            batch: List of dicts with keys ["messages", "target"]

        Returns:
            Dict with processed inputs and labels
        """
        messages_list = [ex["messages"] for ex in batch]
        targets = torch.tensor([ex["target"] for ex in batch], dtype=torch.float32)

        inputs = self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        return inputs | {"labels": targets}
