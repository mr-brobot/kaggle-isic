import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import mlflow
import pandas as pd
import torch
from PIL.Image import Image
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


@dataclass
class ISICDataset(Dataset):
    """
    ISIC Dataset for multimodal skin cancer detection.
    """
    
    hdf5_path: Path
    metadata_path: Path
    _hdf5_file: Optional[h5py.File] = None
    _metadata: Optional[pd.DataFrame] = None

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._metadata = pd.read_csv(self.metadata_path, low_memory=False)
        return self._metadata

    @property
    def is_open(self) -> bool:
        """Check if the HDF5 file is currently open."""
        return self._hdf5_file is not None

    def open(self) -> "ISICDataset":
        """Open the HDF5 file for reading."""
        if self._hdf5_file is not None:
            warnings.warn("Dataset is already open")
            return self

        self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self

    def close(self) -> None:
        """Close the HDF5 file if it's open."""
        if self.is_open:
            self._hdf5_file.close()
            self._hdf5_file = None

    def __enter__(self) -> "ISICDataset":
        """Context manager entry."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()
        # Return None to propagate any exception

    @mlflow.trace(name="ISICDataset.image")
    def image(self, key: str) -> Image:
        from PIL import Image

        if self._hdf5_file is None:
            raise RuntimeError(
                "Dataset is not open. Use 'with ISICDataset(...) as ds:' "
                "or call ds.open() before accessing images."
            )

        b = io.BytesIO(self._hdf5_file[key][()])
        return Image.open(b)

    @mlflow.trace(name="ISICDataset.row")
    def row(self, idx: int) -> pd.Series:
        return self.metadata.iloc[idx]

    def __len__(self):
        return len(self.metadata)

    @mlflow.trace(name="ISICDataset.__getitem__")
    def __getitem__(self, idx: int) -> Tuple[pd.Series, Image, int]:
        if type(idx) is not int:
            raise ValueError(f"ISICDataset: Unexpected index type {idx} ({type(idx)})")

        # ensure int for pandas compat
        idx = int(idx)
        row = self.row(idx)

        key = row["isic_id"]
        image = self.image(key)

        return row, image, row["target"]


@dataclass
class ImageEncoder:
    """
    Image encoder that handles resizing and normalization for sequences of PIL Images.
    """

    image_size: Tuple[int, int]

    @mlflow.trace(name="ImageEncoder")
    def __call__(self, images: Sequence[Image]) -> torch.Tensor:
        """
        Encode a sequence of images as a tensor

        Images are resized & normalized

        Args:
            images: Sequence of PIL Images to process

        Returns:
            torch.Tensor: Batch tensor with shape (N, C, H, W) where N is the number of images
        """
        # resize
        scaled_images = [i.resize(self.image_size) for i in images]

        # stack
        result = torch.stack([pil_to_tensor(i) for i in scaled_images])

        # normalize
        result = result.float() / 255.0

        return result


@dataclass
class MetadataEncoder:
    """
    Metadata encoder that handles preprocessing and encoding for metadata.
    """

    age_scaler: Optional[MinMaxScaler] = None
    sex_encoder: Optional[OneHotEncoder] = None
    anatom_site_general_encoder: Optional[OneHotEncoder] = None
    _is_fitted: bool = False

    def fit(self, metadata: pd.DataFrame) -> "MetadataEncoder":
        """
        Fit encoders and scaler on the full metadata DataFrame.

        Args:
            metadata: Full metadata DataFrame to fit transformers on

        Returns:
            self: For method chaining
        """
        # fit age scaler
        age_df = metadata[["age_approx"]].copy()
        age_df["age_approx"] = pd.to_numeric(
            age_df["age_approx"], errors="coerce"
        ).fillna(0)
        self.age_scaler = MinMaxScaler()
        self.age_scaler.fit(age_df)

        # fit sex encoder
        self.sex_encoder = OneHotEncoder(
            categories=[["male", "female"]], handle_unknown="ignore"
        )
        self.sex_encoder.fit(metadata[["sex"]])

        # fit anatom_site_general encoder
        self.anatom_site_general_encoder = OneHotEncoder(
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
        self.anatom_site_general_encoder.fit(metadata[["anatom_site_general"]])

        self._is_fitted = True
        return self

    @mlflow.trace(name="MetadataEncoder")
    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Encode metadata into tensor using fitted transformers.

        Args:
            df: Metadata to encode

        Returns:
            torch.Tensor: Encoded metadata with shape (N, 8)
        """
        if not self._is_fitted:
            raise ValueError(
                "MetadataEncoder must be fitted before use. Call fit() first."
            )

        df = df[["age_approx", "sex", "anatom_site_general"]].copy()

        # age_approx: convert, impute, normalize using fitted scaler
        df.loc[:, "age_approx"] = pd.to_numeric(
            df["age_approx"], errors="coerce"
        ).fillna(0)
        age_scaled = self.age_scaler.transform(df[["age_approx"]])
        df.loc[:, "age_approx"] = age_scaled.flatten()

        # sex: one-hot encode using fitted encoder
        one_hot_sex = self.sex_encoder.transform(df[["sex"]]).toarray()
        for i, category in enumerate(self.sex_encoder.categories_[0]):
            df[f"is_{category}"] = one_hot_sex[:, i]
        df = df.drop(columns=["sex"])

        # anatom_site_general: one-hot encode using fitted encoder
        one_hot_anatom_site_general = self.anatom_site_general_encoder.transform(
            df[["anatom_site_general"]]
        ).toarray()
        for i, category in enumerate(self.anatom_site_general_encoder.categories_[0]):
            df[f"anatom_site_general_is_{category}"] = one_hot_anatom_site_general[:, i]
        df = df.drop(columns=["anatom_site_general"])

        # ensure all columns are numeric and return tensor
        df = df.astype(float)
        return torch.tensor(df.values, dtype=torch.float32)


@dataclass
class BatchEncoder:
    """
    Batch encoder that orchestrates image and metadata encoding for training batches.
    """

    image_encoder: ImageEncoder
    metadata_encoder: MetadataEncoder

    @mlflow.trace(name="BatchEncoder")
    def __call__(
        self, batch: Sequence[Tuple[pd.Series, Image, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process batch of (metadata, image, target) tuples.

        Args:
            batch: Sequence of (metadata_row, image, target) tuples

        Returns:
            Tuple of (image_tensor, metadata_tensor, target_tensor)
        """
        rows, images, targets = zip(*batch)

        df = pd.DataFrame(rows)

        # encode using fitted encoders
        x_imgs = self.image_encoder(images)
        x_mds = self.metadata_encoder(df)
        y = torch.tensor(targets, dtype=torch.float32)

        return x_imgs, x_mds, y
