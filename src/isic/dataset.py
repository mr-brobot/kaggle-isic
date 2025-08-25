import io
import warnings
from pathlib import Path
from typing import Dict, Sequence, Tuple

import h5py
import mlflow
import pandas as pd
import torch
from PIL.Image import Image
from sklearn.preprocessing import (
    OneHotEncoder,
    minmax_scale,
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class ISICDataset(Dataset):
    """
    ISIC Dataset for multimodal skin cancer detection.
    """

    def __init__(self, hdf5_path: Path, metadata_path: Path):
        self.hdf5_path = hdf5_path
        self.metadata_path = metadata_path
        self._hdf5_file = None
        self._metadata = None

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

    @mlflow.trace
    def image(self, key: str) -> Image:
        from PIL import Image

        if self._hdf5_file is None:
            raise RuntimeError(
                "Dataset is not open. Use 'with ISICDataset(...) as ds:' "
                "or call ds.open() before accessing images."
            )

        b = io.BytesIO(self._hdf5_file[key][()])
        return Image.open(b)

    @mlflow.trace
    def row(self, idx: int) -> pd.Series:
        return self.metadata.iloc[idx]

    def __len__(self):
        return len(self.metadata)

    @mlflow.trace
    def __getitem__(self, idx: int) -> Tuple[pd.Series, Image, int]:
        if type(idx) is not int:
            raise ValueError(f"ISICDataset: Unexpected index type {idx} ({type(idx)})")

        # ensure int for pandas compat
        idx = int(idx)
        row = self.row(idx)

        key = row["isic_id"]
        image = self.image(key)

        return row, image, row["target"]


@mlflow.trace
def collate_images(
    images: Sequence[Image], image_size: Tuple[int, int]
) -> torch.Tensor:
    # resize
    scaled_imgs = [img.resize(image_size) for img in images]

    # to tensor
    result = torch.stack([pil_to_tensor(img) for img in scaled_imgs])

    # normalize
    result = result.float() / 255.0

    return result


@mlflow.trace
def collate_metadata(
    rows: Sequence[pd.Series], encoders: Dict[str, OneHotEncoder]
) -> torch.Tensor:
    df = pd.DataFrame(rows)
    df = df[["age_approx", "sex", "anatom_site_general"]].copy()

    # age_approx: convert, impute, normalize
    df.loc[:, "age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce").fillna(0)
    df.loc[:, "age_approx"] = minmax_scale(df.loc[:, "age_approx"])

    # sex: one-hot
    sex_enc = encoders["sex"]
    one_hot_sex = sex_enc.transform(df[["sex"]]).toarray()
    for i, category in enumerate(sex_enc.categories_[0]):
        df[f"is_{category}"] = one_hot_sex[:, i]
    df = df.drop(columns=["sex"])

    # anatom_site_general: one-hot
    anatom_site_general_enc = encoders["anatom_site_general"]
    one_hot_anatom_site_general = anatom_site_general_enc.transform(
        df[["anatom_site_general"]]
    ).toarray()
    for i, category in enumerate(anatom_site_general_enc.categories_[0]):
        df[f"anatom_site_general_is_{category}"] = one_hot_anatom_site_general[:, i]
    df = df.drop(columns=["anatom_site_general"])

    # ensure all columns are numeric
    df = df.astype(float)
    return torch.tensor(df.values, dtype=torch.float32)


@mlflow.trace
def collate_batch(
    batch: Sequence[Tuple[pd.Series, Image, str]],
    img_size: Tuple[int, int],
    md_encoders: Dict[str, OneHotEncoder],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for training data"""
    rows, images, targets = zip(*batch)

    x_imgs = collate_images(images, img_size)
    x_mds = collate_metadata(rows, md_encoders)
    y = torch.tensor(targets, dtype=torch.float32)

    return x_imgs, x_mds, y
