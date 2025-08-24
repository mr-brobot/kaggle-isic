import io
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
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
    y = torch.tensor(targets, dtype=torch.int8)

    return x_imgs, x_mds, y


class MetadataPreprocessor:
    """
    Experimental metadata preprocessing for ISIC dataset.
    Classifies each column and applies appropriate scaling and encoding.
    """

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        excluded_features: Optional[List[str]] = None,
    ):
        """
        Initialize preprocessor with feature specifications.

        Args:
            numerical_features: List of numerical column names to include
            categorical_features: List of categorical column names to include
            excluded_features: List of column names to exclude from preprocessing
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.excluded_features = excluded_features or [
            "isic_id",
            "target",
            "patient_id",
        ]

        # Preprocessing components
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.label_encoders = {}

        # Fitted state
        self.is_fitted = False
        self.feature_names = []
        self.output_dim = 0

    def _auto_detect_features(self, df: pd.DataFrame) -> None:
        """Auto-detect numerical and categorical features if not specified."""
        if not self.numerical_features and not self.categorical_features:
            # Get all columns except excluded ones
            available_cols = [
                col for col in df.columns if col not in self.excluded_features
            ]

            # Auto-detect based on dtype
            for col in available_cols:
                if df[col].dtype in ["int64", "float64"]:
                    # Check if it's actually categorical (few unique values)
                    if df[col].nunique() <= 10 and col != "age_approx":
                        self.categorical_features.append(col)
                    else:
                        self.numerical_features.append(col)
                else:
                    self.categorical_features.append(col)

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        """
        Fit preprocessor on training data.

        Args:
            df: Training dataframe

        Returns:
            self for method chaining
        """
        df = df.copy()

        # Auto-detect features if not specified
        self._auto_detect_features(df)

        print(
            f"Preprocessing {len(self.numerical_features)} numerical and {len(self.categorical_features)} categorical features"
        )
        print(
            f"Numerical: {self.numerical_features[:5]}{'...' if len(self.numerical_features) > 5 else ''}"
        )
        print(
            f"Categorical: {self.categorical_features[:5]}{'...' if len(self.categorical_features) > 5 else ''}"
        )

        # Fit numerical scaler
        if self.numerical_features:
            numerical_data = df[self.numerical_features].fillna(0)  # Fill NaN with 0
            self.numerical_scaler.fit(numerical_data)

        # Fit categorical encoders
        for feature in self.categorical_features:
            if feature in df.columns:
                # Fill NaN with 'unknown' for categorical features
                data = df[feature].fillna("unknown").astype(str)

                # Use LabelEncoder for high cardinality, OneHotEncoder for low cardinality
                if data.nunique() <= 10:
                    encoder = OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    )
                    encoder.fit(data.values.reshape(-1, 1))
                    self.categorical_encoders[feature] = ("onehot", encoder)
                else:
                    encoder = LabelEncoder()
                    encoder.fit(data)
                    self.categorical_encoders[feature] = ("label", encoder)

        # Calculate output dimension
        self._calculate_output_dim()

        self.is_fitted = True
        return self

    def _calculate_output_dim(self) -> None:
        """Calculate the output dimension of processed features."""
        dim = len(self.numerical_features)  # Numerical features

        # Add categorical feature dimensions
        for feature, (encoder_type, encoder) in self.categorical_encoders.items():
            if encoder_type == "onehot":
                dim += encoder.categories_[0].shape[0]
            else:  # label encoder
                dim += 1

        self.output_dim = dim
        print(f"Total metadata feature dimension: {self.output_dim}")

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Transform dataframe to tensor.

        Args:
            df: Input dataframe (can be single row or multiple rows)

        Returns:
            Tensor of shape (n_samples, feature_dim)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()
        features = []

        # Handle numerical features
        if self.numerical_features:
            numerical_data = df[self.numerical_features].fillna(0)
            scaled_numerical = self.numerical_scaler.transform(numerical_data)
            features.append(scaled_numerical)

        # Handle categorical features
        for feature in self.categorical_features:
            if feature in df.columns:
                data = df[feature].fillna("unknown").astype(str)
                encoder_type, encoder = self.categorical_encoders[feature]

                if encoder_type == "onehot":
                    encoded = encoder.transform(data.values.reshape(-1, 1))
                    features.append(encoded)
                else:  # label encoder
                    # Handle unknown categories
                    encoded_values = []
                    for val in data:
                        try:
                            encoded_values.append(encoder.transform([val])[0])
                        except ValueError:  # Unknown category
                            encoded_values.append(-1)  # Use -1 for unknown
                    features.append(np.array(encoded_values).reshape(-1, 1))

        # Concatenate all features
        if features:
            combined_features = np.concatenate(features, axis=1)
            return torch.tensor(combined_features, dtype=torch.float32)
        else:
            # Return empty tensor if no features
            return torch.zeros((len(df), 0), dtype=torch.float32)


def create_train_val_splits(
    metadata_df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Create stratified train/validation splits.

    Args:
        metadata_df: Full metadata dataframe
        val_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices)
    """
    from sklearn.model_selection import train_test_split

    # Create stratified split
    indices = np.arange(len(metadata_df))
    targets = metadata_df["target"].values

    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=targets, random_state=random_state
    )

    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Train malignant rate: {metadata_df.iloc[train_idx]['target'].mean():.4f}")
    print(f"Val malignant rate: {metadata_df.iloc[val_idx]['target'].mean():.4f}")

    return train_idx.tolist(), val_idx.tolist()


def create_weighted_sampler(
    dataset: ISICDataset,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create weighted sampler for handling class imbalance.

    Args:
        dataset: ISIC dataset

    Returns:
        WeightedRandomSampler for balanced sampling
    """
    targets = dataset.metadata["target"].values
    class_counts = np.bincount(targets.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets.astype(int)]

    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")

    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
