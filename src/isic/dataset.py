import io
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from functools import cached_property, cache

import torch
from torch.utils.data import Dataset
from PIL.Image import Image

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


class MetadataPreprocessor:
    """
    Experimental metadata preprocessing for ISIC dataset.
    Classifies each column and applies appropriate scaling and encoding.
    """

    def __init__(self,
                 numerical_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None,
                 excluded_features: Optional[List[str]] = None):
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
            'isic_id', 'target', 'patient_id']

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
                col for col in df.columns if col not in self.excluded_features]

            # Auto-detect based on dtype
            for col in available_cols:
                if df[col].dtype in ['int64', 'float64']:
                    # Check if it's actually categorical (few unique values)
                    if df[col].nunique() <= 10 and col != 'age_approx':
                        self.categorical_features.append(col)
                    else:
                        self.numerical_features.append(col)
                else:
                    self.categorical_features.append(col)

    def fit(self, df: pd.DataFrame) -> 'MetadataPreprocessor':
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
            f"Preprocessing {len(self.numerical_features)} numerical and {len(self.categorical_features)} categorical features")
        print(
            f"Numerical: {self.numerical_features[:5]}{'...' if len(self.numerical_features) > 5 else ''}")
        print(
            f"Categorical: {self.categorical_features[:5]}{'...' if len(self.categorical_features) > 5 else ''}")

        # Fit numerical scaler
        if self.numerical_features:
            numerical_data = df[self.numerical_features].fillna(
                0)  # Fill NaN with 0
            self.numerical_scaler.fit(numerical_data)

        # Fit categorical encoders
        for feature in self.categorical_features:
            if feature in df.columns:
                # Fill NaN with 'unknown' for categorical features
                data = df[feature].fillna('unknown').astype(str)

                # Use LabelEncoder for high cardinality, OneHotEncoder for low cardinality
                if data.nunique() <= 10:
                    encoder = OneHotEncoder(
                        handle_unknown='ignore', sparse_output=False)
                    encoder.fit(data.values.reshape(-1, 1))
                    self.categorical_encoders[feature] = ('onehot', encoder)
                else:
                    encoder = LabelEncoder()
                    encoder.fit(data)
                    self.categorical_encoders[feature] = ('label', encoder)

        # Calculate output dimension
        self._calculate_output_dim()

        self.is_fitted = True
        return self

    def _calculate_output_dim(self) -> None:
        """Calculate the output dimension of processed features."""
        dim = len(self.numerical_features)  # Numerical features

        # Add categorical feature dimensions
        for feature, (encoder_type, encoder) in self.categorical_encoders.items():
            if encoder_type == 'onehot':
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
                data = df[feature].fillna('unknown').astype(str)
                encoder_type, encoder = self.categorical_encoders[feature]

                if encoder_type == 'onehot':
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


class ISICDataset(Dataset):
    """
    ISIC Dataset for multimodal skin cancer detection
    """

    def __init__(self, hdf5_path: Path, metadata_path: Path):
        self.hdf5_path = hdf5_path
        self.metadata_path = metadata_path

    @cached_property
    def metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.metadata_path, low_memory=False)

    @cache
    def image(self, key: str) -> Image:
        from PIL import Image
        with h5py.File(self.hdf5_path) as f:
            b = io.BytesIO(f[key][()])
            return Image.open(b)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[pd.Series, Image, int]:
        if type(idx) is not int:
            raise ValueError(
                f"ISICDataset: Unexpected index type {idx} ({type(idx)})")

        # ensure int for pandas compat
        idx = int(idx)
        row = self.metadata.iloc[idx]

        key = row["isic_id"]
        image = self.image(key)

        return row, image, row["target"]


def create_train_val_splits(metadata_df: pd.DataFrame,
                            val_size: float = 0.2,
                            random_state: int = 42) -> Tuple[List[int], List[int]]:
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
    targets = metadata_df['target'].values

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=targets,
        random_state=random_state
    )

    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(
        f"Train malignant rate: {metadata_df.iloc[train_idx]['target'].mean():.4f}")
    print(
        f"Val malignant rate: {metadata_df.iloc[val_idx]['target'].mean():.4f}")

    return train_idx.tolist(), val_idx.tolist()


def create_weighted_sampler(dataset: ISICDataset) -> torch.utils.data.WeightedRandomSampler:
    """
    Create weighted sampler for handling class imbalance.

    Args:
        dataset: ISIC dataset

    Returns:
        WeightedRandomSampler for balanced sampling
    """
    targets = dataset.metadata['target'].values
    class_counts = np.bincount(targets.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets.astype(int)]

    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")

    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
