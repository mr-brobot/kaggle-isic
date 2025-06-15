# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for the ISIC 2024 Skin Cancer Detection challenge from Kaggle. The project uses 3D-TBP (Total Body Photography) images to detect skin cancer.

## Data Structure

- `data/train-image.hdf5`: HDF5 file containing 401,059 training images stored as binary data
- `data/train-metadata.csv`: CSV file with 55 columns of metadata including patient demographics and medical features
- Images are square with dimensions ranging from 41x41 to 269x269 pixels
- Most images are between 100x100 and 150x150 pixels
- Target variable is binary classification (0=benign, 1=malignant)

## Key Dependencies

- **PyTorch**: Primary ML framework for model development
- **h5py**: For reading HDF5 image files
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Preprocessing (OneHotEncoder, minmax_scale)
- **PIL/Pillow**: Image processing
- **matplotlib**: Data visualization
- **numpy**: Numerical operations

## Development Environment

- Python >=3.12 required
- CUDA support available (code checks for GPU)
- Uses `uv` for dependency management (uv.lock present)

## Data Access Setup

1. Set up Kaggle API key: https://www.kaggle.com/docs/api#getting-started-installation-&-authentication
2. Download data:
   ```bash
   mkdir data
   cd data
   kaggle competitions download -c isic-2024-challenge
   unzip isic-2024-challenge.zip
   rm isic-2024-challenge.zip
   cd -
   ```

## Notebooks Structure

- `nbs/1__eda.ipynb`: Exploratory data analysis of images and metadata
- `nbs/2__mlp.ipynb`: Multi-layer perceptron model implementation (in progress)

## Model Architecture

The current MLP model combines:

- Image features: Flattened 128x128 images processed through fully connected layers
- Metadata features: Age, sex, and anatomical site (one-hot encoded)
- Final prediction: Binary classification with sigmoid activation

## Data Processing Pipeline

- Images: Resize to 128x128, convert to tensors
- Metadata: Normalize age_approx, one-hot encode sex and anatom_site_general
- Custom Dataset class handles HDF5 file reading and metadata alignment
