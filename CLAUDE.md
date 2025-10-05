# ISIC 2024 - Skin Cancer Detection

## Project Overview

This is a machine learning project for the ISIC 2024 Skin Cancer Detection challenge from Kaggle. The project uses 3D-TBP (Total Body Photography) images to detect skin cancer.

## Data Structure

- `data/train-image.hdf5`: HDF5 file containing 401,059 training images stored as binary data
- `data/train-metadata.csv`: CSV file with 55 columns of metadata including patient demographics and medical features
- Images are square with dimensions ranging from 41x41 to 269x269 pixels
- Most images are between 100x100 and 150x150 pixels
- Target variable is binary classification (0=benign, 1=malignant)

## Key Dependencies

### ML/Data

- **PyTorch**: Primary ML framework for model development
- **h5py**: For reading HDF5 image files
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Preprocessing (OneHotEncoder, minmax_scale)
- **PIL/Pillow**: Image processing
- **matplotlib**: Data visualization
- **numpy**: Numerical operations

### Experiment Tracking

- **trackio**: Lightweight experiment tracking (params, metrics, runs)

### Observability

- **AWS Distro for OpenTelemetry (ADOT)**: Collector-less setup with direct export to AWS X-Ray
- **Auto-instrumentation**: Scripts only (notebooks require manual instrumentation)
- **Configuration**: X-Ray endpoint auto-configured based on AWS region from profile/env vars

## Development Environment

- Python >=3.12 required
- CUDA support available (code checks for GPU)
- Uses `uv` for dependency management (uv.lock present)

## Code Style

- **Design Philosophy**: Prefer simplicity and elegance over complexity; strive for concise, elegant, and readable code; every line of code should have clear and articulable value
- **Strong Typing**: Use strict type checking and make types explicit; rely on the the type system to prevent errors
- **Functional Programming**: Prefer pure functions over classes; separate side effects and mutations
- **Testing Philosophy**: Focus tests on logic that the type system cannot verify; aim for meaningful test coverage rather than arbitrary metrics
- **Error Handling**: Only catch exceptions when you can meaningfully recover or transform; avoid empty catch-log-reraise patterns that add more noise than value
- **Comments**: Only use inline comments to explain context that is not obvious; avoid excessive comments as they add noise; prefer extracting complex logic into well-named functions.

## Notebooks Structure

- `nbs/1__eda.ipynb`: Exploratory data analysis of images and metadata
- `nbs/2__mlp.ipynb`: Multi-layer perceptron model implementation (in progress)

## Model Architecture

The current MLP model combines:

- Image features: Flattened 128x128 images processed through fully connected layers
- Metadata features: Age, sex, and anatomical site (one-hot encoded)
- Final prediction: Binary classification with sigmoid activation
