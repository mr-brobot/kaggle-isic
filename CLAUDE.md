# ISIC 2024 - Skin Cancer Detection

## Project Overview

This is a machine learning project for the ISIC 2024 Skin Cancer Detection challenge from Kaggle. The project uses 3D-TBP (Total Body Photography) images to detect skin cancer.

## Data Structure

This project uses the [ISIC 2024 dataset on HuggingFace](https://huggingface.co/datasets/mrbrobot/isic-2024), converted from the original Kaggle competition data:

- Format: HuggingFace Dataset
- Dataset: `mrbrobot/isic-2024` (401,059 training samples)
- Images: Square dimensions 41x41 to 269x269 pixels (most 100x100-150x150)
- Metadata: 55 columns including demographics and medical features
- Target: Binary classification (0=benign, 1=malignant)
- Class imbalance: ~1020:1 ratio (benign:malignant)

## Key Dependencies

### ML/Data

- **PyTorch**: Primary ML framework
- **torchvision**: Image transformations
- **torcheval**: Evaluation metrics
- **transformers**: HuggingFace models and VLMs
- **datasets[vision]**: HuggingFace datasets with vision support
- **accelerate**: Distributed training support
- **h5py**: Legacy HDF5 file support
- **pandas**: Data manipulation
- **scikit-learn**: Preprocessing utilities
- **PIL/Pillow**: Image processing
- **matplotlib**: Visualization
- **numpy**: Numerical operations
- **pyarrow**: Efficient data serialization

### Infrastructure

- **AWS SDK (boto3)**: AWS service integration
- **pynvml**: NVIDIA GPU monitoring

### Experiment Tracking

- **trackio**: Lightweight experiment tracking

### Observability

- **AWS Distro for OpenTelemetry (ADOT)**: Collector-less setup with direct export to AWS X-Ray
- **Auto-instrumentation**: Scripts only (notebooks require manual instrumentation)
- **Configuration**: X-Ray endpoint auto-configured based on AWS region

### UI/CLI

- **rich**: Terminal formatting and progress bars
- **typer**: CLI framework
- **notebook**: Jupyter notebook interface
- **ipywidgets**: Interactive notebook widgets

### Development Tools

- **ruff**: Fast Python linter and formatter
- **pyrefly**: Type checking
- **pytest**: Testing framework

## Development Environment

- Python >=3.12 required
- CUDA support available
- Uses `uv` for dependency management

## Notebooks Structure

- `nbs/dataset.ipynb`: Dataset conversion from HDF5/CSV to HuggingFace format
- `nbs/eda.ipynb`: Exploratory data analysis of images and metadata
- `nbs/fusion.ipynb`: CNN-MLP fusion model training and evaluation
- `nbs/qwen.ipynb`: Vision-Language Model (Qwen3-VL) fine-tuning for classification

## Source Code Structure

### Core Modules (`src/isic/`)

- **`dataset.py`**: Data preprocessing and transformation utilities

  - `ImageEncoder`: torchvision-based image transformations
  - `MetadataEncoder`: sklearn-based feature encoding (age scaling, one-hot encoding)
  - `MetadataTextFormatter`: Format metadata as markdown for VLMs
  - `MessagesFormatter`: VLM chat message formatting
  - `VLMCollator`: HuggingFace Trainer-compatible collator for VLMs
  - `collate_batch()`: DataLoader collation for standard models

- **`models.py`**: Model architecture definitions

  - `MLP`: Configurable multi-layer perceptron
  - `FusionModel`: CNN + metadata fusion model for binary classification

- **`training.py`**: Training and validation utilities

  - `train()`: Training loop with metrics tracking
  - `validate()`: Validation with comprehensive metrics (accuracy, precision, recall, F1, AUROC)
  - `render_confusion_matrix()`: Rich-formatted confusion matrix display

- **`loss.py`**: Custom loss functions
  - `WeightedFocalLoss`: Focal loss with class balancing for imbalanced datasets
  - `VLMLoss`: Adapter for using PyTorch losses with HuggingFace Trainer

### Scripts (`scripts/`)

- **`bench.py`**: Data loading performance benchmarking

  - Measures batch loading times and throughput
  - Estimates full epoch duration
  - OpenTelemetry instrumentation for distributed tracing

- **`setup-xray.sh`**: One-time AWS X-Ray OTLP endpoint configuration

## Model Architectures

The project implements multiple model architectures for comparison:

### 1. CNN-MLP Fusion (`FusionModel`)

- **Image branch**: Custom CNN with configurable architecture
  - Layers defined as `(out_channels, kernel_size, pool_after)` tuples for easy visualization
  - Conv2d → BatchNorm2d → ReLU pattern per layer
  - MaxPool2d strategically placed for spatial downsampling
  - Flexible kernel sizes per layer (e.g., 5×5 early, 3×3 later)
- **Metadata branch**: MLP processing encoded features (age, sex, anatomical site)
- **Fusion**: Flattened CNN features + metadata → MLP → binary classification
- **Design**: Simple, interpretable baseline; can be replaced with pretrained backbones later

### 2. Vision-Language Model (Qwen3-VL-4B)

- **Base**: Qwen3-VL-4B-Instruct (4B parameters)
- **Task adaptation**: Replace language modeling head with binary classifier
- **Input format**: Images + metadata formatted as markdown tables in chat prompts
- **Training strategy**: Frozen backbone with fine-tuned classification head
- **Loss**: Weighted Focal Loss for extreme class imbalance handling

## Make Commands

- `make bootstrap`: Install OpenTelemetry auto-instrumentation packages
- `make bench`: Run data loading benchmark with OpenTelemetry tracing
- `make check`: Run linter (ruff) and type checker (pyrefly)
- `make format`: Format code with ruff
