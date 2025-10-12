from typing import Tuple

import timm
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.
    """

    def __init__(
        self,
        layer_dims: list[int],
        activation: type[nn.Module] = nn.ReLU,
    ):
        """
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, ..., output_dim]
            activation: Activation function class to use between layers
        """
        super().__init__()

        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least 2 dimensions")

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@torch.compile
class FusionMLPModel(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        image_layer_dims: list[int],
        metadata_layer_dims: list[int],
        fusion_layer_dims: list[int],
    ):
        super().__init__()
        h, w, c = image_shape

        self.image_stack = nn.Sequential(
            nn.Flatten(1, -1),  # (B, H, W, C) -> (B, H*W*C)
            MLP([h * w * c] + image_layer_dims),
        )

        self.metadata_stack = MLP(metadata_layer_dims)

        fusion_input_dim = image_layer_dims[-1] + metadata_layer_dims[-1]
        self.fusion_head = nn.Sequential(
            MLP([fusion_input_dim] + fusion_layer_dims),
            nn.Linear(fusion_layer_dims[-1], 1),
        )

    def forward(self, x_img: torch.Tensor, x_md: torch.Tensor) -> torch.Tensor:
        x_img = self.image_stack(x_img)

        x_md = self.metadata_stack(x_md)

        x = torch.cat([x_img, x_md], dim=1)
        x = self.fusion_head(x)
        return x


class CNNMLPBaseline(nn.Module):
    """
    Two-headed CNN + MLP baseline model.
    - CNN branch: EfficientNet backbone for image features
    - MLP branch: Fully connected layers for metadata features
    - Fusion: Concatenation + final classification layers
    """

    def __init__(
        self,
        metadata_dim: int,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: int = 256,
    ):
        """
        Initialize CNN+MLP baseline model.

        Args:
            metadata_dim: Dimension of metadata features
            backbone_name: Name of the CNN backbone (from timm)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            hidden_dim: Hidden dimension for fusion layers
        """
        super().__init__()

        self.metadata_dim = metadata_dim
        self.backbone_name = backbone_name
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # Image branch - CNN backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(dummy_input)
            self.backbone_dim = backbone_output.shape[1]

        print(f"CNN backbone: {backbone_name}")
        print(f"CNN output dim: {self.backbone_dim}")
        print(f"Metadata input dim: {metadata_dim}")

        # Metadata branch - MLP
        if metadata_dim > 0:
            self.metadata_mlp = nn.Sequential(
                nn.Linear(metadata_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            metadata_output_dim = hidden_dim // 2
        else:
            self.metadata_mlp = nn.Identity()
            metadata_output_dim = 0

        # Fusion and classification layers
        fusion_input_dim = self.backbone_dim + metadata_output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        print(f"Fusion input dim: {fusion_input_dim}")
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Image tensor of shape (batch_size, 3, height, width)
            metadata: Metadata tensor of shape (batch_size, metadata_dim)

        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Image branch
        image_features = self.backbone(images)

        # Metadata branch
        if self.metadata_dim > 0:
            metadata_features = self.metadata_mlp(metadata)
            # Concatenate features
            combined_features = torch.cat([image_features, metadata_features], dim=1)
        else:
            combined_features = image_features

        # Classification
        output = self.classifier(combined_features)

        return output


MODEL_CONFIGS = {
    "efficientnet_b0": {
        "backbone_name": "efficientnet_b0",
        "hidden_dim": 256,
        "dropout_rate": 0.3,
    },
    "efficientnet_b2": {
        "backbone_name": "efficientnet_b2",
        "hidden_dim": 512,
        "dropout_rate": 0.5,
    },
    "efficientnet_b3": {
        "backbone_name": "efficientnet_b3",
        "hidden_dim": 512,
        "dropout_rate": 0.5,
    },
    "resnet50": {"backbone_name": "resnet50", "hidden_dim": 256, "dropout_rate": 0.4},
}
