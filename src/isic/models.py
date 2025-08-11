import torch
import torch.nn as nn
import timm
from typing import Tuple


@torch.compile
class MLP(nn.Module):
    def __init__(self, img_size: Tuple[int, int]):
        super().__init__()
        h, w = img_size

        self.flatten = nn.Flatten(1, -1)  # (B, H, W, 3) -> (B, 128*128*3)
        self.image_stack = nn.Sequential(
            nn.Linear(h * w * 3, 128),  # (B, H*W*3) -> (B, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.metadata_stack = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        self.output_stack = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x_img: torch.Tensor, x_md: torch.Tensor) -> torch.Tensor:
        x_img = self.flatten(x_img)
        x_img = self.image_stack(x_img)

        x_md = self.metadata_stack(x_md)

        x = torch.cat([x_img, x_md], dim=1)
        x = self.output_stack(x)
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
