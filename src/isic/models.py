from collections.abc import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        activation: type[nn.Module] = nn.LeakyReLU,
        batch_norm: bool = False,
    ):
        """
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, ..., output_dim]
            activation: Activation function class to use between layers
            batch_norm: Whether to apply batch normalization after each linear layer
        """
        super().__init__()

        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least 2 dimensions")

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            layers.append(activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@torch.compile
class FusionModel(nn.Module):
    """
    Fusion model combining CNN image processing with metadata MLP.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        cnn_layers: Sequence[tuple[int, int, bool]],
        metadata_layer_dims: Sequence[int],
        fusion_layer_dims: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ):
        """
        Args:
            image_shape: (height, width, channels) of input images
            cnn_layers: List of (out_channels, kernel_size, pool_after) tuples
                e.g., [(16, 5, True), (32, 3, False), (16, 3, True)]
            metadata_layer_dims: MLP layer dimensions for metadata [input_dim, hidden1, ..., output_dim]
            fusion_layer_dims: MLP layer dimensions for fusion [hidden1, ..., output_dim]
            activation: Activation function class to use in CNN and MLPs
        """
        super().__init__()
        h, w, c = image_shape

        # build cnn layers
        cnn_modules = []
        current_channels = c
        for out_channels, kernel_size, pool in cnn_layers:
            cnn_modules.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            cnn_modules.append(nn.BatchNorm2d(out_channels))
            cnn_modules.append(activation())

            if pool:
                cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

            current_channels = out_channels

        self.image_stack = nn.Sequential(*cnn_modules)

        # calculate CNN output size
        num_pools = sum(pool for _, _, pool in cnn_layers)
        spatial_size = h // (2**num_pools)
        final_channels = cnn_layers[-1][0]
        cnn_output_features = final_channels * spatial_size * spatial_size

        self.metadata_stack = MLP(metadata_layer_dims)

        fusion_input_dim = cnn_output_features + metadata_layer_dims[-1]
        self.fusion_head = nn.Sequential(
            MLP([fusion_input_dim] + fusion_layer_dims, batch_norm=True),
            nn.Linear(fusion_layer_dims[-1], 1),
        )

    def forward(self, x_img: torch.Tensor, x_md: torch.Tensor) -> torch.Tensor:
        x_img = self.image_stack(x_img)

        x_md = self.metadata_stack(x_md)

        x = torch.cat([torch.flatten(x_img, 1), x_md], dim=1)
        x = self.fusion_head(x)
        return x
