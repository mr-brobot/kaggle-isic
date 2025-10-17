import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class WeightedFocalLoss(nn.Module):
    """Focal loss for binary classification with class balancing.

    This implementation combines pos_weight (applied within the BCE
    calculation) with the focal modulation factor, providing flexibility
    to handle class imbalance through pos_weight while applying the focal
    modulation to focus on hard examples.

    References:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
        https://arxiv.org/abs/1708.02002

    Args:
        pos_weight: Weight for positive class, typically computed as
                    neg_count / pos_count for class balancing
        gamma: Focusing parameter that down-weights easy examples (default: 2.0)
               Higher gamma increases focus on hard examples
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted focal loss.

        Args:
            logits: Raw model outputs of shape (N, *) before sigmoid
            targets: Binary targets of shape (N, *) with values in {0, 1}

        Returns:
            Scalar loss value (mean over all elements)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.exp(-bce_loss)  # p_t = p if y=1, else 1-p
        focal_loss = (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


class VLMLoss:
    """Adapt PyTorch loss functions for VLM training with HuggingFace Trainer."""

    criterion: nn.Module

    def __init__(self, criterion: nn.Module) -> None:
        """
        Initialize with PyTorch loss function.

        Args:
            criterion: PyTorch loss module (e.g., WeightedFocalLoss, BCEWithLogitsLoss)
        """
        self.criterion = criterion

    def __call__(
        self,
        model_outputs: CausalLMOutputWithPast,
        labels: torch.Tensor,
        num_items_in_batch: int,
    ) -> torch.Tensor:
        """
        Compute loss from VLM model outputs for HuggingFace Trainer.

        Args:
            model_outputs: Model output object with .logits attribute
            labels: Ground truth labels of shape (batch_size,)
            num_items_in_batch: Total items in accumulated batch (unused)

        Returns:
            Scalar loss tensor
        """
        # extract logits from last token position
        # [batch_size, seq_length, 1] -> [batch_size]
        logits = model_outputs.logits[:, -1, 0]
        loss = self.criterion(logits, labels)
        return loss
