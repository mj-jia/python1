"""训练。"""

from .losses import BlockinessLoss, SpatialAdaptiveLoss, flatness_weight

__all__ = ["BlockinessLoss", "SpatialAdaptiveLoss", "flatness_weight"]
