"""
Matryoshka Representation Learning (MRL) for ReDimNet Speaker Recognition

This package provides MRL-enabled speaker embeddings with multi-resolution support.
"""

from .model import ReDimNetMRL, MatryoshkaProjection
from .losses import MatryoshkaLoss, AAMSoftmax
from .pretrained import (
    load_pretrained_redimnet,
    create_mrl_from_pretrained,
    unfreeze_backbone,
    get_model_info,
)

__version__ = "0.1.0"
__all__ = [
    "ReDimNetMRL",
    "MatryoshkaProjection",
    "MatryoshkaLoss",
    "AAMSoftmax",
    "load_pretrained_redimnet",
    "create_mrl_from_pretrained",
    "unfreeze_backbone",
    "get_model_info",
]
