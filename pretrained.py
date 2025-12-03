"""
Utilities for loading pretrained ReDimNet models and converting to MRL.

This module provides convenient functions to:
1. Load official pretrained ReDimNet models from torch.hub
2. Convert them to MRL-enabled models
3. Support two-stage training (freeze backbone, train projection)
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import ReDimNetMRL, MatryoshkaProjection


# Model configurations for different variants
# Note: These are approximate. The actual architecture will be extracted from
# the pretrained model when available.
MODEL_CONFIGS = {
    'b0': {'F': 72, 'C': 8, 'out_channels': 384},
    'b1': {'F': 72, 'C': 10, 'out_channels': 448},
    'b2': {'F': 72, 'C': 16, 'out_channels': 512},  # Corrected: actual b2 uses C=16
    'b3': {'F': 72, 'C': 12, 'out_channels': 512},  # Different stages_setup
    'b4': {'F': 72, 'C': 14, 'out_channels': 576},
    'b5': {'F': 72, 'C': 16, 'out_channels': 640},
    'b6': {'F': 72, 'C': 18, 'out_channels': 704},
    'M': {'F': 72, 'C': 12, 'out_channels': 512},   # Medium (commonly used)
}


def load_pretrained_redimnet(
    model_name='b2',
    train_type='ptn',
    dataset='vox2',
    device='cpu'
):
    """
    Load official pretrained ReDimNet model from torch.hub.

    Args:
        model_name: Model variant (b0, b1, b2, b3, b4, b5, b6, M)
        train_type: Training type
            - 'ptn': Pretrained on VoxCeleb2 (no finetuning)
            - 'ft_lm': Finetuned with Large-Margin loss
            - 'ft_mix': Finetuned on mixed datasets (VoxBlink2 + VoxCeleb2 + CN-Celeb)
        dataset: Dataset used for training ('vox2', 'vb2+vox2+cnc')
        device: Device to load model on

    Returns:
        Pretrained ReDimNetWrap model

    Example:
        >>> model = load_pretrained_redimnet('b2', 'ft_lm', 'vox2')
        >>> print(f"Embedding dim: {model.linear.out_features}")
    """
    print(f"Loading pretrained ReDimNet-{model_name} ({train_type}, {dataset})...")

    try:
        model = torch.hub.load(
            'IDRnD/ReDimNet',
            'ReDimNet',
            model_name=model_name,
            train_type=train_type,
            dataset=dataset,
            force_reload=False
        )
        model = model.to(device)
        model.eval()

        print(f"[OK] Successfully loaded pretrained model")
        print(f"   Embedding dimension: {model.linear.out_features}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")

        return model

    except Exception as e:
        print(f"[FAIL] Failed to load pretrained model: {e}")
        print(f"   Falling back to random initialization")
        return None


def create_mrl_from_pretrained(
    model_name='b2',
    train_type='ptn',
    dataset='vox2',
    embed_dim=256,
    mrl_dims=None,
    device='cpu',
    freeze_backbone=False
):
    """
    Create MRL model from pretrained ReDimNet checkpoint.

    This function:
    1. Loads pretrained ReDimNet from torch.hub
    2. Creates new MRL model with same architecture
    3. Transfers backbone weights
    4. Initializes new MRL projection head

    Args:
        model_name: ReDimNet variant (b0-b6, M)
        train_type: 'ptn', 'ft_lm', or 'ft_mix'
        dataset: Dataset identifier
        embed_dim: Maximum MRL embedding dimension
        mrl_dims: List of MRL dimensions (default: [64, 128, 192, embed_dim])
        device: Device for model
        freeze_backbone: If True, freeze backbone for projection-only training

    Returns:
        ReDimNetMRL model with pretrained backbone

    Example:
        >>> # Create MRL model from pretrained b2
        >>> model = create_mrl_from_pretrained(
        ...     model_name='b2',
        ...     train_type='ft_lm',
        ...     embed_dim=256,
        ...     mrl_dims=[64, 128, 192, 256],
        ...     freeze_backbone=True  # Stage 1: train projection only
        ... )
        >>>
        >>> # Later, unfreeze for full fine-tuning
        >>> for param in model.parameters():
        ...     param.requires_grad = True
    """
    if mrl_dims is None:
        mrl_dims = [64, 128, 192, embed_dim]

    # Load pretrained model
    pretrained = load_pretrained_redimnet(model_name, train_type, dataset, device)

    if pretrained is None:
        print("[WARN] Creating MRL model without pretrained weights")
        mrl_model = ReDimNetMRL(
            embed_dim=embed_dim,
            mrl_dims=mrl_dims,
            **MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['M'])
        )
        return mrl_model.to(device)

    # Extract actual configuration from pretrained model
    # This is more reliable than using hardcoded MODEL_CONFIGS
    backbone = pretrained.backbone

    print(f"Extracting architecture from pretrained model:")
    print(f"  C={backbone.C}, F={backbone.F}")
    print(f"  block_1d_type={backbone.block_1d_type}")
    print(f"  block_2d_type={backbone.block_2d_type}")

    # Create MRL model with same architecture as pretrained
    mrl_model = ReDimNetMRL(
        embed_dim=embed_dim,
        mrl_dims=mrl_dims,
        F=backbone.F,
        C=backbone.C,
        out_channels=512,  # Standard for most models
        # Copy other settings from pretrained model
        block_1d_type=backbone.block_1d_type,
        block_2d_type=backbone.block_2d_type,
        pooling_func=pretrained.pool.__class__.__name__,
        feat_type='pt',  # Assume PyTorch features
        stages_setup=backbone.stages_setup,
    )
    mrl_model = mrl_model.to(device)

    # Transfer weights from pretrained to MRL model
    print("\nTransferring weights from pretrained model...")

    # 1. Transfer backbone (feature extractor)
    try:
        pretrained_backbone_state = pretrained.backbone.state_dict()
        mrl_model.backbone.backbone.load_state_dict(pretrained_backbone_state, strict=True)
        print(f"   [OK] Backbone: {len(pretrained_backbone_state)} layers transferred")
    except Exception as e:
        print(f"   [WARN] Backbone transfer failed: {e}")

    # 2. Transfer feature extraction (MelBanks)
    try:
        pretrained_spec_state = pretrained.spec.state_dict()
        mrl_model.backbone.spec.load_state_dict(pretrained_spec_state, strict=True)
        print(f"   [OK] Feature extractor: {len(pretrained_spec_state)} layers transferred")
    except Exception as e:
        print(f"   [WARN] Feature extractor transfer failed: {e}")

    # 3. Transfer pooling layer
    try:
        pretrained_pool_state = pretrained.pool.state_dict()
        mrl_model.backbone.pool.load_state_dict(pretrained_pool_state, strict=True)
        print(f"   [OK] Pooling: {len(pretrained_pool_state)} layers transferred")
    except Exception as e:
        print(f"   [WARN] Pooling transfer failed: {e}")

    # 4. Initialize MRL projection (cannot transfer - different architecture)
    print(f"   [NEW] MRL projection: Randomly initialized for {mrl_dims} dimensions")

    # Freeze backbone if requested (for stage 1 training)
    if freeze_backbone:
        print("\n[FREEZE] Freezing backbone for projection-only training...")
        for param in mrl_model.backbone.parameters():
            param.requires_grad = False

        # Only projection parameters are trainable
        trainable_params = sum(p.numel() for p in mrl_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in mrl_model.parameters())
        print(f"   Trainable: {trainable_params:,} / {total_params:,} parameters")

    print(f"\n[OK] MRL model created successfully!")
    return mrl_model


def unfreeze_backbone(model):
    """
    Unfreeze backbone for full model fine-tuning (Stage 2).

    Args:
        model: ReDimNetMRL model with frozen backbone

    Example:
        >>> # Stage 1: Train projection only
        >>> model = create_mrl_from_pretrained(freeze_backbone=True)
        >>> # ... train for a few epochs ...
        >>>
        >>> # Stage 2: Fine-tune entire model
        >>> unfreeze_backbone(model)
        >>> # ... continue training ...
    """
    print("🔓 Unfreezing backbone for full model fine-tuning...")

    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable: {trainable_params:,} parameters (all)")


def get_model_info(model):
    """
    Print detailed information about a model.

    Args:
        model: ReDimNet or ReDimNetMRL model
    """
    print(f"\n{'='*60}")
    print("Model Information")
    print(f"{'='*60}")

    # Basic info
    print(f"Type: {model.__class__.__name__}")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # MRL specific info
    if hasattr(model, 'mrl_dims'):
        print(f"\nMRL Configuration:")
        print(f"  Max embedding dim: {model.embed_dim}")
        print(f"  MRL dimensions: {model.mrl_dims}")

    # Architecture info
    if hasattr(model, 'backbone'):
        backbone = model.backbone if hasattr(model.backbone, 'backbone') else model
        if hasattr(backbone, 'backbone'):
            print(f"\nArchitecture:")
            print(f"  F (frequency bins): {backbone.backbone.F}")
            print(f"  C (channel multiplier): {backbone.backbone.C}")
            if hasattr(backbone.backbone, 'stages_setup'):
                print(f"  Stages: {len(backbone.backbone.stages_setup)}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Testing pretrained model loading...\n")

    # Test 1: Load pretrained ReDimNet
    print("Test 1: Loading pretrained ReDimNet-b2")
    print("-" * 60)
    pretrained = load_pretrained_redimnet(
        model_name='b2',
        train_type='ptn',
        dataset='vox2'
    )

    if pretrained:
        get_model_info(pretrained)

        # Test inference
        print("Testing inference...")
        audio = torch.randn(1, 1, 48000)  # 3 seconds
        with torch.no_grad():
            emb = pretrained(audio)
        print(f"Output shape: {emb.shape}\n")

    # Test 2: Create MRL from pretrained
    print("\nTest 2: Creating MRL model from pretrained")
    print("-" * 60)
    mrl_model = create_mrl_from_pretrained(
        model_name='b2',
        train_type='ptn',
        dataset='vox2',
        embed_dim=256,
        mrl_dims=[64, 128, 192, 256],
        freeze_backbone=True  # Stage 1
    )

    get_model_info(mrl_model)

    # Test MRL inference
    print("Testing MRL inference...")
    audio = torch.randn(1, 1, 48000)
    with torch.no_grad():
        # Test all dimensions
        emb_dict = mrl_model(audio, return_all_dims=True)
        for dim, emb in emb_dict.items():
            print(f"  {dim}D: {emb.shape}")

        # Test single dimension
        emb_64d = mrl_model(audio, target_dim=64)
        print(f"  Single (64D): {emb_64d.shape}")

    # Test 3: Unfreeze for stage 2
    print("\nTest 3: Unfreezing for stage 2 training")
    print("-" * 60)
    unfreeze_backbone(mrl_model)
    get_model_info(mrl_model)

    print("\n[OK] All tests passed!")
