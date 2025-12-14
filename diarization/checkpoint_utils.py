"""
Utilities for loading trained MRL checkpoints.

This module provides functions to load trained ReDimNet-MRL models
from checkpoints for use in speaker diarization.
"""

import torch
import yaml
from pathlib import Path
import sys

# Add parent directory to path to import model and pretrained functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from pretrained import create_mrl_from_pretrained


def get_default_checkpoint_path():
    """
    Return path to best trained checkpoint.

    Returns:
        Path: Path to best.pt checkpoint
    """
    script_dir = Path(__file__).parent.parent
    checkpoint_path = script_dir / "checkpoints" / "mrl_redimnet" / "best.pt"
    return str(checkpoint_path)


def get_default_config_path():
    """
    Return path to training config.

    Returns:
        Path: Path to config.yaml
    """
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config.yaml"
    return str(config_path)


def load_mrl_checkpoint(
    checkpoint_path=None,
    config_path=None,
    device='cpu',
    verbose=True
):
    """
    Load trained MRL model from checkpoint.

    Steps:
    1. Load config YAML to get model architecture params
    2. Create ReDimNetMRL model with same architecture
    3. Load trained weights from checkpoint
    4. Set to eval mode
    5. Return initialized model

    Args:
        checkpoint_path: Path to checkpoint file (default: best.pt)
        config_path: Path to config YAML (default: config.yaml)
        device: Device to load model on ('cpu', 'cuda', or torch.device)
        verbose: Print loading information

    Returns:
        ReDimNetMRL model with loaded weights

    Example:
        >>> model = load_mrl_checkpoint(device='cuda')
        >>> audio = torch.randn(1, 1, 48000).cuda()
        >>> embeddings = model(audio, target_dim=256)
    """
    # Use defaults if not specified
    if checkpoint_path is None:
        checkpoint_path = get_default_checkpoint_path()
    if config_path is None:
        config_path = get_default_config_path()

    # Convert to Path objects
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)

    # Validate paths
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if verbose:
        print(f"Loading MRL checkpoint from: {checkpoint_path.name}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']

    if verbose:
        print(f"  Model: {model_config['name']}")
        print(f"  Embedding dim: {model_config['embed_dim']}")
        print(f"  MRL dims: {model_config['mrl_dims']}")

    # Create MRL model from pretrained backbone
    # Use create_mrl_from_pretrained to properly initialize the model
    model = create_mrl_from_pretrained(
        model_name=model_config.get('name', 'b2'),
        train_type=config.get('advanced', {}).get('train_type', 'ptn'),
        dataset=config.get('advanced', {}).get('pretrained_dataset', 'vox2'),
        embed_dim=model_config['embed_dim'],
        mrl_dims=model_config['mrl_dims'],
        device='cpu',  # Load to CPU first
        freeze_backbone=False,  # Don't freeze when loading
    )

    # Load checkpoint (weights_only=False since we trust our own checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        if verbose:
            print(f"  Loaded all weights successfully")
    except Exception as e:
        # Try loading with strict=False
        model.load_state_dict(state_dict, strict=False)
        if verbose:
            print(f"  Loaded weights with some mismatches (this is usually OK)")
            print(f"  Warning: {e}")

    # Move to device and set eval mode
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    model.eval()

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Print checkpoint metadata if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                print(f"  Checkpoint epoch: {checkpoint['epoch']}")
            if 'eer' in checkpoint or 'val_eer' in checkpoint:
                eer = checkpoint.get('eer', checkpoint.get('val_eer', 'N/A'))
                print(f"  Checkpoint EER: {eer}")

        print(f"  Device: {device}")

    return model


def load_checkpoint_from_path(checkpoint_path, device='cpu', verbose=True):
    """
    Load checkpoint with automatic config detection.

    This is a convenience function that looks for config.yaml in the
    parent directories automatically.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        verbose: Print loading information

    Returns:
        ReDimNetMRL model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)

    # Try to find config.yaml in parent directories
    config_path = None
    search_dir = checkpoint_path.parent
    for _ in range(3):  # Search up to 3 levels
        potential_config = search_dir / "config.yaml"
        if potential_config.exists():
            config_path = potential_config
            break
        # Try parent config.yaml
        potential_config = search_dir.parent / "config.yaml"
        if potential_config.exists():
            config_path = potential_config
            break
        search_dir = search_dir.parent

    if config_path is None:
        raise FileNotFoundError(
            f"Could not find config.yaml in parent directories of {checkpoint_path}"
        )

    return load_mrl_checkpoint(checkpoint_path, config_path, device, verbose)


if __name__ == "__main__":
    # Test checkpoint loading
    print("Testing checkpoint loading...\n")

    # Test with default paths
    try:
        model = load_mrl_checkpoint(device='cpu')
        print("\n[OK] Checkpoint loaded successfully")

        # Test inference
        print("\nTesting inference...")
        audio = torch.randn(2, 1, 48000)  # 2 samples, 3 seconds at 16kHz

        # Test single dimension
        with torch.no_grad():
            emb_256 = model(audio, target_dim=256)
            print(f"  256D embedding: {emb_256.shape}")

            # Test all dimensions
            emb_dict = model(audio, return_all_dims=True)
            for dim, emb in emb_dict.items():
                print(f"  {dim}D embedding: {emb.shape}")

        print("\n[OK] All tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
