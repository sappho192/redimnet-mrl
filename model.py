"""
MRL-enabled ReDimNet model for speaker recognition.

This module provides a Matryoshka Representation Learning wrapper around
the original ReDimNet architecture, enabling multi-resolution embeddings.

Note: This module uses torch.hub to load the official ReDimNet model,
so you don't need to have the ReDimNet repository installed locally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatryoshkaProjection(nn.Module):
    """
    Multi-resolution projection head for Matryoshka Representation Learning.

    Enables extracting embeddings at multiple dimensions from a single forward pass.
    Prefix truncations (e.g., first 64, 128 dimensions) are trained to remain
    useful representations on their own.

    Args:
        input_dim: Input feature dimension (e.g., pooling output dimension)
        max_embed_dim: Maximum embedding dimension (full resolution)
        mrl_dims: List of supported MRL dimensions in ascending order
                  e.g., [64, 128, 192, 256]

    Example:
        >>> projection = MatryoshkaProjection(1024, 256, [64, 128, 192, 256])
        >>> x = torch.randn(32, 1024)  # Batch of pooled features
        >>>
        >>> # Get all MRL dimensions for training
        >>> emb_dict = projection(x, return_all_dims=True)
        >>> # emb_dict = {64: tensor[32,64], 128: tensor[32,128], ...}
        >>>
        >>> # Get single dimension for inference
        >>> emb = projection(x, target_dim=64)  # tensor[32, 64]
    """
    def __init__(self, input_dim, max_embed_dim, mrl_dims):
        super().__init__()
        self.input_dim = input_dim
        self.max_embed_dim = max_embed_dim
        self.mrl_dims = sorted(mrl_dims)

        # Validate MRL dimensions
        assert all(dim <= max_embed_dim for dim in mrl_dims), \
            f"All MRL dims must be <= max_embed_dim ({max_embed_dim})"
        assert max_embed_dim in mrl_dims, \
            f"max_embed_dim ({max_embed_dim}) should be in mrl_dims"

        # Batch normalization for input features
        self.bn = nn.BatchNorm1d(input_dim)

        # Single linear projection to maximum dimension
        self.linear = nn.Linear(input_dim, max_embed_dim)

    def forward(self, x, return_all_dims=False, target_dim=None):
        """
        Forward pass with flexible dimension output.

        Args:
            x: Input tensor [batch_size, input_dim]
            return_all_dims: If True, return dict with all MRL dimensions
            target_dim: If specified, return only this dimension (must be in mrl_dims)

        Returns:
            - If return_all_dims=True: dict {dim: tensor[B, dim]}
            - If target_dim specified: tensor[B, target_dim]
            - Otherwise: tensor[B, max_embed_dim] (full dimension)
        """
        # Normalize and project to full dimension
        x = self.bn(x)
        x = self.linear(x)  # [B, max_embed_dim]

        if return_all_dims:
            # Return dictionary with all MRL dimensions
            return {dim: x[:, :dim] for dim in self.mrl_dims}
        elif target_dim is not None:
            # Return specific dimension
            assert target_dim in self.mrl_dims or target_dim == self.max_embed_dim, \
                f"target_dim ({target_dim}) must be in mrl_dims {self.mrl_dims} or max_embed_dim"
            return x[:, :target_dim]
        else:
            # Return full embedding
            return x


class ReDimNetMRL(nn.Module):
    """
    MRL-enabled ReDimNet for speaker recognition.

    This wrapper extends ReDimNetWrap with Matryoshka Representation Learning,
    allowing flexible embedding dimensions for different deployment scenarios.

    Args:
        embed_dim: Maximum embedding dimension (default: 256)
        mrl_dims: List of MRL dimensions (default: [64, 128, 192, 256])
        **kwargs: Additional arguments passed to ReDimNetWrap

    Example:
        >>> model = ReDimNetMRL(embed_dim=256, mrl_dims=[64, 128, 192, 256])
        >>> audio = torch.randn(1, 1, 48000)  # [batch, channels, samples]
        >>>
        >>> # Training: get all dimensions
        >>> emb_dict = model(audio, return_all_dims=True)
        >>> # {64: tensor[1,64], 128: tensor[1,128], ...}
        >>>
        >>> # Inference: get specific dimension
        >>> emb_64d = model(audio, target_dim=64)  # Fast mode
        >>> emb_256d = model(audio, target_dim=256)  # High accuracy mode
    """
    def __init__(
        self,
        embed_dim=256,
        mrl_dims=None,
        **kwargs
    ):
        super().__init__()

        if mrl_dims is None:
            mrl_dims = [64, 128, 192, 256]

        self.embed_dim = embed_dim
        self.mrl_dims = sorted(mrl_dims)

        # Ensure embed_dim is in mrl_dims
        if embed_dim not in self.mrl_dims:
            self.mrl_dims.append(embed_dim)
            self.mrl_dims = sorted(self.mrl_dims)

        # Create base ReDimNet without final projection
        # We'll replace the projection layer with MatryoshkaProjection
        # Note: ReDimNet is loaded via torch.hub in pretrained.py
        # For standalone usage, must use create_mrl_from_pretrained()
        # This constructor is mainly for internal use
        try:
            from redimnet.model import ReDimNetWrap
            self.backbone = ReDimNetWrap(
                embed_dim=embed_dim,
                num_classes=None,  # No classification head by default
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "ReDimNetWrap not found. For standalone usage, please use:\n"
                "  from redimnet_mrl import create_mrl_from_pretrained\n"
                "  model = create_mrl_from_pretrained('b2', 'ft_lm', 'vox2')\n"
                "\nThis will automatically load ReDimNet via torch.hub."
            )

        # Get pooling output dimension from backbone
        pool_out_dim = self.backbone.pool_out_dim

        # Replace standard projection with MRL projection
        self.backbone.bn = nn.Identity()  # Disable original BN (MRL has its own)
        self.backbone.linear = nn.Identity()  # Disable original linear
        self.backbone.bn2 = None  # Disable secondary BN

        # Create MRL projection
        self.projection = MatryoshkaProjection(
            input_dim=pool_out_dim,
            max_embed_dim=embed_dim,
            mrl_dims=self.mrl_dims
        )

    def forward(self, x, return_all_dims=False, target_dim=None):
        """
        Forward pass with MRL support.

        Args:
            x: Input audio waveform [B, 1, T] or [B, T]
            return_all_dims: If True, return dict with all MRL dimensions
            target_dim: If specified, return only this dimension

        Returns:
            - If return_all_dims=True: dict {dim: tensor[B, dim]}
            - If target_dim specified: tensor[B, target_dim]
            - Otherwise: tensor[B, max_embed_dim]
        """
        # Feature extraction and backbone
        x = self.backbone.spec(x)

        if self.backbone.tf_optimized_arch:
            x = x.permute(0, 2, 1)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Backbone forward (without projection)
        features = self.backbone.backbone(x)
        pooled = self.backbone.pool(features)

        # MRL projection
        embeddings = self.projection(
            pooled,
            return_all_dims=return_all_dims,
            target_dim=target_dim
        )

        return embeddings

    def extract_embedding(self, audio_path, target_dim=None, device='cuda'):
        """
        Convenience method to extract embedding from audio file.

        Args:
            audio_path: Path to audio file
            target_dim: Embedding dimension (None = full dimension)
            device: Device to run inference on

        Returns:
            Embedding tensor
        """
        import torchaudio

        self.eval()
        self.to(device)

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(device)

        # Extract embedding
        with torch.no_grad():
            embedding = self(waveform, target_dim=target_dim)

        return embedding.cpu()


def load_pretrained_backbone(model, pretrained_path):
    """
    Load pretrained ReDimNet weights into MRL model.

    Args:
        model: ReDimNetMRL instance
        pretrained_path: Path to pretrained ReDimNetWrap checkpoint

    Returns:
        model with loaded backbone weights
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # Extract state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load backbone weights (excluding projection layers)
    model_dict = model.backbone.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and 'linear' not in k and 'bn2' not in k
    }

    model_dict.update(pretrained_dict)
    model.backbone.load_state_dict(model_dict, strict=False)

    print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} layers from pretrained model")

    return model


if __name__ == "__main__":
    # Test MRL model
    print("Testing ReDimNetMRL...")

    model = ReDimNetMRL(
        embed_dim=256,
        mrl_dims=[64, 128, 192, 256],
        F=72,
        C=12,
        out_channels=512,
    )

    # Test with random audio
    audio = torch.randn(2, 1, 48000)  # 2 samples, 3 seconds at 16kHz

    # Test training mode (all dimensions)
    print("\nTraining mode (all dimensions):")
    emb_dict = model(audio, return_all_dims=True)
    for dim, emb in emb_dict.items():
        print(f"  {dim}D: {emb.shape}")

    # Test inference mode (single dimension)
    print("\nInference mode (single dimension):")
    for dim in [64, 128, 192, 256]:
        emb = model(audio, target_dim=dim)
        print(f"  {dim}D: {emb.shape}")

    print("\n✅ Model test passed!")
