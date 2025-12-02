"""
Examples of using pretrained ReDimNet models with MRL.

This script demonstrates:
1. Loading official pretrained models
2. Converting to MRL
3. Two-stage training strategy
4. Inference with different dimensions
"""

import torch
from mrl import (
    load_pretrained_redimnet,
    create_mrl_from_pretrained,
    unfreeze_backbone,
    get_model_info
)


def example1_load_pretrained():
    """Example 1: Load official pretrained ReDimNet model."""
    print("\n" + "="*70)
    print("Example 1: Loading Pretrained ReDimNet")
    print("="*70)

    # Load pretrained model from torch.hub
    model = load_pretrained_redimnet(
        model_name='b2',      # Model variant: b0, b1, b2, b3, b4, b5, b6, M
        train_type='ptn',     # ptn (pretrained), ft_lm (large-margin), ft_mix (mixed)
        dataset='vox2',       # Dataset: vox2 or vb2+vox2+cnc
        device='cpu'
    )

    if model:
        # Get model information
        get_model_info(model)

        # Test inference
        print("Testing inference with pretrained model...")
        audio = torch.randn(1, 1, 48000)  # 3 seconds at 16kHz
        with torch.no_grad():
            embedding = model(audio)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {embedding.norm(p=2).item():.4f}\n")


def example2_create_mrl_from_pretrained():
    """Example 2: Create MRL model from pretrained checkpoint."""
    print("\n" + "="*70)
    print("Example 2: Creating MRL Model from Pretrained")
    print("="*70)

    # Create MRL model with pretrained backbone
    mrl_model = create_mrl_from_pretrained(
        model_name='b2',
        train_type='ptn',
        dataset='vox2',
        embed_dim=256,
        mrl_dims=[64, 128, 192, 256],
        device='cpu',
        freeze_backbone=False  # Full model trainable
    )

    # Get model information
    get_model_info(mrl_model)

    # Test multi-resolution inference
    print("Testing multi-resolution inference...")
    audio = torch.randn(1, 1, 48000)

    with torch.no_grad():
        # Get all dimensions
        emb_dict = mrl_model(audio, return_all_dims=True)
        print("\nAll dimensions:")
        for dim, emb in emb_dict.items():
            print(f"  {dim}D: shape={emb.shape}, norm={emb.norm(p=2).item():.4f}")

        # Get specific dimension
        print("\nSingle dimension:")
        emb_64d = mrl_model(audio, target_dim=64)
        print(f"  64D: shape={emb_64d.shape}, norm={emb_64d.norm(p=2).item():.4f}")


def example3_two_stage_training():
    """Example 3: Two-stage training strategy."""
    print("\n" + "="*70)
    print("Example 3: Two-Stage Training Strategy")
    print("="*70)

    # Stage 1: Train projection head only (backbone frozen)
    print("\nStage 1: Creating model with frozen backbone...")
    mrl_model = create_mrl_from_pretrained(
        model_name='b2',
        train_type='ft_lm',  # Use fine-tuned model for better starting point
        dataset='vox2',
        embed_dim=256,
        mrl_dims=[64, 128, 192, 256],
        device='cpu',
        freeze_backbone=True  # ⚠️ Freeze backbone
    )

    print("\nStage 1 - Only projection head is trainable:")
    get_model_info(mrl_model)

    # Simulate training for a few epochs...
    print("📚 Training projection head for 5 epochs...")
    print("   (In actual training, you would run your training loop here)")

    # Stage 2: Unfreeze backbone and fine-tune entire model
    print("\nStage 2: Unfreezing backbone for full model fine-tuning...")
    unfreeze_backbone(mrl_model)

    print("\nStage 2 - All parameters are trainable:")
    get_model_info(mrl_model)

    print("📚 Fine-tuning entire model for remaining epochs...")
    print("   (In actual training, you would continue your training loop here)")


def example4_different_model_variants():
    """Example 4: Using different model variants."""
    print("\n" + "="*70)
    print("Example 4: Different Model Variants")
    print("="*70)

    variants = [
        ('b0', '~1M params - fastest, good for edge devices'),
        ('b2', '~5M params - balanced performance/speed'),
        ('b5', '~9M params - high accuracy'),
    ]

    for model_name, description in variants:
        print(f"\n{model_name.upper()}: {description}")
        print("-" * 60)

        try:
            model = create_mrl_from_pretrained(
                model_name=model_name,
                train_type='ptn',
                dataset='vox2',
                embed_dim=256,
                mrl_dims=[64, 128, 192, 256],
                device='cpu',
                freeze_backbone=False
            )

            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")

            # Test inference speed (rough estimate)
            audio = torch.randn(1, 1, 48000)
            import time
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(audio, target_dim=256)
            elapsed = (time.time() - start) / 10
            print(f"Inference time (256D): {elapsed*1000:.2f}ms per utterance")

        except Exception as e:
            print(f"⚠️ Could not load {model_name}: {e}")


def example5_inference_comparison():
    """Example 5: Compare inference at different dimensions."""
    print("\n" + "="*70)
    print("Example 5: Inference Speed vs Accuracy Trade-off")
    print("="*70)

    # Create MRL model
    mrl_model = create_mrl_from_pretrained(
        model_name='b2',
        train_type='ptn',
        dataset='vox2',
        embed_dim=256,
        mrl_dims=[64, 128, 192, 256],
        device='cpu',
        freeze_backbone=False
    )

    print("\nComparing inference at different dimensions...")
    print("-" * 60)

    audio = torch.randn(1, 1, 48000)
    import time

    for dim in [64, 128, 192, 256]:
        # Time inference
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                emb = mrl_model(audio, target_dim=dim)
        elapsed = (time.time() - start) / 100

        # Memory (approximate)
        memory_mb = (dim * 4) / 1024 / 1024  # float32 = 4 bytes

        print(f"{dim}D: {elapsed*1000:.2f}ms/utterance, {memory_mb:.4f}MB memory")

    print("\n💡 Use case recommendations:")
    print("  64D:  Ultra-fast filtering, initial screening")
    print("  128D: Mobile/edge devices, real-time processing")
    print("  192D: Balanced accuracy and speed")
    print("  256D: Maximum accuracy for verification")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MRL-ReDimNet Pretrained Model Examples")
    print("="*70)

    # Run examples
    try:
        example1_load_pretrained()
        example2_create_mrl_from_pretrained()
        example3_two_stage_training()
        example4_different_model_variants()
        example5_inference_comparison()

        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
