"""
Test script for MRL-ReDimNet checkpoints.

This script loads saved checkpoints and verifies:
1. Model architecture and parameters
2. Multi-resolution inference at all MRL dimensions
3. Embedding quality and norms
4. Inference speed comparison
5. Optional: Test on real audio from VoxCeleb
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
import time
import sys
import os
from typing import Dict, List

# Configure FFmpeg DLLs for torchcodec on Windows
if sys.platform == 'win32':
    ffmpeg_dll_paths = [
        r'C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-8.0.1-full_build-shared\bin',
        r'C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-7.1.1-full_build-shared\bin',
    ]
    for dll_path in ffmpeg_dll_paths:
        if Path(dll_path).exists():
            os.add_dll_directory(dll_path)
            print(f"Added FFmpeg DLL directory: {dll_path}")
            break

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import ReDimNetMRL
from pretrained import create_mrl_from_pretrained


def load_checkpoint(checkpoint_path: Path, device='cpu'):
    """Load checkpoint and extract model state."""
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path.name}")
    print(f"{'='*70}")

    if not checkpoint_path.exists():
        print(f"[FAIL] Checkpoint not found: {checkpoint_path}")
        return None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Print checkpoint info
    print(f"Checkpoint information:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    return checkpoint


def create_model_from_checkpoint(config_path: Path, device='cpu'):
    """Create model based on config."""
    print(f"\nLoading configuration from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model parameters
    model_config = config['model']
    advanced_config = config.get('advanced', {})

    print(f"\nModel configuration:")
    print(f"  Model name: {model_config['name']}")
    print(f"  Embedding dimension: {model_config['embed_dim']}")
    print(f"  MRL dimensions: {model_config['mrl_dims']}")
    print(f"  Use pretrained: {advanced_config.get('use_pretrained', False)}")

    # Create model
    if advanced_config.get('use_pretrained', False):
        print(f"\nCreating MRL model from pretrained {model_config['name']}...")
        model = create_mrl_from_pretrained(
            model_name=model_config['name'],
            train_type=advanced_config.get('train_type', 'ptn'),
            dataset=advanced_config.get('pretrained_dataset', 'vox2'),
            embed_dim=model_config['embed_dim'],
            mrl_dims=model_config['mrl_dims'],
            device=device,
            freeze_backbone=False
        )
    else:
        # Create model from scratch (not common for our setup)
        from model import ReDimNetMRL
        model = ReDimNetMRL(
            embed_dim=model_config['embed_dim'],
            mrl_dims=model_config['mrl_dims']
        ).to(device)

    return model, config


def test_model_inference(model, mrl_dims: List[int], device='cpu', num_tests=10):
    """Test model inference at different dimensions."""
    print(f"\n{'='*70}")
    print("Testing Multi-Resolution Inference")
    print(f"{'='*70}")

    model.eval()

    # Generate test audio (3 seconds at 16kHz)
    audio = torch.randn(1, 1, 48000).to(device)

    print(f"\nTest audio shape: {audio.shape}")
    print(f"Running {num_tests} iterations per dimension for speed test...\n")

    results = {}

    for dim in mrl_dims:
        print(f"Testing {dim}D embeddings:")
        print("-" * 60)

        # Warmup
        with torch.no_grad():
            _ = model(audio, target_dim=dim)

        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_tests):
                embedding = model(audio, target_dim=dim)
        elapsed = (time.time() - start_time) / num_tests

        # Check embedding properties
        emb_norm = embedding.norm(p=2).item()
        emb_mean = embedding.mean().item()
        emb_std = embedding.std().item()
        emb_min = embedding.min().item()
        emb_max = embedding.max().item()

        results[dim] = {
            'shape': embedding.shape,
            'norm': emb_norm,
            'mean': emb_mean,
            'std': emb_std,
            'min': emb_min,
            'max': emb_max,
            'time_ms': elapsed * 1000
        }

        print(f"  Shape: {embedding.shape}")
        print(f"  L2 norm: {emb_norm:.4f}")
        print(f"  Mean: {emb_mean:.4f}, Std: {emb_std:.4f}")
        print(f"  Range: [{emb_min:.4f}, {emb_max:.4f}]")
        print(f"  Inference time: {elapsed*1000:.2f}ms")
        print()

    return results


def test_all_dimensions_batch(model, mrl_dims: List[int], device='cpu'):
    """Test getting all dimensions at once."""
    print(f"\n{'='*70}")
    print("Testing Batch Multi-Dimension Output")
    print(f"{'='*70}")

    model.eval()

    # Generate test audio
    audio = torch.randn(4, 1, 48000).to(device)  # Batch of 4

    print(f"Test audio batch shape: {audio.shape}\n")

    with torch.no_grad():
        emb_dict = model(audio, return_all_dims=True)

    print("All dimensions extracted in single forward pass:")
    for dim, emb in emb_dict.items():
        print(f"  {dim}D: shape={emb.shape}, norm={emb[0].norm(p=2).item():.4f}")


def test_speaker_similarity(model, dim=128, device='cpu'):
    """Test speaker similarity computation."""
    print(f"\n{'='*70}")
    print(f"Testing Speaker Similarity (using {dim}D embeddings)")
    print(f"{'='*70}")

    model.eval()

    # Simulate two utterances from same speaker
    audio1_same = torch.randn(1, 1, 48000).to(device)
    audio2_same = audio1_same + torch.randn(1, 1, 48000).to(device) * 0.1  # Similar

    # Simulate utterances from different speakers
    audio1_diff = torch.randn(1, 1, 48000).to(device)
    audio2_diff = torch.randn(1, 1, 48000).to(device)

    with torch.no_grad():
        # Same speaker
        emb1_same = model(audio1_same, target_dim=dim)
        emb2_same = model(audio2_same, target_dim=dim)

        # Different speakers
        emb1_diff = model(audio1_diff, target_dim=dim)
        emb2_diff = model(audio2_diff, target_dim=dim)

        # Normalize embeddings
        emb1_same = F.normalize(emb1_same, p=2, dim=1)
        emb2_same = F.normalize(emb2_same, p=2, dim=1)
        emb1_diff = F.normalize(emb1_diff, p=2, dim=1)
        emb2_diff = F.normalize(emb2_diff, p=2, dim=1)

        # Compute cosine similarity
        sim_same = F.cosine_similarity(emb1_same, emb2_same).item()
        sim_diff = F.cosine_similarity(emb1_diff, emb2_diff).item()

    print(f"\nCosine similarity scores:")
    print(f"  Same speaker (synthetic): {sim_same:.4f}")
    print(f"  Different speakers (synthetic): {sim_diff:.4f}")
    print(f"  Difference: {sim_same - sim_diff:.4f}")

    if sim_same > sim_diff:
        print("  [OK] Model correctly distinguishes same vs different speakers")
    else:
        print("  [WARN] Model may need more training (same speaker similarity should be higher)")


def compare_checkpoints(checkpoint_dir: Path, config_path: Path, device='cpu'):
    """Compare multiple checkpoints."""
    print(f"\n{'='*70}")
    print("Comparing All Checkpoints")
    print(f"{'='*70}")

    checkpoint_files = ['epoch_0.pt', 'epoch_5.pt', 'best.pt', 'latest.pt']

    # Create model once
    model, config = create_model_from_checkpoint(config_path, device)
    mrl_dims = config['model']['mrl_dims']

    results = {}

    for ckpt_name in checkpoint_files:
        ckpt_path = checkpoint_dir / ckpt_name

        if not ckpt_path.exists():
            print(f"\n[WARN] Skipping {ckpt_name} (not found)")
            continue

        # Load checkpoint
        checkpoint = load_checkpoint(ckpt_path, device)

        if checkpoint is None:
            continue

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Test inference
        print(f"\nTesting {ckpt_name} at 128D...")
        model.eval()
        audio = torch.randn(1, 1, 48000).to(device)

        with torch.no_grad():
            emb = model(audio, target_dim=128)
            emb_norm = emb.norm(p=2).item()

        results[ckpt_name] = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'embedding_norm': emb_norm,
            'file_size_mb': ckpt_path.stat().st_size / (1024 * 1024)
        }

        print(f"  Embedding norm: {emb_norm:.4f}")
        print(f"  File size: {results[ckpt_name]['file_size_mb']:.2f} MB")

    # Summary table
    print(f"\n{'='*70}")
    print("Checkpoint Comparison Summary")
    print(f"{'='*70}")
    print(f"{'Checkpoint':<15} {'Epoch':<8} {'Val Loss':<12} {'Emb Norm':<12} {'Size (MB)':<10}")
    print("-" * 70)

    for ckpt_name, info in results.items():
        epoch = str(info['epoch'])
        val_loss = f"{info['val_loss']:.4f}" if info['val_loss'] != float('inf') else "N/A"
        emb_norm = f"{info['embedding_norm']:.4f}"
        size_mb = f"{info['file_size_mb']:.2f}"
        print(f"{ckpt_name:<15} {epoch:<8} {val_loss:<12} {emb_norm:<12} {size_mb:<10}")


def test_real_audio(model, audio_dir: Path, mrl_dims: List[int], device='cpu'):
    """Test on real audio files from VoxCeleb."""
    print(f"\n{'='*70}")
    print("Testing on Real Audio Files")
    print(f"{'='*70}")

    if not audio_dir.exists():
        print(f"[WARN] Audio directory not found: {audio_dir}")
        print("   Skipping real audio test.")
        return

    # Find some audio files
    audio_files = list(audio_dir.rglob("*.wav"))[:3]  # Test on first 3 files

    if not audio_files:
        audio_files = list(audio_dir.rglob("*.m4a"))[:3]

    if not audio_files:
        print("[WARN] No audio files found in directory")
        return

    print(f"Found {len(audio_files)} audio files to test\n")

    model.eval()

    for i, audio_file in enumerate(audio_files, 1):
        print(f"File {i}: {audio_file.name}")

        try:
            # Load audio using torchaudio (with torchcodec backend on Windows)
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_file))

            # Resample if needed
            if sample_rate != 16000:
                import torchaudio
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Take 3-second chunk
            chunk_size = 48000
            if waveform.shape[1] < chunk_size:
                # Pad if too short
                waveform = F.pad(waveform, (0, chunk_size - waveform.shape[1]))
            else:
                # Crop if too long
                waveform = waveform[:, :chunk_size]

            # Add batch dimension
            waveform = waveform.unsqueeze(0).to(device)

            # Test at different dimensions
            with torch.no_grad():
                for dim in mrl_dims:
                    emb = model(waveform, target_dim=dim)
                    norm = emb.norm(p=2).item()
                    print(f"  {dim}D: norm={norm:.4f}")

            print()

        except Exception as e:
            print(f"  [FAIL] Error loading audio: {e}\n")


def main():
    """Main test routine."""
    print(f"\n{'='*70}")
    print("MRL-ReDimNet Checkpoint Testing Suite")
    print(f"{'='*70}")

    # Setup paths
    project_root = Path(__file__).parent
    checkpoint_dir = project_root / "checkpoints" / "mrl_redimnet"
    config_path = project_root / "config.yaml"

    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load configuration and create model
    model, config = create_model_from_checkpoint(config_path, device)
    mrl_dims = config['model']['mrl_dims']

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  MRL dimensions: {mrl_dims}")

    # Test 1: Load and compare all checkpoints
    compare_checkpoints(checkpoint_dir, config_path, device)

    # Test 2: Load best checkpoint for detailed tests
    best_checkpoint_path = checkpoint_dir / "best.pt"

    if best_checkpoint_path.exists():
        print(f"\n{'='*70}")
        print("Loading best checkpoint for detailed testing")
        print(f"{'='*70}")

        checkpoint = load_checkpoint(best_checkpoint_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Test 3: Multi-resolution inference
        inference_results = test_model_inference(model, mrl_dims, device, num_tests=10)

        # Test 4: Batch all dimensions
        test_all_dimensions_batch(model, mrl_dims, device)

        # Test 5: Speaker similarity
        test_speaker_similarity(model, dim=128, device=device)

        # Test 6: Real audio (if available)
        voxceleb_path = Path("G:/DATASET/voxceleb")
        if voxceleb_path.exists():
            # Try different possible paths
            test_paths = [
                voxceleb_path / "dev" / "wav",
                voxceleb_path / "test" / "wav",
                voxceleb_path / "wav",
                voxceleb_path
            ]

            for test_path in test_paths:
                if test_path.exists():
                    test_real_audio(model, test_path, mrl_dims, device)
                    break
        else:
            print(f"\n[WARN] VoxCeleb dataset not found at {voxceleb_path}")
            print("   Skipping real audio test.")

    print(f"\n{'='*70}")
    print("[OK] All tests completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FAIL] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
