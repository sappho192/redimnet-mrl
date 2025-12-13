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
from evaluate import generate_verification_pairs, compute_eer


def load_checkpoint(checkpoint_path: Path, device='cpu'):
    """Load checkpoint and extract model state."""
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path.name}")
    print(f"{'='*70}")

    if not checkpoint_path.exists():
        print(f"[FAIL] Checkpoint not found: {checkpoint_path}")
        return None

    # Load checkpoint (weights_only=False since these are our own trusted checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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


def test_speaker_similarity(model, mrl_dims: List[int], device='cpu', voxceleb_pairs=None):
    """Test speaker similarity computation across all MRL dimensions."""
    print(f"\n{'='*70}")
    print(f"Testing Speaker Similarity Across All Dimensions")
    print(f"{'='*70}")

    model.eval()

    # If real VoxCeleb pairs available, use EER evaluation
    if voxceleb_pairs is not None and len(voxceleb_pairs) > 0:
        print(f"\nUsing real VoxCeleb audio ({len(voxceleb_pairs)} verification pairs)")
        print("Computing EER (Equal Error Rate) for each dimension...")

        eer_results = {}
        for dim in mrl_dims:
            print(f"\nEvaluating {dim}D embeddings...")
            metrics = compute_eer(model, voxceleb_pairs, device=device, target_dim=dim)
            eer_results[dim] = metrics
            print(f"  EER: {metrics['eer']*100:.2f}%")
            print(f"  Threshold: {metrics['threshold']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")

        # Display summary table
        print(f"\n{'='*70}")
        print("EER Summary (Lower is Better)")
        print(f"{'='*70}")
        print(f"{'Dimension':<12} {'EER':<12} {'Accuracy':<12} {'Threshold':<12} {'Status':<10}")
        print("-" * 70)

        for dim in mrl_dims:
            res = eer_results[dim]
            eer_pct = res['eer'] * 100
            acc_pct = res['accuracy'] * 100
            status = "[OK]" if eer_pct < 10.0 else "[WARN]"
            print(f"{dim}D{'':<9} {eer_pct:<12.2f}% {acc_pct:<12.2f}% {res['threshold']:<12.4f} {status:<10}")

        print(f"\n{'='*70}")
        print("Summary:")
        print(f"  EER = Equal Error Rate (industry standard metric)")
        print(f"  Lower EER = better speaker verification")
        print(f"  Good: <5%, Moderate: 5-10%, Poor: >10%")

        avg_eer = sum(r['eer'] for r in eer_results.values()) / len(mrl_dims) * 100
        print(f"  Average EER: {avg_eer:.2f}%")

    else:
        # Fallback to synthetic audio test
        print(f"\n⚠️ No VoxCeleb pairs provided - using synthetic audio")
        print(f"Note: Synthetic audio tests are NOT reliable for speaker verification!")

        # Simulate two utterances from same speaker
        audio1_same = torch.randn(1, 1, 48000).to(device)
        audio2_same = audio1_same + torch.randn(1, 1, 48000).to(device) * 0.1  # Similar

        # Simulate utterances from different speakers
        audio1_diff = torch.randn(1, 1, 48000).to(device)
        audio2_diff = torch.randn(1, 1, 48000).to(device)

        print(f"\nTest setup:")
        print(f"  Same speaker: utterance 2 = utterance 1 + 10% noise")
        print(f"  Different speakers: completely different random audio")

        results = {}

        for dim in mrl_dims:
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

                results[dim] = {
                    'same': sim_same,
                    'diff': sim_diff,
                    'delta': sim_same - sim_diff
                }

        # Display results in a table
        print(f"\n{'Dimension':<12} {'Same Spkr':<12} {'Diff Spkr':<12} {'Delta':<12} {'Status':<10}")
        print("-" * 70)

        for dim in mrl_dims:
            res = results[dim]
            status = "[OK]" if res['delta'] > 0 else "[WARN]"
            print(f"{dim}D{'':<9} {res['same']:<12.4f} {res['diff']:<12.4f} {res['delta']:<12.4f} {status:<10}")

        # Summary
        print(f"\n{'='*70}")
        print("Summary:")
        print(f"  Higher delta means better speaker discrimination")
        print(f"  Expected: Same speaker similarity > Different speaker similarity")
        print(f"  ⚠️ WARNING: These results may be misleading!")

        # Check if all dimensions show correct behavior
        all_correct = all(res['delta'] > 0 for res in results.values())
        if all_correct:
            print(f"  [OK] All dimensions correctly distinguish same vs different speakers")
        else:
            failing_dims = [dim for dim, res in results.items() if res['delta'] <= 0]
            print(f"  [WARN] Dimensions {failing_dims} may need more training")


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
    import argparse

    parser = argparse.ArgumentParser(description='Test MRL-ReDimNet checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing checkpoints (default: ./checkpoints/mrl_redimnet)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: ./config.yaml)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("MRL-ReDimNet Checkpoint Testing Suite")
    print(f"{'='*70}")

    # Setup paths
    project_root = Path(__file__).parent

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    else:
        checkpoint_dir = project_root / "checkpoints" / "mrl_redimnet"

    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        config_path = project_root / "config.yaml"

    print(f"\nPaths:")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Config file: {config_path}")

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

        # Test 5: Generate VoxCeleb verification pairs for real evaluation
        print(f"\n{'='*70}")
        print("Generating VoxCeleb Verification Pairs")
        print(f"{'='*70}")

        voxceleb_pairs = []
        vox1_test = Path.home() / "dataset" / "voxceleb" / "test" / "wav"
        vox2_test = Path.home() / "dataset" / "voxceleb2" / "test" / "aac"

        if vox1_test.exists():
            print(f"\nGenerating pairs from VoxCeleb1 test: {vox1_test}")
            vox1_pairs = generate_verification_pairs(str(vox1_test), num_pairs=250, seed=42)
            voxceleb_pairs.extend(vox1_pairs)
            print(f"  Generated {len(vox1_pairs)} VoxCeleb1 pairs")
        else:
            print(f"\n⚠️ VoxCeleb1 test not found at {vox1_test}")

        if vox2_test.exists():
            print(f"\nGenerating pairs from VoxCeleb2 test: {vox2_test}")
            vox2_pairs = generate_verification_pairs(str(vox2_test), num_pairs=250, seed=43)
            voxceleb_pairs.extend(vox2_pairs)
            print(f"  Generated {len(vox2_pairs)} VoxCeleb2 pairs")
        else:
            print(f"\n⚠️ VoxCeleb2 test not found at {vox2_test}")

        if voxceleb_pairs:
            print(f"\nTotal verification pairs: {len(voxceleb_pairs)}")

        # Test 6: Speaker similarity/verification with real audio
        test_speaker_similarity(model, mrl_dims, device=device, voxceleb_pairs=voxceleb_pairs if voxceleb_pairs else None)

        # Test 7: Individual audio file testing (optional, if alternative path exists)
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
