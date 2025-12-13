"""
Compare two specific checkpoints side by side using real VoxCeleb audio.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import ReDimNetMRL
from pretrained import create_mrl_from_pretrained
from evaluate import generate_verification_pairs, compute_eer


def load_and_test(checkpoint_path: Path, config_path: Path, voxceleb_pairs=None, device='cpu'):
    """Load checkpoint and run tests."""
    print(f"\n{'='*70}")
    print(f"Testing: {checkpoint_path}")
    print(f"{'='*70}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    advanced_config = config.get('advanced', {})
    mrl_dims = model_config['mrl_dims']

    # Create model
    model = create_mrl_from_pretrained(
        model_name=model_config['name'],
        train_type=advanced_config.get('train_type', 'ptn'),
        dataset=advanced_config.get('pretrained_dataset', 'vox2'),
        embed_dim=model_config['embed_dim'],
        mrl_dims=model_config['mrl_dims'],
        device=device,
        freeze_backbone=False
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"  File size: {checkpoint_path.stat().st_size / (1024 * 1024):.2f} MB")

    # Test embeddings at all dimensions (with random audio for norm analysis)
    print(f"\n{'='*70}")
    print("Embedding Norms (Random Audio - For Reference Only)")
    print(f"{'='*70}")

    audio = torch.randn(1, 1, 48000).to(device)

    results = {}
    for dim in mrl_dims:
        with torch.no_grad():
            emb = model(audio, target_dim=dim)
            results[dim] = {
                'norm': emb.norm(p=2).item(),
                'mean': emb.mean().item(),
                'std': emb.std().item()
            }

    print(f"\n{'Dim':<8} {'L2 Norm':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 50)
    for dim, res in results.items():
        print(f"{dim}D{'':<5} {res['norm']:<12.4f} {res['mean']:<12.4f} {res['std']:<12.4f}")

    # Real speaker verification test using VoxCeleb
    eer_results = {}
    if voxceleb_pairs is not None and len(voxceleb_pairs) > 0:
        print(f"\n{'='*70}")
        print(f"Real Speaker Verification (VoxCeleb Test Set)")
        print(f"{'='*70}")
        print(f"Testing on {len(voxceleb_pairs)} verification pairs from VoxCeleb")

        for dim in mrl_dims:
            print(f"\nEvaluating {dim}D embeddings...")
            metrics = compute_eer(model, voxceleb_pairs, device=device, target_dim=dim)
            eer_results[dim] = metrics
            print(f"  EER: {metrics['eer']*100:.2f}%")
            print(f"  Threshold: {metrics['threshold']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    else:
        print(f"\n⚠️ VoxCeleb test pairs not available - skipping EER evaluation")

    return {
        'checkpoint_info': checkpoint,
        'embedding_results': results,
        'eer_results': eer_results
    }


def main():
    """Compare two checkpoints."""
    print("\n" + "="*70)
    print("Checkpoint Comparison Tool")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    config_path = Path(__file__).parent / "config.yaml"

    # Local checkpoint (epoch 14 best)
    local_checkpoint = Path(__file__).parent / "checkpoints" / "mrl_redimnet" / "best.pt"

    # Temp checkpoint (epoch 42 latest)
    temp_checkpoint = Path.home() / "temp" / "redimnet-mrl" / "checkpoints" / "mrl_redimnet" / "latest.pt"

    # Generate VoxCeleb verification pairs
    print("\n" + "="*70)
    print("Generating VoxCeleb Test Pairs")
    print("="*70)

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

    print(f"\nTotal verification pairs: {len(voxceleb_pairs)}")

    print("\n" + "="*70)
    print("CHECKPOINT 1: Local Best (Epoch 14)")
    print("="*70)
    local_results = load_and_test(local_checkpoint, config_path, voxceleb_pairs, device)

    print("\n" + "="*70)
    print("CHECKPOINT 2: Temp Latest (Epoch 42)")
    print("="*70)
    temp_results = load_and_test(temp_checkpoint, config_path, voxceleb_pairs, device)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print("\n1. Embedding Norms Comparison (Random Audio - Reference Only):")
    print(f"{'Dim':<8} {'Local (E14)':<15} {'Temp (E42)':<15} {'Difference':<12}")
    print("-" * 60)

    mrl_dims = [64, 128, 192, 256]
    for dim in mrl_dims:
        local_norm = local_results['embedding_results'][dim]['norm']
        temp_norm = temp_results['embedding_results'][dim]['norm']
        diff = temp_norm - local_norm
        print(f"{dim}D{'':<5} {local_norm:<15.4f} {temp_norm:<15.4f} {diff:<12.4f}")

    # Compare EER results (real verification performance)
    if local_results['eer_results'] and temp_results['eer_results']:
        print("\n2. Speaker Verification Performance (EER on Real VoxCeleb Audio):")
        print(f"{'Dim':<8} {'Local (E14)':<15} {'Temp (E42)':<15} {'Difference':<12} {'Winner':<10}")
        print("-" * 75)

        for dim in mrl_dims:
            local_eer = local_results['eer_results'][dim]['eer'] * 100
            temp_eer = temp_results['eer_results'][dim]['eer'] * 100
            diff = temp_eer - local_eer
            winner = "Local ✓" if local_eer < temp_eer else "Temp ✓" if temp_eer < local_eer else "Tie"
            print(f"{dim}D{'':<5} {local_eer:<15.2f}% {temp_eer:<15.2f}% {diff:<12.2f}% {winner:<10}")

        print("\n3. Key Findings:")
        print("-" * 70)

        # Compare epochs
        local_epoch = local_results['checkpoint_info'].get('epoch', 0)
        temp_epoch = temp_results['checkpoint_info'].get('epoch', 0)
        print(f"  • Epoch difference: {temp_epoch - local_epoch} epochs ({local_epoch} vs {temp_epoch})")

        # Compare average EER (lower is better)
        local_avg_eer = sum(r['eer'] for r in local_results['eer_results'].values()) / len(mrl_dims) * 100
        temp_avg_eer = sum(r['eer'] for r in temp_results['eer_results'].values()) / len(mrl_dims) * 100
        print(f"  • Average EER: Local={local_avg_eer:.2f}%, Temp={temp_avg_eer:.2f}% ({temp_avg_eer-local_avg_eer:+.2f}%)")

        if local_avg_eer < temp_avg_eer:
            print(f"  ✓ Local checkpoint (E14) is BETTER - {temp_avg_eer - local_avg_eer:.2f}% lower EER")
            print(f"  → Confirms projection-only training (frozen backbone) is superior")
        elif temp_avg_eer < local_avg_eer:
            print(f"  ✓ Temp checkpoint (E42) is BETTER - {local_avg_eer - temp_avg_eer:.2f}% lower EER")
            print(f"  → Suggests continued training improved performance")
        else:
            print(f"  → Similar speaker verification performance")

        print(f"\n  Note: Lower EER = better speaker verification performance")
        print(f"        This metric directly measures ability to distinguish speakers")
    else:
        print("\n⚠️ EER comparison not available (VoxCeleb test data not found)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
