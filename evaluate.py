"""
Evaluation module for speaker verification using EER (Equal Error Rate).

This module provides proper validation for speaker recognition models
by computing similarity-based metrics instead of classification loss.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import random
from tqdm import tqdm
import torchaudio


def generate_verification_pairs(
    data_dir: str,
    num_pairs: int = 1000,
    positive_ratio: float = 0.5,
    seed: int = 42
) -> List[Tuple[str, str, int]]:
    """
    Generate verification pairs from a speaker dataset.

    Args:
        data_dir: Root directory with speaker subdirectories
        num_pairs: Number of pairs to generate
        positive_ratio: Ratio of positive (same speaker) pairs
        seed: Random seed

    Returns:
        List of (file1_path, file2_path, label) tuples
        label: 1 = same speaker, 0 = different speakers
    """
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)

    # Build speaker -> files mapping
    speaker_files = {}
    for speaker_dir in sorted(data_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        files = list(speaker_dir.rglob('*.wav')) + \
                list(speaker_dir.rglob('*.m4a')) + \
                list(speaker_dir.rglob('*.flac'))

        if len(files) >= 2:  # Need at least 2 files for pairs
            speaker_files[speaker_dir.name] = files

    speakers = list(speaker_files.keys())
    num_positive = int(num_pairs * positive_ratio)
    num_negative = num_pairs - num_positive

    pairs = []

    # Generate positive pairs (same speaker)
    for _ in range(num_positive):
        speaker = random.choice(speakers)
        if len(speaker_files[speaker]) < 2:
            continue
        file1, file2 = random.sample(speaker_files[speaker], 2)
        pairs.append((str(file1), str(file2), 1))

    # Generate negative pairs (different speakers)
    for _ in range(num_negative):
        if len(speakers) < 2:
            break
        speaker1, speaker2 = random.sample(speakers, 2)
        file1 = random.choice(speaker_files[speaker1])
        file2 = random.choice(speaker_files[speaker2])
        pairs.append((str(file1), str(file2), 0))

    return pairs


def load_official_voxceleb_pairs(pairs_file: str, data_root: str) -> List[Tuple[str, str, int]]:
    """
    Load official VoxCeleb verification pairs from veri_test2.txt format.

    Format:
        1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
        0 id10309/0cYFdtyWVds/00001.wav id10296/q-8fGPszYYI/00001.wav

    Args:
        pairs_file: Path to pairs file (e.g., veri_test2.txt)
        data_root: Root directory containing audio files

    Returns:
        List of (file1_path, file2_path, label) tuples
    """
    pairs = []
    data_root = Path(data_root)

    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            label, path1, path2 = parts
            full_path1 = data_root / path1
            full_path2 = data_root / path2

            if full_path1.exists() and full_path2.exists():
                pairs.append((str(full_path1), str(full_path2), int(label)))

    return pairs


def compute_eer(
    model: torch.nn.Module,
    pairs: List[Tuple[str, str, int]],
    device: str = 'cuda',
    batch_size: int = 1,
    target_dim: int = None
) -> Dict[str, float]:
    """
    Compute Equal Error Rate (EER) for speaker verification.

    Args:
        model: Speaker recognition model
        pairs: List of (file1, file2, label) tuples
        device: Device to run on
        batch_size: Batch size for inference
        target_dim: Target MRL dimension (None = use full embedding)

    Returns:
        Dictionary with 'eer', 'threshold', 'accuracy' metrics
    """
    model.eval()

    similarities = []
    labels = []

    with torch.no_grad():
        for file1, file2, label in tqdm(pairs, desc="Computing EER"):
            # Load audio files
            try:
                wav1, sr1 = torchaudio.load(file1)
                wav2, sr2 = torchaudio.load(file2)

                # Resample if needed
                if sr1 != 16000:
                    wav1 = torchaudio.functional.resample(wav1, sr1, 16000)
                if sr2 != 16000:
                    wav2 = torchaudio.functional.resample(wav2, sr2, 16000)

                # Convert to mono
                if wav1.shape[0] > 1:
                    wav1 = wav1.mean(dim=0, keepdim=True)
                if wav2.shape[0] > 1:
                    wav2 = wav2.mean(dim=0, keepdim=True)

                # Move to device
                wav1 = wav1.unsqueeze(0).to(device)  # [1, 1, T]
                wav2 = wav2.unsqueeze(0).to(device)

                # Extract embeddings
                if target_dim is not None:
                    emb1 = model(wav1, target_dim=target_dim)
                    emb2 = model(wav2, target_dim=target_dim)
                else:
                    emb1 = model(wav1)
                    emb2 = model(wav2)

                # Normalize embeddings
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)

                # Compute cosine similarity
                similarity = F.cosine_similarity(emb1, emb2).item()

                similarities.append(similarity)
                labels.append(label)

            except Exception as e:
                # Skip problematic files
                continue

    # Convert to numpy
    similarities = np.array(similarities)
    labels = np.array(labels)

    # Compute EER
    eer, threshold = calculate_eer(similarities, labels)

    # Compute accuracy at EER threshold
    predictions = (similarities >= threshold).astype(int)
    accuracy = (predictions == labels).mean()

    return {
        'eer': eer,
        'threshold': threshold,
        'accuracy': accuracy,
        'num_pairs': len(labels)
    }


def calculate_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) from similarity scores and labels.

    Args:
        scores: Similarity scores (higher = more similar)
        labels: Ground truth labels (1 = same speaker, 0 = different)

    Returns:
        eer: Equal Error Rate (as fraction, not percentage)
        threshold: Threshold at EER
    """
    # Sort by scores
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Compute FAR and FRR at each threshold
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)

    # FRR: False Rejection Rate (missed positive matches)
    # FAR: False Acceptance Rate (false positive matches)

    far = []
    frr = []
    thresholds = []

    for threshold in np.linspace(scores.min(), scores.max(), 1000):
        # Predictions at this threshold
        predictions = (scores >= threshold).astype(int)

        # False rejections: label=1 but pred=0
        false_rejections = np.sum((labels == 1) & (predictions == 0))
        frr_value = false_rejections / num_positive if num_positive > 0 else 0

        # False acceptances: label=0 but pred=1
        false_acceptances = np.sum((labels == 0) & (predictions == 1))
        far_value = false_acceptances / num_negative if num_negative > 0 else 0

        far.append(far_value)
        frr.append(frr_value)
        thresholds.append(threshold)

    far = np.array(far)
    frr = np.array(frr)
    thresholds = np.array(thresholds)

    # Find EER (where FAR = FRR)
    eer_idx = np.argmin(np.abs(far - frr))
    eer = (far[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold


def evaluate_mrl_all_dims(
    model: torch.nn.Module,
    pairs: List[Tuple[str, str, int]],
    mrl_dims: List[int],
    device: str = 'cuda'
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate EER for all MRL dimensions.

    Args:
        model: MRL-enabled speaker recognition model
        pairs: Verification pairs
        mrl_dims: List of MRL dimensions to evaluate
        device: Device to run on

    Returns:
        Dictionary mapping dimension -> metrics
        {64: {'eer': 0.012, 'threshold': 0.55, ...}, ...}
    """
    results = {}

    for dim in mrl_dims:
        print(f"\nEvaluating {dim}D embeddings...")
        metrics = compute_eer(model, pairs, device=device, target_dim=dim)
        results[dim] = metrics

        print(f"  EER: {metrics['eer']*100:.3f}%")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")

    return results


if __name__ == "__main__":
    # Test EER computation
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <test_data_dir>")
        print("Example: python evaluate.py /home/tikim/dataset/voxceleb/test/wav")
        sys.exit(1)

    test_dir = sys.argv[1]

    print(f"Generating verification pairs from {test_dir}")
    pairs = generate_verification_pairs(test_dir, num_pairs=500)
    print(f"Generated {len(pairs)} pairs")

    # Print sample pairs
    print("\nSample pairs:")
    for i, (f1, f2, label) in enumerate(pairs[:3]):
        print(f"{i+1}. {'SAME' if label else 'DIFF'}: {Path(f1).parent.name} vs {Path(f2).parent.name}")

    # Test with dummy model (for development)
    print("\n⚠️ This is a test run with random embeddings")
    print("For real evaluation, use with trained model")
