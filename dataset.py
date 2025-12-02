"""
Dataset loaders for MRL speaker recognition training.

Supports VoxCeleb1/2 and other speaker recognition datasets.
"""

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import numpy as np
from typing import Optional, List, Tuple


class VoxCelebDataset(Dataset):
    """
    VoxCeleb speaker recognition dataset for MRL training.

    Supports VoxCeleb1 and VoxCeleb2 datasets with audio augmentation.

    Args:
        data_dir: Root directory containing speaker subdirectories
        sample_rate: Target sample rate (default: 16000)
        chunk_duration: Audio chunk duration in seconds (default: 3.0)
        augmentation: Enable data augmentation (default: True)
        min_duration: Minimum audio duration to include (default: 1.0 seconds)

    Directory structure expected:
        data_dir/
            id00001/
                video_id/
                    00001.wav
                    00002.wav
            id00002/
                ...

    Example:
        >>> dataset = VoxCelebDataset(
        ...     data_dir='/data/voxceleb2/dev/aac',
        ...     chunk_duration=3.0,
        ...     augmentation=True
        ... )
        >>> waveform, label = dataset[0]
        >>> print(waveform.shape, label)  # torch.Size([1, 48000]), 0
    """
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        augmentation: bool = True,
        min_duration: float = 1.0,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.min_samples = int(min_duration * sample_rate)
        self.augmentation = augmentation

        # Build file list
        self.audio_files = []
        self.labels = []
        self.speaker_to_idx = {}
        self._build_file_list()

        print(f"VoxCelebDataset initialized:")
        print(f"  - Total files: {len(self.audio_files)}")
        print(f"  - Total speakers: {len(self.speaker_to_idx)}")
        print(f"  - Chunk duration: {chunk_duration}s ({self.chunk_samples} samples)")
        print(f"  - Augmentation: {augmentation}")

    def _build_file_list(self):
        """Scan directory and build file list with speaker labels."""
        # Find all speaker directories (e.g., id00012, id10270)
        speaker_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        for speaker_idx, speaker_dir in enumerate(speaker_dirs):
            speaker_id = speaker_dir.name
            self.speaker_to_idx[speaker_id] = speaker_idx

            # Find all audio files for this speaker
            audio_files = list(speaker_dir.rglob('*.wav')) + \
                          list(speaker_dir.rglob('*.flac')) + \
                          list(speaker_dir.rglob('*.m4a'))

            for audio_file in audio_files:
                # Check if file is valid
                try:
                    info = torchaudio.info(str(audio_file))
                    if info.num_frames >= self.min_samples:
                        self.audio_files.append(audio_file)
                        self.labels.append(speaker_idx)
                except Exception as e:
                    # Skip corrupted files
                    continue

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Load and process audio file.

        Returns:
            waveform: Tensor of shape [1, num_samples]
            label: Integer speaker label
        """
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Random chunk extraction or padding
        if waveform.shape[1] > self.chunk_samples:
            # Random start point
            start = random.randint(0, waveform.shape[1] - self.chunk_samples)
            waveform = waveform[:, start:start + self.chunk_samples]
        else:
            # Pad if too short
            pad_amount = self.chunk_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Apply augmentation
        if self.augmentation and self.training:
            waveform = self._augment(waveform)

        return waveform, label

    def _augment(self, waveform):
        """
        Apply data augmentation to waveform.

        Includes:
        - Additive Gaussian noise
        - Volume perturbation
        - Speed perturbation (optional, commented out for efficiency)
        """
        # Additive noise (SNR: 20-40 dB)
        if random.random() < 0.5:
            snr_db = random.uniform(20, 40)
            noise = torch.randn_like(waveform)
            signal_power = waveform.norm(p=2)
            noise_power = noise.norm(p=2)
            snr = 10 ** (snr_db / 10)
            scale = signal_power / (snr * noise_power)
            waveform = waveform + scale * noise

        # Volume perturbation (±3 dB)
        if random.random() < 0.5:
            db_change = random.uniform(-3, 3)
            waveform = waveform * (10 ** (db_change / 20))

        # Clamp to prevent clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

    @property
    def training(self):
        """Check if dataset is in training mode (for augmentation)."""
        return self.augmentation

    def get_speaker_id(self, label):
        """Convert integer label to original speaker ID."""
        idx_to_speaker = {v: k for k, v in self.speaker_to_idx.items()}
        return idx_to_speaker.get(label, None)


class PairedSpeakerDataset(Dataset):
    """
    Dataset that returns pairs of audio samples (for verification tasks).

    Args:
        data_dir: Root directory
        sample_rate: Sample rate
        chunk_duration: Chunk duration
        positive_ratio: Ratio of positive pairs (same speaker) (default: 0.5)
    """
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        positive_ratio: float = 0.5,
    ):
        self.base_dataset = VoxCelebDataset(
            data_dir=data_dir,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            augmentation=False,
        )
        self.positive_ratio = positive_ratio

        # Group files by speaker
        self.speaker_files = {}
        for idx, label in enumerate(self.base_dataset.labels):
            if label not in self.speaker_files:
                self.speaker_files[label] = []
            self.speaker_files[label].append(idx)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Return a pair of audio samples and whether they're the same speaker.

        Returns:
            audio1, audio2: Waveform tensors
            is_same: 1 if same speaker, 0 if different
        """
        # Get first sample
        audio1, label1 = self.base_dataset[idx]

        # Decide whether to sample same or different speaker
        if random.random() < self.positive_ratio:
            # Positive pair: same speaker
            speaker_indices = self.speaker_files[label1]
            idx2 = random.choice(speaker_indices)
            while idx2 == idx:  # Ensure different utterance
                idx2 = random.choice(speaker_indices)
            audio2, label2 = self.base_dataset[idx2]
            is_same = 1
        else:
            # Negative pair: different speaker
            label2 = random.choice(list(self.speaker_files.keys()))
            while label2 == label1:
                label2 = random.choice(list(self.speaker_files.keys()))
            idx2 = random.choice(self.speaker_files[label2])
            audio2, _ = self.base_dataset[idx2]
            is_same = 0

        return audio1, audio2, torch.tensor(is_same)


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    chunk_duration: float = 3.0,
    shuffle: bool = True,
    augmentation: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for VoxCeleb dataset.

    Args:
        data_dir: Path to dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        chunk_duration: Audio chunk duration
        shuffle: Shuffle data
        augmentation: Enable augmentation
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance

    Example:
        >>> train_loader = create_dataloader(
        ...     data_dir='/data/voxceleb2/dev',
        ...     batch_size=64,
        ...     num_workers=8,
        ...     augmentation=True
        ... )
    """
    dataset = VoxCelebDataset(
        data_dir=data_dir,
        chunk_duration=chunk_duration,
        augmentation=augmentation,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for stable training
    )

    return dataloader


if __name__ == "__main__":
    # Test dataset
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path_to_voxceleb_data>")
        sys.exit(1)

    data_dir = sys.argv[1]

    print("Testing VoxCelebDataset...")
    dataset = VoxCelebDataset(
        data_dir=data_dir,
        chunk_duration=3.0,
        augmentation=True,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of speakers: {len(dataset.speaker_to_idx)}")

    # Test loading a sample
    print("\nLoading sample...")
    waveform, label = dataset[0]
    print(f"Waveform shape: {waveform.shape}")
    print(f"Label: {label}")
    print(f"Speaker ID: {dataset.get_speaker_id(label)}")

    # Test dataloader
    print("\nTesting DataLoader...")
    loader = create_dataloader(
        data_dir=data_dir,
        batch_size=4,
        num_workers=2,
        shuffle=True,
    )

    for batch_idx, (waveforms, labels) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Waveforms shape: {waveforms.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        if batch_idx >= 2:
            break

    print("\n✅ Dataset test passed!")
