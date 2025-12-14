#!/usr/bin/env python3
"""
Debug script to understand the embedding flow.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os

sys.path.insert(0, str(Path(__file__).parent))

# Load HF token
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key == 'HF_TOKEN':
                    os.environ['HF_TOKEN'] = value
                    try:
                        from huggingface_hub import login
                        login(token=value)
                    except Exception:
                        pass

import torch
import numpy as np

from checkpoint_utils import get_default_checkpoint_path, get_default_config_path
from redimnet_wrapper import ReDimNetMRLSpeakerEmbedding
from hierarchical_mrl_clustering import PyannoteStyleClustering

from pyannote.audio import Pipeline

# Wrap the embedding model to track calls
class DebugEmbeddingWrapper:
    def __init__(self, model):
        self.model = model
        self.call_count = 0
        self.all_embeddings = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        result = self.model(*args, **kwargs)

        # Log what we received
        print(f"\n[Embedding Call #{self.call_count}]")
        print(f"  Input shape: {args[0].shape if args else 'N/A'}")
        print(f"  Output shape: {result.shape}")
        print(f"  Output type: {type(result)}")

        # Check if multi-dim embeddings are available
        if hasattr(self.model, '_last_embeddings_dict') and self.model._last_embeddings_dict:
            print(f"  Multi-dim available: {list(self.model._last_embeddings_dict.keys())}")
            for dim, emb in self.model._last_embeddings_dict.items():
                print(f"    {dim}D: {emb.shape}")
        else:
            print(f"  Multi-dim available: NO")

        return result

    def __getattr__(self, name):
        return getattr(self.model, name)

# Wrap the clustering to track calls
class DebugClusteringWrapper:
    def __init__(self, clustering):
        self.clustering = clustering
        self.call_count = 0

    def __call__(self, embeddings, **kwargs):
        self.call_count += 1

        print(f"\n[Clustering Call #{self.call_count}]")
        print(f"  Input type: {type(embeddings)}")
        if isinstance(embeddings, dict):
            print(f"  Dict keys: {list(embeddings.keys())}")
            for k, v in embeddings.items():
                print(f"    {k}: {v.shape}")
        else:
            print(f"  Array shape: {embeddings.shape}")

        # Check if we can access multi-dim embeddings
        if hasattr(self.clustering, 'embedding_model') and self.clustering.embedding_model:
            if hasattr(self.clustering.embedding_model, '_last_embeddings_dict'):
                ed = self.clustering.embedding_model._last_embeddings_dict
                if ed:
                    print(f"  Retrieved from embedding_model._last_embeddings_dict:")
                    for k, v in ed.items():
                        print(f"    {k}D: {v.shape}")
                else:
                    print(f"  embedding_model._last_embeddings_dict is None/empty")
            else:
                print(f"  embedding_model does not have _last_embeddings_dict")
        else:
            print(f"  No embedding_model reference")

        result = self.clustering(embeddings, **kwargs)

        # Handle tuple return (hard_clusters, soft_clusters, centroids)
        if isinstance(result, tuple):
            hard_clusters = result[0]
            print(f"  Output: {len(np.unique(hard_clusters))} clusters")
        else:
            print(f"  Output: {len(np.unique(result))} clusters")

        return result

    def __getattr__(self, name):
        return getattr(self.clustering, name)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Debug embedding and clustering flow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--audio',
        required=True,
        help='Input audio file'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (default: cuda if available, else cpu)'
    )
    args = parser.parse_args()

    audio_file = Path(args.audio)
    if not audio_file.exists():
        print(f"ERROR: Audio file not found: {audio_file}")
        return

    print("="*80)
    print("Debug: Embedding and Clustering Flow")
    print("="*80)
    print(f"Audio: {audio_file.name}")
    print(f"Device: {args.device}")
    print()

    # Create embedding model
    print("Loading embedding model...")
    embedding_model = ReDimNetMRLSpeakerEmbedding(
        checkpoint_path=get_default_checkpoint_path(),
        config_path=get_default_config_path(),
        embedding_dim=256,
        extract_all_dims=True,
        device=torch.device(args.device)
    )
    print()

    # Wrap it
    wrapped_embedding = DebugEmbeddingWrapper(embedding_model)

    # Create pipeline
    print("Creating pipeline...")
    hf_token = os.environ.get('HF_TOKEN')

    import functools
    original_load = torch.load

    @functools.wraps(original_load)
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token
        )
    finally:
        torch.load = original_load

    # Replace embedding with wrapped version
    pipeline._embedding = wrapped_embedding

    # Create and wrap clustering
    clustering = PyannoteStyleClustering(
        method='hierarchical_mrl',
        embedding_model=embedding_model,  # Pass original, not wrapper
        coarse_threshold=0.7,
        refined_threshold=0.5,
        boundary_threshold=0.7,
    )
    wrapped_clustering = DebugClusteringWrapper(clustering)
    pipeline.clustering = wrapped_clustering

    print()
    print("="*80)
    print("Running diarization...")
    print("="*80)

    diarization = pipeline(str(audio_file))

    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Embedding calls: {wrapped_embedding.call_count}")
    print(f"Clustering calls: {wrapped_clustering.call_count}")

    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    elif hasattr(diarization, 'labels'):
        annotation = diarization
    else:
        annotation = diarization

    print(f"Detected speakers: {len(annotation.labels())}")
    print()

if __name__ == "__main__":
    main()
