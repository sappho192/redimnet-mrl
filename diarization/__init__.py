"""
Speaker Diarization with ReDimNet-MRL

This package provides speaker diarization functionality using ReDimNet-MRL
embeddings integrated with the pyannote.audio pipeline.

Key Components:
- ReDimNetMRLSpeakerEmbedding: Pyannote-compatible wrapper for ReDimNet-MRL
- HierarchicalMRLClustering: Novel multi-stage clustering using MRL embeddings
- checkpoint_utils: Utilities for loading trained models

Example:
    >>> from diarization import ReDimNetMRLSpeakerEmbedding
    >>> from diarization import checkpoint_utils
    >>>
    >>> # Load embedding model
    >>> embedding = ReDimNetMRLSpeakerEmbedding(
    ...     checkpoint_path=checkpoint_utils.get_default_checkpoint_path(),
    ...     config_path=checkpoint_utils.get_default_config_path(),
    ... )
    >>>
    >>> # Use with pyannote pipeline
    >>> from pyannote.audio.pipelines import SpeakerDiarization
    >>> pipeline = SpeakerDiarization(segmentation="pyannote/segmentation-3.0")
    >>> pipeline._embedding = embedding
    >>> diarization = pipeline("audio.wav")
"""

__version__ = "0.1.0"

from .checkpoint_utils import (
    load_mrl_checkpoint,
    get_default_checkpoint_path,
    get_default_config_path,
)

from .redimnet_wrapper import ReDimNetMRLSpeakerEmbedding

from .hierarchical_mrl_clustering import (
    HierarchicalMRLClustering,
    PyannoteStyleClustering,
)

__all__ = [
    "load_mrl_checkpoint",
    "get_default_checkpoint_path",
    "get_default_config_path",
    "ReDimNetMRLSpeakerEmbedding",
    "HierarchicalMRLClustering",
    "PyannoteStyleClustering",
]
