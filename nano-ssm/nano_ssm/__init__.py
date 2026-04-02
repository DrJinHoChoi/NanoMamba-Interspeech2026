# nano_ssm/__init__.py
# Nano AI SDK: Ultra-lightweight NC-SSM for edge keyword spotting
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.
#
# Usage:
#   import nano_ssm
#   model = nano_ssm.create("ncssm")
#   result = model.predict(audio)
#
#   # Streaming
#   from nano_ssm.streaming import StreamingEngine
#   engine = StreamingEngine(model, chunk_ms=100)
#   for chunk in audio_stream:
#       result = engine.feed(chunk)

"""
Nano AI SDK - NC-SSM (Noise-Conditioned State Space Model)
==========================================================

Ultra-lightweight keyword spotting models for edge devices.
7.4K-20K parameters with state-of-the-art noise robustness.

Quick start:
    >>> import nano_ssm
    >>> nano_ssm.list_models()
    ['ncssm', 'ncssm-large', 'ncssm-15k', 'ncssm-20k', 'ncssm-nanose']
    >>> model = nano_ssm.create("ncssm", n_classes=12)
    >>> print(model.summary())
"""

__version__ = "0.1.0"
__author__ = "Jin Ho Choi"

from .models.registry import create, list_models, model_info, MODEL_REGISTRY
from .models.core import NCSSM, GSC_LABELS_12
from .audio.features import load_audio, load_audio_stream
from .utils.checkpoint import load_checkpoint, save_checkpoint

# Convenience re-exports for direct factory access
from .models.factory import (
    create_ncssm,
    create_ncssm_large,
    create_ncssm_15k,
    create_ncssm_20k,
    create_ncssm_nanose,
)

__all__ = [
    # Top-level API
    "create", "list_models", "model_info",
    "load_audio", "load_audio_stream",
    "load_checkpoint", "save_checkpoint",
    # Model class
    "NCSSM", "GSC_LABELS_12",
    # Direct factory functions
    "create_ncssm", "create_ncssm_large",
    "create_ncssm_15k", "create_ncssm_20k",
    "create_ncssm_nanose",
    # Registry
    "MODEL_REGISTRY",
    # Version
    "__version__",
]
