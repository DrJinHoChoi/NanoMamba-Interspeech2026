# nano_ssm/models/registry.py
# String-based model registry for convenient model creation
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.

from typing import Optional, List
from .factory import (
    create_ncssm, create_ncssm_large,
    create_ncssm_15k, create_ncssm_20k,
    create_ncssm_nanose,
)
from .core import NCSSM

MODEL_REGISTRY = {
    "ncssm": {
        "factory": create_ncssm,
        "params": "7,443",
        "description": "NC-SSM base: d=20, N=6, 2 layers",
    },
    "ncssm-large": {
        "factory": create_ncssm_large,
        "params": "~10.2K",
        "description": "NC-SSM-Large: d=24, N=8, 2 layers",
    },
    "ncssm-15k": {
        "factory": create_ncssm_15k,
        "params": "~15.8K",
        "description": "NC-SSM-15K: scaled mid-range",
    },
    "ncssm-20k": {
        "factory": create_ncssm_20k,
        "params": "~20K",
        "description": "NC-SSM-20K: matches DS-CNN-S accuracy",
    },
    "ncssm-nanose": {
        "factory": create_ncssm_nanose,
        "params": "~7.4K",
        "description": "NC-SSM + NanoSE v3: integrated speech enhancement",
    },
}


def list_models() -> List[str]:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def create(name: str, n_classes: int = 12,
           labels: Optional[List[str]] = None,
           pretrained: Optional[str] = None,
           **kwargs) -> NCSSM:
    """Create a model by name.

    Args:
        name: Model name (see list_models())
        n_classes: Number of output classes
        labels: Optional class label list
        pretrained: Path to checkpoint file
        **kwargs: Extra arguments passed to factory function

    Returns:
        NCSSM wrapper model

    Example:
        >>> model = create("ncssm", n_classes=12)
        >>> model = create("ncssm-large", use_nasg=True)
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    factory = MODEL_REGISTRY[name]["factory"]
    return factory(n_classes=n_classes, labels=labels,
                   pretrained=pretrained, **kwargs)


def model_info(name: str) -> dict:
    """Get info about a model variant."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'")
    info = MODEL_REGISTRY[name].copy()
    del info["factory"]
    return info
