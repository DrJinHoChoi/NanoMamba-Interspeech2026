# nano_ssm/models/factory.py
# Factory functions for creating NC-SSM model variants
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.

import sys
from pathlib import Path
from typing import Optional, List

_REPO_ROOT = Path(__file__).resolve().parents[2].parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanomamba import (  # noqa: E402
    create_nanomamba_nc_matched,
    create_nanomamba_nc_large,
    create_nanomamba_nc_15k,
    create_nanomamba_nc_20k,
    create_nanomamba_nc_nanose_v3,
)
from .core import NCSSM


def create_ncssm(n_classes: int = 12,
                 labels: Optional[List[str]] = None,
                 pretrained: Optional[str] = None) -> NCSSM:
    """Create NC-SSM base model (7,443 params).

    The flagship model: matches BC-ResNet-1 accuracy with 5.1x fewer MACs.
    d_model=20, d_state=6, 2 layers with weight sharing.

    Args:
        n_classes: Number of output classes (default: 12 for GSC V2)
        labels: Optional class label list
        pretrained: Path to checkpoint file, or None
    Returns:
        NCSSM wrapper model
    """
    model = create_nanomamba_nc_matched(n_classes=n_classes)
    ncssm = NCSSM(model, labels=labels)
    if pretrained:
        from ..utils.checkpoint import load_checkpoint
        load_checkpoint(ncssm, pretrained)
    return ncssm


def create_ncssm_large(n_classes: int = 12,
                       use_nasg: bool = False,
                       labels: Optional[List[str]] = None,
                       pretrained: Optional[str] = None) -> NCSSM:
    """Create NC-SSM-Large model (~10.2K params).

    Scaled up: d_model=24, d_state=8 with 4.1x MAC advantage over BC-ResNet-1.

    Args:
        n_classes: Number of output classes
        use_nasg: Enable Noise-Aware Selective Gating (adds 4 params)
        labels: Optional class label list
        pretrained: Path to checkpoint file
    Returns:
        NCSSM wrapper model
    """
    model = create_nanomamba_nc_large(n_classes=n_classes, use_nasg=use_nasg)
    ncssm = NCSSM(model, labels=labels)
    if pretrained:
        from ..utils.checkpoint import load_checkpoint
        load_checkpoint(ncssm, pretrained)
    return ncssm


def create_ncssm_15k(n_classes: int = 12,
                     labels: Optional[List[str]] = None,
                     pretrained: Optional[str] = None) -> NCSSM:
    """Create NC-SSM-15K model (~15.8K params).

    Mid-scale variant with higher accuracy.

    Args:
        n_classes: Number of output classes
        labels: Optional class label list
        pretrained: Path to checkpoint file
    Returns:
        NCSSM wrapper model
    """
    model = create_nanomamba_nc_15k(n_classes=n_classes)
    ncssm = NCSSM(model, labels=labels)
    if pretrained:
        from ..utils.checkpoint import load_checkpoint
        load_checkpoint(ncssm, pretrained)
    return ncssm


def create_ncssm_20k(n_classes: int = 12,
                     labels: Optional[List[str]] = None,
                     pretrained: Optional[str] = None) -> NCSSM:
    """Create NC-SSM-20K model (~20K params).

    Largest variant: matches DS-CNN-S accuracy (96.4%) with 5.1x fewer MACs.

    Args:
        n_classes: Number of output classes
        labels: Optional class label list
        pretrained: Path to checkpoint file
    Returns:
        NCSSM wrapper model
    """
    model = create_nanomamba_nc_20k(n_classes=n_classes)
    ncssm = NCSSM(model, labels=labels)
    if pretrained:
        from ..utils.checkpoint import load_checkpoint
        load_checkpoint(ncssm, pretrained)
    return ncssm


def create_ncssm_nanose(n_classes: int = 12,
                        labels: Optional[List[str]] = None,
                        pretrained: Optional[str] = None) -> NCSSM:
    """Create NC-SSM + NanoSE v3 (~7.4K params, with speech enhancement).

    Parameter-matched to base NC-SSM but with integrated mel-domain
    speech enhancement for extreme low-SNR scenarios.

    Args:
        n_classes: Number of output classes
        labels: Optional class label list
        pretrained: Path to checkpoint file
    Returns:
        NCSSM wrapper model
    """
    model = create_nanomamba_nc_nanose_v3(n_classes=n_classes)
    ncssm = NCSSM(model, labels=labels)
    if pretrained:
        from ..utils.checkpoint import load_checkpoint
        load_checkpoint(ncssm, pretrained)
    return ncssm
