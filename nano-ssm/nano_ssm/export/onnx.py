# nano_ssm/export/onnx.py
# ONNX export utility for NC-SSM models
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.

import torch
from pathlib import Path
from typing import Union, Optional


def export_onnx(model, path: Union[str, Path],
                sr: int = 16000,
                duration_ms: int = 1000,
                opset_version: int = 17,
                dynamic_batch: bool = True) -> Path:
    """Export NC-SSM model to ONNX format.

    Args:
        model: NCSSM or NanoMamba model
        path: Output ONNX file path
        sr: Sample rate
        duration_ms: Input audio duration in ms
        opset_version: ONNX opset version (default: 17)
        dynamic_batch: Allow dynamic batch size (default: True)

    Returns:
        Path to exported ONNX file
    """
    from ..models.core import NCSSM

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, NCSSM):
        inner = model.model
    else:
        inner = model

    inner.eval()
    device = next(inner.parameters()).device

    # Create dummy input
    T = int(sr * duration_ms / 1000)
    dummy = torch.randn(1, T, device=device)

    # Dynamic axes
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            'audio': {0: 'batch'},
            'logits': {0: 'batch'},
        }

    torch.onnx.export(
        inner, dummy,
        str(path),
        input_names=['audio'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    return path
