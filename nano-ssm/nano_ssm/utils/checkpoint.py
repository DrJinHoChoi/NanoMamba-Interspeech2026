# nano_ssm/utils/checkpoint.py
# Checkpoint loading and saving utilities
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.

import torch
from pathlib import Path
from typing import Union, Optional


def load_checkpoint(model, path: Union[str, Path],
                    strict: bool = True,
                    map_location: Optional[str] = None) -> dict:
    """Load a checkpoint into an NCSSM or NanoMamba model.

    Handles both NCSSM wrapper and raw NanoMamba checkpoints.

    Args:
        model: NCSSM or NanoMamba model instance
        path: Path to .pt/.pth checkpoint file
        strict: Whether to strictly match state dict keys (default: True)
        map_location: Device mapping (default: auto-detect)

    Returns:
        Checkpoint dict (may contain extra keys like 'epoch', 'optimizer')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(str(path), map_location=map_location,
                            weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Try loading into NCSSM wrapper first, then into inner model
    from ..models.core import NCSSM
    if isinstance(model, NCSSM):
        try:
            model.model.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            # Keys might include 'model.' prefix
            cleaned = {k.replace('model.', '', 1): v
                       for k, v in state_dict.items()}
            model.model.load_state_dict(cleaned, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)

    return checkpoint if isinstance(checkpoint, dict) else {}


def save_checkpoint(model, path: Union[str, Path],
                    epoch: Optional[int] = None,
                    optimizer=None,
                    extra: Optional[dict] = None):
    """Save a checkpoint.

    Args:
        model: NCSSM or NanoMamba model
        path: Output path
        epoch: Optional epoch number
        optimizer: Optional optimizer state
        extra: Optional extra metadata
    """
    from ..models.core import NCSSM

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, NCSSM):
        state_dict = model.model.state_dict()
    else:
        state_dict = model.state_dict()

    checkpoint = {'model_state_dict': state_dict}
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, str(path))
