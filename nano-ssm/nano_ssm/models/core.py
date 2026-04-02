# nano_ssm/models/core.py
# NC-SSM: Noise-Conditioned State Space Model wrapper for SDK
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.

"""
NCSSM - High-level wrapper around NanoMamba for easy inference.

Provides:
  - predict(): single-shot classification from raw audio
  - extract_features(): get mel + SNR features
  - get_ssm_state(): access internal SSM hidden state (for streaming)
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Union

import torch
import torch.nn as nn
import numpy as np

# Add parent repo to path so we can import nanomamba.py directly
_REPO_ROOT = Path(__file__).resolve().parents[2].parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanomamba import NanoMamba  # noqa: E402

# Google Speech Commands V2 12-class labels
GSC_LABELS_12 = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'stop', 'go', 'silence', 'unknown'
]


class NCSSM(nn.Module):
    """NC-SSM: High-level wrapper for NanoMamba inference.

    Adds convenience methods on top of NanoMamba:
      - predict(audio) -> {"label": str, "confidence": float, "logits": Tensor}
      - Automatic device handling
      - Label mapping

    Args:
        model: Underlying NanoMamba model instance
        labels: List of class labels (default: GSC V2 12-class)
        sr: Sample rate (default: 16000)
    """

    def __init__(self, model: NanoMamba, labels: Optional[List[str]] = None,
                 sr: int = 16000):
        super().__init__()
        self.model = model
        self.labels = labels or GSC_LABELS_12
        self.sr = sr

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def n_params_kb(self) -> float:
        """Model size in KB (INT8 quantized)."""
        return self.n_params / 1024

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - returns raw logits.

        Args:
            audio: (B, T) raw waveform at self.sr Hz
        Returns:
            logits: (B, n_classes)
        """
        return self.model(audio, **kwargs)

    @torch.no_grad()
    def predict(self, audio: torch.Tensor,
                top_k: int = 1) -> Union[Dict, List[Dict]]:
        """Classify audio and return human-readable results.

        Args:
            audio: (T,) or (B, T) raw waveform at self.sr Hz.
                   If 1D, treated as single utterance.
            top_k: Number of top predictions to return (default: 1)

        Returns:
            If top_k == 1:
                {"label": str, "confidence": float, "class_id": int}
            If top_k > 1:
                [{"label": str, "confidence": float, "class_id": int}, ...]
        """
        self.model.eval()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        logits = self.model(audio)  # (B, C)
        probs = torch.softmax(logits, dim=-1)

        if audio.size(0) == 1:
            probs = probs[0]  # (C,)
            if top_k == 1:
                idx = probs.argmax().item()
                return {
                    "label": self.labels[idx],
                    "confidence": probs[idx].item(),
                    "class_id": idx,
                }
            else:
                vals, idxs = probs.topk(top_k)
                return [
                    {"label": self.labels[i.item()],
                     "confidence": v.item(),
                     "class_id": i.item()}
                    for v, i in zip(vals, idxs)
                ]
        else:
            # Batch mode
            results = []
            for b in range(probs.size(0)):
                if top_k == 1:
                    idx = probs[b].argmax().item()
                    results.append({
                        "label": self.labels[idx],
                        "confidence": probs[b, idx].item(),
                        "class_id": idx,
                    })
                else:
                    vals, idxs = probs[b].topk(top_k)
                    results.append([
                        {"label": self.labels[i.item()],
                         "confidence": v.item(),
                         "class_id": i.item()}
                        for v, i in zip(vals, idxs)
                    ])
            return results

    @torch.no_grad()
    def predict_file(self, path: str, **kwargs) -> Dict:
        """Classify an audio file.

        Args:
            path: Path to audio file (wav, flac, etc.)
        Returns:
            Prediction dict from predict()
        """
        from ..audio.features import load_audio
        audio = load_audio(path, sr=self.sr)
        return self.predict(audio, **kwargs)

    def summary(self) -> str:
        """Return model summary string."""
        lines = [
            f"NC-SSM Model Summary",
            f"  Parameters:  {self.n_params:,} ({self.n_params_kb:.1f} KB INT8)",
            f"  d_model:     {self.model.d_model}",
            f"  n_layers:    {self.model.n_repeats}",
            f"  n_classes:   {len(self.labels)}",
            f"  Labels:      {self.labels}",
            f"  Sample rate: {self.sr} Hz",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (f"NCSSM(params={self.n_params:,}, "
                f"d_model={self.model.d_model}, "
                f"classes={len(self.labels)})")
