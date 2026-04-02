# nano_ssm/audio/features.py
# Audio loading and feature extraction utilities
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.

import torch
import numpy as np
from typing import Optional, Union
from pathlib import Path


def load_audio(path: Union[str, Path], sr: int = 16000,
               duration_ms: int = 1000,
               mono: bool = True) -> torch.Tensor:
    """Load an audio file and return as a torch tensor.

    Handles resampling, mono conversion, and padding/trimming
    to fixed duration for keyword spotting.

    Args:
        path: Path to audio file (wav, flac, mp3, etc.)
        sr: Target sample rate (default: 16000)
        duration_ms: Target duration in ms (default: 1000)
        mono: Convert to mono (default: True)

    Returns:
        audio: (1, T) tensor where T = sr * duration_ms / 1000
    """
    import torchaudio

    waveform, orig_sr = torchaudio.load(str(path))

    # Convert to mono
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)

    # Pad or trim to target duration
    target_len = int(sr * duration_ms / 1000)
    if waveform.size(1) < target_len:
        pad = target_len - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.size(1) > target_len:
        waveform = waveform[:, :target_len]

    return waveform


def load_audio_stream(path: Union[str, Path], sr: int = 16000,
                      chunk_ms: int = 100) -> 'AudioStreamIterator':
    """Load audio file and yield chunks for streaming inference.

    Args:
        path: Path to audio file
        sr: Sample rate
        chunk_ms: Chunk size in milliseconds

    Returns:
        Iterator yielding (chunk_samples,) tensors
    """
    import torchaudio

    waveform, orig_sr = torchaudio.load(str(path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)  # (T,)
    chunk_size = int(sr * chunk_ms / 1000)

    return AudioStreamIterator(waveform, chunk_size)


class AudioStreamIterator:
    """Iterator that yields fixed-size audio chunks."""

    def __init__(self, waveform: torch.Tensor, chunk_size: int):
        self.waveform = waveform
        self.chunk_size = chunk_size
        self.pos = 0

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self.pos >= len(self.waveform):
            raise StopIteration
        end = min(self.pos + self.chunk_size, len(self.waveform))
        chunk = self.waveform[self.pos:end]
        # Pad last chunk if needed
        if len(chunk) < self.chunk_size:
            chunk = torch.nn.functional.pad(
                chunk, (0, self.chunk_size - len(chunk)))
        self.pos = end
        return chunk

    def __len__(self):
        """Number of chunks."""
        return (len(self.waveform) + self.chunk_size - 1) // self.chunk_size
