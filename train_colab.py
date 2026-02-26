#!/usr/bin/env python3
# NanoMamba: Noise-Robust State Space Models for Keyword Spotting
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.
# Dual License: Free for academic/research use. Commercial use requires license.
# See LICENSE file. Contact: jinhochoi@smartear.co.kr for commercial licensing.
"""
NanoMamba Colab Training Script — Structural Noise Robustness
=============================================================

Standalone script for Google Colab. No external dependencies beyond
nanomamba.py and PyTorch/torchaudio.

Usage (Colab):
  1. Upload nanomamba.py to /content/drive/MyDrive/NanoMamba/
  2. Run this script:
     !python train_colab.py --models NanoMamba-Tiny --epochs 30
     !python train_colab.py --models NanoMamba-Tiny-TC --epochs 30
     !python train_colab.py --models NanoMamba-Tiny-WS-TC --epochs 30

  Or train all at once:
     !python train_colab.py --models NanoMamba-Tiny,NanoMamba-Tiny-TC,NanoMamba-Tiny-WS-TC --epochs 30

  Eval only (after training):
     !python train_colab.py --models NanoMamba-Tiny --eval_only --noise_types factory,white,babble,street,pink
"""

import os
import sys
import json
import time
import math
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings('ignore')

# ============================================================================
# Import NanoMamba (from same directory or Drive)
# ============================================================================

try:
    from nanomamba import (
        NanoMamba,
        create_nanomamba_tiny,
        create_nanomamba_small,
        create_nanomamba_tiny_tc,
        create_nanomamba_tiny_ws_tc,
        create_nanomamba_tiny_ws,
        create_nanomamba_tiny_pcen,
        create_nanomamba_small_pcen,
        create_nanomamba_tiny_pcen_tc,
        create_nanomamba_tiny_dualpcen,
        create_nanomamba_small_dualpcen,
    )
    print("  [OK] nanomamba.py loaded successfully")
except ImportError:
    print("  [ERROR] Cannot import nanomamba.py!")
    print("  Make sure nanomamba.py is in the same directory or on sys.path")
    sys.exit(1)


# ============================================================================
# Google Speech Commands V2 Dataset (12-class) — torchaudio-based
# ============================================================================

import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

GSC_LABELS_12 = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'stop', 'go', 'silence', 'unknown'
]

CORE_WORDS = set(['yes', 'no', 'up', 'down', 'left', 'right',
                  'on', 'off', 'stop', 'go'])


class _SubsetSC(SPEECHCOMMANDS):
    """torchaudio SPEECHCOMMANDS with proper train/val/test split."""

    def __init__(self, root, subset):
        super().__init__(root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return set(f.read().strip().splitlines())

        if subset == "validation":
            self._walker = [
                w for w in self._walker
                if os.path.relpath(w, self._path) in load_list("validation_list.txt")
            ]
        elif subset == "testing":
            self._walker = [
                w for w in self._walker
                if os.path.relpath(w, self._path) in load_list("testing_list.txt")
            ]
        elif subset == "training":
            excludes = load_list("validation_list.txt") | load_list("testing_list.txt")
            self._walker = [
                w for w in self._walker
                if os.path.relpath(w, self._path) not in excludes
            ]


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands V2 — 12-class wrapper over torchaudio.

    Uses torchaudio.datasets.SPEECHCOMMANDS for reliable downloading
    and train/val/test splitting. Maps 35 words to 12 classes:
    10 core keywords + silence + unknown.
    """

    def __init__(self, root, subset='training', n_mels=40, sr=16000,
                 clip_duration_ms=1000, augment=False):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.target_length = int(sr * clip_duration_ms / 1000)
        self.augment = augment
        self.subset = subset

        self.labels = GSC_LABELS_12
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

        # Load via torchaudio (handles download + split)
        print(f"  Loading {subset} split via torchaudio...")
        self._dataset = _SubsetSC(root, subset)

        # Build (path, label_idx) list with 12-class mapping
        self.samples = []
        for item in self._dataset._walker:
            keyword = os.path.basename(os.path.dirname(item))
            if keyword in CORE_WORDS:
                label = keyword
            else:
                label = 'unknown'
            self.samples.append((item, self.label_to_idx[label]))

        # Add silence samples from background noise
        bg_dir = os.path.join(self._dataset._path, '_background_noise_')
        if os.path.isdir(bg_dir):
            noise_files = [os.path.join(bg_dir, f)
                           for f in os.listdir(bg_dir) if f.endswith('.wav')]
            n_silence = 2000 if subset == 'training' else 500
            silence_idx = self.label_to_idx['silence']
            for i in range(n_silence):
                nf = noise_files[i % len(noise_files)]
                self.samples.append((nf + f'#silence_{i}', silence_idx))

        # Mel spectrogram parameters
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.mel_fb = self._create_mel_fb()

        # Count per class
        class_counts = {}
        for _, idx in self.samples:
            lbl = self.labels[idx]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        print(f"  [{subset}] {len(self.samples)} samples, {len(self.labels)} classes")
        print(f"    Per-class: { {k: v for k, v in sorted(class_counts.items())} }")

    def _create_mel_fb(self):
        n_freq = self.n_fft // 2 + 1
        mel_low = 0
        mel_high = 2595 * np.log10(1 + self.sr / 2 / 700)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)

        fb = np.zeros((self.n_mels, n_freq), dtype=np.float32)
        for i in range(self.n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                if j < n_freq:
                    fb[i, j] = ((j - bin_points[i]) /
                                max(bin_points[i + 1] - bin_points[i], 1))
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                if j < n_freq:
                    fb[i, j] = ((bin_points[i + 2] - j) /
                                max(bin_points[i + 2] - bin_points[i + 1], 1))
        return torch.from_numpy(fb)

    def _load_audio(self, path):
        actual_path = path.split('#')[0]
        try:
            waveform, sr = torchaudio.load(actual_path)
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            audio = waveform[0]
        except Exception:
            audio = torch.zeros(self.target_length)

        # For silence: random segment, scaled down
        if '#silence' in path:
            if len(audio) > self.target_length:
                start = np.random.randint(0, len(audio) - self.target_length)
                audio = audio[start:start + self.target_length]
            audio = audio * 0.1

        # Pad or trim to 1 second
        if len(audio) < self.target_length:
            audio = F.pad(audio, (0, self.target_length - len(audio)))
        elif len(audio) > self.target_length:
            audio = audio[:self.target_length]

        return audio

    def _compute_mel(self, audio):
        window = torch.hann_window(self.win_length)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          self.win_length, window=window,
                          return_complex=True)
        mag = spec.abs()
        mel = torch.matmul(self.mel_fb, mag)
        mel = torch.log(mel + 1e-8)
        return mel

    def _augment(self, audio):
        shift = np.random.randint(-1600, 1600)
        if shift > 0:
            audio = F.pad(audio[shift:], (0, shift))
        elif shift < 0:
            audio = F.pad(audio[:shift], (-shift, 0))
        vol = np.random.uniform(0.8, 1.2)
        audio = audio * vol
        if np.random.random() < 0.3:
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise
        return audio

    def cache_all(self):
        """Pre-load all audio into RAM for fast training."""
        print(f"  Caching {len(self.samples)} samples to RAM...", end=" ",
              flush=True)
        self._cache_audio = []
        self._cache_mel = []
        self._cache_labels = []
        for i, (path, label) in enumerate(self.samples):
            audio = self._load_audio(path)
            mel = self._compute_mel(audio)
            self._cache_audio.append(audio)
            self._cache_mel.append(mel)
            self._cache_labels.append(label)
            if (i + 1) % 10000 == 0:
                print(f"{i+1}", end=" ", flush=True)
        self._cache_audio = torch.stack(self._cache_audio)
        self._cache_mel = torch.stack(self._cache_mel)
        self._cache_labels = torch.tensor(self._cache_labels, dtype=torch.long)
        self._cached = True
        mem_mb = (self._cache_audio.nelement() * 4 +
                  self._cache_mel.nelement() * 4) / 1024**2
        print(f"Done! ({mem_mb:.0f} MB)", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if hasattr(self, '_cached') and self._cached:
            audio = self._cache_audio[idx]
            mel = self._cache_mel[idx]
            label = self._cache_labels[idx].item()
            if self.augment:
                audio = self._augment(audio.clone())
                mel = self._compute_mel(audio)
            return mel, label, audio

        path, label = self.samples[idx]
        audio = self._load_audio(path)
        if self.augment:
            audio = self._augment(audio)
        mel = self._compute_mel(audio)
        return mel, label, audio


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, scheduler, device,
                    label_smoothing=0.1, epoch=0, model_name=""):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (mel, labels, audio) in enumerate(train_loader):
        labels = labels.to(device)
        audio = audio.to(device)

        # NanoMamba takes raw audio
        logits = model(audio)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            acc = 100. * correct / total
            print(f"    [{model_name}] Batch {batch_idx+1}/{len(train_loader)} "
                  f"Loss: {total_loss/total:.4f} Acc: {acc:.1f}%",
                  flush=True)

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    for mel, labels, audio in val_loader:
        labels = labels.to(device)
        audio = audio.to(device)
        logits = model(audio)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


# ============================================================================
# Noise Generation (Audio-Domain)
# ============================================================================

def generate_noise_signal(noise_type, length, sr=16000, dataset_audios=None):
    if noise_type == 'factory':
        return _generate_factory_noise(length, sr)
    elif noise_type == 'white':
        noise = torch.randn(length)
        return noise / (noise.abs().max() + 1e-8) * 0.7
    elif noise_type == 'babble':
        return _generate_babble_noise(length, sr, dataset_audios)
    elif noise_type == 'street':
        return _generate_street_noise(length, sr)
    elif noise_type == 'pink':
        return _generate_pink_noise(length, sr)
    else:
        return _generate_factory_noise(length, sr)


def _generate_factory_noise(length, sr=16000):
    t = torch.arange(length, dtype=torch.float32) / sr
    hum = torch.zeros(length)
    for h in [50, 100, 150, 200, 250]:
        amp = 0.3 / (h / 50)
        phase = torch.rand(1).item() * 2 * math.pi
        hum += amp * torch.sin(2 * math.pi * h * t + phase)

    rumble_np = np.random.randn(length).astype(np.float32) * 0.2
    fft = np.fft.rfft(rumble_np)
    freqs = np.fft.rfftfreq(length, 1 / sr)
    mask = ((freqs >= 200) & (freqs <= 800)).astype(np.float32)
    mask = np.convolve(mask, np.ones(20) / 20, mode='same')
    rumble = torch.from_numpy(
        np.fft.irfft(fft * mask, n=length).astype(np.float32))

    impacts = torch.zeros(length)
    n_impacts = np.random.randint(5, 15)
    for _ in range(n_impacts):
        pos = np.random.randint(0, max(1, length - 1000))
        dur = np.random.randint(50, 500)
        amp = np.random.uniform(0.3, 0.8)
        env = torch.from_numpy(np.hanning(dur).astype(np.float32))
        impulse = amp * env * torch.randn(dur)
        end = min(pos + dur, length)
        impacts[pos:end] += impulse[:end - pos]

    pink = _generate_pink_noise(length, sr) * 0.15
    noise = hum + rumble + impacts + pink
    noise = noise / (noise.abs().max() + 1e-8) * 0.7
    return noise


def _generate_babble_noise(length, sr=16000, dataset_audios=None):
    n_talkers = np.random.randint(5, 9)
    babble = torch.zeros(length)

    if dataset_audios is not None and len(dataset_audios) > 0:
        indices = np.random.choice(len(dataset_audios), n_talkers, replace=True)
        for idx in indices:
            sample = dataset_audios[idx]
            if len(sample) < length:
                sample = F.pad(sample, (0, length - len(sample)))
            elif len(sample) > length:
                start = np.random.randint(0, len(sample) - length)
                sample = sample[start:start + length]
            babble += sample
    else:
        for _ in range(n_talkers):
            t = torch.arange(length, dtype=torch.float32) / sr
            f0 = np.random.uniform(100, 300)
            sig = 0.3 * torch.sin(2 * math.pi * f0 * t)
            sig += 0.1 * torch.sin(2 * math.pi * f0 * 2 * t)
            for fc in [730, 1090, 2440]:
                sig += 0.15 * torch.sin(2 * math.pi * fc * t)
            onset = int(np.random.uniform(0.05, 0.3) * sr)
            dur = int(np.random.uniform(0.3, 0.8) * sr)
            dur = min(dur, length - onset)
            env = torch.zeros(length)
            if dur > 0:
                env[onset:onset + dur] = torch.from_numpy(
                    np.hanning(dur).astype(np.float32))
            babble += sig * env

    babble = babble / (babble.abs().max() + 1e-8) * 0.7
    return babble


def _generate_street_noise(length, sr=16000):
    t = torch.arange(length, dtype=torch.float32) / sr
    rumble_np = np.random.randn(length).astype(np.float32) * 0.3
    fft = np.fft.rfft(rumble_np)
    freqs = np.fft.rfftfreq(length, 1 / sr)
    mask = ((freqs >= 20) & (freqs <= 200)).astype(np.float32)
    mask = np.convolve(mask, np.ones(10) / 10, mode='same')
    rumble = torch.from_numpy(
        np.fft.irfft(fft * mask, n=length).astype(np.float32))

    horns = torch.zeros(length)
    for _ in range(np.random.randint(1, 4)):
        pos = np.random.randint(0, max(1, length - 3000))
        dur = np.random.randint(1000, 3000)
        freq = np.random.uniform(300, 600)
        amp = np.random.uniform(0.3, 0.6)
        horn_t = torch.arange(dur, dtype=torch.float32) / sr
        horn = amp * torch.sin(2 * math.pi * freq * horn_t)
        env = torch.from_numpy(np.hanning(dur).astype(np.float32))
        horn = horn * env
        end = min(pos + dur, length)
        horns[pos:end] += horn[:end - pos]

    road = torch.randn(length) * 0.15
    engine_freq = np.random.uniform(80, 150)
    engine = 0.2 * torch.sin(2 * math.pi * engine_freq * t)
    engine += 0.1 * torch.sin(2 * math.pi * engine_freq * 2 * t)

    noise = rumble + horns + road + engine
    noise = noise / (noise.abs().max() + 1e-8) * 0.7
    return noise


def _generate_pink_noise(length, sr=16000):
    white = np.random.randn(length).astype(np.float32)
    fft_w = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, 1 / sr)
    freqs[0] = 1
    pink = np.fft.irfft(fft_w / np.sqrt(freqs), n=length).astype(np.float32)
    pink_t = torch.from_numpy(pink)
    pink_t = pink_t / (pink_t.abs().max() + 1e-8) * 0.7
    return pink_t


# ============================================================================
# Reverberation (Synthetic RIR)
# ============================================================================

def generate_synthetic_rir(rt60, sr=16000, seed=None):
    """Generate synthetic Room Impulse Response via exponential decay model.

    h(t) = gaussian_noise * exp(-6.908 * t / RT60)
    where 6.908 = ln(1000) ensures 60 dB decay at RT60.

    Args:
        rt60: Reverberation time in seconds (e.g., 0.2, 0.4, 0.6, 0.8)
        sr: Sample rate (default 16kHz)
        seed: Random seed for reproducibility
    Returns:
        rir: (L,) torch tensor, normalized RIR
    """
    rir_length = int(rt60 * sr)
    if rir_length < 1:
        return torch.ones(1)
    t = np.arange(rir_length, dtype=np.float32) / sr
    envelope = np.exp(-6.908 / rt60 * t)
    rng = np.random.RandomState(seed) if seed else np.random
    rir = rng.randn(rir_length).astype(np.float32) * envelope
    rir[0] = abs(rir[0])  # Ensure causal (direct path positive)
    rir = rir / (np.sum(np.abs(rir)) + 1e-10)
    return torch.from_numpy(rir)


def apply_reverb(audio, rir):
    """Apply Room Impulse Response to audio via causal convolution.

    Uses F.conv1d with left-padding to preserve original length.

    Args:
        audio: (B, T) or (T,) waveform tensor
        rir: (L,) impulse response tensor
    Returns:
        reverberant: same shape as input
    """
    squeeze = False
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeeze = True
    rir = rir.to(audio.device)
    # F.conv1d expects weight as (out_ch, in_ch/groups, kW)
    rir_flipped = rir.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    audio_3d = audio.unsqueeze(1)  # (B, 1, T)
    pad_len = rir.size(0) - 1
    audio_padded = F.pad(audio_3d, (pad_len, 0))  # Left pad for causal
    reverberant = F.conv1d(audio_padded, rir_flipped).squeeze(1)  # (B, T)
    if squeeze:
        reverberant = reverberant.squeeze(0)
    return reverberant


def spectral_subtraction_enhance(noisy_audio, n_fft=512, hop_length=160,
                                  oversubtract=2.0, floor=0.1):
    """Real-time spectral subtraction enhancer (classical, 0 trainable params).

    Identical enhancer applied to ALL models for fair comparison.
    Estimates noise spectrum from first 5 frames, subtracts from magnitude.

    Args:
        noisy_audio: (B, T) or (T,) waveform tensor
        n_fft: FFT size (512 = 32ms @ 16kHz)
        hop_length: hop size (160 = 10ms @ 16kHz)
        oversubtract: over-subtraction factor (2.0 = aggressive)
        floor: spectral floor to prevent musical noise (0.1 = -20dB)
    Returns:
        enhanced: (B, T) or (T,) enhanced waveform
    """
    squeeze = False
    if noisy_audio.dim() == 1:
        noisy_audio = noisy_audio.unsqueeze(0)
        squeeze = True

    window = torch.hann_window(n_fft, device=noisy_audio.device)
    spec = torch.stft(noisy_audio, n_fft, hop_length, window=window,
                      return_complex=True)  # (B, F, T)
    mag = spec.abs()
    phase = spec.angle()

    # Noise estimation: average of first 5 frames
    n_noise_frames = min(5, mag.size(-1))
    noise_est = mag[..., :n_noise_frames].mean(dim=-1, keepdim=True)  # (B, F, 1)

    # Spectral subtraction with over-subtraction and flooring
    enhanced_mag = mag - oversubtract * noise_est
    enhanced_mag = torch.maximum(enhanced_mag, floor * mag)

    # Reconstruct waveform
    enhanced_spec = enhanced_mag * torch.exp(1j * phase)
    enhanced = torch.istft(enhanced_spec, n_fft, hop_length, window=window,
                           length=noisy_audio.size(-1))

    if squeeze:
        enhanced = enhanced.squeeze(0)
    return enhanced


# ============================================================================
# GTCRN Pre-trained Enhancer (23.7K params, ICASSP 2024)
# ============================================================================

_GTCRN_MODEL = None  # Cached global instance


def load_gtcrn_enhancer(gtcrn_dir='/content/gtcrn', device='cpu'):
    """Load pre-trained GTCRN speech enhancement model.

    GTCRN: 23.7K params, 33 MMACs/s — lightest pre-trained SE model.
    Trained on DNS3 dataset (diverse noise conditions).

    Setup on Colab:
      !git clone https://github.com/Xiaobin-Rong/gtcrn.git /content/gtcrn

    Args:
        gtcrn_dir: Path to cloned GTCRN repository
        device: torch device
    Returns:
        GTCRN model in eval mode
    """
    global _GTCRN_MODEL
    if _GTCRN_MODEL is not None:
        return _GTCRN_MODEL

    import importlib.util
    gtcrn_path = os.path.join(gtcrn_dir, 'gtcrn.py')
    if not os.path.exists(gtcrn_path):
        raise FileNotFoundError(
            f"GTCRN not found at {gtcrn_dir}. Run:\n"
            f"  !git clone https://github.com/Xiaobin-Rong/gtcrn.git {gtcrn_dir}")

    spec = importlib.util.spec_from_file_location("gtcrn", gtcrn_path)
    gtcrn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gtcrn_module)

    model = gtcrn_module.GTCRN().eval().to(device)
    ckpt_path = os.path.join(gtcrn_dir, 'checkpoints', 'model_trained_on_dns3.tar')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"GTCRN checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    params = sum(p.numel() for p in model.parameters())
    print(f"  [GTCRN] Loaded pre-trained enhancer: {params:,} params")

    _GTCRN_MODEL = model
    return model


def gtcrn_enhance(noisy_audio, gtcrn_model, n_fft=512, hop_length=256):
    """Apply GTCRN pre-trained enhancer to audio batch.

    Args:
        noisy_audio: (B, T) or (T,) waveform tensor @ 16kHz
        gtcrn_model: loaded GTCRN model
        n_fft: 512 (GTCRN default)
        hop_length: 256 (GTCRN default)
    Returns:
        enhanced: same shape as input
    """
    squeeze = False
    if noisy_audio.dim() == 1:
        noisy_audio = noisy_audio.unsqueeze(0)
        squeeze = True

    device = noisy_audio.device
    window = torch.hann_window(n_fft, device=device).pow(0.5)  # sqrt-Hann (GTCRN convention)

    enhanced_list = []
    for i in range(noisy_audio.size(0)):
        x = noisy_audio[i]  # (T,)
        # STFT → (F, T, 2) real-valued
        spec = torch.stft(x, n_fft, hop_length, n_fft, window, return_complex=False)
        # GTCRN expects (1, F, T, 2)
        with torch.no_grad():
            out = gtcrn_model(spec.unsqueeze(0))[0]  # (F, T, 2)
        # iSTFT back to waveform
        enh = torch.istft(out, n_fft, hop_length, n_fft, window,
                          return_complex=False)
        # Match original length
        if enh.size(-1) < x.size(-1):
            enh = F.pad(enh, (0, x.size(-1) - enh.size(-1)))
        else:
            enh = enh[:x.size(-1)]
        enhanced_list.append(enh)

    enhanced = torch.stack(enhanced_list)
    if squeeze:
        enhanced = enhanced.squeeze(0)
    return enhanced


def mix_audio_at_snr(clean_audio, noise, snr_db):
    if clean_audio.dim() == 2:
        clean_rms = torch.sqrt(torch.mean(clean_audio ** 2, dim=-1, keepdim=True) + 1e-10)
    else:
        clean_rms = torch.sqrt(torch.mean(clean_audio ** 2) + 1e-10)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-10)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return clean_audio + scaled_noise


@torch.no_grad()
def evaluate_noisy(model, val_loader, device, noise_type='factory',
                   snr_db=0, dataset_audios=None,
                   use_enhancer=False, enhancer_type='spectral',
                   gtcrn_model=None):
    """Evaluate under noisy conditions with optional front-end enhancer.

    Args:
        enhancer_type: 'spectral' (0 params, classical) or 'gtcrn' (23.7K pre-trained)
        gtcrn_model: loaded GTCRN model (required if enhancer_type='gtcrn')
    """
    model.eval()
    correct = 0
    total = 0

    for mel, labels, audio in val_loader:
        labels = labels.to(device)
        audio = audio.to(device)

        noise = generate_noise_signal(
            noise_type, audio.size(-1), sr=16000,
            dataset_audios=dataset_audios).to(device)

        noisy_audio = mix_audio_at_snr(audio, noise, snr_db)
        if use_enhancer:
            if enhancer_type == 'gtcrn' and gtcrn_model is not None:
                noisy_audio = gtcrn_enhance(noisy_audio, gtcrn_model)
            else:
                noisy_audio = spectral_subtraction_enhance(noisy_audio)
        logits = model(noisy_audio)

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


@torch.no_grad()
def evaluate_reverb(model, val_loader, device, rt60=0.5,
                    noise_type=None, snr_db=None, dataset_audios=None,
                    use_enhancer=False, enhancer_type='spectral',
                    gtcrn_model=None):
    """Evaluate under reverberant conditions with optional noise and enhancer.

    Processing chain: clean → reverb → [noise] → [enhancer] → model

    Args:
        rt60: Reverberation time in seconds
        noise_type: If set, adds noise after reverb (e.g., 'factory', 'babble')
        snr_db: SNR in dB (required if noise_type is set)
        use_enhancer: Apply front-end enhancer
        enhancer_type: 'spectral' or 'gtcrn'
        gtcrn_model: Loaded GTCRN model
    Returns:
        accuracy (float)
    """
    model.eval()
    correct = 0
    total = 0

    rir = generate_synthetic_rir(rt60, sr=16000, seed=42)

    for mel, labels, audio in val_loader:
        labels = labels.to(device)
        audio = audio.to(device)

        # Step 1: Apply reverb
        reverberant = apply_reverb(audio, rir)

        # Step 2: Optionally add noise on top of reverberant signal
        if noise_type is not None and snr_db is not None:
            noise = generate_noise_signal(
                noise_type, audio.size(-1), sr=16000,
                dataset_audios=dataset_audios).to(device)
            reverberant = mix_audio_at_snr(reverberant, noise, snr_db)

        # Step 3: Optionally enhance
        if use_enhancer:
            if enhancer_type == 'gtcrn' and gtcrn_model is not None:
                reverberant = gtcrn_enhance(reverberant, gtcrn_model)
            else:
                reverberant = spectral_subtraction_enhance(reverberant)

        logits = model(reverberant)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


# ============================================================================
# Noise Evaluation Runner
# ============================================================================

@torch.no_grad()
def run_noise_evaluation(models_dict, val_loader, device,
                         noise_types=None, snr_levels=None,
                         dataset_audios=None, use_enhancer=False,
                         enhancer_type='spectral', gtcrn_model=None):
    if noise_types is None:
        noise_types = ['factory', 'white', 'babble', 'street', 'pink']
    if snr_levels is None:
        snr_levels = [-15, -10, -5, 0, 5, 10, 15, 'clean']

    enhancer_names = {'spectral': 'SPECTRAL SUBTRACTION (0 params)',
                      'gtcrn': 'GTCRN PRE-TRAINED (23.7K params)'}
    enhancer_str = f" [WITH {enhancer_names.get(enhancer_type, enhancer_type)}]" if use_enhancer else ""
    print("\n" + "=" * 80)
    print(f"  NOISE ROBUSTNESS EVALUATION{enhancer_str}")
    print(f"  Noise types: {noise_types}")
    print(f"  SNR levels: {snr_levels}")
    print("=" * 80)

    results = {}
    for model_name, model in models_dict.items():
        model.eval()
        results[model_name] = {}
        print(f"\n  Evaluating: {model_name}", flush=True)

        for noise_type in noise_types:
            results[model_name][noise_type] = {}
            for snr in snr_levels:
                if snr == 'clean':
                    acc = evaluate(model, val_loader, device)
                else:
                    acc = evaluate_noisy(
                        model, val_loader, device, noise_type, snr,
                        dataset_audios=dataset_audios,
                        use_enhancer=use_enhancer,
                        enhancer_type=enhancer_type,
                        gtcrn_model=gtcrn_model)
                results[model_name][noise_type][str(snr)] = acc

            clean_acc = results[model_name][noise_type].get('clean', 0)
            zero_acc = results[model_name][noise_type].get('0', 0)
            m15_acc = results[model_name][noise_type].get('-15', 0)
            print(f"    {noise_type:<10} | Clean: {clean_acc:.1f}% | "
                  f"0dB: {zero_acc:.1f}% | -15dB: {m15_acc:.1f}%", flush=True)

    # Print summary tables
    for noise_type in noise_types:
        numeric_snrs = [s for s in snr_levels if s != 'clean']
        print(f"\n  === {noise_type.upper()} Noise Summary ===")
        print(f"  {'Model':<25} | {'Clean':>7} | " +
              " | ".join(f"{s:>6}dB" for s in numeric_snrs))
        print("  " + "-" * (30 + 9 * len(numeric_snrs)))

        for model_name, noise_data in results.items():
            if noise_type not in noise_data:
                continue
            clean = noise_data[noise_type].get('clean', 0)
            snrs = [noise_data[noise_type].get(str(s), 0) for s in numeric_snrs]
            print(f"  {model_name:<25} | {clean:>6.1f}% | " +
                  " | ".join(f"{s:>6.1f}%" for s in snrs))

    return results


# ============================================================================
# Reverb Evaluation Runner
# ============================================================================

@torch.no_grad()
def run_reverb_evaluation(models_dict, val_loader, device,
                          rt60_list=None, noise_types_reverb=None,
                          snr_levels_reverb=None, dataset_audios=None,
                          use_enhancer=False, enhancer_type='spectral',
                          gtcrn_model=None):
    """Run full reverb evaluation: reverb-only + noise+reverb combined.

    Conditions evaluated:
      C. Reverb only: each RT60 value, no noise
      E. Noise+Reverb: selected noise types × SNRs × RT60s

    Args:
        rt60_list: List of RT60 values (default: [0.2, 0.4, 0.6, 0.8])
        noise_types_reverb: Noise types for combined test (default: ['factory', 'babble'])
        snr_levels_reverb: SNR levels for combined test (default: [0, 5])
    Returns:
        dict with 'reverb_only' and 'noise_reverb' results
    """
    if rt60_list is None:
        rt60_list = [0.2, 0.4, 0.6, 0.8]
    if noise_types_reverb is None:
        noise_types_reverb = ['factory', 'babble']
    if snr_levels_reverb is None:
        snr_levels_reverb = [0, 5]

    enhancer_names = {'spectral': 'SPECTRAL SUBTRACTION (0 params)',
                      'gtcrn': 'GTCRN PRE-TRAINED (23.7K params)'}
    enhancer_str = f" [WITH {enhancer_names.get(enhancer_type, enhancer_type)}]" if use_enhancer else ""

    # ---- C. Reverb-only evaluation ----
    print("\n" + "=" * 80)
    print(f"  REVERB-ONLY EVALUATION{enhancer_str}")
    print(f"  RT60 values: {rt60_list}")
    print("=" * 80)

    reverb_only_results = {}
    for model_name, model in models_dict.items():
        model.eval()
        reverb_only_results[model_name] = {}
        print(f"\n  Evaluating: {model_name}", flush=True)

        for rt60 in rt60_list:
            acc = evaluate_reverb(
                model, val_loader, device, rt60=rt60,
                use_enhancer=use_enhancer,
                enhancer_type=enhancer_type,
                gtcrn_model=gtcrn_model)
            reverb_only_results[model_name][str(rt60)] = acc
            print(f"    RT60={rt60:.1f}s | Acc: {acc:.1f}%", flush=True)

    # Print reverb-only summary table
    print(f"\n  === REVERB-ONLY Summary ===")
    print(f"  {'Model':<25} | " +
          " | ".join(f"RT60={r:.1f}s" for r in rt60_list))
    print("  " + "-" * (28 + 12 * len(rt60_list)))
    for model_name in reverb_only_results:
        accs = [reverb_only_results[model_name].get(str(r), 0) for r in rt60_list]
        print(f"  {model_name:<25} | " +
              " | ".join(f"  {a:>5.1f}%" for a in accs))

    # ---- E. Noise+Reverb combined evaluation ----
    # RT60 subset for combined: 0.3, 0.6 (representative room sizes)
    combined_rt60s = [0.3, 0.6]

    print(f"\n  === NOISE+REVERB COMBINED{enhancer_str} ===")
    print(f"  Noise: {noise_types_reverb}, SNR: {snr_levels_reverb}dB, "
          f"RT60: {combined_rt60s}s")
    print("=" * 80)

    noise_reverb_results = {}
    for model_name, model in models_dict.items():
        model.eval()
        noise_reverb_results[model_name] = {}
        print(f"\n  Evaluating: {model_name}", flush=True)

        for noise_type in noise_types_reverb:
            noise_reverb_results[model_name][noise_type] = {}
            for snr_db in snr_levels_reverb:
                for rt60 in combined_rt60s:
                    key = f"snr{snr_db}_rt{rt60}"
                    acc = evaluate_reverb(
                        model, val_loader, device, rt60=rt60,
                        noise_type=noise_type, snr_db=snr_db,
                        dataset_audios=dataset_audios,
                        use_enhancer=use_enhancer,
                        enhancer_type=enhancer_type,
                        gtcrn_model=gtcrn_model)
                    noise_reverb_results[model_name][noise_type][key] = acc
                    print(f"    {noise_type:<10} SNR={snr_db:>3}dB RT60={rt60:.1f}s | "
                          f"Acc: {acc:.1f}%", flush=True)

    # Print noise+reverb summary
    print(f"\n  === NOISE+REVERB Summary ===")
    for noise_type in noise_types_reverb:
        print(f"\n  --- {noise_type.upper()} + Reverb ---")
        combos = [f"SNR={s}dB/RT60={r}s"
                  for s in snr_levels_reverb for r in combined_rt60s]
        combo_keys = [f"snr{s}_rt{r}"
                      for s in snr_levels_reverb for r in combined_rt60s]
        print(f"  {'Model':<25} | " + " | ".join(f"{c:>16}" for c in combos))
        print("  " + "-" * (28 + 19 * len(combos)))
        for model_name in noise_reverb_results:
            accs = [noise_reverb_results[model_name].get(noise_type, {}).get(k, 0)
                    for k in combo_keys]
            print(f"  {model_name:<25} | " +
                  " | ".join(f"       {a:>5.1f}%" for a in accs))

    return {
        'reverb_only': reverb_only_results,
        'noise_reverb': noise_reverb_results,
    }


# ============================================================================
# Baseline Models (CNN) — for fair comparison experiments
# ============================================================================

class DSCNN_S(nn.Module):
    """DS-CNN Small baseline (ARM, 2017).
    Depthwise Separable CNN for keyword spotting.
    ~23.7K params, 96.6% on GSC V2 12-class.
    """
    def __init__(self, n_mels=40, n_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (10, 4), stride=(2, 2), padding=(5, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def _F_pad(x, pad):
    """Wrapper for F.pad to avoid name collision."""
    return F.pad(x, pad)


class SubSpectralNorm(nn.Module):
    """Sub-Spectral Normalization for BC-ResNet."""
    def __init__(self, num_features, num_sub_bands=5):
        super().__init__()
        self.num_sub_bands = num_sub_bands
        self.bn = nn.BatchNorm2d(num_features * num_sub_bands)

    def forward(self, x):
        B, C, Fr, T = x.shape
        S = self.num_sub_bands
        pad = (S - Fr % S) % S
        if pad > 0:
            x = _F_pad(x, (0, 0, 0, pad))
            Fr_new = Fr + pad
        else:
            Fr_new = Fr
        x = x.reshape(B, C, S, Fr_new // S, T).reshape(B, C * S, Fr_new // S, T)
        x = self.bn(x)
        x = x.reshape(B, C, S, Fr_new // S, T).reshape(B, C, Fr_new, T)
        if pad > 0:
            x = x[:, :, :Fr_new - pad, :]
        return x


class BCResBlock(nn.Module):
    """BC-ResNet block with broadcasted residual connection."""
    def __init__(self, in_ch, out_ch, kernel_size=3,
                 stride=(1, 1), dilation=1, num_sub_bands=5):
        super().__init__()
        self.use_residual = (in_ch == out_ch and stride == (1, 1))
        self.freq_conv1 = nn.Conv2d(in_ch, out_ch, (1, 1))
        self.ssn1 = SubSpectralNorm(out_ch, num_sub_bands)
        padding = (0, (kernel_size - 1) * dilation // 2)
        self.temp_dw_conv = nn.Conv2d(
            out_ch, out_ch, (1, kernel_size), stride=(1, stride[1]),
            padding=padding, dilation=(1, dilation), groups=out_ch)
        self.ssn2 = SubSpectralNorm(out_ch, num_sub_bands)
        self.freq_conv2 = nn.Conv2d(out_ch, out_ch, (1, 1))
        self.ssn3 = SubSpectralNorm(out_ch, num_sub_bands)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        if not self.use_residual and in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (1, 1), stride=stride),
                nn.BatchNorm2d(out_ch))
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = F.relu(self.ssn1(self.freq_conv1(x)))
        out = F.relu(self.ssn2(self.temp_dw_conv(out)))
        out = self.ssn3(self.freq_conv2(out))
        out = out + self.freq_pool(out)
        if self.use_residual:
            out = out + identity
        elif self.skip is not None:
            out = out + self.skip(identity)
        return F.relu(out)


class BCResNet(nn.Module):
    """BC-ResNet: Broadcasted Residual Network (Qualcomm, 2021).
    BC-ResNet-1: ~7.5K params, 96.0% on GSC V2 12-class.
    """
    def __init__(self, n_mels=40, n_classes=12, scale=1, num_sub_bands=5):
        super().__init__()
        c = max(int(8 * scale), 8)
        self.conv1 = nn.Conv2d(1, c, (5, 5), stride=(2, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(c)
        self.stage1 = nn.Sequential(
            BCResBlock(c, c, num_sub_bands=num_sub_bands),
            BCResBlock(c, c, num_sub_bands=num_sub_bands))
        c2 = int(c * 1.5)
        self.stage2 = nn.Sequential(
            BCResBlock(c, c2, stride=(1, 2), num_sub_bands=num_sub_bands),
            BCResBlock(c2, c2, dilation=2, num_sub_bands=num_sub_bands))
        c3 = c * 2
        self.stage3 = nn.Sequential(
            BCResBlock(c2, c3, stride=(1, 2), num_sub_bands=num_sub_bands),
            BCResBlock(c3, c3, dilation=4, num_sub_bands=num_sub_bands))
        c4 = int(c * 2.5)
        self.stage4 = BCResBlock(c3, c4, num_sub_bands=num_sub_bands)
        self.head_conv = nn.Conv2d(c4, c4, (1, 1))
        self.head_bn = nn.BatchNorm2d(c4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c4, n_classes)

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.head_bn(self.head_conv(x)))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ============================================================================
# Model Registry — NanoMamba + CNN Baselines
# ============================================================================

MODEL_REGISTRY = {
    'NanoMamba-Tiny': create_nanomamba_tiny,
    'NanoMamba-Small': create_nanomamba_small,
    'NanoMamba-Tiny-TC': create_nanomamba_tiny_tc,
    'NanoMamba-Tiny-WS-TC': create_nanomamba_tiny_ws_tc,
    'NanoMamba-Tiny-WS': create_nanomamba_tiny_ws,
    'NanoMamba-Tiny-PCEN': create_nanomamba_tiny_pcen,
    'NanoMamba-Small-PCEN': create_nanomamba_small_pcen,
    'NanoMamba-Tiny-PCEN-TC': create_nanomamba_tiny_pcen_tc,
    'NanoMamba-Tiny-DualPCEN': create_nanomamba_tiny_dualpcen,
    'NanoMamba-Small-DualPCEN': create_nanomamba_small_dualpcen,
    'DS-CNN-S': lambda n=12: DSCNN_S(n_classes=n),
    'BC-ResNet-1': lambda n=12: BCResNet(n_classes=n, scale=1),
}


def create_model(name, n_classes=12):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](n_classes)
    else:
        print(f"  [ERROR] Unknown model: {name}")
        print(f"  Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)


# ============================================================================
# Training Pipeline
# ============================================================================

def train_model(model, model_name, train_dataset, val_dataset,
                checkpoint_dir, device, epochs=30, batch_size=128, lr=3e-3):
    """Full training loop for NanoMamba."""
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"{'='*70}")

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01)

    best_acc = 0
    best_epoch = 0
    model_dir = Path(checkpoint_dir) / model_name.replace(' ', '_')
    model_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch, model_name=model_name)

        val_acc = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        marker = " *** BEST ***" if val_acc > best_acc else ""
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"Time: {elapsed:.1f}s{marker}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'model_name': model_name,
            }, model_dir / 'best.pt')

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
        'val_acc': val_acc,
        'model_name': model_name,
    }, model_dir / 'final.pt')

    with open(model_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best: {best_acc:.2f}% @ epoch {best_epoch}")
    print(f"  Saved to {model_dir}")

    # Load best checkpoint
    ckpt = torch.load(model_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    return best_acc, model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NanoMamba Colab Training — Structural Noise Robustness")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results save directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run noise evaluation (load checkpoints)')
    parser.add_argument('--models', type=str,
                        default='NanoMamba-Tiny',
                        help='Comma-separated model names')
    parser.add_argument('--noise_types', type=str,
                        default='factory,white,babble,street,pink',
                        help='Comma-separated noise types')
    parser.add_argument('--snr_range', type=str,
                        default='-15,-10,-5,0,5,10,15',
                        help='Comma-separated SNR levels (dB)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache', action='store_true',
                        help='Cache val dataset in RAM (needs ~1GB)')
    parser.add_argument('--use_enhancer', action='store_true',
                        help='Apply front-end enhancer to all models (fair comparison)')
    parser.add_argument('--enhancer_type', type=str, default='spectral',
                        choices=['spectral', 'gtcrn'],
                        help='Enhancer type: spectral (0 params) or gtcrn (23.7K pre-trained)')
    parser.add_argument('--gtcrn_dir', type=str, default='/content/gtcrn',
                        help='Path to cloned GTCRN repo (for --enhancer_type gtcrn)')
    parser.add_argument('--use_reverb', action='store_true',
                        help='Run reverberation evaluation (reverb-only + noise+reverb)')
    parser.add_argument('--rt60', type=str, default='0.2,0.4,0.6,0.8',
                        help='Comma-separated RT60 values in seconds')
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"  NanoMamba Training — Structural Noise Robustness")
    print(f"  Device: {device}")
    print(f"  Models: {args.models}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # ===== 1. Load dataset =====
    print("\n  Loading Google Speech Commands V2...")
    os.makedirs(args.data_dir, exist_ok=True)

    train_dataset = SpeechCommandsDataset(
        args.data_dir, subset='training', augment=True)
    val_dataset = SpeechCommandsDataset(
        args.data_dir, subset='validation', augment=False)
    test_dataset = SpeechCommandsDataset(
        args.data_dir, subset='testing', augment=False)

    print(f"\n  Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Test: {len(test_dataset)}")

    # Optional RAM caching for val set
    if args.cache:
        try:
            val_dataset.cache_all()
        except Exception as e:
            print(f"  [WARNING] Caching failed: {e}")

    # ===== 2. Create models =====
    model_names = [m.strip() for m in args.models.split(',')]
    models = {}
    for name in model_names:
        model = create_model(name)
        params = sum(p.numel() for p in model.parameters())
        fp32_kb = params * 4 / 1024
        print(f"  {name}: {params:,} params ({fp32_kb:.1f} KB FP32)")
        models[name] = model

    # ===== 3. Train (or load) =====
    trained_models = {}
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    for model_name, model in models.items():
        ckpt_path = (Path(args.checkpoint_dir) /
                     model_name.replace(' ', '_') / 'best.pt')

        if args.eval_only:
            if ckpt_path.exists():
                print(f"\n  Loading checkpoint: {model_name}")
                model = model.to(device)
                ckpt = torch.load(ckpt_path, map_location=device,
                                  weights_only=True)
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                print(f"  Loaded: val_acc={ckpt.get('val_acc', 0):.2f}%")
            else:
                print(f"\n  [SKIP] No checkpoint: {ckpt_path}")
                continue
        else:
            best_acc, model = train_model(
                model, model_name, train_dataset, val_dataset,
                args.checkpoint_dir, device,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

        trained_models[model_name] = model

    if not trained_models:
        print("\n  [ERROR] No models to evaluate!")
        return

    # ===== 4. Test set evaluation =====
    print("\n" + "=" * 80)
    print("  TEST SET EVALUATION (Clean)")
    print("=" * 80)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True)

    test_results = {}
    for model_name, model in trained_models.items():
        test_acc = evaluate(model, test_loader, device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name:<25} | Test: {test_acc:.2f}% | Params: {params:,}")
        test_results[model_name] = test_acc

    # ===== 5. Noise robustness evaluation =====
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True)

    noise_types = [t.strip() for t in args.noise_types.split(',')]
    snr_levels = [int(s.strip()) for s in args.snr_range.split(',')]
    snr_levels.append('clean')

    # Load GTCRN enhancer if requested
    gtcrn_model = None
    if args.use_enhancer and args.enhancer_type == 'gtcrn':
        try:
            gtcrn_model = load_gtcrn_enhancer(args.gtcrn_dir, device=device)
        except FileNotFoundError as e:
            print(f"\n  [ERROR] {e}")
            print("  Falling back to spectral subtraction enhancer.")
            args.enhancer_type = 'spectral'

    noise_results = run_noise_evaluation(
        trained_models, val_loader, device,
        noise_types=noise_types, snr_levels=snr_levels,
        use_enhancer=args.use_enhancer,
        enhancer_type=args.enhancer_type,
        gtcrn_model=gtcrn_model)

    # ===== 5b. Reverb evaluation (if requested) =====
    reverb_results = {}
    if args.use_reverb:
        rt60_list = [float(r.strip()) for r in args.rt60.split(',')]
        reverb_results = run_reverb_evaluation(
            trained_models, val_loader, device,
            rt60_list=rt60_list,
            noise_types_reverb=[t for t in noise_types if t in ['factory', 'babble']],
            snr_levels_reverb=[0, 5],
            use_enhancer=args.use_enhancer,
            enhancer_type=args.enhancer_type,
            gtcrn_model=gtcrn_model)

    # ===== 6. Save results =====
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'lr': args.lr,
        'seed': args.seed,
        'models': {}
    }

    for model_name, model in trained_models.items():
        params = sum(p.numel() for p in model.parameters())
        model_result = {
            'params': params,
            'size_fp32_kb': round(params * 4 / 1024, 1),
            'size_int8_kb': round(params / 1024, 1),
            'test_acc': test_results.get(model_name, 0),
            'noise_robustness': noise_results.get(model_name, {}),
        }
        if reverb_results:
            model_result['reverb_robustness'] = {
                'reverb_only': reverb_results.get('reverb_only', {}).get(model_name, {}),
                'noise_reverb': reverb_results.get('noise_reverb', {}).get(model_name, {}),
            }
        final_results['models'][model_name] = model_result

    results_path = Path(args.results_dir) / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n  Results saved to: {results_path}")

    # ===== 7. Print structural params =====
    print("\n" + "=" * 80)
    print("  STRUCTURAL NOISE ROBUSTNESS PARAMETERS")
    print("=" * 80)
    for model_name, model in trained_models.items():
        print(f"\n  {model_name}:")
        # Print learnable parameters (alpha)
        for pname, p in model.named_parameters():
            if 'alpha' in pname or 'log_s' in pname or 'log_delta' in pname or 'log_r' in pname:
                print(f"    {pname}: mean={p.mean().item():.4f}, std={p.std().item():.4f}")
        # Print fixed structural buffers (delta_floor, epsilon)
        for bname, buf in model.named_buffers():
            if any(k in bname for k in ['delta_floor', 'epsilon']):
                print(f"    {bname}: mean={buf.mean().item():.4f} (fixed, non-learnable)")

    return final_results


if __name__ == '__main__':
    main()
