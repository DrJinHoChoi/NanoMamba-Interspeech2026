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

  === Basic training (clean only) ===
     !python train_colab.py --models NanoMamba-Tiny-DualPCEN --epochs 30

  === Multi-Condition Noise-Aug Training (RECOMMENDED) ===
  Per-sample noise mixing: each sample gets random noise type x random SNR
  Reveals structural advantage of DualPCEN dynamic routing vs CNN fixed kernels

     # NanoMamba vs CNN baselines with noise-aug training:
     !python train_colab.py --models NanoMamba-Tiny-DualPCEN,DS-CNN-S,BC-ResNet-1 \\
         --epochs 30 --noise_aug --calibrate

     # Full system with all enhancements:
     !python train_colab.py --models NanoMamba-Tiny-DualPCEN,DS-CNN-S,BC-ResNet-1 \\
         --epochs 30 --noise_aug --calibrate --use_enhancer --enhancer_bypass

  === Eval only (after training) ===
     !python train_colab.py --models NanoMamba-Tiny-DualPCEN --eval_only \\
         --noise_types factory,white,babble,street,pink --calibrate
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
        create_nanomamba_matched_dualpcen,
        create_nanomamba_tiny_tripcen,
        create_nanomamba_matched_tripcen,
        # v2 Enhanced Routing variants
        create_nanomamba_tiny_dualpcen_v2,
        create_nanomamba_matched_dualpcen_v2,
        create_nanomamba_tiny_tripcen_v2,
        create_nanomamba_matched_tripcen_v2,
        # v2 + SSM v2 (full stack: PCEN v2 routing + SA-SSM v2 dynamics)
        create_nanomamba_tiny_dualpcen_v2_ssmv2,
        create_nanomamba_matched_dualpcen_v2_ssmv2,
        create_nanomamba_tiny_tripcen_v2_ssmv2,
        create_nanomamba_matched_tripcen_v2_ssmv2,
        # Complete model: v2 + SSMv2 + Integrated Spectral Enhancement
        create_nanomamba_tiny_dualpcen_v2_ssmv2_se,
        create_nanomamba_matched_dualpcen_v2_ssmv2_se,
        # Learnable Spectral Enhancement (differentiable Wiener gain, 516 params)
        create_nanomamba_tiny_dualpcen_v2_ssmv2_lse,
        create_nanomamba_matched_dualpcen_v2_ssmv2_lse,
        # FI add-on: SpectralMambaBlock on existing NanoMamba architectures
        create_nanomamba_matched_dualpcen_v2_ssmv2_fi,
        create_nanomamba_tiny_dualpcen_v2_ssmv2_fi,
        # SM-SSM: Selectivity-Modulated SA-SSM (CNN↔Mamba blend based on SNR)
        create_nanomamba_matched_dualpcen_v2_smssm,
        create_nanomamba_tiny_dualpcen_v2_smssm,
        # FI-Mamba: Frequency-Interleaved Mamba (spectral + temporal SSM)
        create_fimamba_matched,
        create_fimamba_small,
        # v3: Pure representation efficiency (beat BC-ResNet-1)
        create_nanomamba_v3_matched,
        create_nanomamba_v3_deep,
        # NanoApple: Frequency-Aware SSM (CNN freq processing + SSM streaming)
        create_nanoapple,
        create_nanoapple_v2,
        # NanoApple-v3: DualPCEN v2 + BC-ResNet backbone
        create_nanoapple_v3,
        # SAGN: SNR-Adaptive Gated Network (CNN backbone + learned spectral gate)
        create_sagn,
        # NC-SSM: Noise-Conditioned SM-SSM (per-sub-band selectivity + LSG)
        create_nanomamba_nc_matched,
        create_nanomamba_nc_large,
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
# Model EMA (Exponential Moving Average)
# ============================================================================

class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow weights = running average of training weights.
    At eval, shadow weights provide better generalization (~0.3-0.5%p).

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # After each optimizer.step():
        ema.update(model)
        # For validation:
        ema.apply_shadow(model)
        val_acc = evaluate(model, val_loader, device)
        ema.restore(model)
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, scheduler, device,
                    label_smoothing=0.1, epoch=0, model_name="",
                    noise_aug=False, noise_ratio=0.5, is_cnn=False,
                    dataset_audios=None, mel_fb=None,
                    n_fft=512, hop_length=160, n_mels=40,
                    total_epochs=30, noise_curriculum_v2=False,
                    ss_train=False,
                    ema=None):
    """Train one epoch with Per-Sample Multi-Condition Noise Augmentation.

    [KEY INSIGHT] Why noise-aug helps NanoMamba MORE than CNN:

    NanoMamba (SA-SSM + DualPCEN):
      - DualPCEN has two experts: Expert1 (nonstat, delta=2.0) for clean,
        Expert2 (stationary, delta=0.01) for noise
      - SF (Spectral Flatness) + Spectral Tilt → dynamic routing at INFERENCE
      - Clean-only training: Expert2 path dormant, SNR estimator unexposed
      - Per-sample noise-aug: every sample gets different noise type × SNR
        → DualPCEN learns rich routing manifold (not just one noise condition)
        → Adaptive delta/epsilon/B-gate paths activate with diverse gradients
        → Two SEPARATE pathways → Clean preserved + Noise improved

    CNN (DS-CNN-S, BC-ResNet-1):
      - BC-ResNet's Sub-Spectral Norm: sub-band normalize at training time
        → but normalize stats are FROZEN at inference (running mean/var)
        → partial frequency adaptation, but NOT dynamic per-input
      - DS-CNN-S: standard BatchNorm → single global mean/var at inference
      - Fixed kernels must encode BOTH clean and noisy patterns simultaneously
      - No routing mechanism → same weights serve ALL conditions
      - Result: Clean degrades, Noise improves → TRADE-OFF (zero-sum)

    === Per-Sample vs Per-Batch Noise Mixing ===
    Per-batch (old): one noise type × one SNR per mini-batch
      → sparse gradient: DualPCEN sees one routing target per step
    Per-sample (new): each sample gets independent noise type × SNR
      → rich gradient: DualPCEN sees B different routing targets per step
      → 5 noise types × continuous SNR range → combinatorial diversity
      → Each Expert pathway gets gradients from its natural domain

    === Training Phases ===
    Phase 0 (warm-up, epoch 0-2):    Clean-only training
      → Stabilize Expert1 path, establish clean feature baseline
      → CNN establishes BatchNorm statistics on clean distribution
    Phase 1 (gentle, epoch 3-9):     noise_ratio ramps 0→target, SNR 5~15dB
      → Gradually expose noise paths without disrupting clean features
    Phase 2 (moderate, epoch 10-19):  full noise_ratio, SNR 0~10dB
      → Full multi-condition training, DualPCEN routing maturing
    Phase 3 (hard, epoch 20+):        full noise_ratio, SNR -5~10dB
      → Extreme noise conditions, hardening both pathways
    """
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_loss = 0
    correct = 0
    total = 0

    # ================================================================
    # Noise Configuration: Per-sample diversity with curriculum
    # ================================================================

    # Phase-based curriculum
    WARM_UP_EPOCHS = 3   # Clean-only warm-up
    GENTLE_END = 10      # Easy noise phase
    MODERATE_END = 20    # Moderate noise phase

    if epoch < WARM_UP_EPOCHS:
        # Phase 0: Clean-only warm-up — stabilize Expert1 / CNN baseline
        effective_noise_aug = False
        effective_ratio = 0.0
        snr_range = (15, 20)  # Not used, but set for logging
        phase_name = "WARM-UP (clean-only)"
        # v2 curriculum: no noise types in warm-up
        noise_types_pool = []
    elif epoch < GENTLE_END:
        effective_noise_aug = noise_aug
        ramp_progress = (epoch - WARM_UP_EPOCHS) / (GENTLE_END - WARM_UP_EPOCHS)
        effective_ratio = noise_ratio * ramp_progress
        snr_range = (5, 15)
        if noise_curriculum_v2:
            # [v2] Phase 1: Stationary noise ONLY — let PCEN stat expert stabilize
            noise_types_pool = ['white', 'pink', 'factory']
            phase_name = f"GENTLE-STAT (ratio={effective_ratio:.2f}, SNR {snr_range[0]}~{snr_range[1]}dB)"
        else:
            noise_types_pool = ['factory', 'white', 'babble', 'street', 'pink']
            phase_name = f"GENTLE (ratio={effective_ratio:.2f}, SNR {snr_range[0]}~{snr_range[1]}dB)"
    elif epoch < MODERATE_END:
        effective_noise_aug = noise_aug
        effective_ratio = noise_ratio
        snr_range = (0, 10)
        if noise_curriculum_v2:
            # [v2] Phase 2: Add non-stationary — routing must differentiate
            noise_types_pool = ['white', 'pink', 'factory', 'street', 'babble']
            phase_name = f"MODERATE-ALL (ratio={effective_ratio:.2f}, SNR {snr_range[0]}~{snr_range[1]}dB)"
        else:
            noise_types_pool = ['factory', 'white', 'babble', 'street', 'pink']
            phase_name = f"MODERATE (ratio={effective_ratio:.2f}, SNR {snr_range[0]}~{snr_range[1]}dB)"
    else:
        effective_noise_aug = noise_aug
        effective_ratio = noise_ratio
        if noise_curriculum_v2:
            # [v2] Phase 3: Extend to evaluation-matching range (-15dB).
            # Gaussian annealing ensures gradual exposure — extreme samples
            # are initially rare, becoming frequent as _snr_center drops.
            snr_range = (-15, 10)
            noise_types_pool = ['factory', 'white', 'babble', 'street', 'pink']
            phase_name = f"HARD-ALL (ratio={effective_ratio:.2f}, SNR {snr_range[0]}~{snr_range[1]}dB)"
        else:
            snr_range = (-5, 10)
            noise_types_pool = ['factory', 'white', 'babble', 'street', 'pink']
            phase_name = f"HARD (ratio={effective_ratio:.2f}, SNR {snr_range[0]}~{snr_range[1]}dB)"

    for batch_idx, (mel, labels, audio) in enumerate(train_loader):
        labels = labels.to(device)
        audio = audio.to(device)
        mel = mel.to(device)

        if effective_noise_aug and effective_ratio > 0:
            B = audio.size(0)
            n_noisy = int(B * effective_ratio)

            if n_noisy > 0:
                noisy_audio = audio.clone()

                # ============================================
                # PER-SAMPLE noise mixing: each sample gets
                # independent noise_type × SNR
                # → Maximizes DualPCEN routing diversity
                # ============================================
                # Track per-sample noise metadata (for aux routing loss)
                noise_types_per_sample = []
                snr_dbs_per_sample = []

                # [v2] Compute SNR annealing center for this epoch
                if noise_curriculum_v2:
                    if epoch < GENTLE_END:
                        _phase_prog = (epoch - WARM_UP_EPOCHS) / max(1, GENTLE_END - WARM_UP_EPOCHS)
                    elif epoch < MODERATE_END:
                        _phase_prog = (epoch - GENTLE_END) / max(1, MODERATE_END - GENTLE_END)
                    else:
                        _phase_prog = min(1.0, (epoch - MODERATE_END) / 10.0)
                    # Center moves from easy (high SNR) to hard (low SNR)
                    _snr_center = snr_range[1] - _phase_prog * (snr_range[1] - snr_range[0])
                else:
                    _snr_center = None

                for i in range(n_noisy):
                    # Random noise type per sample
                    noise_type_i = noise_types_pool[
                        np.random.randint(len(noise_types_pool))]
                    # Random SNR per sample
                    if noise_curriculum_v2 and _snr_center is not None:
                        # [v2] Gaussian centered at current difficulty frontier
                        # std=4.0 covers the wider Phase 3 range (-15~10dB = 25dB)
                        snr_db_i = np.clip(
                            np.random.normal(_snr_center, 4.0),
                            snr_range[0], snr_range[1])
                    else:
                        # [v1] Uniform random
                        snr_db_i = np.random.uniform(snr_range[0], snr_range[1])

                    noise_types_per_sample.append(noise_type_i)
                    snr_dbs_per_sample.append(snr_db_i)

                    # Generate noise for this sample
                    noise_i = generate_noise_signal(
                        noise_type_i, audio.size(-1), sr=16000,
                        dataset_audios=dataset_audios).to(device)

                    # Mix at target SNR
                    clean_i = audio[i:i+1]  # (1, T)
                    noisy_i = mix_audio_at_snr(clean_i, noise_i, snr_db_i)
                    noisy_audio[i] = noisy_i.squeeze(0)

                # [SS-Train] Apply Spectral Subtraction to noisy audio
                # during training so model learns to exploit SS-enhanced input.
                # Key insight: SS helps NanoMamba (+20%p broadband) but
                # hurts BC-ResNet-1 (-7%p) → asymmetric advantage.
                if ss_train:
                    with torch.no_grad():
                        noisy_audio[:n_noisy] = spectral_subtraction_v2(
                            noisy_audio[:n_noisy])

                if is_cnn:
                    noisy_mel = _compute_mel_batch(
                        noisy_audio, n_fft, hop_length, mel_fb, device)
                    logits = model(noisy_mel)
                else:
                    logits = model(noisy_audio)
            else:
                if is_cnn:
                    logits = model(mel)
                else:
                    logits = model(audio)
        else:
            # Clean training (warm-up phase or noise_aug disabled)
            if is_cnn:
                logits = model(mel)
            else:
                logits = model(audio)

        loss = criterion(logits, labels)

        # ============================================================
        # NaN safety: detect NaN in logits/loss and skip batch
        # ============================================================
        if torch.isnan(logits).any() or torch.isnan(loss) or torch.isinf(loss):
            nan_skip_count = getattr(train_one_epoch, '_nan_skips', 0) + 1
            train_one_epoch._nan_skips = nan_skip_count
            if nan_skip_count <= 5:  # Only print first 5 warnings
                print(f"    ⚠ NaN/Inf detected at batch {batch_idx+1}, "
                      f"skipping (total skipped: {nan_skip_count})", flush=True)
            optimizer.zero_grad()  # Clear any partial state
            continue

        # [v2] Auxiliary routing loss: explicit gate supervision
        # During noise-aug, we KNOW noise type → soft gate targets → MSE loss
        if (effective_noise_aug and effective_ratio > 0
                and not is_cnn and hasattr(model, 'get_routing_gate')):
            # Ramp up: 0 during warm-up, delayed 2 epochs after first noise
            routing_weight = min(0.1, 0.02 * max(0, epoch - 5))

            # Level 1: stationary vs non-stationary (all PCEN v2 models)
            gate_pred = model.get_routing_gate()
            if gate_pred is not None and n_noisy > 0 and routing_weight > 0:
                gate_targets = torch.tensor(
                    [_compute_gate_target(nt, snr)
                     for nt, snr in zip(noise_types_per_sample, snr_dbs_per_sample)],
                    device=device, dtype=torch.float32)
                routing_loss = F.mse_loss(gate_pred[:n_noisy], gate_targets)
                loss = loss + routing_weight * routing_loss

            # Level 2: broadband vs colored (TriPCEN v2 only)
            if hasattr(model, 'get_routing_gate_l2'):
                gate_l2_pred = model.get_routing_gate_l2()
                if gate_l2_pred is not None and n_noisy > 0 and routing_weight > 0:
                    gate_l2_targets = torch.tensor(
                        [_compute_gate_l2_target(nt, snr)
                         for nt, snr in zip(noise_types_per_sample, snr_dbs_per_sample)],
                        device=device, dtype=torch.float32)
                    routing_l2_loss = F.mse_loss(gate_l2_pred[:n_noisy], gate_l2_targets)
                    # Half weight for L2 — less critical than L1 routing
                    loss = loss + (routing_weight * 0.5) * routing_l2_loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient sanitization: replace Inf/NaN with 0, then clip
        # This is standard practice for recurrent models (SSM/RNN)
        # Previously: detect NaN/Inf → skip batch → clip never runs → 100% skip
        # Now: sanitize → clip → step → model always learns
        grad_has_inf = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    grad_has_inf = True
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        if grad_has_inf:
            nan_skip_count = getattr(train_one_epoch, '_nan_skips', 0) + 1
            train_one_epoch._nan_skips = nan_skip_count
            if nan_skip_count <= 10:
                print(f"    ⓘ Inf/NaN grad sanitized at batch {batch_idx+1} "
                      f"(total: {nan_skip_count})", flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update (per-step for smooth averaging)
        if ema is not None:
            ema.update(model)

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            acc = 100. * correct / total
            aug_str = f" [{phase_name}]" if noise_aug else ""
            print(f"    [{model_name}] Batch {batch_idx+1}/{len(train_loader)} "
                  f"Loss: {total_loss/total:.4f} Acc: {acc:.1f}%{aug_str}",
                  flush=True)

    return total_loss / total, 100. * correct / total


def _compute_mel_batch(audio, n_fft, hop_length, mel_fb, device):
    """Compute log-mel spectrogram for CNN baselines from raw audio batch.

    Args:
        audio: (B, T) raw waveform
        mel_fb: (n_mels, n_freq) mel filterbank tensor
    Returns:
        log_mel: (B, n_mels, T_frames) log-mel spectrogram
    """
    window = torch.hann_window(n_fft, device=device)
    spec = torch.stft(audio, n_fft, hop_length, window=window,
                      return_complex=True)
    mag = spec.abs()  # (B, F, T_frames)
    mel = torch.matmul(mel_fb.to(device), mag)  # (B, n_mels, T_frames)
    return torch.log(mel + 1e-8)


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


def spectral_subtraction_v2(noisy_audio, n_fft=512, hop_length=160):
    """Improved Spectral Subtraction: adaptive oversubtract + freq-weighted floor.

    Three improvements over v1:
      1. Running minimum statistics for noise estimation (not just first 5 frames)
      2. Per-frame SNR-adaptive oversubtraction (low SNR → more aggressive)
      3. Frequency-weighted spectral floor (protect low-freq speech fundamentals)

    Still 0 trainable parameters — purely classical signal processing.

    Args:
        noisy_audio: (B, T) or (T,) waveform tensor
        n_fft: FFT size (512 = 32ms @ 16kHz)
        hop_length: hop size (160 = 10ms @ 16kHz)
    Returns:
        enhanced: (B, T) or (T,) enhanced waveform
    """
    squeeze = False
    if noisy_audio.dim() == 1:
        noisy_audio = noisy_audio.unsqueeze(0)
        squeeze = True

    window = torch.hann_window(n_fft, device=noisy_audio.device)
    spec = torch.stft(noisy_audio, n_fft, hop_length, window=window,
                      return_complex=True)  # (B, F, T_frames)
    mag = spec.abs()
    phase = spec.angle()
    B, F, T_frames = mag.shape

    # [Improvement 1] Running minimum statistics noise estimation
    # Initialize from first 5 frames, then track running minimum
    n_init = min(5, T_frames)
    noise_est = mag[..., :n_init].mean(dim=-1, keepdim=True).expand_as(mag).clone()
    alpha_noise = 0.95  # smoothing factor for noise estimate update

    for t in range(1, T_frames):
        # Smooth minimum: tracks slowly-varying noise floor
        frame_power = mag[..., t:t+1]
        # Update noise estimate: blend previous estimate with frame minimum
        local_min = torch.minimum(frame_power, noise_est[..., t-1:t])
        noise_est[..., t:t+1] = (alpha_noise * noise_est[..., t-1:t] +
                                  (1.0 - alpha_noise) * local_min)

    # [Improvement 2] Per-frame SNR → adaptive oversubtraction
    # Estimate frame-level SNR across frequency bands
    frame_power_avg = mag.pow(2).mean(dim=1, keepdim=True)       # (B, 1, T)
    noise_power_avg = noise_est.pow(2).mean(dim=1, keepdim=True) # (B, 1, T)
    frame_snr = 10.0 * torch.log10(
        frame_power_avg / (noise_power_avg + 1e-10) + 1e-10)    # (B, 1, T) dB

    # Sigmoid mapping: low SNR → oversubtract≈3.0, high SNR → oversubtract≈0.5
    # Floor of 0.5 prevents over-suppression at moderate SNR
    oversubtract = 0.5 + 2.5 * torch.sigmoid(-0.3 * (frame_snr - 5.0))  # (B, 1, T)

    # [Improvement 3] SF-weighted subtraction strength
    # Stationary noise (high SF) → full subtraction (sf_weight ≈ 1.0)
    # Non-stationary noise (low SF) → reduced subtraction (sf_weight ≈ 0.3)
    # This prevents over-subtraction artifacts for babble/street noise
    geo_mean = torch.exp(torch.log(mag + 1e-10).mean(dim=1, keepdim=True))  # (B,1,T)
    arith_mean = mag.mean(dim=1, keepdim=True)                               # (B,1,T)
    sf_per_frame = (geo_mean / (arith_mean + 1e-10)).clamp(0, 1)            # (B,1,T)
    sf_weight = 0.3 + 0.7 * sf_per_frame  # range [0.3, 1.0]
    oversubtract = oversubtract * sf_weight

    # [Improvement 4] Frequency-weighted spectral floor
    # More protection at low frequencies (speech F0, formants)
    # Less protection at high frequencies (allow more noise removal)
    freq_floor = torch.linspace(0.15, 0.03, F, device=mag.device)
    freq_floor = freq_floor.unsqueeze(0).unsqueeze(-1)  # (1, F, 1)

    # Spectral subtraction with adaptive parameters
    enhanced_mag = mag - oversubtract * noise_est
    enhanced_mag = torch.maximum(enhanced_mag, freq_floor * mag)

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
        # Convert real-valued (F, T, 2) to complex tensor for istft
        out_complex = torch.complex(out[..., 0], out[..., 1])
        enh = torch.istft(out_complex, n_fft, hop_length, n_fft, window)
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


def estimate_snr_simple(audio, n_noise_frames=5, hop_length=160):
    """Estimate SNR from audio signal (first N frames = noise floor).

    Used for SNR-adaptive enhancer bypass: high SNR → skip enhancer (preserve Clean),
    low SNR → apply enhancer (noise removal).

    Args:
        audio: (B, T) waveform tensor
        n_noise_frames: number of initial frames assumed to be noise-only
        hop_length: frame hop size in samples
    Returns:
        snr_db: (B, 1) estimated SNR in dB
    """
    frame_size = hop_length * 2
    noise_samples = min(n_noise_frames * frame_size, audio.size(-1) // 4)
    noise_floor = audio[:, :noise_samples].pow(2).mean(dim=-1, keepdim=True) + 1e-10
    signal_power = audio.pow(2).mean(dim=-1, keepdim=True)
    snr_linear = signal_power / noise_floor
    snr_db_est = 10 * torch.log10(snr_linear + 1e-10)
    return snr_db_est  # (B, 1)


def compute_spectral_flatness_audio(audio, n_fft=512, hop_length=160):
    """Compute utterance-level spectral flatness from audio waveform.

    Spectral Flatness (SF) distinguishes noise stationarity:
      High SF (≈0.9) → flat spectrum → stationary (white, pink)
      Low SF (≈0.3)  → peaked spectrum → non-stationary (babble, speech)

    Used by noise_aware_bypass to adapt threshold per noise type.

    Args:
        audio: (B, T) waveform tensor
    Returns:
        sf: (B,) spectral flatness values ∈ [0, 1]
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    window = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(audio, n_fft, hop_length, window=window,
                      return_complex=True)  # (B, F, T)
    mag = spec.abs().mean(dim=-1)  # Average over time → (B, F)

    log_mag = torch.log(mag + 1e-8)
    geo_mean = torch.exp(log_mag.mean(dim=-1))   # (B,) geometric mean
    arith_mean = mag.mean(dim=-1) + 1e-8          # (B,) arithmetic mean
    sf = (geo_mean / arith_mean).clamp(0, 1)

    return sf  # (B,)


def noise_aware_bypass(original, enhanced, bypass_threshold=-2.0,
                       bypass_scale=3.0, sf_range=2.0):
    """Noise-type-aware SNR-adaptive bypass (v4 all-noise-safe).

    Optimized to eliminate 0dB degradation for ALL noise types.
    sf_range=2.0 ensures gate > 0.85 at 0dB for all noise types
    (Factory/Street/Babble included), unlike sf_range=8.0 which
    only fixed White/Pink.

    Expected gate values (sf_range=2.0):
      Noise     SF    thresh  -15dB  -5dB   0dB    5dB
      White     0.9   -1.8    0.00   0.00   0.996  1.00
      Factory   0.5   -1.0    0.00   0.00   0.953  1.00
      Babble    0.3   -0.6    0.00   0.00   0.858  1.00

    Args:
        original: (B, T) noisy audio before enhancement
        enhanced: (B, T) enhanced audio after SS
        bypass_threshold: base threshold in dB (default -2.0)
        bypass_scale: sigmoid steepness (default 3.0)
        sf_range: spectral flatness adaptive range (default 2.0)
    Returns:
        output: (B, T) adaptively blended audio
    """
    snr_est = estimate_snr_simple(original)               # (B, 1)
    sf = compute_spectral_flatness_audio(original)        # (B,)

    # Adaptive threshold based on noise type (sf_range=2.0):
    # White (SF=0.9): thresh = -2 + 2*0.1 = -1.8 → enhance below ~-2dB
    # Factory (SF=0.5): thresh = -2 + 2*0.5 = -1.0 → enhance below ~-1dB
    # Babble (SF=0.3): thresh = -2 + 2*0.7 = -0.6 → enhance below ~-1dB
    # ALL noises: gate > 0.85 at 0dB → bypass → no degradation
    adaptive_threshold = bypass_threshold + sf_range * (1.0 - sf.unsqueeze(1))  # (B, 1)

    # Sharp gate for clean on/off transition (~3dB window)
    gate = torch.sigmoid(bypass_scale * (snr_est - adaptive_threshold))

    return gate * original + (1 - gate) * enhanced


# ============================================================================
# Auxiliary Routing Loss: Gate Target Computation (DualPCEN v2)
# ============================================================================

# Soft gate targets: noise type → desired gate value
# High gate = stationary expert, low gate = non-stationary expert
NOISE_GATE_TARGETS = {
    'clean': 0.2,     # mostly nonstat expert (preserves speech dynamics)
    'babble': 0.15,   # strongly nonstat (babble = non-stationary)
    'street': 0.45,   # mixed (street has both stationary hum + transients)
    'factory': 0.65,  # mostly stat (factory hum = stationary)
    'pink': 0.80,     # strongly stat (broadband stationary)
    'white': 0.90,    # very strongly stat (maximally stationary)
}

# Level 2 gate targets (TriPCEN): broadband vs colored within stationary
# High p_broad = broadband (white/pink → Expert 1), low = colored (factory/street → Expert 2)
# Only meaningful for stationary noise types; non-stationary gets center target (0.5)
NOISE_GATE_L2_TARGETS = {
    'clean': 0.5,     # irrelevant (routed to nonstat at L1), use center
    'babble': 0.5,    # irrelevant (routed to nonstat at L1), use center
    'white': 0.85,    # broadband stationary → Expert 1
    'pink': 0.80,     # broadband stationary → Expert 1
    'factory': 0.20,  # colored stationary → Expert 2
    'street': 0.30,   # colored/structured → Expert 2
}


def _compute_gate_target(noise_type, snr_db):
    """Compute soft gate target based on noise type and SNR.

    At low SNR, noise dominates → push target toward more extreme values.
    At high SNR, speech dominates → pull target toward center (0.5).

    Uses softer multiplier range [0.6, 1.2] to avoid unreachable 0.0/1.0
    targets (sigmoid gates practically output [0.05, 0.95]).

    Returns:
        float: target gate value in [0.05, 0.95] (sigmoid-reachable range)
    """
    import math
    base = NOISE_GATE_TARGETS.get(noise_type, 0.5)
    # snr_factor: ~0.88 at -15dB, ~0.5 at 5dB, ~0.12 at 15dB
    snr_factor = 1.0 / (1.0 + math.exp((snr_db - 5.0) / 5.0))
    # Softer multiplier: [0.6, 1.2] instead of [0.5, 1.5]
    # Prevents clipping to 0.0/1.0 which sigmoid gates can't reach
    target = 0.5 + (base - 0.5) * (0.6 + 0.6 * snr_factor)
    return max(0.05, min(0.95, target))


def _compute_gate_l2_target(noise_type, snr_db):
    """Compute Level 2 gate target for TriPCEN: broadband vs colored.

    Only meaningful for stationary noise types. Non-stationary types
    are handled at Level 1 (stat vs non-stat), so L2 target is 0.5.

    Returns:
        float: target gate value in [0.05, 0.95]
    """
    import math
    base = NOISE_GATE_L2_TARGETS.get(noise_type, 0.5)
    snr_factor = 1.0 / (1.0 + math.exp((snr_db - 5.0) / 5.0))
    target = 0.5 + (base - 0.5) * (0.6 + 0.6 * snr_factor)
    return max(0.05, min(0.95, target))


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
                   gtcrn_model=None, enhancer_bypass=False,
                   bypass_threshold=-2.0, bypass_scale=3.0,
                   ss_version='v2', bypass_version='v2',
                   is_cnn=False, mel_fb=None):
    """Evaluate under noisy conditions with optional front-end enhancer.

    Supports both NanoMamba (raw audio input) and CNN baselines (mel input).

    Args:
        enhancer_type: 'spectral' (0 params, classical) or 'gtcrn' (23.7K pre-trained)
        gtcrn_model: loaded GTCRN model (required if enhancer_type='gtcrn')
        enhancer_bypass: if True, apply SNR-adaptive bypass (blend original + enhanced)
        bypass_threshold: SNR threshold in dB for bypass gate center
        bypass_scale: sigmoid steepness (higher = sharper transition)
        ss_version: 'v1' (fixed oversubtract) or 'v2' (adaptive oversubtract + freq floor)
        bypass_version: 'v1' (fixed threshold) or 'v2' (noise-type-aware threshold)
        is_cnn: if True, model expects mel input → compute mel from noisy audio
        mel_fb: mel filterbank tensor (required if is_cnn=True)
    """
    # Select SS function based on version
    ss_fn = spectral_subtraction_v2 if ss_version == 'v2' else spectral_subtraction_enhance

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
            if enhancer_bypass:
                original = noisy_audio.clone()
                if enhancer_type == 'gtcrn' and gtcrn_model is not None:
                    enhanced = gtcrn_enhance(noisy_audio, gtcrn_model)
                else:
                    enhanced = ss_fn(noisy_audio)

                if bypass_version == 'v2':
                    # [v2] Noise-type-aware adaptive bypass
                    noisy_audio = noise_aware_bypass(
                        original, enhanced,
                        bypass_threshold=bypass_threshold,
                        bypass_scale=bypass_scale)
                else:
                    # [v1] Fixed-threshold bypass (original)
                    snr_est = estimate_snr_simple(original)  # (B, 1)
                    gate = torch.sigmoid(bypass_scale * (snr_est - bypass_threshold))
                    noisy_audio = gate * original + (1 - gate) * enhanced
            else:
                if enhancer_type == 'gtcrn' and gtcrn_model is not None:
                    noisy_audio = gtcrn_enhance(noisy_audio, gtcrn_model)
                else:
                    noisy_audio = ss_fn(noisy_audio)

        if is_cnn and mel_fb is not None:
            # CNN: compute mel from (possibly enhanced) noisy audio
            noisy_mel = _compute_mel_batch(noisy_audio, 512, 160, mel_fb, device)
            logits = model(noisy_mel)
        else:
            # NanoMamba: raw audio → internal STFT/SNR/DualPCEN
            logits = model(noisy_audio)

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


@torch.no_grad()
def evaluate_reverb(model, val_loader, device, rt60=0.5,
                    noise_type=None, snr_db=None, dataset_audios=None,
                    use_enhancer=False, enhancer_type='spectral',
                    gtcrn_model=None,
                    is_cnn=False, mel_fb=None):
    """Evaluate under reverberant conditions with optional noise and enhancer.

    Processing chain: clean → reverb → [noise] → [enhancer] → model

    Args:
        rt60: Reverberation time in seconds
        noise_type: If set, adds noise after reverb (e.g., 'factory', 'babble')
        snr_db: SNR in dB (required if noise_type is set)
        use_enhancer: Apply front-end enhancer
        enhancer_type: 'spectral' or 'gtcrn'
        gtcrn_model: Loaded GTCRN model
        is_cnn: if True, model expects mel input → compute mel from reverberant audio
        mel_fb: mel filterbank tensor (required if is_cnn=True)
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

        # Step 4: Model inference (CNN needs mel, SSM uses raw audio)
        if is_cnn and mel_fb is not None:
            reverb_mel = _compute_mel_batch(reverberant, 512, 160, mel_fb, device)
            logits = model(reverb_mel)
        else:
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
                         enhancer_type='spectral', gtcrn_model=None,
                         enhancer_bypass=False, bypass_threshold=-2.0,
                         bypass_scale=3.0,
                         ss_version='v2', bypass_version='v2'):
    if noise_types is None:
        noise_types = ['factory', 'white', 'babble', 'street', 'pink']
    if snr_levels is None:
        snr_levels = [-15, -10, -5, 0, 5, 10, 15, 'clean']

    enhancer_names = {'spectral': 'SPECTRAL SUBTRACTION (0 params)',
                      'gtcrn': 'GTCRN PRE-TRAINED (23.7K params)'}
    bypass_str = " + SNR-ADAPTIVE BYPASS" if enhancer_bypass else ""
    version_str = ""
    if ss_version == 'v2' or bypass_version == 'v2':
        version_str = f" [SS:{ss_version}, Bypass:{bypass_version}]"
    enhancer_str = f" [WITH {enhancer_names.get(enhancer_type, enhancer_type)}{bypass_str}{version_str}]" if use_enhancer else ""
    print("\n" + "=" * 80)
    print(f"  NOISE ROBUSTNESS EVALUATION{enhancer_str}")
    print(f"  Noise types: {noise_types}")
    print(f"  SNR levels: {snr_levels}")
    if enhancer_bypass:
        print(f"  Bypass: threshold={bypass_threshold}dB, scale={bypass_scale}")
        if ss_version == 'v2':
            print(f"  SS v2: adaptive oversubtract + freq-weighted floor + running noise est")
        if bypass_version == 'v2':
            print(f"  Bypass v2: noise-type-aware threshold (SF-adaptive)")
    print("=" * 80)

    # Prepare mel filterbank for CNN baselines
    mel_fb = _create_mel_fb_tensor()

    results = {}
    for model_name, model in models_dict.items():
        model.eval()
        is_cnn = _is_cnn_model(model_name)
        # Skip external SS if model has built-in SpectralEnhancer
        has_builtin_se = getattr(model, 'use_spectral_enhancer', False)
        _use_enhancer = use_enhancer and not has_builtin_se
        se_tag = " [built-in SE]" if has_builtin_se else ""
        results[model_name] = {}
        print(f"\n  Evaluating: {model_name}" +
              (" [CNN: mel input]" if is_cnn else " [SSM: raw audio]") +
              se_tag, flush=True)

        for noise_type in noise_types:
            results[model_name][noise_type] = {}
            for snr in snr_levels:
                if snr == 'clean':
                    if is_cnn:
                        acc = _evaluate_cnn(model, val_loader, device)
                    else:
                        acc = evaluate(model, val_loader, device)
                else:
                    acc = evaluate_noisy(
                        model, val_loader, device, noise_type, snr,
                        dataset_audios=dataset_audios,
                        use_enhancer=_use_enhancer,
                        enhancer_type=enhancer_type,
                        gtcrn_model=gtcrn_model,
                        enhancer_bypass=enhancer_bypass,
                        bypass_threshold=bypass_threshold,
                        bypass_scale=bypass_scale,
                        ss_version=ss_version,
                        bypass_version=bypass_version,
                        is_cnn=is_cnn,
                        mel_fb=mel_fb)
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

    # ================================================================
    # Structural Advantage Analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("  STRUCTURAL ADVANTAGE ANALYSIS")
    print("  (NanoMamba dynamic routing vs CNN fixed-stats comparison)")
    print("  NanoMamba: DualPCEN SF/Tilt routing = DYNAMIC per-input at inference")
    print("  BC-ResNet: Sub-Spectral Norm = sub-band BN, FROZEN stats at inference")
    print("  DS-CNN-S:  Standard BatchNorm = global BN, FROZEN stats at inference")
    print("=" * 80)

    # Identify NanoMamba and CNN models
    nanomamba_models = [n for n in results if not _is_cnn_model(n)]
    cnn_models = [n for n in results if _is_cnn_model(n)]

    if nanomamba_models and cnn_models:
        for nm_name in nanomamba_models:
            for cnn_name in cnn_models:
                nm_params = sum(p.numel() for p in models_dict[nm_name].parameters())
                cnn_params = sum(p.numel() for p in models_dict[cnn_name].parameters())
                param_ratio = cnn_params / max(nm_params, 1)
                print(f"\n  {nm_name} ({nm_params:,} params) vs "
                      f"{cnn_name} ({cnn_params:,} params, {param_ratio:.1f}x larger):")

                total_nm_advantage = 0
                total_clean_diff = 0
                n_comparisons = 0
                for noise_type in noise_types:
                    if noise_type not in results[nm_name] or noise_type not in results[cnn_name]:
                        continue
                    nm_clean = results[nm_name][noise_type].get('clean', 0)
                    cnn_clean = results[cnn_name][noise_type].get('clean', 0)
                    # Average over extreme SNRs (-15, -10, -5)
                    extreme_snrs = [s for s in snr_levels if isinstance(s, int) and s <= -5]
                    if extreme_snrs:
                        nm_extreme = np.mean([results[nm_name][noise_type].get(str(s), 0)
                                              for s in extreme_snrs])
                        cnn_extreme = np.mean([results[cnn_name][noise_type].get(str(s), 0)
                                               for s in extreme_snrs])
                        advantage = nm_extreme - cnn_extreme
                        clean_diff = nm_clean - cnn_clean
                        total_nm_advantage += advantage
                        total_clean_diff += clean_diff
                        n_comparisons += 1
                        winner = "NanoMamba WINS" if advantage > 0 else "CNN leads"
                        print(f"    {noise_type:<10} | Clean: {clean_diff:+.1f}%p | "
                              f"Extreme SNR avg: NM {nm_extreme:.1f}% vs CNN {cnn_extreme:.1f}% "
                              f"({advantage:+.1f}%p) [{winner}]")

                if n_comparisons > 0:
                    avg_advantage = total_nm_advantage / n_comparisons
                    avg_clean = total_clean_diff / n_comparisons
                    print(f"    {'OVERALL':<10} | Avg clean diff: {avg_clean:+.1f}%p | "
                          f"Avg extreme advantage: {avg_advantage:+.1f}%p "
                          f"(with {param_ratio:.1f}x FEWER params)")

    return results


# ============================================================================
# DualPCEN Routing Analysis — Evidence for Structural Advantage
# ============================================================================

@torch.no_grad()
def analyze_dualpcen_routing(model, val_loader, device, dataset_audios=None):
    """Analyze DualPCEN routing behavior for clean vs noisy inputs.

    This function provides DIRECT EVIDENCE for the structural advantage paper claim:
      - Clean inputs → DualPCEN gate ≈ 0 (Expert1/nonstat dominates)
      - Noisy inputs → DualPCEN gate ≈ 1 (Expert2/stat dominates)
      - CNN has NO equivalent routing → same computation for all inputs

    Key metrics:
      1. Gate Separation: mean_noisy_gate - mean_clean_gate (higher = better routing)
      2. Routing Consistency: std of gate values (lower = more confident routing)
      3. Per-noise-type routing profile: which expert handles which noise

    Returns:
        dict with routing statistics for paper analysis
    """
    model.eval()

    # Find DualPCEN module
    dualpcen = None
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'DualPCEN':
            dualpcen = module
            break

    if dualpcen is None:
        print("  [SKIP] Model does not have DualPCEN (not applicable to CNN)")
        return None

    print("\n" + "=" * 80)
    print("  DualPCEN ROUTING ANALYSIS")
    print("  Expert1 (nonstat, delta=2.0): speech-focused, AGC disabled")
    print("  Expert2 (stat, delta=0.01):   noise-robust, strong AGC")
    print("  Gate: 0=Expert1(nonstat), 1=Expert2(stat)")
    print("  Evidence: routing separation = structural advantage over CNN")
    print("=" * 80)

    noise_types_test = ['factory', 'white', 'babble', 'street', 'pink']
    snr_test = [-15, 0, 15]

    results = {'clean': {}, 'noisy': {}}

    # Collect clean gate statistics
    clean_gates = []
    for mel, labels, audio in val_loader:
        audio = audio.to(device)
        # Run through model to get internal activations
        # We need to hook into DualPCEN's forward
        gate_values = _extract_dualpcen_gates(model, dualpcen, audio, device)
        if gate_values is not None:
            clean_gates.append(gate_values)
        if len(clean_gates) >= 5:  # 5 batches sufficient
            break

    if clean_gates:
        clean_all = torch.cat(clean_gates, dim=0)
        clean_mean = clean_all.mean().item()
        clean_std = clean_all.std().item()
        results['clean'] = {'mean_gate': clean_mean, 'std_gate': clean_std}
        print(f"\n  Clean:   gate mean={clean_mean:.4f}, std={clean_std:.4f}")
        expert_str = "Expert1 (nonstat)" if clean_mean < 0.5 else "Expert2 (stat)"
        print(f"           → Routing to {expert_str}")

    # Collect noisy gate statistics
    for noise_type in noise_types_test:
        for snr_db in snr_test:
            noisy_gates = []
            for mel, labels, audio in val_loader:
                audio = audio.to(device)
                noise = generate_noise_signal(
                    noise_type, audio.size(-1), sr=16000,
                    dataset_audios=dataset_audios).to(device)
                noisy_audio = mix_audio_at_snr(audio, noise, snr_db)
                gate_values = _extract_dualpcen_gates(
                    model, dualpcen, noisy_audio, device)
                if gate_values is not None:
                    noisy_gates.append(gate_values)
                if len(noisy_gates) >= 3:
                    break

            if noisy_gates:
                noisy_all = torch.cat(noisy_gates, dim=0)
                noisy_mean = noisy_all.mean().item()
                noisy_std = noisy_all.std().item()
                key = f"{noise_type}_{snr_db}dB"
                results['noisy'][key] = {
                    'mean_gate': noisy_mean,
                    'std_gate': noisy_std,
                    'separation': noisy_mean - clean_mean if clean_gates else 0,
                }
                separation = noisy_mean - clean_mean if clean_gates else 0
                expert_str = "Expert1 (nonstat)" if noisy_mean < 0.5 else "Expert2 (stat)"
                print(f"  {noise_type:<8} {snr_db:>4}dB: gate mean={noisy_mean:.4f}, "
                      f"std={noisy_std:.4f}, separation={separation:+.4f} → {expert_str}")

    # Summary
    if results.get('noisy') and clean_gates:
        separations = [v['separation'] for v in results['noisy'].values()]
        avg_sep = np.mean(separations)
        print(f"\n  ROUTING SUMMARY:")
        print(f"    Average gate separation (noisy - clean): {avg_sep:+.4f}")
        print(f"    {'GOOD' if abs(avg_sep) > 0.05 else 'WEAK'}: ",
              end="")
        if abs(avg_sep) > 0.05:
            print(f"DualPCEN is routing differently for clean vs noisy inputs")
            print(f"    → This is the structural advantage CNN CANNOT replicate")
            print(f"    → BC-ResNet SSN: frozen sub-band stats cannot do per-input routing")
        else:
            print(f"Routing separation is weak — consider more noise-aug training")

    return results


def _extract_dualpcen_gates(model, dualpcen, audio, device):
    """Extract DualPCEN gate values by hooking into the forward pass."""
    gate_storage = []

    def hook_fn(module, input, output):
        # DualPCEN computes gate internally; we need to re-compute it
        mel_linear = input[0]  # (B, n_mels, T)
        log_mel = torch.log(mel_linear + 1e-8)
        geo_mean = torch.exp(log_mel.mean(dim=1, keepdim=True))
        arith_mean = mel_linear.mean(dim=1, keepdim=True) + 1e-8
        sf = (geo_mean / arith_mean).clamp(0, 1)

        n_mels = mel_linear.size(1)
        low_energy = mel_linear[:, :n_mels // 3, :].mean(dim=1, keepdim=True)
        high_energy = mel_linear[:, 2 * n_mels // 3:, :].mean(dim=1, keepdim=True)
        spectral_tilt = (low_energy / (low_energy + high_energy + 1e-8)).clamp(0, 1)

        sf_adjusted = sf + (1.0 - sf) * torch.relu(spectral_tilt - 0.6)
        gate = torch.sigmoid(dualpcen.gate_temp * (sf_adjusted - 0.5))
        gate_storage.append(gate.mean(dim=(1, 2)).cpu())  # (B,)

    handle = dualpcen.register_forward_hook(hook_fn)
    try:
        model(audio)
    except Exception:
        pass
    handle.remove()

    if gate_storage:
        return gate_storage[0]
    return None


# ============================================================================
# Per-Sub-Band Selectivity Analysis — Per-Band Multimodality Evidence
# ============================================================================

@torch.no_grad()
def analyze_selectivity_gates(model, val_loader, device, dataset_audios=None,
                               n_batches=5):
    """Analyze SM-SSM/NC-SSM per-sub-band selectivity gate (σ) behavior.

    Provides DIRECT EVIDENCE for per-band multimodality:
      - Clean bands → σ ≈ 1 (selective SSM mode, BETTER than CNN)
      - Noisy bands → σ ≈ 0 (LTI/CNN-like mode, MATCHES CNN)
      - Result: NC-SSM ≥ CNN in every sub-band

    Key metrics:
      1. Per-sub-band σ values across noise types and SNR levels
      2. Sub-band separation: σ_high_freq - σ_low_freq (higher = better per-band adaptation)
      3. Multimodality index: std(σ across bands) — NC-SSM should be high, SM-SSM low

    Returns:
        dict with per-condition, per-sub-band sigma statistics
    """
    model.eval()

    # Find SM-SSM or NC-SSM modules
    ssm_modules = []
    ssm_type = 'unknown'
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == 'NoiseCondSMSSM':
            ssm_modules.append((name, module))
            ssm_type = 'NC-SSM'
        elif cls_name == 'SelectivityModulatedSSM':
            ssm_modules.append((name, module))
            if ssm_type != 'NC-SSM':  # NC-SSM is subclass of SM-SSM
                ssm_type = 'SM-SSM'

    if not ssm_modules:
        print("  [SKIP] Model does not have SM-SSM or NC-SSM (not applicable)")
        return None

    # Keep only the most specific type
    if ssm_type == 'NC-SSM':
        ssm_modules = [(n, m) for n, m in ssm_modules
                       if m.__class__.__name__ == 'NoiseCondSMSSM']

    n_blocks = len(ssm_modules)
    n_states = ssm_modules[0][1].d_state  # 5 or 6
    n_mels = 40

    # Sub-band frequency labels (dynamic based on d_state)
    mel_freqs = [0, 500, 1000, 2000, 4000, 6000, 8000]  # approximate mel boundaries
    band_labels = []
    for i in range(n_states):
        lo_idx = int(i * n_mels / n_states)
        hi_idx = int((i + 1) * n_mels / n_states) - 1
        lo_hz = mel_freqs[min(i, len(mel_freqs) - 1)]
        hi_hz = mel_freqs[min(i + 1, len(mel_freqs) - 1)]
        band_labels.append(f'{lo_hz}-{hi_hz}Hz')

    print("\n" + "=" * 80)
    print(f"  PER-SUB-BAND SELECTIVITY ANALYSIS ({ssm_type}, {n_blocks} blocks)")
    print(f"  σ ≈ 1: Selective SSM mode (adaptive, exploits clean signal)")
    print(f"  σ ≈ 0: LTI/CNN mode (fixed, noise-robust)")
    print(f"  Per-band multimodality = σ varies across bands = competitive advantage")
    print("=" * 80)

    noise_types_test = ['factory', 'white', 'babble', 'street', 'pink']
    snr_levels_test = [-15, -5, 0, 5, 15]

    results = {}

    # ─── Clean condition ────────────────────────────────────
    sigma_accum = {i: [] for i in range(n_blocks)}
    batch_count = 0
    for mel, labels, audio in val_loader:
        audio = audio.to(device)
        model(audio)
        for blk_idx, (name, ssm) in enumerate(ssm_modules):
            if hasattr(ssm, '_last_sigma_BC'):
                # (B, L, N) → mean over batch and time → (N,)
                sigma_accum[blk_idx].append(
                    ssm._last_sigma_BC.mean(dim=(0, 1)).cpu())
        batch_count += 1
        if batch_count >= n_batches:
            break

    clean_sigmas = {}
    for blk_idx in range(n_blocks):
        if sigma_accum[blk_idx]:
            clean_sigmas[blk_idx] = torch.stack(
                sigma_accum[blk_idx]).mean(dim=0).numpy()

    results['clean'] = clean_sigmas

    if clean_sigmas:
        print(f"\n  {'Condition':<20} |", end="")
        for b in range(n_states):
            print(f" {band_labels[b]:>10}", end="")
        print(f" | {'Multi-idx':>10} | {'Separation':>10}")
        print("  " + "-" * (22 + 12 * n_states + 26))

        for blk_idx in range(n_blocks):
            s = clean_sigmas.get(blk_idx)
            if s is not None:
                multi_idx = np.std(s)
                separation = s[-1] - s[0]  # high_freq - low_freq
                row = f"  {'Clean (blk'+str(blk_idx)+')':<20} |"
                for b in range(n_states):
                    row += f" {s[b]:>10.4f}"
                row += f" | {multi_idx:>10.4f} | {separation:>+10.4f}"
                print(row)

    # ─── Noisy conditions ───────────────────────────────────
    for noise_type in noise_types_test:
        for snr_db in snr_levels_test:
            key = f"{noise_type}_{snr_db}dB"
            sigma_accum = {i: [] for i in range(n_blocks)}
            snr_accum = {i: [] for i in range(n_blocks)}
            batch_count = 0

            for mel, labels, audio in val_loader:
                audio = audio.to(device)
                noise = generate_noise_signal(
                    noise_type, audio.size(-1), sr=16000,
                    dataset_audios=dataset_audios).to(device)
                noisy_audio = mix_audio_at_snr(audio, noise, snr_db)
                model(noisy_audio)

                for blk_idx, (name, ssm) in enumerate(ssm_modules):
                    if hasattr(ssm, '_last_sigma_BC'):
                        sigma_accum[blk_idx].append(
                            ssm._last_sigma_BC.mean(dim=(0, 1)).cpu())
                    if hasattr(ssm, '_last_snr_sub'):
                        snr_accum[blk_idx].append(
                            ssm._last_snr_sub.mean(dim=(0, 1)).cpu())
                batch_count += 1
                if batch_count >= n_batches:
                    break

            condition_sigmas = {}
            condition_snrs = {}
            for blk_idx in range(n_blocks):
                if sigma_accum[blk_idx]:
                    condition_sigmas[blk_idx] = torch.stack(
                        sigma_accum[blk_idx]).mean(dim=0).numpy()
                if snr_accum[blk_idx]:
                    condition_snrs[blk_idx] = torch.stack(
                        snr_accum[blk_idx]).mean(dim=0).numpy()

            results[key] = {
                'sigma': condition_sigmas,
                'snr_sub': condition_snrs,
            }

            # Print only block 0 for brevity (block 1 in full report)
            s = condition_sigmas.get(0)
            if s is not None:
                multi_idx = np.std(s)
                separation = s[-1] - s[0]
                row = f"  {key:<20} |"
                for b in range(n_states):
                    row += f" {s[b]:>10.4f}"
                row += f" | {multi_idx:>10.4f} | {separation:>+10.4f}"
                print(row)

    # ─── Summary: Per-Band Multimodality Evidence ───────────
    print(f"\n  {'='*80}")
    print(f"  PER-BAND MULTIMODALITY SUMMARY")
    print(f"  {'='*80}")

    for noise_type in noise_types_test:
        print(f"\n  --- {noise_type.upper()} ---")
        for snr_db in snr_levels_test:
            key = f"{noise_type}_{snr_db}dB"
            if key in results and 'sigma' in results[key]:
                s = results[key]['sigma'].get(0)
                if s is not None:
                    multi_idx = np.std(s)
                    min_band = np.argmin(s)
                    max_band = np.argmax(s)
                    print(f"    {snr_db:>4}dB: "
                          f"σ_min={s[min_band]:.3f} ({band_labels[min_band]}) "
                          f"σ_max={s[max_band]:.3f} ({band_labels[max_band]}) "
                          f"multimodality={multi_idx:.4f} "
                          f"{'★ STRONG' if multi_idx > 0.05 else '○ weak'}")

    # ─── Factory noise special: per-band multimodality proof ─
    factory_0 = results.get('factory_0dB', {}).get('sigma', {}).get(0)
    white_0 = results.get('white_0dB', {}).get('sigma', {}).get(0)
    if factory_0 is not None and white_0 is not None:
        print(f"\n  KEY FINDING: Per-Band Multimodality under Factory vs White Noise (0dB)")
        print(f"    Factory: low-freq NOISY, high-freq CLEAN")
        print(f"      σ pattern: [{', '.join(f'{v:.3f}' for v in factory_0)}]")
        print(f"      → Low bands LTI (CNN-like), High bands Selective (SSM advantage)")
        print(f"    White: ALL bands NOISY (broadband)")
        print(f"      σ pattern: [{', '.join(f'{v:.3f}' for v in white_0)}]")
        print(f"      → All bands LTI (CNN-like protection everywhere)")
        print(f"    Separation: Factory {factory_0[-1]-factory_0[0]:+.3f} vs "
              f"White {white_0[-1]-white_0[0]:+.3f}")
        if abs(factory_0[-1] - factory_0[0]) > abs(white_0[-1] - white_0[0]):
            print(f"    ✓ NC-SSM correctly differentiates per-band — "
                  f"MULTIMODALITY CONFIRMED")
        else:
            print(f"    ✗ Per-band differentiation weak — "
                  f"need stronger sub-band training signal")

    return results


@torch.no_grad()
def analyze_subband_perturbation(model, val_loader, device,
                                  noise_type='factory', snr_db=0,
                                  dataset_audios=None, n_batches=10):
    """Measure per-sub-band importance via frequency masking.

    Zeroes out each sub-band (8 mel bins) in turn and measures accuracy drop.
    This is the most direct "per-band accuracy" proxy possible.

    Hypothesis for NC-SSM vs CNN:
      - NC-SSM: robust to masking NOISY bands (already in LTI mode),
                sensitive to masking CLEAN bands (selective mode carries info)
      - CNN: similar sensitivity across all bands (no per-band specialization)

    If NC-SSM shows differentiated per-band sensitivity while CNN doesn't,
    this proves per-band multimodality is REAL and functional.

    Returns:
        dict with per-sub-band accuracy and drop for each model
    """
    # Dynamic sub-band configuration (supports d_state=5, 6, etc.)
    n_mels = 40
    # Detect d_state from model if it's an NC-SSM/SM-SSM
    n_sub_bands = 5  # default
    for m in model.modules():
        if hasattr(m, 'n_sub_bands'):
            n_sub_bands = m.n_sub_bands
            break
    # Compute band boundaries using adaptive pooling logic (same as NC-SSM)
    band_boundaries = [int(i * n_mels / n_sub_bands) for i in range(n_sub_bands + 1)]
    bands_per_sub_list = [band_boundaries[i+1] - band_boundaries[i] for i in range(n_sub_bands)]
    mel_freqs = [0, 500, 1000, 2000, 4000, 6000, 8000]
    band_labels = []
    for i in range(n_sub_bands):
        lo_hz = mel_freqs[min(i, len(mel_freqs) - 1)]
        hi_hz = mel_freqs[min(i + 1, len(mel_freqs) - 1)]
        band_labels.append(f'{lo_hz}-{hi_hz}Hz')

    print(f"\n" + "=" * 80)
    print(f"  SUB-BAND PERTURBATION ANALYSIS")
    print(f"  Noise: {noise_type} at {snr_db}dB")
    print(f"  Method: zero out each sub-band → measure accuracy drop")
    print(f"  NC-SSM hypothesis: robust to noisy-band masking, sensitive to clean-band")
    print("=" * 80)

    model.eval()

    # Step 1: Baseline accuracy (no masking)
    correct_base = 0
    total = 0
    for batch_idx, (mel, labels, audio) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        labels = labels.to(device)
        audio = audio.to(device)
        noise = generate_noise_signal(
            noise_type, audio.size(-1), sr=16000,
            dataset_audios=dataset_audios).to(device)
        noisy_audio = mix_audio_at_snr(audio, noise, snr_db)
        logits = model(noisy_audio)
        _, predicted = logits.max(1)
        correct_base += predicted.eq(labels).sum().item()
        total += labels.size(0)
    base_acc = 100. * correct_base / total if total > 0 else 0

    # Step 2: Per-sub-band masking
    # We need to hook into the mel spectrogram and zero out sub-bands
    # Strategy: hook into the model's STFT/mel pipeline to mask after mel computation
    subband_accs = []

    for sb in range(n_sub_bands):
        mask_start = band_boundaries[sb]
        mask_end = band_boundaries[sb + 1]

        # Register hook to zero out this sub-band in mel spectrogram
        mel_hooks = []

        def make_hook(start, end):
            def hook_fn(module, input, output):
                # output is mel spectrogram (B, n_mels, T) or similar
                if isinstance(output, torch.Tensor) and output.dim() >= 2:
                    n_freq = output.shape[-2] if output.dim() == 3 else output.shape[1]
                    if n_freq == 40:  # n_mels dimension
                        if output.dim() == 3:  # (B, n_mels, T)
                            output = output.clone()
                            output[:, start:end, :] = 0
                        return output
                return output
            return hook_fn

        # Find the SNR estimator or mel computation module to hook into
        # The mel is computed inside the model, so we hook into the
        # first module that processes it
        hook_handle = None
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            # Hook into LSG (Learned Spectral Gate) or SNREstimator's output
            if cls_name in ('LearnedSpectralGate',):
                hook_handle = module.register_forward_hook(
                    make_hook(mask_start, mask_end))
                break

        if hook_handle is None:
            # Fallback: hook into the SNR estimator to mask mel_spec
            for name, module in model.named_modules():
                if module.__class__.__name__ == 'SNREstimator':
                    # SNR estimator processes mel internally
                    # We need a different approach: modify the input audio
                    # to simulate sub-band masking
                    break

        # If no suitable hook point, use frequency-domain masking on audio
        correct_masked = 0
        total_masked = 0

        for batch_idx, (mel, labels, audio) in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            labels = labels.to(device)
            audio = audio.to(device)
            noise = generate_noise_signal(
                noise_type, audio.size(-1), sr=16000,
                dataset_audios=dataset_audios).to(device)
            noisy_audio = mix_audio_at_snr(audio, noise, snr_db)

            if hook_handle is not None:
                # Hook-based masking
                logits = model(noisy_audio)
            else:
                # Audio-domain frequency masking via STFT
                n_fft = 512
                hop = 160
                spec = torch.stft(noisy_audio, n_fft=n_fft, hop_length=hop,
                                  return_complex=True,
                                  window=torch.hann_window(n_fft, device=device))
                # Map mel sub-band to STFT bins (approximate)
                n_freq_bins = spec.shape[-2]
                # Mel bands are roughly log-spaced; approximate linear mapping
                mel_to_fft_start = int(mask_start / 40 * n_freq_bins)
                mel_to_fft_end = int(mask_end / 40 * n_freq_bins)
                spec_masked = spec.clone()
                spec_masked[:, mel_to_fft_start:mel_to_fft_end, :] = 0
                masked_audio = torch.istft(spec_masked, n_fft=n_fft,
                                           hop_length=hop,
                                           window=torch.hann_window(n_fft, device=device),
                                           length=noisy_audio.shape[-1])
                logits = model(masked_audio)

            _, predicted = logits.max(1)
            correct_masked += predicted.eq(labels).sum().item()
            total_masked += labels.size(0)

        if hook_handle is not None:
            hook_handle.remove()

        masked_acc = 100. * correct_masked / total_masked if total_masked > 0 else 0
        drop = base_acc - masked_acc
        subband_accs.append({
            'band': sb,
            'label': band_labels[sb],
            'masked_acc': masked_acc,
            'drop': drop,
        })

    # Print results
    print(f"\n  Baseline accuracy ({noise_type} {snr_db}dB): {base_acc:.1f}%")
    print(f"\n  {'Sub-band':<12} | {'Freq Range':<12} | {'Masked Acc':>10} | "
          f"{'Drop':>8} | {'Importance':>10}")
    print("  " + "-" * 62)

    max_drop = max(r['drop'] for r in subband_accs) if subband_accs else 1
    for r in subband_accs:
        importance = r['drop'] / max(max_drop, 0.01) * 100
        bar = '█' * int(importance / 5) + '░' * (20 - int(importance / 5))
        print(f"  Band {r['band']:<6} | {r['label']:<12} | {r['masked_acc']:>9.1f}% | "
              f"{r['drop']:>+7.1f}% | {bar}")

    # Compute differentiation index: std of drops
    drops = [r['drop'] for r in subband_accs]
    diff_index = np.std(drops) if drops else 0
    print(f"\n  Differentiation index (std of drops): {diff_index:.3f}")
    if diff_index > 1.0:
        print(f"  ✓ Model treats sub-bands DIFFERENTLY → per-band multimodality ACTIVE")
    else:
        print(f"  ○ Model treats sub-bands similarly → limited per-band specialization")

    return {
        'baseline_acc': base_acc,
        'subband_results': subband_accs,
        'differentiation_index': diff_index,
        'noise_type': noise_type,
        'snr_db': snr_db,
    }


@torch.no_grad()
def compare_subband_perturbation(models_dict, val_loader, device,
                                  noise_type='factory', snr_db=0,
                                  dataset_audios=None, n_batches=10):
    """Compare per-sub-band perturbation sensitivity across models.

    Head-to-head comparison proving NC-SSM's per-band multimodality advantage.

    Expected result:
      NC-SSM: high differentiation (different bands have different importance)
      BC-ResNet-1: low differentiation (all bands similarly important)
    """
    # Dynamic sub-band count from first model
    n_sub_bands = 5
    for model in models_dict.values():
        for m in model.modules():
            if hasattr(m, 'n_sub_bands'):
                n_sub_bands = m.n_sub_bands
                break
        break

    print(f"\n" + "=" * 80)
    print(f"  COMPARATIVE SUB-BAND PERTURBATION ANALYSIS ({n_sub_bands} sub-bands)")
    print(f"  Noise: {noise_type} at {snr_db}dB")
    print(f"  Hypothesis: NC-SSM has HIGHER differentiation than CNN")
    print("=" * 80)

    all_results = {}
    for model_name, model in models_dict.items():
        print(f"\n  --- {model_name} ---")
        result = analyze_subband_perturbation(
            model, val_loader, device,
            noise_type=noise_type, snr_db=snr_db,
            dataset_audios=dataset_audios, n_batches=n_batches)
        all_results[model_name] = result

    # Comparative summary
    print(f"\n  {'='*80}")
    print(f"  COMPARATIVE SUMMARY ({noise_type} {snr_db}dB)")
    print(f"  {'='*80}")
    print(f"\n  {'Model':<25} | {'Base Acc':>8} | {'Diff Index':>10} | "
          f"{'Most Important':>15} | {'Least Important':>15}")
    print("  " + "-" * 85)

    for model_name, result in all_results.items():
        if result is None:
            continue
        base = result['baseline_acc']
        diff = result['differentiation_index']
        drops = [(r['label'], r['drop']) for r in result['subband_results']]
        most_imp = max(drops, key=lambda x: x[1])
        least_imp = min(drops, key=lambda x: x[1])
        print(f"  {model_name:<25} | {base:>7.1f}% | {diff:>10.3f} | "
              f"{most_imp[0]:>9} ({most_imp[1]:+.1f}%) | "
              f"{least_imp[0]:>9} ({least_imp[1]:+.1f}%)")

    return all_results


# ============================================================================
# Runtime Calibration Evaluation
# ============================================================================

# Calibration profiles: noise environment → optimal parameter preset
CALIBRATION_PROFILES = {
    # 'clean' profile must match training defaults (delta_floor_min=0.05)
    # Previous bug: delta_floor_min=0.15 caused Clean acc to drop 93.7%→82.8%
    'clean':    dict(delta_floor_min=0.05, delta_floor_max=0.15,
                    epsilon_min=0.08, epsilon_max=0.20, bgate_floor=0.0),
    'light':    dict(delta_floor_min=0.05, delta_floor_max=0.15,
                    epsilon_min=0.08, epsilon_max=0.20, bgate_floor=0.1),
    'moderate': dict(delta_floor_min=0.04, delta_floor_max=0.15,
                    epsilon_min=0.10, epsilon_max=0.25, bgate_floor=0.2),
    'extreme':  dict(delta_floor_min=0.02, delta_floor_max=0.15,
                    epsilon_min=0.15, epsilon_max=0.30, bgate_floor=0.4),
}

# SNR → profile mapping (simulates silence-period estimation)
def snr_to_profile(snr_db):
    """Map known SNR to calibration profile (simulates VAD estimation)."""
    if snr_db == 'clean' or (isinstance(snr_db, (int, float)) and snr_db >= 20):
        return 'clean'
    elif isinstance(snr_db, (int, float)) and snr_db >= 10:
        return 'light'
    elif isinstance(snr_db, (int, float)) and snr_db >= 0:
        return 'moderate'
    else:
        return 'extreme'


def calibrate_continuous(snr_db, is_ssm_v2=False):
    """Continuous calibration: SNR → smooth parameter interpolation.

    Replaces discrete 4-profile system with differentiable parameter curves.
    No hard transitions at SNR boundaries — parameters vary smoothly.

    Separate curves for v1 and v2 SSM models: v2 has wider training defaults
    (delta_floor_min=0.03 vs 0.05, epsilon ranges differ), so clean-end anchor
    points must match v2 defaults to avoid train/eval mismatch.

    Args:
        snr_db: estimated SNR in dB ('clean' treated as 25dB)
        is_ssm_v2: if True, use v2 parameter curves (wider default ranges)
    Returns:
        dict: calibration parameters for set_calibration(**params)
    """
    if snr_db == 'clean':
        snr_db = 25.0

    # Normalize: [-20, 25] → [0, 1]
    t = (float(snr_db) + 20.0) / 45.0
    t = max(0.0, min(1.0, t))

    if is_ssm_v2:
        # v2 curves: clean-end matches SSM v2 training defaults
        # delta_floor_min: extreme(-20dB)=0.01, clean(25dB)=0.03 (v2 default)
        delta_floor_min = 0.01 + 0.02 * t
        # epsilon_min: extreme=0.03, clean=0.05 (v2 default)
        epsilon_min = 0.03 + 0.02 * t
        # epsilon_max: extreme=0.40, clean=0.30 (v2 default, wider rescue)
        epsilon_max = 0.40 - 0.10 * t
        # bgate_floor: extreme=0.50, clean=0.00 (same as v1)
        bgate_floor = 0.50 * (1.0 - t)
    else:
        # v1 curves: original parameter ranges
        # delta_floor_min: extreme(-20dB)=0.01, clean(25dB)=0.05
        delta_floor_min = 0.01 + 0.04 * t
        # epsilon_min: extreme=0.05, clean=0.08
        epsilon_min = 0.05 + 0.03 * t
        # epsilon_max: extreme=0.35, clean=0.20
        epsilon_max = 0.35 - 0.15 * t
        # bgate_floor: extreme=0.50, clean=0.00
        bgate_floor = 0.50 * (1.0 - t)

    # delta_floor_max: constant (same for v1/v2)
    delta_floor_max = 0.15

    return dict(
        delta_floor_min=round(delta_floor_min, 4),
        delta_floor_max=round(delta_floor_max, 4),
        epsilon_min=round(epsilon_min, 4),
        epsilon_max=round(epsilon_max, 4),
        bgate_floor=round(bgate_floor, 4),
    )


@torch.no_grad()
def run_calibrated_evaluation(models_dict, val_loader, device,
                              noise_types=None, snr_levels=None,
                              dataset_audios=None,
                              use_enhancer=False, enhancer_type='spectral',
                              gtcrn_model=None, enhancer_bypass=False,
                              bypass_threshold=-2.0, bypass_scale=3.0,
                              ss_version='v2', bypass_version='v2',
                              use_continuous_calibration=False):
    """Evaluate with Runtime Parameter Calibration.

    For each SNR level, sets the optimal calibration profile BEFORE evaluation.
    Simulates real deployment: silence → estimate noise → calibrate → infer.
    """
    if noise_types is None:
        noise_types = ['factory', 'white', 'babble', 'street', 'pink']
    if snr_levels is None:
        snr_levels = [-15, -10, -5, 0, 5, 10, 15, 'clean']

    cal_mode = "CONTINUOUS" if use_continuous_calibration else "DISCRETE"
    print("\n" + "=" * 80)
    print("  RUNTIME CALIBRATION EVALUATION")
    print("  Noise types:", noise_types)
    print("  SNR levels:", snr_levels)
    print(f"  Mode: {cal_mode}")
    if not use_continuous_calibration:
        print("  Profiles: clean(20dB+), light(10-20dB), moderate(0-10dB), extreme(<0dB)")
    print("=" * 80)

    # Prepare mel filterbank for CNN baselines
    mel_fb = _create_mel_fb_tensor()

    results = {}
    for model_name, model in models_dict.items():
        model.eval()
        is_cnn = _is_cnn_model(model_name)
        # Skip external SS if model has built-in SpectralEnhancer
        has_builtin_se = getattr(model, 'use_spectral_enhancer', False)
        _use_enhancer = use_enhancer and not has_builtin_se
        se_tag = ", built-in SE" if has_builtin_se else ""
        results[model_name] = {}
        tag = f"[CNN: mel, no calibration]" if is_cnn else f"[SSM: raw, calibrated{se_tag}]"
        print(f"\n  Evaluating: {model_name} {tag}", flush=True)

        for noise_type in noise_types:
            results[model_name][noise_type] = {}
            for snr in snr_levels:
                # [KEY] Set calibration based on SNR (NanoMamba only)
                if hasattr(model, 'set_calibration'):
                    if use_continuous_calibration:
                        # Continuous: smooth interpolation, no profile boundaries
                        # Detect SSM v2 for correct parameter curves
                        _is_v2 = getattr(model, 'use_ssm_v2', False)
                        cal_params = calibrate_continuous(snr, is_ssm_v2=_is_v2)
                        model.set_calibration(profile='custom', **cal_params)
                    else:
                        # Discrete: 4 profiles with hard SNR boundaries
                        profile = snr_to_profile(snr)
                        model.set_calibration(profile=profile)

                if snr == 'clean':
                    if is_cnn:
                        acc = _evaluate_cnn(model, val_loader, device)
                    else:
                        acc = evaluate(model, val_loader, device)
                else:
                    acc = evaluate_noisy(
                        model, val_loader, device, noise_type, snr,
                        dataset_audios=dataset_audios,
                        use_enhancer=_use_enhancer,
                        enhancer_type=enhancer_type,
                        gtcrn_model=gtcrn_model,
                        enhancer_bypass=enhancer_bypass,
                        bypass_threshold=bypass_threshold,
                        bypass_scale=bypass_scale,
                        ss_version=ss_version,
                        bypass_version=bypass_version,
                        is_cnn=is_cnn, mel_fb=mel_fb)
                results[model_name][noise_type][str(snr)] = acc

            # Reset to default after evaluation
            if hasattr(model, 'set_calibration'):
                model.set_calibration(profile='default')

            clean_acc = results[model_name][noise_type].get('clean', 0)
            zero_acc = results[model_name][noise_type].get('0', 0)
            m15_acc = results[model_name][noise_type].get('-15', 0)
            profile_m15 = snr_to_profile(-15)
            cal_info = f"[profile: {profile_m15}]" if not is_cnn else "[no calibration]"
            print(f"    {noise_type:<10} | Clean: {clean_acc:.1f}% | "
                  f"0dB: {zero_acc:.1f}% | -15dB: {m15_acc:.1f}% "
                  f"{cal_info}", flush=True)

    # Print summary tables
    for noise_type in noise_types:
        numeric_snrs = [s for s in snr_levels if s != 'clean']
        print(f"\n  === {noise_type.upper()} Noise + CALIBRATION Summary ===")
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

    # Prepare mel filterbank for CNN baselines
    mel_fb = _create_mel_fb_tensor()

    reverb_only_results = {}
    for model_name, model in models_dict.items():
        model.eval()
        is_cnn = _is_cnn_model(model_name)
        reverb_only_results[model_name] = {}
        tag = "[CNN: mel]" if is_cnn else "[SSM: raw]"
        print(f"\n  Evaluating: {model_name} {tag}", flush=True)

        for rt60 in rt60_list:
            acc = evaluate_reverb(
                model, val_loader, device, rt60=rt60,
                use_enhancer=use_enhancer,
                enhancer_type=enhancer_type,
                gtcrn_model=gtcrn_model,
                is_cnn=is_cnn, mel_fb=mel_fb)
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
        is_cnn = _is_cnn_model(model_name)
        noise_reverb_results[model_name] = {}
        tag = "[CNN: mel]" if is_cnn else "[SSM: raw]"
        print(f"\n  Evaluating: {model_name} {tag}", flush=True)

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
                        gtcrn_model=gtcrn_model,
                        is_cnn=is_cnn, mel_fb=mel_fb)
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
    'NanoMamba-Matched-DualPCEN': create_nanomamba_matched_dualpcen,
    'NanoMamba-Tiny-TriPCEN': create_nanomamba_tiny_tripcen,
    'NanoMamba-Matched-TriPCEN': create_nanomamba_matched_tripcen,
    # v2 Enhanced Routing (TMI + SNR-Conditioned, 0 extra params)
    'NanoMamba-Tiny-DualPCEN-v2': create_nanomamba_tiny_dualpcen_v2,
    'NanoMamba-Matched-DualPCEN-v2': create_nanomamba_matched_dualpcen_v2,
    'NanoMamba-Tiny-TriPCEN-v2': create_nanomamba_tiny_tripcen_v2,
    'NanoMamba-Matched-TriPCEN-v2': create_nanomamba_matched_tripcen_v2,
    # v2 + SSM v2 (full stack: PCEN v2 routing + SA-SSM v2 dynamics, 0 extra params)
    'NanoMamba-Tiny-DualPCEN-v2-SSMv2': create_nanomamba_tiny_dualpcen_v2_ssmv2,
    'NanoMamba-Matched-DualPCEN-v2-SSMv2': create_nanomamba_matched_dualpcen_v2_ssmv2,
    'NanoMamba-Tiny-TriPCEN-v2-SSMv2': create_nanomamba_tiny_tripcen_v2_ssmv2,
    'NanoMamba-Matched-TriPCEN-v2-SSMv2': create_nanomamba_matched_tripcen_v2_ssmv2,
    # Complete model: v2 + SSMv2 + Integrated Spectral Enhancement (0 extra params)
    'NanoMamba-Tiny-SE': create_nanomamba_tiny_dualpcen_v2_ssmv2_se,
    'NanoMamba-Matched-SE': create_nanomamba_matched_dualpcen_v2_ssmv2_se,
    # Learnable Spectral Enhancement: differentiable Wiener gain (516 extra params)
    'NanoMamba-Tiny-LSE': create_nanomamba_tiny_dualpcen_v2_ssmv2_lse,
    'NanoMamba-Matched-LSE': create_nanomamba_matched_dualpcen_v2_ssmv2_lse,
    # SM-SSM: Selectivity-Modulated SA-SSM (CNN↔Mamba blend based on SNR, +26/+22 params)
    'NanoMamba-Matched-SM': create_nanomamba_matched_dualpcen_v2_smssm,
    'NanoMamba-Tiny-SM': create_nanomamba_tiny_dualpcen_v2_smssm,
    # FI add-on: SpectralMambaBlock on existing NanoMamba (freq-axis SSM scanning)
    'NanoMamba-Matched-FI': create_nanomamba_matched_dualpcen_v2_ssmv2_fi,
    'NanoMamba-Tiny-FI': create_nanomamba_tiny_dualpcen_v2_ssmv2_fi,
    # FI-Mamba: Frequency-Interleaved Mamba (unified spectral+temporal SSM)
    'FI-Mamba': create_fimamba_matched,
    'FI-Mamba-Small': create_fimamba_small,
    # v3: Pure representation efficiency (beat BC-ResNet-1)
    'NanoMamba-v3-Matched': create_nanomamba_v3_matched,
    'NanoMamba-v3-Deep': create_nanomamba_v3_deep,
    # NanoApple: Frequency-Aware SSM (CNN freq processing + SSM streaming)
    'NanoApple': create_nanoapple,
    'NanoApple-v2': create_nanoapple_v2,
    # NanoApple-v3: DualPCEN v2 + BC-ResNet backbone + SS Training
    'NanoApple-v3': create_nanoapple_v3,
    # SAGN: SNR-Adaptive Gated Network (CNN backbone + learned spectral gate)
    'SAGN': create_sagn,
    # NC-SSM: Noise-Conditioned SM-SSM (per-sub-band selectivity + LSG)
    'NanoMamba-NC-Matched': create_nanomamba_nc_matched,
    'NanoMamba-NC-Large': create_nanomamba_nc_large,
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

def _create_mel_fb_tensor(sr=16000, n_fft=512, n_mels=40):
    """Create mel filterbank as a torch tensor (for CNN baseline mel computation)."""
    n_freq = n_fft // 2 + 1
    mel_low = 0
    mel_high = 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            if j < n_freq:
                fb[i, j] = (j - bin_points[i]) / max(
                    bin_points[i + 1] - bin_points[i], 1)
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            if j < n_freq:
                fb[i, j] = (bin_points[i + 2] - j) / max(
                    bin_points[i + 2] - bin_points[i + 1], 1)
    return torch.from_numpy(fb)


def _is_cnn_model(model_name):
    """Check if model is a CNN baseline (expects mel input, not raw audio)."""
    cnn_names = ['DS-CNN-S', 'BC-ResNet-1', 'DSCNN', 'BCResNet', 'TC-ResNet']
    return any(cn in model_name for cn in cnn_names)


def _adjust_bn_momentum(model, momentum):
    """Adjust BatchNorm momentum for CNN models during noise-aug training.

    When training with noisy data, CNN's BatchNorm running statistics
    shift to reflect the noisy distribution. Higher momentum makes BN
    adapt faster but also makes it more sensitive to distribution shifts.

    During clean warm-up: standard momentum (0.1)
    During noise-aug: reduced momentum (0.05) to stabilize BN stats
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.momentum = momentum


def train_model(model, model_name, train_dataset, val_dataset,
                checkpoint_dir, device, epochs=30, batch_size=128, lr=3e-3,
                noise_aug=False, noise_ratio=0.5, noise_curriculum_v2=False,
                ss_train=False,
                use_ema=False, ema_decay=0.999):
    """Full training loop with Per-Sample Multi-Condition Noise Augmentation.

    [NOVEL] Per-Sample Multi-Condition Training reveals structural differences:

    === NanoMamba (DualPCEN + SA-SSM) ===
      Structure: Two PCEN experts with SF/Tilt routing + adaptive SSM
      - Expert1 (nonstat, delta=2.0): handles clean speech (low SF, steep tilt)
      - Expert2 (stationary, delta=0.01): handles noisy input (high SF, flat tilt)
      - Routing is DYNAMIC per-input: computed from SF and Spectral Tilt at runtime
      - Per-sample noise → each sample activates different routing path
        → DualPCEN learns RICH routing manifold, not just binary clean/noise
        → Result: Clean preserved via Expert1, Noise improved via Expert2, NO TRADE-OFF

    === BC-ResNet-1 (Sub-Spectral Norm) ===
      Structure: Sub-band BatchNorm → frequency-aware normalization
      - Training: BN computes sub-band statistics (mean/var per sub-band)
      - Inference: running_mean/running_var are FROZEN → fixed normalization
      - Per-sample noise → BN running stats = AVERAGE of clean + noisy
        → Normalization is an average compromise, not per-input adaptive
        → Moderate noise improvement, moderate clean degradation

    === DS-CNN-S (Standard BatchNorm) ===
      Structure: Global BatchNorm → single mean/var across all frequencies
      - Noise shifts global mean/var → affects ALL frequency bands equally
      - Fixed kernels = same convolution weights for clean and noisy
      - Result: Strongest trade-off (clean degrades most, noise gains limited)

    Args:
        noise_aug: if True, per-sample multi-condition noise augmentation
        noise_ratio: target fraction of batch to corrupt (ramps up from 0)
    """
    is_cnn = _is_cnn_model(model_name)
    aug_str = " + Per-Sample Noise-Aug" if noise_aug else ""

    print(f"\n{'='*70}")
    print(f"  Training: {model_name}{aug_str}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    if noise_aug:
        print(f"  Noise-Aug: PER-SAMPLE mixing (5 types x continuous SNR)")
        print(f"    Epoch  0- 2: WARM-UP (clean-only, stabilize Expert1 / BN)")
        print(f"    Epoch  3- 9: GENTLE  (ratio ramps 0->{noise_ratio:.1f}, SNR 5~15dB)")
        print(f"    Epoch 10-19: MODERATE (ratio={noise_ratio:.1f}, SNR 0~10dB)")
        print(f"    Epoch 20+  : HARD     (ratio={noise_ratio:.1f}, SNR -5~10dB)")
    if is_cnn:
        print(f"  Mode: CNN baseline (mel input)")
        if 'BC-ResNet' in model_name:
            print(f"    Sub-Spectral Norm: sub-band BN (fixed stats at inference)")
        else:
            print(f"    Standard BatchNorm: global BN (fixed stats at inference)")
    else:
        print(f"  Mode: NanoMamba (raw audio → DualPCEN dynamic routing → SA-SSM)")
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

    # Mel filterbank for CNN noise-aug mel computation
    mel_fb = _create_mel_fb_tensor() if is_cnn else None

    # Collect some audio samples for babble noise generation
    dataset_audios = None
    if noise_aug and hasattr(train_dataset, '_cache_audio'):
        dataset_audios = train_dataset._cache_audio[:500]

    # EMA: Exponential Moving Average for better generalization
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"    EMA enabled (decay={ema_decay})")

    best_acc = 0
    best_epoch = 0
    model_dir = Path(checkpoint_dir) / model_name.replace(' ', '_')
    model_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        t0 = time.time()

        # Adjust CNN BatchNorm momentum during noise-aug phases
        if is_cnn and noise_aug:
            if epoch < 3:
                _adjust_bn_momentum(model, 0.1)   # Standard during warm-up
            else:
                _adjust_bn_momentum(model, 0.05)  # Slower during noise-aug

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch, model_name=model_name,
            noise_aug=noise_aug, noise_ratio=noise_ratio,
            is_cnn=is_cnn, dataset_audios=dataset_audios,
            mel_fb=mel_fb, total_epochs=epochs,
            noise_curriculum_v2=noise_curriculum_v2,
            ss_train=ss_train,
            ema=ema)

        # Evaluate on CLEAN val set (always clean, fair comparison)
        # Use EMA weights for evaluation if enabled
        if ema is not None:
            ema.apply_shadow(model)

        if is_cnn:
            val_acc = _evaluate_cnn(model, val_loader, device)
        else:
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
            # Save EMA weights if enabled (already applied above)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'model_name': model_name,
                'noise_aug': noise_aug,
                'use_ema': use_ema,
            }, model_dir / 'best.pt')

        # Restore original weights for continued training
        if ema is not None:
            ema.restore(model)

    # Save final (use EMA weights if enabled)
    if ema is not None:
        ema.apply_shadow(model)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
        'val_acc': val_acc,
        'model_name': model_name,
        'noise_aug': noise_aug,
        'use_ema': use_ema,
    }, model_dir / 'final.pt')
    if ema is not None:
        ema.restore(model)

    with open(model_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best: {best_acc:.2f}% @ epoch {best_epoch}")
    print(f"  Saved to {model_dir}")

    # Load best checkpoint
    ckpt = torch.load(model_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    return best_acc, model


@torch.no_grad()
def _evaluate_cnn(model, val_loader, device):
    """Evaluate CNN baseline model (uses mel input)."""
    model.eval()
    correct = 0
    total = 0
    for mel, labels, audio in val_loader:
        labels = labels.to(device)
        mel = mel.to(device)
        logits = model(mel)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total


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
    parser.add_argument('--enhancer_bypass', action='store_true',
                        help='SNR-adaptive bypass: high SNR → skip enhancer (preserve Clean)')
    parser.add_argument('--bypass_threshold', type=float, default=-2.0,
                        help='SNR threshold (dB) for bypass gate center (default: -2.0)')
    parser.add_argument('--bypass_scale', type=float, default=3.0,
                        help='Bypass gate sigmoid steepness (default: 3.0)')
    parser.add_argument('--sf_range', type=float, default=2.0,
                        help='Spectral flatness adaptive range for bypass (default: 2.0)')
    parser.add_argument('--ss_version', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='SS version: v1 (fixed oversubtract) or v2 (adaptive)')
    parser.add_argument('--bypass_version', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='Bypass version: v1 (fixed threshold) or v2 (noise-aware)')
    parser.add_argument('--use_reverb', action='store_true',
                        help='Run reverberation evaluation (reverb-only + noise+reverb)')
    parser.add_argument('--rt60', type=str, default='0.2,0.4,0.6,0.8',
                        help='Comma-separated RT60 values in seconds')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run Runtime Parameter Calibration evaluation '
                             '(sets optimal profile per SNR level)')
    parser.add_argument('--noise_aug', action='store_true',
                        help='Multi-Condition Noise-Aug Training. '
                             'Mixes 50%% of each batch with random noise. '
                             'NanoMamba: unlocks DualPCEN routing + SNR paths. '
                             'CNN: fixed kernel trade-off (clean vs noise).')
    parser.add_argument('--noise_ratio', type=float, default=0.5,
                        help='Fraction of each batch to corrupt with noise '
                             '(default: 0.5 = 50%%)')
    parser.add_argument('--noise_curriculum_v2', action='store_true',
                        help='Use v2 noise curriculum: type-staged introduction '
                             '(stationary first, then non-stationary) + '
                             'SNR annealing (Gaussian centered at difficulty frontier)')
    parser.add_argument('--ss_train', action='store_true',
                        help='Apply Spectral Subtraction (SS v2) to noisy audio '
                             'DURING training. Model learns to exploit SS-enhanced '
                             'input. NanoMamba gains +20%%p broadband robustness '
                             'while BC-ResNet-1 loses -7%%p (asymmetric advantage).')
    parser.add_argument('--calibrate_continuous', action='store_true',
                        help='Use continuous calibration interpolation instead of '
                             'discrete 4-profile system. Smooth SNR-to-parameter curves.')
    # Training enhancements (universal, works with all models)
    parser.add_argument('--spec_augment', action='store_true',
                        help='Enable SpecAugment (freq+time masking). '
                             'Improves generalization, 0 extra params.')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use Exponential Moving Average of model weights. '
                             'Smooths training, better generalization.')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate (default: 0.999)')
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
        # Enable SpecAugment if requested (works with any NanoMamba model)
        if args.spec_augment and hasattr(model, 'use_spec_augment'):
            model.use_spec_augment = True
        params = sum(p.numel() for p in model.parameters())
        fp32_kb = params * 4 / 1024
        aug_info = " +SpecAug" if args.spec_augment and hasattr(model, 'use_spec_augment') else ""
        ema_info = " +EMA" if args.use_ema else ""
        print(f"  {name}: {params:,} params ({fp32_kb:.1f} KB FP32){aug_info}{ema_info}")
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
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                noise_aug=args.noise_aug, noise_ratio=args.noise_ratio,
                noise_curriculum_v2=args.noise_curriculum_v2,
                ss_train=args.ss_train,
                use_ema=args.use_ema, ema_decay=args.ema_decay)

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
        if _is_cnn_model(model_name):
            test_acc = _evaluate_cnn(model, test_loader, device)
        else:
            test_acc = evaluate(model, test_loader, device)
        params = sum(p.numel() for p in model.parameters())
        cnn_str = " [CNN]" if _is_cnn_model(model_name) else ""
        print(f"  {model_name:<25} | Test: {test_acc:.2f}% | Params: {params:,}{cnn_str}")
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
        gtcrn_model=gtcrn_model,
        enhancer_bypass=args.enhancer_bypass,
        bypass_threshold=args.bypass_threshold,
        bypass_scale=args.bypass_scale,
        ss_version=args.ss_version,
        bypass_version=args.bypass_version)

    # ===== 5a. Runtime Calibration evaluation (if requested) =====
    calibrated_results = {}
    if args.calibrate:
        calibrated_results = run_calibrated_evaluation(
            trained_models, val_loader, device,
            noise_types=noise_types, snr_levels=snr_levels,
            use_enhancer=args.use_enhancer,
            enhancer_type=args.enhancer_type,
            gtcrn_model=gtcrn_model,
            enhancer_bypass=args.enhancer_bypass,
            bypass_threshold=args.bypass_threshold,
            bypass_scale=args.bypass_scale,
            ss_version=args.ss_version,
            bypass_version=args.bypass_version,
            use_continuous_calibration=getattr(args, 'calibrate_continuous', False))

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

    # ===== 5c. DualPCEN Routing Analysis =====
    routing_results = {}
    for model_name, model in trained_models.items():
        if not _is_cnn_model(model_name):
            routing = analyze_dualpcen_routing(
                model, val_loader, device,
                dataset_audios=train_dataset._cache_audio[:500]
                if hasattr(train_dataset, '_cache_audio') else None)
            if routing:
                routing_results[model_name] = routing

    # ===== 6. Save results =====
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'lr': args.lr,
        'seed': args.seed,
        'noise_aug': args.noise_aug,
        'noise_aug_config': {
            'method': 'per-sample',
            'noise_ratio': args.noise_ratio,
            'warm_up_epochs': 3,
            'curriculum': {
                'phase0_clean': 'epoch 0-2',
                'phase1_gentle': 'epoch 3-9 (SNR 5~15dB, ratio ramp)',
                'phase2_moderate': 'epoch 10-19 (SNR 0~10dB)',
                'phase3_hard': 'epoch 20+ (SNR -5~10dB)',
            },
            'noise_types': ['factory', 'white', 'babble', 'street', 'pink'],
            'cnn_bn_momentum': '0.1 (warm-up) → 0.05 (noise-aug)',
        } if args.noise_aug else None,
        'models': {}
    }

    for model_name, model in trained_models.items():
        params = sum(p.numel() for p in model.parameters())
        is_cnn = _is_cnn_model(model_name)
        model_result = {
            'params': params,
            'size_fp32_kb': round(params * 4 / 1024, 1),
            'size_int8_kb': round(params / 1024, 1),
            'model_type': 'CNN' if is_cnn else 'SSM (NanoMamba)',
            'noise_adaptation': (
                'Fixed (BatchNorm frozen stats at inference)' if 'DS-CNN' in model_name
                else 'Partial (Sub-Spectral Norm, frozen sub-band stats)' if 'BC-ResNet' in model_name
                else 'Dynamic (DualPCEN SF/Tilt routing, per-input adaptive)'),
            'test_acc': test_results.get(model_name, 0),
            'noise_robustness': noise_results.get(model_name, {}),
        }
        if calibrated_results and model_name in calibrated_results:
            model_result['calibrated_robustness'] = calibrated_results.get(model_name, {})
        if model_name in routing_results:
            model_result['dualpcen_routing'] = routing_results[model_name]
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
