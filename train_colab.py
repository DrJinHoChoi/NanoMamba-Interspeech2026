#!/usr/bin/env python3
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
                   snr_db=0, dataset_audios=None):
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
        logits = model(noisy_audio)

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
                         dataset_audios=None):
    if noise_types is None:
        noise_types = ['factory', 'white', 'babble', 'street', 'pink']
    if snr_levels is None:
        snr_levels = [-15, -10, -5, 0, 5, 10, 15, 'clean']

    print("\n" + "=" * 80)
    print("  NOISE ROBUSTNESS EVALUATION")
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
                        dataset_audios=dataset_audios)
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
# Model Registry (NanoMamba only — no external dependencies)
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
    model.load_state_dict(ckpt['model_state_dict'])

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
                model.load_state_dict(ckpt['model_state_dict'])
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

    noise_results = run_noise_evaluation(
        trained_models, val_loader, device,
        noise_types=noise_types, snr_levels=snr_levels)

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
        final_results['models'][model_name] = {
            'params': params,
            'size_fp32_kb': round(params * 4 / 1024, 1),
            'size_int8_kb': round(params / 1024, 1),
            'test_acc': test_results.get(model_name, 0),
            'noise_robustness': noise_results.get(model_name, {}),
        }

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
            if 'alpha' in pname:
                print(f"    {pname}: {p.item():.4f}")
        # Print fixed structural buffers (delta_floor, epsilon)
        for bname, buf in model.named_buffers():
            if any(k in bname for k in ['delta_floor', 'epsilon']):
                print(f"    {bname}: {buf.item():.4f} (fixed, non-learnable)")

    return final_results


if __name__ == '__main__':
    main()
