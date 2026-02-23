#!/usr/bin/env python3
# coding=utf-8
"""
SMARTEAR KWS - Complete Training & Evaluation Pipeline
=====================================================

Trains all SOTA KWS models on Google Speech Commands V2 (12-class)
and evaluates under factory noise conditions for paper results.

Models:
  1. BC-ResNet-1/2/3/6/8 (Qualcomm, 2021)
  2. DS-CNN-S baseline
  3. MatchboxNet baseline
  4. Keyword Mamba Small (KWM, 2025)
  5. Joint AEC+KWS (Proposed)

Target: Voice remote control in adverse factory environments (SNR -10~+5dB)
Platform: Jetson Nano (Maxwell GPU, TensorRT INT8)

Usage:
  python train_all_models.py --data_dir ./data --epochs 30
  python train_all_models.py --eval_only --checkpoint_dir ./checkpoints
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
# Google Speech Commands V2 Dataset (12-class)
# ============================================================================

GSC_LABELS_12 = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'stop', 'go', 'silence', 'unknown'
]

# 10 core keywords + silence + unknown (all other words mapped to 'unknown')
CORE_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right',
              'on', 'off', 'stop', 'go']

class SpeechCommandsDataset(Dataset):
    """Google Speech Commands V2 dataset wrapper with mel spectrogram.

    Downloads and processes GSC V2 automatically.
    12-class: 10 keywords + silence + unknown.
    """

    def __init__(self, root, subset='training', n_mels=40, sr=16000,
                 clip_duration_ms=1000, augment=False):
        super().__init__()
        self.root = Path(root)
        self.subset = subset
        self.n_mels = n_mels
        self.sr = sr
        self.target_length = int(sr * clip_duration_ms / 1000)
        self.augment = augment

        # Try torchaudio first for download
        self._prepare_data()

        # Build file list
        self.samples = []
        self.labels = GSC_LABELS_12
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self._scan_files()

        # Mel spectrogram parameters
        self.n_fft = 512
        self.hop_length = 160  # 10ms at 16kHz
        self.win_length = 400  # 25ms at 16kHz

        # Create mel filterbank
        self.mel_fb = self._create_mel_fb()

        print(f"  [{subset}] {len(self.samples)} samples, "
              f"{len(self.labels)} classes")

    def _prepare_data(self):
        """Download GSC V2 if not present."""
        data_dir = self.root / 'SpeechCommands' / 'speech_commands_v0.02'
        if data_dir.exists() and len(list(data_dir.iterdir())) > 5:
            return  # Already downloaded

        print(f"  Downloading Google Speech Commands V2 to {self.root}...")
        try:
            from torchaudio.datasets import SPEECHCOMMANDS

            class SubsetSC(SPEECHCOMMANDS):
                def __init__(self, root, subset=None):
                    super().__init__(root, download=True)
                    if subset == "validation":
                        self._walker = [s for s in self._walker
                                       if self._is_val(s)]
                    elif subset == "testing":
                        self._walker = [s for s in self._walker
                                       if self._is_test(s)]
                    elif subset == "training":
                        self._walker = [s for s in self._walker
                                       if not self._is_val(s)
                                       and not self._is_test(s)]

                def _is_val(self, path):
                    return path in self._val_list

                def _is_test(self, path):
                    return path in self._test_list

                @property
                def _val_list(self):
                    if not hasattr(self, '_val_list_cache'):
                        vf = self._path / 'validation_list.txt'
                        self._val_list_cache = set()
                        if vf.exists():
                            with open(vf) as f:
                                for line in f:
                                    self._val_list_cache.add(
                                        os.path.join(self._path, line.strip()))
                    return self._val_list_cache

                @property
                def _test_list(self):
                    if not hasattr(self, '_test_list_cache'):
                        tf = self._path / 'testing_list.txt'
                        self._test_list_cache = set()
                        if tf.exists():
                            with open(tf) as f:
                                for line in f:
                                    self._test_list_cache.add(
                                        os.path.join(self._path, line.strip()))
                    return self._test_list_cache

            _ = SubsetSC(str(self.root), 'training')
            print("  Download complete!")
        except Exception as e:
            print(f"  torchaudio download failed: {e}")
            print("  Trying manual download...")
            self._manual_download()

    def _manual_download(self):
        """Manual download fallback."""
        import urllib.request
        import tarfile

        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        tar_path = self.root / "speech_commands_v0.02.tar.gz"
        extract_dir = self.root / 'SpeechCommands' / 'speech_commands_v0.02'

        if not tar_path.exists():
            print(f"  Downloading from {url}...")
            os.makedirs(self.root, exist_ok=True)
            urllib.request.urlretrieve(url, str(tar_path))

        if not extract_dir.exists():
            print(f"  Extracting...")
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(str(tar_path), 'r:gz') as tar:
                tar.extractall(str(extract_dir))

        print("  Manual download complete!")

    def _scan_files(self):
        """Scan directory for audio files and build sample list."""
        data_dir = self.root / 'SpeechCommands' / 'speech_commands_v0.02'

        # Read validation/testing lists for split
        val_list = set()
        test_list = set()

        val_file = data_dir / 'validation_list.txt'
        test_file = data_dir / 'testing_list.txt'

        if val_file.exists():
            with open(val_file) as f:
                val_list = {line.strip() for line in f}
        if test_file.exists():
            with open(test_file) as f:
                test_list = {line.strip() for line in f}

        # Scan all keyword directories
        for keyword_dir in sorted(data_dir.iterdir()):
            if not keyword_dir.is_dir():
                continue
            keyword = keyword_dir.name

            if keyword.startswith('_'):
                if keyword == '_background_noise_':
                    # Generate silence samples from background noise
                    if self.subset == 'training':
                        self._add_silence_samples(keyword_dir)
                continue

            # Map to label
            if keyword in CORE_WORDS:
                label = keyword
            else:
                label = 'unknown'

            label_idx = self.label_to_idx[label]

            for wav_file in keyword_dir.glob('*.wav'):
                rel_path = f"{keyword}/{wav_file.name}"

                # Determine split
                if rel_path in val_list:
                    if self.subset != 'validation':
                        continue
                elif rel_path in test_list:
                    if self.subset != 'testing':
                        continue
                else:
                    if self.subset != 'training':
                        continue

                self.samples.append((str(wav_file), label_idx))

        # Add silence samples for validation/testing too
        if self.subset in ('validation', 'testing'):
            self._add_silence_from_noise(data_dir / '_background_noise_')

    def _add_silence_samples(self, noise_dir, n_samples=2000):
        """Generate silence samples from background noise."""
        label_idx = self.label_to_idx['silence']
        noise_files = list(noise_dir.glob('*.wav'))
        if noise_files:
            for i in range(min(n_samples, 2000)):
                nf = noise_files[i % len(noise_files)]
                self.samples.append((str(nf) + f'#silence_{i}', label_idx))

    def _add_silence_from_noise(self, noise_dir, n_samples=500):
        """Add silence samples for val/test."""
        label_idx = self.label_to_idx['silence']
        noise_files = list(noise_dir.glob('*.wav')) if noise_dir.exists() else []
        if noise_files:
            for i in range(min(n_samples, 500)):
                nf = noise_files[i % len(noise_files)]
                self.samples.append((str(nf) + f'#silence_{i}', label_idx))

    def _create_mel_fb(self):
        """Create mel filterbank matrix."""
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
        """Load audio file and pad/trim to target length."""
        actual_path = path.split('#')[0]  # Remove silence marker

        try:
            import torchaudio
            waveform, sr = torchaudio.load(actual_path)
            if sr != self.sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            audio = waveform[0]  # mono
        except Exception:
            try:
                import scipy.io.wavfile as wav
                sr, data = wav.read(actual_path)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                if len(data.shape) > 1:
                    data = data[:, 0]
                if sr != self.sr:
                    # Simple resampling
                    ratio = self.sr / sr
                    new_len = int(len(data) * ratio)
                    indices = np.linspace(0, len(data) - 1, new_len)
                    data = np.interp(indices, np.arange(len(data)), data)
                audio = torch.from_numpy(data.astype(np.float32))
            except Exception:
                audio = torch.zeros(self.target_length)

        # For silence, take random segment and scale down
        if '#silence' in path:
            if len(audio) > self.target_length:
                start = np.random.randint(0, len(audio) - self.target_length)
                audio = audio[start:start + self.target_length]
            audio = audio * 0.1  # Scale down for silence

        # Pad or trim
        if len(audio) < self.target_length:
            audio = F.pad(audio, (0, self.target_length - len(audio)))
        elif len(audio) > self.target_length:
            audio = audio[:self.target_length]

        return audio

    def _compute_mel(self, audio):
        """Compute log-mel spectrogram."""
        window = torch.hann_window(self.win_length)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          self.win_length, window=window,
                          return_complex=True)
        mag = spec.abs()  # (F, T)

        # Apply mel filterbank
        mel = torch.matmul(self.mel_fb, mag)  # (n_mels, T)

        # Log compression
        mel = torch.log(mel + 1e-8)

        return mel

    def _augment(self, audio):
        """Apply data augmentation."""
        # Time shift (random shift up to 100ms)
        shift = np.random.randint(-1600, 1600)
        if shift > 0:
            audio = F.pad(audio[shift:], (0, shift))
        elif shift < 0:
            audio = F.pad(audio[:shift], (-shift, 0))

        # Volume perturbation
        vol = np.random.uniform(0.8, 1.2)
        audio = audio * vol

        # Random noise (small)
        if np.random.random() < 0.3:
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise

        return audio

    def cache_all(self):
        """Pre-load all audio into RAM for fast training."""
        import sys
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
        # Stack into tensors for fastest access
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

        mel = self._compute_mel(audio)  # (n_mels, T)
        return mel, label, audio


# ============================================================================
# Baseline Models
# ============================================================================

class DSCNN_S(nn.Module):
    """DS-CNN Small baseline (ARM, 2017).

    Depthwise Separable CNN for keyword spotting.
    ~26K params, 94.4% on GSC V2 12-class.
    """
    def __init__(self, n_mels=40, n_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (10, 4), stride=(2, 2), padding=(5, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # DS conv 1
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # DS conv 2
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # DS conv 3
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # DS conv 4
            nn.Conv2d(64, 64, (3, 3), padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, mel):
        x = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class MatchboxNet(nn.Module):
    """MatchboxNet baseline (NVIDIA, 2020).

    1D time-channel separable convolutions.
    ~93K params, 97.48% on GSC V1 35-class.
    """
    def __init__(self, n_mels=40, n_classes=12, B_param=3, R=2, C=64):
        super().__init__()

        # Prologue
        self.prologue = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Main blocks (B blocks, each with R sub-blocks)
        blocks = []
        in_c = 128
        kernels = [13, 15, 17]
        for b in range(B_param):
            k = kernels[b % len(kernels)]
            for r in range(R):
                blocks.extend([
                    nn.Conv1d(in_c if r == 0 else C, C, kernel_size=k,
                              padding=k // 2, groups=C if r > 0 else 1),
                    nn.BatchNorm1d(C),
                    nn.ReLU(),
                    nn.Conv1d(C, C, 1),
                    nn.BatchNorm1d(C),
                    nn.ReLU(),
                ])
                in_c = C

        self.blocks = nn.Sequential(*blocks)

        # Epilogue
        self.epilogue = nn.Sequential(
            nn.Conv1d(C, 128, kernel_size=29, padding=14, groups=C),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, mel):
        # mel: (B, n_mels, T)
        x = self.prologue(mel)
        x = self.blocks(x)
        x = self.epilogue(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, scheduler, device,
                    label_smoothing=0.1, epoch=0, model_name="",
                    teacher_model=None, kd_alpha=0.5, kd_temperature=4.0):
    """Train for one epoch, optionally with Knowledge Distillation.

    Args:
        teacher_model: if provided, use KD loss with this teacher.
        kd_alpha: weight for hard label loss (1-kd_alpha for KD loss).
        kd_temperature: softmax temperature for KD.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (mel, labels, audio) in enumerate(train_loader):
        mel = mel.to(device)
        labels = labels.to(device)

        # Forward (student)
        if hasattr(model, 'feature_ext') or hasattr(model, 'spectral_gate') or hasattr(model, 'snr_estimator'):
            audio = audio.to(device)
            logits = model(audio)
        else:
            logits = model(mel)

        # Loss computation
        if teacher_model is not None:
            # Knowledge Distillation loss
            with torch.no_grad():
                if hasattr(teacher_model, 'feature_ext') or hasattr(teacher_model, 'spectral_gate') or hasattr(teacher_model, 'snr_estimator'):
                    audio = audio.to(device) if not audio.is_cuda else audio
                    teacher_logits = teacher_model(audio)
                else:
                    teacher_logits = teacher_model(mel)

            # Soft targets (KD loss)
            T = kd_temperature
            kd_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean') * (T * T)

            # Combined loss
            hard_loss = criterion(logits, labels)
            loss = kd_alpha * hard_loss + (1.0 - kd_alpha) * kd_loss
        else:
            loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Stats
        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            acc = 100. * correct / total
            kd_str = " [KD]" if teacher_model else ""
            print(f"    [{model_name}{kd_str}] Batch {batch_idx+1}/{len(train_loader)} "
                  f"Loss: {total_loss/total:.4f} Acc: {acc:.1f}%",
                  flush=True)

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for mel, labels, audio in val_loader:
        mel = mel.to(device)
        labels = labels.to(device)

        if hasattr(model, 'feature_ext') or hasattr(model, 'spectral_gate') or hasattr(model, 'snr_estimator'):
            audio = audio.to(device)
            logits = model(audio)
        else:
            logits = model(mel)

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100. * correct / total
    return acc, np.array(all_preds), np.array(all_labels)


# ============================================================================
# Noise Generation (Audio-Domain, Physically Accurate)
# ============================================================================

def generate_noise_signal(noise_type, length, sr=16000, dataset_audios=None):
    """Generate noise signal of specified type in audio domain.

    Args:
        noise_type: 'factory', 'white', 'babble', 'street', 'pink'
        length: number of audio samples
        sr: sample rate
        dataset_audios: list of audio tensors for babble noise
    Returns:
        noise: tensor of shape (length,)
    """
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
    """Realistic factory noise: hum harmonics + conveyor + impacts + pink."""
    t = torch.arange(length, dtype=torch.float32) / sr

    # Machine hum (50Hz fundamental + harmonics with decay)
    hum = torch.zeros(length)
    for h in [50, 100, 150, 200, 250]:
        amp = 0.3 / (h / 50)
        phase = torch.rand(1).item() * 2 * math.pi
        hum += amp * torch.sin(2 * math.pi * h * t + phase)

    # Conveyor belt rumble (bandpass 200-800Hz filtered noise)
    rumble_np = np.random.randn(length).astype(np.float32) * 0.2
    fft = np.fft.rfft(rumble_np)
    freqs = np.fft.rfftfreq(length, 1 / sr)
    mask = ((freqs >= 200) & (freqs <= 800)).astype(np.float32)
    mask = np.convolve(mask, np.ones(20) / 20, mode='same')
    rumble = torch.from_numpy(
        np.fft.irfft(fft * mask, n=length).astype(np.float32))

    # Impact noise (random impulses)
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

    # Pink noise (1/f)
    pink = _generate_pink_noise(length, sr) * 0.15

    # Combine and normalize
    noise = hum + rumble + impacts + pink
    noise = noise / (noise.abs().max() + 1e-8) * 0.7
    return noise


def _generate_babble_noise(length, sr=16000, dataset_audios=None):
    """Multi-talker babble from summing random speech signals."""
    n_talkers = np.random.randint(5, 9)
    babble = torch.zeros(length)

    if dataset_audios is not None and len(dataset_audios) > 0:
        # Use actual dataset audio samples
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
        # Synthetic speech-like signals
        for _ in range(n_talkers):
            t = torch.arange(length, dtype=torch.float32) / sr
            f0 = np.random.uniform(100, 300)
            sig = 0.3 * torch.sin(2 * math.pi * f0 * t)
            sig += 0.1 * torch.sin(2 * math.pi * f0 * 2 * t)
            for fc in [730, 1090, 2440]:
                sig += 0.15 * torch.sin(2 * math.pi * fc * t)
            # Random onset/offset envelope
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
    """Street/traffic noise: low-freq rumble + horns + broadband."""
    t = torch.arange(length, dtype=torch.float32) / sr

    # Traffic rumble (20-200Hz)
    rumble_np = np.random.randn(length).astype(np.float32) * 0.3
    fft = np.fft.rfft(rumble_np)
    freqs = np.fft.rfftfreq(length, 1 / sr)
    mask = ((freqs >= 20) & (freqs <= 200)).astype(np.float32)
    mask = np.convolve(mask, np.ones(10) / 10, mode='same')
    rumble = torch.from_numpy(
        np.fft.irfft(fft * mask, n=length).astype(np.float32))

    # Horn impulses (2-3 random honks)
    horns = torch.zeros(length)
    n_horns = np.random.randint(1, 4)
    for _ in range(n_horns):
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

    # Broadband road noise
    road = torch.randn(length) * 0.15

    # Engine harmonics
    engine_freq = np.random.uniform(80, 150)
    engine = 0.2 * torch.sin(2 * math.pi * engine_freq * t)
    engine += 0.1 * torch.sin(2 * math.pi * engine_freq * 2 * t)

    noise = rumble + horns + road + engine
    noise = noise / (noise.abs().max() + 1e-8) * 0.7
    return noise


def _generate_pink_noise(length, sr=16000):
    """1/f pink noise via spectral shaping."""
    white = np.random.randn(length).astype(np.float32)
    fft_w = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, 1 / sr)
    freqs[0] = 1  # avoid div by zero
    pink = np.fft.irfft(fft_w / np.sqrt(freqs), n=length).astype(np.float32)
    pink_t = torch.from_numpy(pink)
    pink_t = pink_t / (pink_t.abs().max() + 1e-8) * 0.7
    return pink_t


def mix_audio_at_snr(clean_audio, noise, snr_db):
    """Mix clean audio with noise at target SNR using RMS-based calculation.

    Args:
        clean_audio: (B, T) or (T,) clean signal
        noise: (T,) noise signal (will be broadcast to batch)
        snr_db: target SNR in dB
    Returns:
        noisy_audio: same shape as clean_audio
    """
    # Compute RMS
    if clean_audio.dim() == 2:
        clean_rms = torch.sqrt(torch.mean(clean_audio ** 2, dim=-1, keepdim=True) + 1e-10)
    else:
        clean_rms = torch.sqrt(torch.mean(clean_audio ** 2) + 1e-10)

    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-10)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)

    return clean_audio + scaled_noise


def compute_mel_from_audio(audio, n_fft=512, hop_length=160,
                           win_length=400, n_mels=40, sr=16000):
    """Compute log-mel spectrogram from raw audio tensor.

    Args:
        audio: (B, T) raw audio
    Returns:
        mel: (B, n_mels, T_frames) log-mel spectrogram
    """
    # Create mel filterbank
    mel_fb = _create_mel_filterbank(n_mels, n_fft, sr)
    mel_fb = mel_fb.to(audio.device)
    window = torch.hann_window(win_length).to(audio.device)

    mels = []
    for i in range(audio.size(0)):
        spec = torch.stft(audio[i], n_fft, hop_length, win_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # (F, T)
        mel = torch.matmul(mel_fb, mag)  # (n_mels, T)
        mel = torch.log(mel + 1e-8)
        mels.append(mel)
    return torch.stack(mels)  # (B, n_mels, T)


def _create_mel_filterbank(n_mels=40, n_fft=512, sr=16000):
    """Create mel filterbank matrix."""
    n_freqs = n_fft // 2 + 1
    f_min, f_max = 0.0, sr / 2

    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fb[i, j] = (right - j) / (right - center)
    return torch.from_numpy(fb)


@torch.no_grad()
def evaluate_noisy(model, val_loader, device, noise_type='factory',
                   snr_db=0, dataset_audios=None):
    """Evaluate model under noise - ALL models get audio-domain noise.

    This ensures fair comparison: noise is always added in audio domain,
    then mel-domain models receive mel computed from noisy audio.

    Args:
        model: trained model
        val_loader: validation data loader
        device: torch device
        noise_type: 'factory', 'white', 'babble', 'street', 'pink'
        snr_db: target SNR in dB
        dataset_audios: list of audio tensors for babble noise generation
    Returns:
        accuracy: float (percentage)
    """
    model.eval()
    correct = 0
    total = 0

    for mel, labels, audio in val_loader:
        labels = labels.to(device)
        audio = audio.to(device)
        batch_size = audio.size(0)

        # Generate noise in audio domain (shared across all model types)
        noise = generate_noise_signal(
            noise_type, audio.size(-1), sr=16000,
            dataset_audios=dataset_audios).to(device)

        # Mix at target SNR (RMS-based)
        noisy_audio = mix_audio_at_snr(audio, noise, snr_db)

        # Route to model based on its input type
        if (hasattr(model, 'snr_estimator') or
            hasattr(model, 'feature_ext') or
                hasattr(model, 'spectral_gate')):
            # Audio-domain models (NanoMamba, NanoKWS, Joint pipelines)
            logits = model(noisy_audio)
        else:
            # Mel-domain models (BC-ResNet, DS-CNN, MatchboxNet, KWM)
            # Compute mel from noisy audio for fair comparison
            noisy_mel = compute_mel_from_audio(noisy_audio)
            logits = model(noisy_mel.to(device))

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


@torch.no_grad()
def evaluate_noisy_per_class(model, val_loader, device, noise_type='factory',
                              snr_db=0, n_classes=12, dataset_audios=None):
    """Evaluate with per-class accuracy and confusion matrix.

    Returns:
        accuracy: overall accuracy (%)
        per_class_acc: list of per-class accuracies (%)
        confusion_matrix: (n_classes, n_classes) numpy array
    """
    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for mel, labels, audio in val_loader:
        labels = labels.to(device)
        audio = audio.to(device)

        noise = generate_noise_signal(
            noise_type, audio.size(-1), sr=16000,
            dataset_audios=dataset_audios).to(device)
        noisy_audio = mix_audio_at_snr(audio, noise, snr_db)

        if (hasattr(model, 'snr_estimator') or
            hasattr(model, 'feature_ext') or
                hasattr(model, 'spectral_gate')):
            logits = model(noisy_audio)
        else:
            noisy_mel = compute_mel_from_audio(noisy_audio)
            logits = model(noisy_mel.to(device))

        _, predicted = logits.max(1)
        for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            confusion[t][p] += 1

    # Compute accuracies
    per_class_acc = []
    for i in range(n_classes):
        total_i = confusion[i].sum()
        if total_i > 0:
            per_class_acc.append(100.0 * confusion[i][i] / total_i)
        else:
            per_class_acc.append(0.0)

    overall_acc = 100.0 * np.trace(confusion) / confusion.sum()
    return overall_acc, per_class_acc, confusion


def train_model(model, model_name, train_dataset, val_dataset,
                checkpoint_dir, device, epochs=30, batch_size=128,
                lr=1e-3, weight_decay=1e-4,
                teacher_model=None, kd_alpha=0.5, kd_temperature=4.0):
    """Full training loop for a single model, optionally with KD.

    Args:
        teacher_model: if provided, use Knowledge Distillation.
        kd_alpha: weight for hard labels (0.5 = equal CE + KD).
        kd_temperature: softmax temperature for soft targets.
    """
    print(f"\n{'='*70}")
    kd_str = f" (KD from teacher, α={kd_alpha}, T={kd_temperature})" if teacher_model else ""
    print(f"  Training: {model_name}{kd_str}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"{'='*70}")

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01)

    best_acc = 0
    best_epoch = 0
    model_dir = Path(checkpoint_dir) / model_name.replace(' ', '_')
    model_dir.mkdir(parents=True, exist_ok=True)

    # epochs=0: load checkpoint and skip training (eval-only mode)
    if epochs == 0:
        ckpt_path = model_dir / 'best.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            best_acc = ckpt.get('val_acc', 0)
            best_epoch = ckpt.get('epoch', 0)
            print(f"  Loaded checkpoint: {ckpt_path} (val_acc={best_acc:.2f}%)")
        else:
            print(f"  WARNING: No checkpoint found at {ckpt_path}")
        return best_acc, model

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch, model_name=model_name,
            teacher_model=teacher_model, kd_alpha=kd_alpha,
            kd_temperature=kd_temperature)

        val_acc, _, _ = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"Time: {elapsed:.1f}s", flush=True)

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

    # Save history
    with open(model_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best: {best_acc:.2f}% @ epoch {best_epoch}")
    print(f"  Saved to {model_dir}")

    return best_acc, model


# ============================================================================
# Factory Noise Evaluation
# ============================================================================

@torch.no_grad()
def run_noise_evaluation(models_dict, val_loader, device,
                         noise_types=None, snr_levels=None,
                         dataset_audios=None, per_class=False):
    """Evaluate all trained models under multiple noise conditions.

    All models evaluated with audio-domain noise for fair comparison.

    Args:
        models_dict: {name: model} dict
        val_loader: validation DataLoader
        device: torch device
        noise_types: list of noise types (default: 5 types)
        snr_levels: list of SNR dB values (default: extended range)
        dataset_audios: audio tensors for babble noise
        per_class: if True, also collect per-class confusion matrices
    """
    if noise_types is None:
        noise_types = ['factory', 'white', 'babble', 'street', 'pink']
    if snr_levels is None:
        snr_levels = [-15, -10, -5, 0, 5, 10, 15, 'clean']

    print("\n" + "=" * 80)
    print("  NOISE ROBUSTNESS EVALUATION (Audio-Domain, Unified)")
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
                    acc, _, _ = evaluate(model, val_loader, device)
                else:
                    if per_class:
                        acc, per_cls, cm = evaluate_noisy_per_class(
                            model, val_loader, device, noise_type, snr,
                            dataset_audios=dataset_audios)
                        results[model_name][noise_type][str(snr)] = {
                            'accuracy': acc,
                            'per_class': per_cls,
                            'confusion_matrix': cm.tolist()
                        }
                        continue
                    else:
                        acc = evaluate_noisy(
                            model, val_loader, device, noise_type, snr,
                            dataset_audios=dataset_audios)
                results[model_name][noise_type][str(snr)] = acc
            # Progress indicator
            numeric_snrs = [s for s in snr_levels if s != 'clean']
            clean_acc = results[model_name][noise_type].get('clean', 0)
            if isinstance(clean_acc, dict):
                clean_acc = clean_acc['accuracy']
            zero_acc = results[model_name][noise_type].get('0', 0)
            if isinstance(zero_acc, dict):
                zero_acc = zero_acc['accuracy']
            print(f"    {noise_type:<10} | Clean: {clean_acc:.1f}% | "
                  f"0dB: {zero_acc:.1f}%", flush=True)

    # Print summary table (factory noise)
    print(f"\n  === Factory Noise Summary ===")
    numeric_snrs = [s for s in snr_levels if s != 'clean']
    print(f"  {'Model':<22} | {'Clean':>7} | " +
          " | ".join(f"{s:>6}dB" for s in numeric_snrs))
    print("  " + "-" * (28 + 9 * len(numeric_snrs)))

    for model_name, noise_data in results.items():
        if 'factory' not in noise_data:
            continue
        clean = noise_data['factory'].get('clean', 0)
        if isinstance(clean, dict):
            clean = clean['accuracy']
        snrs = []
        for s in numeric_snrs:
            val = noise_data['factory'].get(str(s), 0)
            if isinstance(val, dict):
                val = val['accuracy']
            snrs.append(val)
        print(f"  {model_name:<22} | {clean:>6.1f}% | " +
              " | ".join(f"{s:>6.1f}%" for s in snrs))

    return results


# ============================================================================
# Model Registry
# ============================================================================

def create_all_models(n_classes=12):
    """Create all models for comparison."""
    from paper_models import (BCResNet, KeywordMambaSmall,
                              JointAECKWSPipeline)
    from proposed_model import (create_nanokws_tiny, create_nanokws_small,
                                create_nanokws_base)
    from nanomamba import (create_nanomamba_tiny, create_nanomamba_small,
                           create_nanomamba_base, create_ablation_models,
                           create_nanomamba_tiny_ff, create_nanomamba_small_ff,
                           create_nanomamba_tiny_fc, create_nanomamba_small_fc,
                           create_nanomamba_tiny_moe, create_nanomamba_tiny_ws_moe,
                           create_nanomamba_tiny_tc, create_nanomamba_tiny_ws_tc,
                           create_nanomamba_tiny_ws, create_nanomamba_tiny_ws_ff)

    models = {
        # ===== Proposed NanoKWS (Joint AEC+KWS) =====
        'NanoKWS-Tiny': create_nanokws_tiny(n_classes),
        'NanoKWS-Small': create_nanokws_small(n_classes),
        'NanoKWS-Base': create_nanokws_base(n_classes),

        # ===== Proposed NanoMamba (SA-SSM, Joint AEC+KWS) =====
        'NanoMamba-Tiny': create_nanomamba_tiny(n_classes),
        'NanoMamba-Small': create_nanomamba_small(n_classes),
        'NanoMamba-Base': create_nanomamba_base(n_classes),

        # ===== NanoMamba + Frequency Filter variants =====
        'NanoMamba-Tiny-FF': create_nanomamba_tiny_ff(n_classes),
        'NanoMamba-Small-FF': create_nanomamba_small_ff(n_classes),

        # ===== NanoMamba + FreqConv variants (CNN freq transplant) =====
        'NanoMamba-Tiny-FC': create_nanomamba_tiny_fc(n_classes),
        'NanoMamba-Small-FC': create_nanomamba_small_fc(n_classes),

        # ===== NanoMamba + MoE-Freq variants (SNR-conditioned) =====
        'NanoMamba-Tiny-MoE': create_nanomamba_tiny_moe(n_classes),
        'NanoMamba-Tiny-WS-MoE': create_nanomamba_tiny_ws_moe(n_classes),

        # ===== NanoMamba + TinyConv2D variants (Hybrid CNN-SSM) =====
        'NanoMamba-Tiny-TC': create_nanomamba_tiny_tc(n_classes),
        'NanoMamba-Tiny-WS-TC': create_nanomamba_tiny_ws_tc(n_classes),

        # ===== NanoMamba + Weight Sharing variants =====
        'NanoMamba-Tiny-WS': create_nanomamba_tiny_ws(n_classes),
        'NanoMamba-Tiny-WS-FF': create_nanomamba_tiny_ws_ff(n_classes),

        # ===== Baselines =====
        'DS-CNN-S': DSCNN_S(n_classes=n_classes),
        'MatchboxNet': MatchboxNet(n_classes=n_classes),
        'KWM-Small': KeywordMambaSmall(n_classes=n_classes),

        # ===== BC-ResNet family (SOTA efficient) =====
        'BC-ResNet-1': BCResNet(scale=1, n_classes=n_classes),
        'BC-ResNet-2': BCResNet(scale=2, n_classes=n_classes),
        'BC-ResNet-3': BCResNet(scale=3, n_classes=n_classes),
        'BC-ResNet-6': BCResNet(scale=6, n_classes=n_classes),
        'BC-ResNet-8': BCResNet(scale=8, n_classes=n_classes),

        # ===== Joint pipelines =====
        'Joint-BCRes3': JointAECKWSPipeline(
            'bcresnet', scale=3, n_classes=n_classes),
        'Joint-BCRes6': JointAECKWSPipeline(
            'bcresnet', scale=6, n_classes=n_classes),
    }

    # ===== SA-SSM Ablation variants (NanoMamba-Tiny config) =====
    ablation_models = create_ablation_models(n_classes)
    models.update(ablation_models)

    print("\n  Model Summary:")
    print(f"  {'Name':<22} | {'Params':>10} | {'Size (KB)':>10}")
    print("  " + "-" * 50)
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        size_kb = sum(p.numel() * p.element_size()
                      for p in model.parameters()) / 1024
        print(f"  {name:<22} | {params:>10,} | {size_kb:>9.1f}")

    return models


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SmartEar KWS - Train & Evaluate All Models")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation (load checkpoints)')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model names or "all"')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: small subset for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--noise_types', type=str,
                        default='factory,white,babble,street,pink',
                        help='Comma-separated noise types for evaluation')
    parser.add_argument('--snr_range', type=str,
                        default='-15,-10,-5,0,5,10,15',
                        help='Comma-separated SNR levels (dB)')
    parser.add_argument('--per_class', action='store_true',
                        help='Enable per-class confusion matrix collection')
    parser.add_argument('--teacher', type=str, default=None,
                        help='Teacher model name for Knowledge Distillation '
                             '(e.g., "BC-ResNet-3"). Requires trained checkpoint.')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to teacher checkpoint dir (default: '
                             '<checkpoint_dir>/<teacher_name>/best.pt)')
    parser.add_argument('--kd_alpha', type=float, default=0.5,
                        help='KD: weight for hard label loss (default: 0.5)')
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                        help='KD: softmax temperature (default: 4.0)')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"  SmartEar KWS - Complete Training Pipeline")
    print(f"  Device: {device}")
    print(f"  Data: {args.data_dir}")
    print(f"  Epochs: {args.epochs}, Seed: {args.seed}")
    print(f"  Noise types: {args.noise_types}")
    print(f"  SNR range: {args.snr_range}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # 1. Load dataset
    print("\n  Loading Google Speech Commands V2...")
    os.makedirs(args.data_dir, exist_ok=True)

    train_dataset = SpeechCommandsDataset(
        args.data_dir, subset='training', augment=True)
    val_dataset = SpeechCommandsDataset(
        args.data_dir, subset='validation', augment=False)
    test_dataset = SpeechCommandsDataset(
        args.data_dir, subset='testing', augment=False)

    # Quick mode: use subset
    if args.quick:
        print("  [QUICK MODE] Using 5% subset for testing")
        n_train = max(len(train_dataset) // 20, 1000)
        n_val = max(len(val_dataset) // 20, 200)
        train_dataset = Subset(train_dataset,
                               list(range(min(n_train, len(train_dataset)))))
        val_dataset = Subset(val_dataset,
                             list(range(min(n_val, len(val_dataset)))))

    print(f"\n  Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Test: {len(test_dataset)}")

    # Cache val/test data in RAM — DISABLED to prevent OOM on Windows
    # (val+test caching uses ~1.8GB which crashes the process)
    # if not args.quick:
    #     try:
    #         if hasattr(val_dataset, 'cache_all'):
    #             val_dataset.cache_all()
    #         if hasattr(test_dataset, 'cache_all'):
    #             test_dataset.cache_all()
    #     except Exception as e:
    #         print(f"  [WARNING] RAM caching failed: {e}. Continuing without cache.",
    #               flush=True)

    # 2. Create models
    all_models = create_all_models(n_classes=12)

    # Filter models if specified
    if args.models != 'all':
        selected = [m.strip() for m in args.models.split(',')]
        all_models = {k: v for k, v in all_models.items() if k in selected}

    # 3. Load teacher model for Knowledge Distillation (if specified)
    teacher_model = None
    if args.teacher:
        print(f"\n  [KD] Loading teacher: {args.teacher}")
        if args.teacher in all_models:
            teacher_model = all_models[args.teacher]
        else:
            # Teacher not in selected models, create it
            all_available = create_all_models(n_classes=12)
            if args.teacher in all_available:
                teacher_model = all_available[args.teacher]
            else:
                print(f"  [ERROR] Teacher '{args.teacher}' not found!")
                return

        # Load teacher checkpoint
        if args.teacher_checkpoint:
            teacher_ckpt_path = Path(args.teacher_checkpoint)
        else:
            teacher_ckpt_path = (Path(args.checkpoint_dir) /
                                 args.teacher.replace(' ', '_') / 'best.pt')

        if teacher_ckpt_path.exists():
            ckpt = torch.load(teacher_ckpt_path, map_location=device,
                              weights_only=True)
            teacher_model.load_state_dict(ckpt['model_state_dict'])
            teacher_model = teacher_model.to(device)
            teacher_model.eval()
            teacher_acc = ckpt.get('val_acc', 0)
            print(f"  [KD] Teacher loaded: {args.teacher} ({teacher_acc:.2f}%)")
            print(f"  [KD] α={args.kd_alpha}, T={args.kd_temperature}")
        else:
            print(f"  [ERROR] Teacher checkpoint not found: {teacher_ckpt_path}")
            print(f"  [INFO] Train teacher first, then use --teacher flag")
            teacher_model = None

    # Train or load each model
    results = {}
    trained_models = {}

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for model_name, model in all_models.items():
        # Don't re-train the teacher model
        if teacher_model is not None and model_name == args.teacher:
            continue
        ckpt_path = (Path(args.checkpoint_dir) /
                     model_name.replace(' ', '_') / 'best.pt')

        if args.eval_only and ckpt_path.exists():
            # Load checkpoint
            print(f"\n  Loading checkpoint: {model_name}")
            ckpt = torch.load(ckpt_path, map_location=device,
                              weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device)
            best_acc = ckpt['val_acc']
            print(f"  Loaded: {best_acc:.2f}%")
        elif args.eval_only and not ckpt_path.exists():
            # Skip model if eval_only but no checkpoint
            print(f"\n  Skipping {model_name}: no checkpoint at {ckpt_path}")
            continue
        else:
            # Determine training hyperparameters based on model size
            params = sum(p.numel() for p in model.parameters())
            if params < 20000:
                lr = 3e-3  # Smaller models need larger LR
            elif params > 300000:
                lr = 5e-4  # Larger models need smaller LR
            else:
                lr = args.lr

            best_acc, model = train_model(
                model, model_name, train_dataset, val_dataset,
                args.checkpoint_dir, device,
                epochs=args.epochs, batch_size=args.batch_size, lr=lr,
                teacher_model=teacher_model,
                kd_alpha=args.kd_alpha, kd_temperature=args.kd_temperature)

        results[model_name] = {'val_acc': best_acc}
        trained_models[model_name] = model

    # 4. Test set evaluation
    print("\n" + "=" * 80)
    print("  TEST SET EVALUATION")
    print("=" * 80)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0)

    for model_name, model in trained_models.items():
        test_acc, _, _ = evaluate(model, test_loader, device)
        results[model_name]['test_acc'] = test_acc
        params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name:<22} | Test: {test_acc:.2f}% | "
              f"Params: {params:,}")

    # 5. Noise robustness evaluation
    val_loader = DataLoader(
        val_dataset if not args.quick else val_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    noise_types = [t.strip() for t in args.noise_types.split(',')]
    snr_levels = [int(s.strip()) for s in args.snr_range.split(',')]
    snr_levels.append('clean')

    noise_results = run_noise_evaluation(
        trained_models, val_loader, device,
        noise_types=noise_types, snr_levels=snr_levels,
        per_class=args.per_class)

    # 6. Save all results
    results_dir = Path(args.checkpoint_dir) / 'results'
    results_dir.mkdir(exist_ok=True)

    # Compile final results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'models': {}
    }

    for model_name, model in trained_models.items():
        params = sum(p.numel() for p in model.parameters())
        size_kb = sum(p.numel() * p.element_size()
                      for p in model.parameters()) / 1024

        final_results['models'][model_name] = {
            'params': params,
            'size_fp32_kb': round(size_kb, 1),
            'size_trt_int8_kb': round(params * 1 / 1024, 1),
            'val_acc': results.get(model_name, {}).get('val_acc', 0),
            'test_acc': results.get(model_name, {}).get('test_acc', 0),
            'noise_robustness': noise_results.get(model_name, {}),
        }

    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # 7. Print final paper-ready table
    print("\n" + "=" * 80)
    print("  FINAL RESULTS - Paper Table")
    print("=" * 80)
    print(f"\n  {'Model':<22} | {'Params':>8} | {'FP32':>7} | "
          f"{'INT8*':>7} | {'Val':>6} | {'Test':>6} | "
          f"{'Noisy -5dB':>10}")
    print("  " + "-" * 90)

    for name, data in final_results['models'].items():
        noisy_raw = data['noise_robustness'].get('factory', {}).get('-5', 0)
        noisy = noisy_raw['accuracy'] if isinstance(noisy_raw, dict) else noisy_raw
        print(f"  {name:<22} | {data['params']:>8,} | "
              f"{data['size_fp32_kb']:>6.1f}K | "
              f"{data['size_trt_int8_kb']:>6.1f}K | "
              f"{data['val_acc']:>5.1f}% | "
              f"{data['test_acc']:>5.1f}% | "
              f"{noisy:>9.1f}%")

    print(f"\n  Results saved to: {results_dir / 'final_results.json'}")
    print(f"  * INT8 = estimated TensorRT INT8 size for Jetson Nano")

    return final_results


if __name__ == '__main__':
    main()
