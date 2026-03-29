#!/usr/bin/env python3
# coding=utf-8
"""
NC-Vibration Training Pipeline — CWRU Bearing Fault Detection
=============================================================

Leakage-free evaluation protocols for honest benchmarking.

Protocols:
  A. Load-split: Train on 0,1,2 HP -> Test on 3 HP (cross-load)
  B. Standard:   Train/Test split per recording (80/20, no overlap)
  C. Severity:   Train on 0.014,0.021 -> Test on 0.007 (unseen severity)

Usage:
  # Download CWRU data + train NC-Vib-TCN-20K
  python train_vibration.py --download --models NC-Vib-TCN-20K --epochs 50

  # Train all models with cross-load protocol
  python train_vibration.py --models NC-Vib-TCN-Tiny,NC-Vib-TCN-20K,NC-Vib-SSM-20K \\
      --protocol load_split --epochs 50

  # Noise robustness evaluation
  python train_vibration.py --models NC-Vib-TCN-20K --eval_only --noise_eval

  # Profile (params, MACs, latency)
  python train_vibration.py --models NC-Vib-TCN-20K --profile
"""

import os
import sys
import json
import time
import math
import argparse
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CWRU Dataset — File Mapping
# ============================================================================

# (file_number, fault_type, fault_diameter_inch, description)
CWRU_FILES = {
    # Normal baseline
    'normal': {
        0: {'file': '97',  'rpm': 1797},
        1: {'file': '98',  'rpm': 1772},
        2: {'file': '99',  'rpm': 1750},
        3: {'file': '100', 'rpm': 1730},
    },
    # Inner race fault
    'inner_007': {
        0: {'file': '105', 'rpm': 1797},
        1: {'file': '106', 'rpm': 1772},
        2: {'file': '107', 'rpm': 1750},
        3: {'file': '108', 'rpm': 1730},
    },
    'inner_014': {
        0: {'file': '169', 'rpm': 1797},
        1: {'file': '170', 'rpm': 1772},
        2: {'file': '171', 'rpm': 1750},
        3: {'file': '172', 'rpm': 1730},
    },
    'inner_021': {
        0: {'file': '209', 'rpm': 1797},
        1: {'file': '210', 'rpm': 1772},
        2: {'file': '211', 'rpm': 1750},
        3: {'file': '212', 'rpm': 1730},
    },
    # Ball fault
    'ball_007': {
        0: {'file': '118', 'rpm': 1797},
        1: {'file': '119', 'rpm': 1772},
        2: {'file': '120', 'rpm': 1750},
        3: {'file': '121', 'rpm': 1730},
    },
    'ball_014': {
        0: {'file': '185', 'rpm': 1797},
        1: {'file': '186', 'rpm': 1772},
        2: {'file': '187', 'rpm': 1750},
        3: {'file': '188', 'rpm': 1730},
    },
    'ball_021': {
        0: {'file': '222', 'rpm': 1797},
        1: {'file': '223', 'rpm': 1772},
        2: {'file': '224', 'rpm': 1750},
        3: {'file': '225', 'rpm': 1730},
    },
    # Outer race fault (6 o'clock position)
    'outer_007': {
        0: {'file': '130', 'rpm': 1797},
        1: {'file': '131', 'rpm': 1772},
        2: {'file': '132', 'rpm': 1750},
        3: {'file': '133', 'rpm': 1730},
    },
    'outer_014': {
        0: {'file': '197', 'rpm': 1797},
        1: {'file': '198', 'rpm': 1772},
        2: {'file': '199', 'rpm': 1750},
        3: {'file': '200', 'rpm': 1730},
    },
    'outer_021': {
        0: {'file': '234', 'rpm': 1797},
        1: {'file': '235', 'rpm': 1772},
        2: {'file': '236', 'rpm': 1750},
        3: {'file': '237', 'rpm': 1730},
    },
}

# Classification schemes
LABEL_MAP_4CLASS = {
    'normal': 0,
    'inner_007': 1, 'inner_014': 1, 'inner_021': 1,
    'ball_007': 2, 'ball_014': 2, 'ball_021': 2,
    'outer_007': 3, 'outer_014': 3, 'outer_021': 3,
}
LABEL_NAMES_4CLASS = ['Normal', 'Inner', 'Ball', 'Outer']

LABEL_MAP_10CLASS = {
    'normal': 0,
    'inner_007': 1, 'inner_014': 2, 'inner_021': 3,
    'ball_007': 4, 'ball_014': 5, 'ball_021': 6,
    'outer_007': 7, 'outer_014': 8, 'outer_021': 9,
}
LABEL_NAMES_10CLASS = [
    'Normal',
    'IR007', 'IR014', 'IR021',
    'B007', 'B014', 'B021',
    'OR007', 'OR014', 'OR021'
]


# ============================================================================
# Data Download
# ============================================================================

CWRU_BASE_URL = 'https://engineering.case.edu/bearingdatacenter/download-data-file/'


def download_cwru(data_dir):
    """Download CWRU .mat files from official site."""
    os.makedirs(data_dir, exist_ok=True)

    all_files = set()
    for fault_type, loads in CWRU_FILES.items():
        for load, info in loads.items():
            all_files.add(info['file'])

    print(f"Downloading {len(all_files)} CWRU .mat files to {data_dir}...")

    for file_num in sorted(all_files, key=int):
        dst = os.path.join(data_dir, f'{file_num}.mat')
        if os.path.exists(dst):
            continue

        # CWRU URL format
        url = f'{CWRU_BASE_URL}{file_num}'
        print(f'  Downloading {file_num}.mat ...', end=' ', flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
            size_kb = os.path.getsize(dst) / 1024
            print(f'OK ({size_kb:.0f} KB)')
        except Exception as e:
            print(f'FAILED: {e}')
            # Try alternative URL format
            alt_url = f'https://engineering.case.edu/sites/default/files/{file_num}.mat'
            try:
                urllib.request.urlretrieve(alt_url, dst)
                size_kb = os.path.getsize(dst) / 1024
                print(f'  -> Alt URL OK ({size_kb:.0f} KB)')
            except Exception as e2:
                print(f'  -> Alt URL also FAILED: {e2}')

    print(f'Download complete. Files in: {data_dir}')


# ============================================================================
# Data Loading
# ============================================================================

def load_mat_file(filepath):
    """Load .mat file and extract drive-end vibration signal."""
    try:
        from scipy.io import loadmat
    except ImportError:
        print("scipy required: pip install scipy")
        sys.exit(1)

    data = loadmat(filepath)

    # Find DE (Drive End) time-series variable
    de_key = None
    for key in data.keys():
        if key.endswith('_DE_time'):
            de_key = key
            break

    if de_key is None:
        # Try alternative: just look for largest numeric array
        max_len = 0
        for key, val in data.items():
            if key.startswith('__'):
                continue
            if hasattr(val, 'shape') and len(val.shape) >= 1:
                if val.size > max_len:
                    max_len = val.size
                    de_key = key

    if de_key is None:
        raise ValueError(f"No vibration data found in {filepath}")

    signal = data[de_key].flatten().astype(np.float32)
    return signal


def segment_signal(signal, window_size=2048, stride=None, normalize=True):
    """Segment continuous signal into fixed-length windows.

    Args:
        signal: 1D numpy array
        window_size: samples per window
        stride: step between windows (default: window_size = no overlap)
        normalize: z-score normalize each window
    Returns:
        windows: (N, window_size) numpy array
    """
    if stride is None:
        stride = window_size

    n_windows = (len(signal) - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        w = signal[start:start + window_size]
        if normalize:
            mu = w.mean()
            std = w.std()
            if std > 1e-8:
                w = (w - mu) / std
        windows[i] = w

    return windows


# ============================================================================
# Dataset Class
# ============================================================================

class CWRUDataset(Dataset):
    """CWRU Bearing Dataset with leakage-free splits."""

    def __init__(self, data_dir, split='train', protocol='load_split',
                 n_classes=4, window_size=2048, stride=None,
                 train_loads=(0, 1, 2), test_loads=(3,),
                 train_severities=('014', '021'), test_severities=('007',),
                 train_ratio=0.8, noise_aug=False, noise_snr_range=(-5, 15)):
        """
        Args:
            data_dir: path to .mat files
            split: 'train' or 'test'
            protocol: 'load_split', 'standard', 'severity_split'
            n_classes: 4 (fault type) or 10 (fault type + severity)
            window_size: samples per window
            stride: window stride (None = window_size)
            train_loads: HP loads for training (protocol=load_split)
            test_loads: HP loads for testing (protocol=load_split)
            train_ratio: train fraction (protocol=standard)
            noise_aug: add noise augmentation during training
            noise_snr_range: (min_snr, max_snr) in dB for noise aug
        """
        self.window_size = window_size
        self.noise_aug = noise_aug and (split == 'train')
        self.noise_snr_range = noise_snr_range

        label_map = LABEL_MAP_4CLASS if n_classes == 4 else LABEL_MAP_10CLASS

        windows_list = []
        labels_list = []

        for fault_type, loads in CWRU_FILES.items():
            label = label_map[fault_type]

            # Determine which loads to use
            if protocol == 'load_split':
                target_loads = train_loads if split == 'train' else test_loads
            elif protocol == 'severity_split':
                # Filter by fault severity
                severity = fault_type.split('_')[-1] if '_' in fault_type else None
                if fault_type == 'normal':
                    target_loads = train_loads if split == 'train' else test_loads
                elif severity in train_severities and split == 'train':
                    target_loads = (0, 1, 2, 3)
                elif severity in test_severities and split == 'test':
                    target_loads = (0, 1, 2, 3)
                else:
                    continue
            else:
                target_loads = (0, 1, 2, 3)

            for load in target_loads:
                if load not in loads:
                    continue
                info = loads[load]
                filepath = os.path.join(data_dir, f"{info['file']}.mat")

                if not os.path.exists(filepath):
                    print(f"  Warning: {filepath} not found, skipping")
                    continue

                signal = load_mat_file(filepath)
                windows = segment_signal(signal, window_size, stride)

                if protocol == 'standard':
                    # Split within each recording
                    n = len(windows)
                    n_train = int(n * train_ratio)
                    if split == 'train':
                        windows = windows[:n_train]
                    else:
                        windows = windows[n_train:]

                windows_list.append(windows)
                labels_list.extend([label] * len(windows))

        if len(windows_list) == 0:
            raise ValueError(f"No data loaded for split={split}, protocol={protocol}. "
                             f"Check data_dir={data_dir}")

        self.windows = np.concatenate(windows_list, axis=0)
        self.labels = np.array(labels_list, dtype=np.int64)

        print(f"  [{split:5s}] {len(self.windows):>6,} windows, "
              f"{n_classes} classes, protocol={protocol}")
        # Class distribution
        for i in range(n_classes):
            count = (self.labels == i).sum()
            name = LABEL_NAMES_4CLASS[i] if n_classes == 4 else LABEL_NAMES_10CLASS[i]
            print(f"         class {i} ({name:>7s}): {count:>5,}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx].copy())
        y = self.labels[idx]

        # Noise augmentation
        if self.noise_aug:
            snr_db = np.random.uniform(*self.noise_snr_range)
            noise = torch.randn_like(x)
            signal_power = (x ** 2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10) + 1e-8)
            x = x + noise * noise_power.sqrt()

        return x, y


# ============================================================================
# Model Registry
# ============================================================================

def get_model_registry():
    """Lazy import to avoid circular deps."""
    from nc_vibration import (
        create_nc_vib_ssm_20k, create_nc_vib_tcn_20k,
        create_nc_vib_ssm_matched, create_nc_vib_tcn_matched,
        create_nc_vib_tcn_tiny,
    )

    return {
        'NC-Vib-TCN-Tiny': create_nc_vib_tcn_tiny,
        'NC-Vib-TCN-Matched': create_nc_vib_tcn_matched,
        'NC-Vib-SSM-Matched': create_nc_vib_ssm_matched,
        'NC-Vib-TCN-20K': create_nc_vib_tcn_20k,
        'NC-Vib-SSM-20K': create_nc_vib_ssm_20k,
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    if scheduler is not None:
        scheduler.step()

    acc = correct / total * 100
    avg_loss = total_loss / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = correct / total * 100
    return acc, np.array(all_preds), np.array(all_labels)


def confusion_matrix(preds, labels, n_classes):
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def per_class_metrics(cm):
    """Compute per-class precision, recall, F1 from confusion matrix."""
    n = cm.shape[0]
    metrics = {}
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[i] = {'precision': precision, 'recall': recall, 'f1': f1}
    return metrics


# ============================================================================
# Noise Robustness Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_noise_robustness(model, data_dir, device, n_classes=4,
                              window_size=2048, protocol='load_split'):
    """Evaluate model under various noise types and SNR levels."""
    model.eval()

    noise_types = ['gaussian', 'pink', 'periodic_impulse', 'uniform']
    snr_levels = [-10, -5, 0, 5, 10, 15]

    # Load clean test set
    test_ds = CWRUDataset(data_dir, split='test', protocol=protocol,
                          n_classes=n_classes, window_size=window_size,
                          noise_aug=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=0)

    # Clean accuracy
    clean_acc, _, _ = evaluate(model, test_loader, device)

    print(f"\n{'='*70}")
    print(f"  Noise Robustness Evaluation")
    print(f"{'='*70}")
    print(f"  Clean accuracy: {clean_acc:.1f}%\n")

    header = f"  {'Noise Type':<18}"
    for snr in snr_levels:
        header += f"  {snr:>4}dB"
    print(header)
    print("  " + "-" * (18 + 7 * len(snr_levels)))

    results = {}

    for noise_type in noise_types:
        row = f"  {noise_type:<18}"
        results[noise_type] = {}

        for snr_db in snr_levels:
            # Create noisy dataset
            noisy_windows = []
            for x, y in test_loader:
                x_np = x.numpy()
                for i in range(len(x_np)):
                    signal = x_np[i]
                    noise = _generate_noise(noise_type, len(signal))
                    sig_power = (signal ** 2).mean()
                    noise_power = sig_power / (10 ** (snr_db / 10) + 1e-8)
                    noisy = signal + noise * np.sqrt(noise_power)
                    noisy_windows.append(noisy)

            # Evaluate
            noisy_tensor = torch.tensor(np.array(noisy_windows), dtype=torch.float32)
            labels_tensor = torch.tensor(test_ds.labels, dtype=torch.long)

            noisy_ds = torch.utils.data.TensorDataset(noisy_tensor, labels_tensor)
            noisy_loader = DataLoader(noisy_ds, batch_size=128, shuffle=False)

            acc, _, _ = evaluate(model, noisy_loader, device)
            results[noise_type][snr_db] = acc
            row += f"  {acc:5.1f}%"

        print(row)

    print(f"\n  Clean: {clean_acc:.1f}%")
    return results


def _generate_noise(noise_type, length):
    """Generate noise signal of given type."""
    if noise_type == 'gaussian':
        return np.random.randn(length).astype(np.float32)
    elif noise_type == 'pink':
        # 1/f noise via filtering white noise
        white = np.random.randn(length)
        # Simple IIR approximation of 1/f
        pink = np.zeros(length, dtype=np.float32)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        from scipy.signal import lfilter
        pink = lfilter(b, a, white).astype(np.float32)
        return pink
    elif noise_type == 'periodic_impulse':
        # Periodic impulses (simulates adjacent machine impacts)
        noise = np.random.randn(length).astype(np.float32) * 0.1
        period = np.random.randint(200, 500)
        for i in range(0, length, period):
            width = np.random.randint(3, 8)
            noise[i:min(i + width, length)] += np.random.randn(min(width, length - i)) * 3.0
        return noise
    elif noise_type == 'uniform':
        return (np.random.rand(length).astype(np.float32) - 0.5) * 2
    else:
        return np.random.randn(length).astype(np.float32)


# ============================================================================
# Profiling
# ============================================================================

def profile_model(model, device, window_size=2048):
    """Profile model: params, speed, memory."""
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    fp32_kb = params * 4 / 1024
    int8_kb = params / 1024

    x = torch.randn(1, window_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    N = 100
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1000

    return {
        'params': params,
        'fp32_kb': fp32_kb,
        'int8_kb': int8_kb,
        'latency_ms': ms,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NC-Vibration Training')
    parser.add_argument('--models', type=str, default='NC-Vib-TCN-20K',
                        help='Comma-separated model names')
    parser.add_argument('--data_dir', type=str, default='./data/cwru',
                        help='Path to CWRU .mat files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vibration',
                        help='Checkpoint save directory')
    parser.add_argument('--download', action='store_true',
                        help='Download CWRU dataset before training')

    # Training params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--window_size', type=int, default=2048,
                        help='Samples per window (2048 @ 12kHz = 0.17s)')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='4 (fault type) or 10 (fault type + severity)')

    # Protocol
    parser.add_argument('--protocol', type=str, default='load_split',
                        choices=['load_split', 'standard', 'severity_split'],
                        help='Evaluation protocol')
    parser.add_argument('--train_loads', type=str, default='0,1,2',
                        help='Training loads (HP)')
    parser.add_argument('--test_loads', type=str, default='3',
                        help='Test loads (HP)')

    # Noise augmentation
    parser.add_argument('--noise_aug', action='store_true',
                        help='Enable noise augmentation during training')
    parser.add_argument('--noise_snr_min', type=float, default=-5)
    parser.add_argument('--noise_snr_max', type=float, default=15)

    # Eval modes
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--noise_eval', action='store_true',
                        help='Run noise robustness evaluation')
    parser.add_argument('--profile', action='store_true',
                        help='Profile models (params, speed, memory)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Download data
    if args.download:
        download_cwru(args.data_dir)

    model_names = [m.strip() for m in args.models.split(',')]
    registry = get_model_registry()

    train_loads = tuple(int(x) for x in args.train_loads.split(','))
    test_loads = tuple(int(x) for x in args.test_loads.split(','))

    # ── Profile mode ──
    if args.profile:
        print(f"\n{'='*60}")
        print(f"  Model Profiling")
        print(f"{'='*60}")
        print(f"  {'Model':<24} {'Params':>8} {'INT8':>8} {'Latency':>10}")
        print(f"  {'-'*54}")

        for name in model_names:
            if name not in registry:
                print(f"  {name}: unknown model, skipping")
                continue
            model = registry[name](n_classes=args.n_classes).to(device)
            info = profile_model(model, device, args.window_size)
            print(f"  {name:<24} {info['params']:>8,} "
                  f"{info['int8_kb']:>6.1f}KB "
                  f"{info['latency_ms']:>8.1f}ms")
        return

    # ── Train / Eval each model ──
    for model_name in model_names:
        if model_name not in registry:
            print(f"\nUnknown model: {model_name}")
            print(f"Available: {list(registry.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        model = registry[model_name](n_classes=args.n_classes).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,} ({params*4/1024:.1f} KB FP32, "
              f"{params/1024:.1f} KB INT8)")

        # Checkpoint path
        ckpt_dir = os.path.join(args.checkpoint_dir, model_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'best.pt')

        # Load existing checkpoint
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state['model_state_dict'])
            print(f"  Loaded checkpoint: {state.get('test_acc', '?')}% "
                  f"(epoch {state.get('epoch', '?')})")

        if args.eval_only:
            if args.noise_eval:
                evaluate_noise_robustness(model, args.data_dir, device,
                                          args.n_classes, args.window_size,
                                          args.protocol)
            else:
                test_ds = CWRUDataset(
                    args.data_dir, split='test', protocol=args.protocol,
                    n_classes=args.n_classes, window_size=args.window_size,
                    train_loads=train_loads, test_loads=test_loads)
                test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
                acc, preds, labels = evaluate(model, test_loader, device)
                cm = confusion_matrix(preds, labels, args.n_classes)
                metrics = per_class_metrics(cm)

                names = LABEL_NAMES_4CLASS if args.n_classes == 4 else LABEL_NAMES_10CLASS
                print(f"\n  Test Accuracy: {acc:.2f}%\n")
                print(f"  {'Class':<10} {'Prec':>6} {'Recall':>6} {'F1':>6}")
                print(f"  {'-'*30}")
                for i, name in enumerate(names):
                    m = metrics[i]
                    print(f"  {name:<10} {m['precision']:>5.1%} "
                          f"{m['recall']:>5.1%} {m['f1']:>5.1%}")
            continue

        # ── Training ──
        print(f"\n  Loading data (protocol={args.protocol})...")

        train_ds = CWRUDataset(
            args.data_dir, split='train', protocol=args.protocol,
            n_classes=args.n_classes, window_size=args.window_size,
            train_loads=train_loads, test_loads=test_loads,
            noise_aug=args.noise_aug,
            noise_snr_range=(args.noise_snr_min, args.noise_snr_max))

        test_ds = CWRUDataset(
            args.data_dir, split='test', protocol=args.protocol,
            n_classes=args.n_classes, window_size=args.window_size,
            train_loads=train_loads, test_loads=test_loads)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0,
                                   drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_acc = 0
        print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Test Acc':>8} | {'Best':>6} | {'LR':>10}")
        print(f"  {'-'*65}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch)
            test_acc, _, _ = evaluate(model, test_loader, device)

            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                    'model_name': model_name,
                    'protocol': args.protocol,
                    'n_classes': args.n_classes,
                }, ckpt_path)

            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            marker = ' *' if is_best else ''
            print(f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1f}% | "
                  f"{test_acc:>7.1f}% | {best_acc:>5.1f}% | {lr:>10.6f}{marker}")

        print(f"\n  Best test accuracy: {best_acc:.2f}%")
        print(f"  Checkpoint saved: {ckpt_path}")

        # Final confusion matrix
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict'])
        test_acc, preds, labels = evaluate(model, test_loader, device)
        cm = confusion_matrix(preds, labels, args.n_classes)
        metrics = per_class_metrics(cm)

        names = LABEL_NAMES_4CLASS if args.n_classes == 4 else LABEL_NAMES_10CLASS
        print(f"\n  Final Confusion Matrix:")
        header = "  " + " " * 8
        for name in names:
            header += f" {name[:5]:>5}"
        print(header)
        for i, name in enumerate(names):
            row = f"  {name:<8}"
            for j in range(len(names)):
                row += f" {cm[i,j]:>5}"
            print(row)

        print(f"\n  Per-class metrics:")
        print(f"  {'Class':<10} {'Prec':>6} {'Recall':>6} {'F1':>6}")
        print(f"  {'-'*30}")
        for i, name in enumerate(names):
            m = metrics[i]
            print(f"  {name:<10} {m['precision']:>5.1%} "
                  f"{m['recall']:>5.1%} {m['f1']:>5.1%}")

        # Run noise evaluation if requested
        if args.noise_eval:
            evaluate_noise_robustness(model, args.data_dir, device,
                                      args.n_classes, args.window_size,
                                      args.protocol)


if __name__ == '__main__':
    main()
