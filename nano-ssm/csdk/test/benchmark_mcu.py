#!/usr/bin/env python3
"""
MCU Benchmark: NC-SSM vs DS-CNN-S
=================================
Compares inference latency across:
  1. Python (PyTorch) - both models
  2. C SDK (host x86)  - NC-SSM only (already built)
  3. Cortex-M7 estimate - MAC-based projection

Also runs the C executable on real audio and compares predictions.

Usage:
    python benchmark_mcu.py
"""

import sys
import os
import time
import subprocess
import struct
import numpy as np
import torch
from pathlib import Path

# Paths
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'nano-ssm'))

CKPT_DIR = REPO / 'checkpoints_full'
CSDK_DIR = REPO / 'nano-ssm' / 'csdk'
C_EXE = CSDK_DIR / 'ncssm_20k.exe'

LABELS = ['yes', 'no', 'up', 'down', 'left', 'right',
          'on', 'off', 'stop', 'go', 'silence', 'unknown']


def load_ncssm():
    """Load NC-SSM-20K model."""
    from nanomamba import create_nanomamba_nc_20k
    model = create_nanomamba_nc_20k()
    ckpt = torch.load(CKPT_DIR / 'NanoMamba-NC-20K' / 'best.pt',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def load_dscnn():
    """Load DS-CNN-S model."""
    sys.path.insert(0, str(REPO))
    from train_colab import DSCNN_S
    model = DSCNN_S(n_classes=12)
    ckpt = torch.load(CKPT_DIR / 'DS-CNN-S' / 'best.pt',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def prepare_cnn_input(audio, sr=16000):
    """Prepare log-mel spectrogram for DS-CNN-S (same as SimpleEngine)."""
    n_fft = 512
    hop_length = 160
    n_mels = 40
    n_freq = n_fft // 2 + 1

    window = torch.hann_window(n_fft)
    spec = torch.stft(audio.unsqueeze(0), n_fft, hop_length,
                      window=window, return_complex=True)
    mag = spec.abs()

    # Build mel filterbank
    low_mel = 2595 * np.log10(1 + 20 / 700)
    high_mel = 2595 * np.log10(1 + sr / 2 / 700)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    mel_fb = np.zeros((n_mels, n_freq))
    for i in range(n_mels):
        for j in range(bins[i], bins[i+1]):
            mel_fb[i, j] = (j - bins[i]) / max(bins[i+1] - bins[i], 1)
        for j in range(bins[i+1], bins[i+2]):
            mel_fb[i, j] = (bins[i+2] - j) / max(bins[i+2] - bins[i+1], 1)
    mel_fb_t = torch.from_numpy(mel_fb).float()

    mel = torch.matmul(mel_fb_t, mag)
    log_mel = torch.log(mel + 1e-8)
    return log_mel


def benchmark_python(model, prepare_fn, audio, name, n_runs=100, warmup=10):
    """Benchmark Python inference latency."""
    with torch.no_grad():
        for _ in range(warmup):
            inp = prepare_fn(audio)
            if isinstance(inp, torch.Tensor):
                model(inp)
            else:
                model(inp)

        times = []
        for _ in range(n_runs):
            inp = prepare_fn(audio)
            t0 = time.perf_counter()
            if isinstance(inp, torch.Tensor):
                model(inp)
            else:
                model(inp)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times.sort()
    # Remove outliers (top/bottom 10%)
    trimmed = times[n_runs // 10: -n_runs // 10]
    avg = sum(trimmed) / len(trimmed)
    median = trimmed[len(trimmed) // 2]
    best = trimmed[0]
    worst = trimmed[-1]
    return {'avg': avg, 'median': median, 'min': best, 'max': worst}


def benchmark_c(audio_path, n_runs=20):
    """Benchmark C executable latency."""
    if not C_EXE.exists():
        return None

    times = []
    for _ in range(n_runs):
        result = subprocess.run(
            [str(C_EXE), str(audio_path)],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split('\n'):
            if 'Latency:' in line:
                ms = float(line.split(':')[1].strip().replace('ms', ''))
                times.append(ms)
                break

    if not times:
        return None
    times.sort()
    return {
        'avg': sum(times) / len(times),
        'median': times[len(times) // 2],
        'min': times[0],
        'max': times[-1],
    }


def estimate_cortex_m7(macs, clock_mhz=480):
    """Estimate Cortex-M7 latency from MAC count.

    Cortex-M7 can do 1 MAC/cycle with single-precision FPU.
    With CMSIS-DSP optimization, ~0.8 MAC/cycle effective.
    """
    cycles = macs / 0.8  # Effective MAC/cycle
    seconds = cycles / (clock_mhz * 1e6)
    return seconds * 1000  # ms


def count_macs_ncssm(d_model=37, d_inner=55, d_state=10, n_mels=40,
                     n_frames=100, n_layers=2, n_fft=512, n_classes=12):
    """Count MACs for NC-SSM-20K forward pass."""
    n_freq = n_fft // 2 + 1  # 257

    macs = 0

    # 1. STFT: n_frames * n_fft * log2(n_fft) (FFT)
    macs += n_frames * n_fft * np.log2(n_fft)  # ~460K

    # 2. Mel projection: n_frames * n_mels * n_freq
    macs += n_frames * n_mels * n_freq  # ~1.03M

    # 3. SNR estimation: similar to mel projection + some ops
    macs += n_frames * n_mels * n_freq  # ~1.03M

    # 4. Spectral gate: 3 * n_mels * n_frames (w*snr + b + floor)
    macs += 3 * n_mels * n_frames  # ~12K

    # 5. DualPCEN: ~5 * n_mels * n_frames (IIR + AGC)
    macs += 5 * n_mels * n_frames  # ~20K

    # 6. Instance norm: 3 * n_mels * n_frames
    macs += 3 * n_mels * n_frames  # ~12K

    # 7. Patch projection: n_frames * n_mels * d_model
    macs += n_frames * n_mels * d_model  # ~148K

    # 8. Per SSM block:
    for _ in range(n_layers):
        # LayerNorm: 3 * d_model * n_frames
        macs += 3 * d_model * n_frames  # ~11.1K

        # In projection: n_frames * d_model * 2*d_inner
        macs += n_frames * d_model * 2 * d_inner  # ~407K

        # DWConv1d: n_frames * d_inner * 3 (kernel=3)
        macs += n_frames * d_inner * 3  # ~16.5K

        # SiLU: n_frames * d_inner * 3 (x*sigmoid(x))
        macs += n_frames * d_inner * 3  # ~16.5K

        # SSM scan per frame:
        #   x_proj: d_inner * (2*d_state+1) = 55*21 = 1155
        #   snr_proj: n_mels * (d_state+1) = 40*11 = 440
        #   dt_proj: d_inner = 55
        #   State update: d_inner * d_state * 5 = 55*10*5 = 2750
        #   Output: d_inner * d_state = 550
        ssm_per_frame = (d_inner * (2*d_state+1) +
                         n_mels * (d_state+1) +
                         d_inner +
                         d_inner * d_state * 5 +
                         d_inner * d_state)
        macs += n_frames * ssm_per_frame  # ~495K

        # Gate: n_frames * d_inner * 3
        macs += n_frames * d_inner * 3  # ~16.5K

        # Out projection: n_frames * d_inner * d_model
        macs += n_frames * d_inner * d_model  # ~203.5K

    # 9. Final norm: 3 * d_model * n_frames
    macs += 3 * d_model * n_frames  # ~11.1K

    # 10. GAP: d_model * n_frames
    macs += d_model * n_frames  # ~3.7K

    # 11. Classifier: d_model * n_classes
    macs += d_model * n_classes  # ~444

    return int(macs)


def count_macs_dscnn(n_mels=40, n_frames=101, n_classes=12):
    """Count MACs for DS-CNN-S forward pass.

    Architecture: DS-CNN-S from ARM paper.
    Conv2D(1,64,10x4) → 4x DS blocks (64→64) → AvgPool → FC(64,12)
    """
    macs = 0

    # Feature extraction (STFT + mel) - same as NC-SSM
    n_fft = 512
    n_freq = n_fft // 2 + 1
    macs += n_frames * n_fft * np.log2(n_fft)  # STFT
    macs += n_frames * n_mels * n_freq  # Mel projection

    # First conv: Conv2D(1, 64, (10,4), stride=(2,2), padding=(4,1))
    # Output: ~20 x 20 x 64
    h_out, w_out = 20, 20
    macs += h_out * w_out * 64 * 1 * 10 * 4  # ~1.024M

    # 4x DS blocks: DepthwiseConv2D(64, 3x3) + PointwiseConv2D(64, 64)
    for i in range(4):
        # Depthwise: h * w * 64 * 3 * 3
        macs += h_out * w_out * 64 * 3 * 3  # ~230K
        # Pointwise: h * w * 64 * 64
        macs += h_out * w_out * 64 * 64  # ~1.64M

    # Average pooling: 64
    macs += h_out * w_out * 64

    # FC: 64 * 12
    macs += 64 * n_classes

    return int(macs)


def save_audio_raw(audio_tensor, path):
    """Save float32 audio as raw binary."""
    data = audio_tensor.numpy().astype(np.float32)
    with open(path, 'wb') as f:
        f.write(data.tobytes())


def main():
    print("=" * 70)
    print("  NC-SSM vs DS-CNN-S: MCU Benchmark")
    print("  Simulating Cortex-M7 @ 480MHz deployment")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    ncssm, ncssm_params = load_ncssm()
    dscnn, dscnn_params = load_dscnn()
    print(f"  NC-SSM-20K:  {ncssm_params:>6,} params ({ncssm_params/1024:.1f} KB INT8)")
    print(f"  DS-CNN-S:    {dscnn_params:>6,} params ({dscnn_params/1024:.1f} KB INT8)")

    # Load test audio files
    test_files = sorted(CSDK_DIR.glob('test/audio_*.raw'))
    if not test_files:
        print("No test audio files found! Generating synthetic...")
        torch.manual_seed(42)
        test_audio = torch.randn(16000) * 0.1
        raw_path = CSDK_DIR / 'test' / 'audio_test.raw'
        save_audio_raw(test_audio, raw_path)
        test_files = [raw_path]

    # Load first audio for benchmarking
    with open(test_files[0], 'rb') as f:
        audio_data = np.frombuffer(f.read(), dtype=np.float32)
    audio = torch.from_numpy(audio_data.copy())

    # -- 1. Python Benchmark --
    print("\n" + "-" * 70)
    print("  1. Python (PyTorch CPU) Latency")
    print("-" * 70)

    def ncssm_prepare(a):
        return a.unsqueeze(0)

    ncssm_stats = benchmark_python(ncssm, ncssm_prepare, audio, 'NC-SSM', n_runs=100)
    print(f"  NC-SSM-20K:  avg={ncssm_stats['avg']:.1f}ms  "
          f"median={ncssm_stats['median']:.1f}ms  "
          f"min={ncssm_stats['min']:.1f}ms  max={ncssm_stats['max']:.1f}ms")

    def dscnn_prepare(a):
        return prepare_cnn_input(a)

    dscnn_stats = benchmark_python(dscnn, dscnn_prepare, audio, 'DS-CNN-S', n_runs=100)
    print(f"  DS-CNN-S:    avg={dscnn_stats['avg']:.1f}ms  "
          f"median={dscnn_stats['median']:.1f}ms  "
          f"min={dscnn_stats['min']:.1f}ms  max={dscnn_stats['max']:.1f}ms")

    ratio_py = ncssm_stats['median'] / dscnn_stats['median']
    if ratio_py > 1:
        print(f"\n  [Python] DS-CNN-S is {ratio_py:.1f}x faster (Python overhead hurts SSM)")
    else:
        print(f"\n  [Python] NC-SSM is {1/ratio_py:.1f}x faster")

    # -- 2. C SDK Benchmark --
    print("\n" + "-" * 70)
    print("  2. C SDK (x86, -O3) Latency - NC-SSM only")
    print("-" * 70)

    if C_EXE.exists():
        c_stats = benchmark_c(test_files[0], n_runs=20)
        if c_stats:
            print(f"  NC-SSM-20K C: avg={c_stats['avg']:.1f}ms  "
                  f"median={c_stats['median']:.1f}ms  "
                  f"min={c_stats['min']:.1f}ms  max={c_stats['max']:.1f}ms")
            speedup = ncssm_stats['median'] / c_stats['median']
            print(f"  C vs Python speedup: {speedup:.1f}x")
        else:
            print("  [WARN] Could not parse C output")
            c_stats = None
    else:
        print(f"  [SKIP] C executable not found: {C_EXE}")
        c_stats = None

    # -- 3. Accuracy on test audio --
    print("\n" + "-" * 70)
    print("  3. Prediction Accuracy (test audio samples)")
    print("-" * 70)
    print(f"  {'File':<20} {'NC-SSM (Py)':<20} {'DS-CNN-S (Py)':<20} {'NC-SSM (C)':<20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

    ncssm_correct = 0
    dscnn_correct = 0
    c_correct = 0
    n_files = 0

    for fp in test_files:
        fname = fp.stem.replace('audio_', '')
        with open(fp, 'rb') as f:
            aud = np.frombuffer(f.read(), dtype=np.float32)
        aud_t = torch.from_numpy(aud.copy())

        # Python NC-SSM
        with torch.no_grad():
            logits = ncssm(aud_t.unsqueeze(0))
            probs = torch.softmax(logits, -1)[0]
            ncssm_pred = LABELS[probs.argmax()]
            ncssm_conf = probs.max().item()

        # Python DS-CNN-S
        with torch.no_grad():
            log_mel = prepare_cnn_input(aud_t)
            logits = dscnn(log_mel)
            probs = torch.softmax(logits, -1)[0]
            dscnn_pred = LABELS[probs.argmax()]
            dscnn_conf = probs.max().item()

        # C NC-SSM
        c_pred = '?'
        c_conf = 0.0
        if C_EXE.exists():
            try:
                result = subprocess.run(
                    [str(C_EXE), str(fp)],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.split('\n'):
                    if 'Prediction:' in line:
                        parts = line.split(':')[1].strip()
                        c_pred = parts.split('(')[0].strip()
                        c_conf = float(parts.split('(')[1].replace('%)', '')) / 100
                        break
            except Exception:
                pass

        n_files += 1
        if ncssm_pred == fname:
            ncssm_correct += 1
        if dscnn_pred == fname:
            dscnn_correct += 1
        if c_pred == fname:
            c_correct += 1

        m_n = 'O' if ncssm_pred == fname else 'X'
        m_d = 'O' if dscnn_pred == fname else 'X'
        m_c = 'O' if c_pred == fname else 'X'

        print(f"  {fname:<20} {ncssm_pred:>8} {ncssm_conf:.0%} [{m_n}]  "
              f"{dscnn_pred:>8} {dscnn_conf:.0%} [{m_d}]  "
              f"{c_pred:>8} {c_conf:.0%} [{m_c}]")

    if n_files > 0:
        print(f"\n  Accuracy:  NC-SSM(Py)={ncssm_correct}/{n_files}  "
              f"DS-CNN-S(Py)={dscnn_correct}/{n_files}  "
              f"NC-SSM(C)={c_correct}/{n_files}")

    # -- 4. MAC Count & Cortex-M7 Estimate --
    print("\n" + "-" * 70)
    print("  4. MAC Count & Cortex-M7 Projection")
    print("-" * 70)

    ncssm_macs = count_macs_ncssm()
    dscnn_macs = count_macs_dscnn()

    ncssm_m7_ms = estimate_cortex_m7(ncssm_macs)
    dscnn_m7_ms = estimate_cortex_m7(dscnn_macs)

    print(f"  NC-SSM-20K:  {ncssm_macs:>10,} MACs  →  Cortex-M7: ~{ncssm_m7_ms:.1f} ms")
    print(f"  DS-CNN-S:    {dscnn_macs:>10,} MACs  →  Cortex-M7: ~{dscnn_m7_ms:.1f} ms")
    print(f"  MAC ratio:   DS-CNN-S / NC-SSM = {dscnn_macs/ncssm_macs:.1f}x")
    print(f"  Speed ratio: NC-SSM is ~{dscnn_m7_ms/ncssm_m7_ms:.1f}x faster on Cortex-M7")

    # -- 5. Summary --
    print("\n" + "=" * 70)
    print("  SUMMARY: Why NC-SSM wins on MCU")
    print("=" * 70)
    print(f"""
  Python (PyTorch CPU):
    NC-SSM: {ncssm_stats['median']:.1f}ms  |  DS-CNN-S: {dscnn_stats['median']:.1f}ms
    → SSM sequential scan has Python loop overhead
    → CNN parallel ops benefit from PyTorch/MKL

  C SDK (x86 -O3):
    NC-SSM: {c_stats['median'] if c_stats else '?':.1f}ms  |  DS-CNN-S: not implemented
    → {ncssm_stats['median']/c_stats['median'] if c_stats else '?':.0f}x faster than Python

  Cortex-M7 @ 480MHz (estimated):
    NC-SSM: ~{ncssm_m7_ms:.1f}ms  |  DS-CNN-S: ~{dscnn_m7_ms:.1f}ms
    → NC-SSM is {dscnn_m7_ms/ncssm_m7_ms:.1f}x faster (pure MAC advantage)

  Model Size:
    NC-SSM: {ncssm_params:,} params ({ncssm_params/1024:.1f} KB)
    DS-CNN-S: {dscnn_params:,} params ({dscnn_params/1024:.1f} KB)

  Key Insight:
    SSM's sequential scan is a Python/interpreter bottleneck,
    NOT a hardware bottleneck. In C/MCU, each scan step is just
    ~{55*10*5} MACs - trivial for a 480MHz Cortex-M7.
    CNN's parallel convolutions don't help on single-core MCU.
""")

    # -- 6. Per-keyword latency test (C) --
    if C_EXE.exists():
        print("-" * 70)
        print("  6. NC-SSM C: Per-keyword latency breakdown")
        print("-" * 70)
        for fp in test_files:
            fname = fp.stem.replace('audio_', '')
            result = subprocess.run(
                [str(C_EXE), str(fp)],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split('\n'):
                if 'Prediction:' in line:
                    pred = line.split(':')[1].strip()
                if 'Latency:' in line:
                    lat = line.split(':')[1].strip()
            print(f"  {fname:<10} → {pred:<20} latency: {lat}")


if __name__ == '__main__':
    main()
