#!/usr/bin/env python3
"""Live mic demo for baseline models (BC-ResNet-1, DS-CNN-S).
These models take mel spectrogram input, not raw audio.

Usage:
    python demo_mic_baseline.py --model BC-ResNet-1
    python demo_mic_baseline.py --model DS-CNN-S
"""

import argparse
import sys
import time
import queue
from pathlib import Path
from collections import deque

import torch
import torch.nn.functional as F
import sounddevice as sd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_colab import DSCNN_S, BCResNet

LABELS = ['yes','no','up','down','left','right','on','off','stop','go','silence','unknown']
SR = 16000
N_FFT = 512
HOP = 160
N_MELS = 40


def create_mel_fb(sr=16000, n_fft=512, n_mels=40):
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
                fb[i, j] = (j - bin_points[i]) / max(bin_points[i + 1] - bin_points[i], 1)
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            if j < n_freq:
                fb[i, j] = (bin_points[i + 2] - j) / max(bin_points[i + 2] - bin_points[i + 1], 1)
    return torch.from_numpy(fb)


def audio_to_mel(audio, mel_fb):
    """Convert raw audio to log-mel spectrogram."""
    window = torch.hann_window(N_FFT)
    spec = torch.stft(audio, N_FFT, HOP, window=window, return_complex=True)
    mag = spec.abs()
    mel = torch.matmul(mel_fb, mag)
    mel = torch.log(mel + 1e-8)
    return mel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['BC-ResNet-1', 'DS-CNN-S'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--chunk-ms', type=int, default=200)
    args = parser.parse_args()

    # Create model
    if args.model == 'DS-CNN-S':
        model = DSCNN_S()
    else:
        model = BCResNet(n_classes=12, scale=1)

    ckpt = torch.load(f'../checkpoints_full/{args.model}/best.pt',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    acc = ckpt.get('val_acc', 0)
    print(f"Model: {args.model} ({params:,} params, {acc:.2f}% val acc)")

    mel_fb = create_mel_fb()
    audio_queue = queue.Queue()
    chunk_samples = int(SR * args.chunk_ms / 1000)
    accum = torch.zeros(0)
    history = deque(maxlen=3)

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    print(f"\nListening... (say: yes, no, up, down, left, right, on, off, stop, go)")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                            blocksize=chunk_samples, callback=callback):
            while True:
                try:
                    chunk_np = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                chunk = torch.from_numpy(chunk_np).float()
                accum = torch.cat([accum, chunk])

                if len(accum) < SR:
                    bar = '#' * int(len(accum) / SR * 20) + '-' * (20 - int(len(accum) / SR * 20))
                    print(f"\r  Buffering [{bar}] {len(accum)/SR*1000:.0f}ms",
                          end='', flush=True)
                    continue

                audio = accum[-SR:].unsqueeze(0)
                mel = audio_to_mel(audio, mel_fb)

                with torch.no_grad():
                    logits = model(mel)
                    probs = torch.softmax(logits, dim=-1)[0]

                history.append(probs)
                avg = torch.stack(list(history)).mean(0)
                conf, idx = avg.max(0)
                label = LABELS[idx.item()]
                conf = conf.item()

                bar = '|' * int(conf * 20) + '.' * (20 - int(conf * 20))

                if conf >= args.threshold and label not in ('silence', 'unknown'):
                    print(f"\r  >>> DETECTED: {label:>10s}  [{bar}] {conf:.1%}  <<<")
                else:
                    print(f"\r  {label:>10s}  [{bar}] {conf:.1%}    ", end='', flush=True)

                if len(accum) > SR + chunk_samples:
                    accum = accum[-(SR + chunk_samples):]

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == '__main__':
    main()
