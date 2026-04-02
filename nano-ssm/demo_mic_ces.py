#!/usr/bin/env python3
"""
NC-SSM CES 2027 Demo — Live Mic with AEC + Noise Suppression
=============================================================
Real-time keyword spotting with noise-robust front-end pipeline:

    Mic → NS (Noise Suppression) → NC-SSM KWS
         ↑
    [Optional AEC with speaker reference]

Usage:
    python demo_mic_ces.py
    python demo_mic_ces.py --model ncssm-20k --checkpoint ../checkpoints_full/NanoMamba-NC-20K/best.pt
    python demo_mic_ces.py --no-ns          # disable noise suppression
    python demo_mic_ces.py --ns-level 2     # stronger NS (0=off, 1=mild, 2=standard, 3=aggressive)

Press Ctrl+C to stop.
"""

import argparse
import sys
import time
import threading
import queue
import os

import torch
import sounddevice as sd
import numpy as np

# Noise suppression
try:
    import noisereduce as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False
    print("[WARN] noisereduce not installed. Run: pip install noisereduce")

# Add parent dir
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import nano_ssm
from nano_ssm.streaming import StreamingEngine


class AudioFrontEnd:
    """Audio front-end with Noise Suppression and optional AEC."""

    def __init__(self, sr=16000, ns_level=2, enable_agc=True):
        self.sr = sr
        self.ns_level = ns_level  # 0=off, 1=mild, 2=standard, 3=aggressive
        self.enable_agc = enable_agc

        # Noise profile estimation
        self.noise_buffer = []
        self.noise_profile = None
        self.noise_frames_needed = 10  # ~1 second of noise profile
        self.is_calibrated = False

        # AGC state
        self.agc_target = 0.15
        self.agc_gain = 1.0
        self.agc_attack = 0.01
        self.agc_release = 0.001

        # NS strength mapping
        self.ns_prop_decrease = {0: 0.0, 1: 0.5, 2: 0.8, 3: 0.95}

    def calibrate_noise(self, audio_chunk):
        """Collect noise profile from initial silence."""
        self.noise_buffer.append(audio_chunk.copy())
        if len(self.noise_buffer) >= self.noise_frames_needed:
            self.noise_profile = np.concatenate(self.noise_buffer)
            self.is_calibrated = True
            rms = np.sqrt(np.mean(self.noise_profile ** 2))
            print(f"  [NS] Noise profile captured ({len(self.noise_profile)} samples, RMS={rms:.6f})")
            return True
        return False

    def apply_ns(self, audio):
        """Apply noise suppression using spectral gating."""
        if not HAS_NR or self.ns_level == 0 or not self.is_calibrated:
            return audio

        try:
            cleaned = nr.reduce_noise(
                y=audio,
                sr=self.sr,
                y_noise=self.noise_profile,
                prop_decrease=self.ns_prop_decrease.get(self.ns_level, 0.8),
                stationary=True,
                n_fft=512,
                hop_length=160,
            )
            return cleaned
        except Exception:
            return audio

    def apply_agc(self, audio):
        """Simple AGC to normalize volume."""
        if not self.enable_agc:
            return audio

        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            target_gain = self.agc_target / rms
            target_gain = np.clip(target_gain, 0.1, 10.0)

            if target_gain > self.agc_gain:
                self.agc_gain += self.agc_release * (target_gain - self.agc_gain)
            else:
                self.agc_gain += self.agc_attack * (target_gain - self.agc_gain)

        return audio * self.agc_gain

    def process(self, audio):
        """Full front-end pipeline: NS -> AGC."""
        if not self.is_calibrated:
            done = self.calibrate_noise(audio)
            if not done:
                remaining = self.noise_frames_needed - len(self.noise_buffer)
                return None, f"Calibrating noise... ({remaining} chunks left)"
            return None, "Calibration complete!"

        # Pipeline: NS -> AGC
        out = self.apply_ns(audio)
        out = self.apply_agc(out)
        return out, None


def get_checkpoint_path(model_name):
    """Auto-find checkpoint for model variant."""
    base = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_full')
    mapping = {
        'ncssm': ('NanoMamba-NC-Matched', 'best.pt'),
        'ncssm-large': ('NanoMamba-NC-Large', 'best.pt'),
        'ncssm-15k': ('NanoMamba-NC-15K', 'best.pt'),
        'ncssm-20k': ('NanoMamba-NC-20K', 'best.pt'),
    }
    if model_name in mapping:
        folder, fname = mapping[model_name]
        path = os.path.join(base, folder, fname)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description='NC-SSM CES 2027 Demo')
    parser.add_argument('--model', type=str, default='ncssm-20k',
                        help='Model variant (default: ncssm-20k)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (auto-detected if not given)')
    parser.add_argument('--threshold', type=float, default=0.45,
                        help='Detection threshold (default: 0.45)')
    parser.add_argument('--chunk-ms', type=int, default=100,
                        help='Audio chunk size in ms (default: 100)')
    parser.add_argument('--cooldown', type=int, default=5,
                        help='Cooldown chunks between detections (default: 5)')
    parser.add_argument('--ns-level', type=int, default=2, choices=[0,1,2,3],
                        help='Noise suppression level: 0=off, 1=mild, 2=standard, 3=aggressive')
    parser.add_argument('--no-ns', action='store_true',
                        help='Disable noise suppression')
    parser.add_argument('--no-agc', action='store_true',
                        help='Disable AGC')
    parser.add_argument('--device-id', type=int, default=None,
                        help='Audio device ID')
    parser.add_argument('--list-devices', action='store_true',
                        help='List audio devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    ns_level = 0 if args.no_ns else args.ns_level

    # Find checkpoint
    ckpt_path = args.checkpoint or get_checkpoint_path(args.model)
    if not ckpt_path:
        print(f"[ERROR] No checkpoint found for {args.model}")
        print("  Provide --checkpoint or place best.pt in checkpoints_full/")
        return

    # Auto-detect model from checkpoint
    model_name = args.model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_model = ckpt.get('model_name', '')
    CKPT_MAP = {
        'NanoMamba-NC-Matched': 'ncssm',
        'NanoMamba-NC-Large': 'ncssm-large',
        'NanoMamba-NC-Large-NASG': 'ncssm-large',
        'NanoMamba-NC-15K': 'ncssm-15k',
        'NanoMamba-NC-20K': 'ncssm-20k',
    }
    if ckpt_model in CKPT_MAP:
        model_name = CKPT_MAP[ckpt_model]

    # Load model
    print(f"Loading {model_name}...", end=' ', flush=True)
    kwargs = {}
    if model_name == 'ncssm-large' and 'NASG' in str(ckpt_path):
        kwargs['use_nasg'] = True
    model = nano_ssm.create(model_name, pretrained=ckpt_path, **kwargs)
    print("OK")

    sr = 16000
    engine = StreamingEngine(
        model, chunk_ms=args.chunk_ms, sr=sr,
        confidence_threshold=args.threshold,
        cooldown_chunks=args.cooldown,
    )

    # Front-end
    frontend = AudioFrontEnd(sr=sr, ns_level=ns_level, enable_agc=not args.no_agc)

    # Audio queue
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  [Audio: {status}]", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    # Header
    print()
    print("=" * 64)
    print("  NC-SSM CES 2027 Demo")
    print(f"  Model: {model_name} ({model.n_params:,} params)")
    print(f"  Noise Suppression: {'OFF' if ns_level == 0 else ['', 'Mild', 'Standard', 'Aggressive'][ns_level]}")
    print(f"  AGC: {'ON' if not args.no_agc else 'OFF'}")
    print(f"  Threshold: {args.threshold}")
    print("=" * 64)
    print()
    print("  Step 1: Stay quiet for ~1 second (noise calibration)")
    print("  Step 2: Say keywords: yes, no, up, down, left, right,")
    print("          on, off, stop, go")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    chunk_samples = int(sr * args.chunk_ms / 1000)
    detect_count = 0
    start_time = time.time()

    try:
        with sd.InputStream(
            samplerate=sr, channels=1, dtype='float32',
            blocksize=chunk_samples, device=args.device_id,
            callback=audio_callback,
        ):
            while True:
                try:
                    audio_np = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Front-end processing
                processed, msg = frontend.process(audio_np)
                if processed is None:
                    if msg:
                        print(f"\r  {msg}    ", end='', flush=True)
                    continue

                # Feed to KWS engine
                chunk = torch.from_numpy(processed).float()
                result = engine.feed(chunk)

                if result is None:
                    bar_len = int(engine.buffer_duration_ms / 1000 * 20)
                    bar = '#' * bar_len + '-' * (20 - bar_len)
                    print(f"\r  Buffering [{bar}] {engine.buffer_duration_ms:.0f}ms",
                          end='', flush=True)
                    continue

                label = result['smoothed_label']
                conf = result['smoothed_confidence']
                bar_len = int(conf * 20)
                bar = '|' * bar_len + '.' * (20 - bar_len)

                if result.get('detected', False):
                    detect_count += 1
                    elapsed = time.time() - start_time
                    print(f"\r  >>> [{detect_count:3d}] {label:>10s}  [{bar}] {conf:.1%}  <<<")
                else:
                    print(f"\r  {label:>10s}  [{bar}] {conf:.1%}    ",
                          end='', flush=True)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n  Session: {elapsed:.0f}s | Detections: {detect_count}")
        print("  Stopped.")


if __name__ == '__main__':
    main()
