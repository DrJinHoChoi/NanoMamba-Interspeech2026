#!/usr/bin/env python3
"""
Nano AI SDK - Live Microphone Demo
===================================
Real-time keyword spotting from your laptop microphone.

Usage:
    python demo_mic.py
    python demo_mic.py --model ncssm-large
    python demo_mic.py --model ncssm --threshold 0.4 --checkpoint path/to/model.pt

Commands (12-class GSC V2):
    yes, no, up, down, left, right, on, off, stop, go

Press Ctrl+C to stop.
"""

import argparse
import sys
import time
import threading
import queue

import torch
import sounddevice as sd
import numpy as np

# Add parent dir for nano_ssm
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import nano_ssm
from nano_ssm.streaming import StreamingEngine


def main():
    parser = argparse.ArgumentParser(description='NC-SSM Live Microphone Demo')
    parser.add_argument('--model', type=str, default='ncssm',
                        help='Model variant (default: ncssm). Use list_models() for options, or "auto" to detect from checkpoint.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--chunk-ms', type=int, default=100,
                        help='Audio chunk size in ms (default: 100)')
    parser.add_argument('--cooldown', type=int, default=5,
                        help='Cooldown chunks between detections (default: 5)')
    parser.add_argument('--device-id', type=int, default=None,
                        help='Audio input device ID (default: system default)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Create model - auto-detect from checkpoint if needed
    model_name = args.model
    if args.checkpoint and (model_name == 'auto' or model_name == 'ncssm'):
        # Try to auto-detect model from checkpoint
        import torch as _torch
        ckpt = _torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        ckpt_name = ckpt.get('model_name', '')
        # Map checkpoint model_name to factory
        CKPT_TO_FACTORY = {
            'NanoMamba-NC-Matched': 'ncssm',
            'NanoMamba-NC-Large': 'ncssm-large',
            'NanoMamba-NC-Large-NASG': 'ncssm-large',
            'NanoMamba-NC-15K': 'ncssm-15k',
            'NanoMamba-NC-20K': 'ncssm-20k',
            'NanoMamba-NC-NanoSE': 'ncssm-nanose',
        }
        if ckpt_name in CKPT_TO_FACTORY:
            model_name = CKPT_TO_FACTORY[ckpt_name]
            print(f"Auto-detected model: {ckpt_name} -> {model_name}")

    print(f"Loading {model_name}...", end=' ', flush=True)
    kwargs = {}
    if model_name == 'ncssm-large' and args.checkpoint and 'NASG' in args.checkpoint:
        kwargs['use_nasg'] = True
    model = nano_ssm.create(model_name, pretrained=args.checkpoint, **kwargs)
    print("OK")
    print(model.summary())
    print()

    # Use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        print("Using CPU")

    # Create streaming engine
    sr = 16000
    engine = StreamingEngine(
        model,
        chunk_ms=args.chunk_ms,
        sr=sr,
        confidence_threshold=args.threshold,
        cooldown_chunks=args.cooldown,
    )

    # Audio queue for thread-safe communication
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"  [Audio status: {status}]", file=sys.stderr)
        # Convert to mono float32 tensor
        audio = indata[:, 0].copy()  # (frames,)
        audio_queue.put(audio)

    # Print header
    print()
    print("=" * 60)
    print("  NC-SSM Live Keyword Spotting")
    print(f"  Model: {args.model} ({model.n_params:,} params)")
    print(f"  Threshold: {args.threshold}")
    print(f"  Chunk: {args.chunk_ms}ms")
    print("=" * 60)
    print()
    print("Listening... (say: yes, no, up, down, left, right, on, off, stop, go)")
    print("Press Ctrl+C to stop.")
    print()

    chunk_samples = int(sr * args.chunk_ms / 1000)

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype='float32',
            blocksize=chunk_samples,
            device=args.device_id,
            callback=audio_callback,
        ):
            while True:
                try:
                    audio_np = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                chunk = torch.from_numpy(audio_np).float()
                result = engine.feed(chunk)

                if result is None:
                    # Still accumulating
                    bar_len = int(engine.buffer_duration_ms / 1000 * 20)
                    bar = '#' * bar_len + '-' * (20 - bar_len)
                    print(f"\r  Buffering [{bar}] {engine.buffer_duration_ms:.0f}ms",
                          end='', flush=True)
                    continue

                # Show live prediction
                label = result['smoothed_label']
                conf = result['smoothed_confidence']

                # Color-coded confidence bar
                bar_len = int(conf * 20)
                bar = '|' * bar_len + '.' * (20 - bar_len)

                if result.get('detected', False):
                    # Detection! Show prominently
                    print(f"\r  >>> DETECTED: {label:>10s}  [{bar}] {conf:.1%}  <<<")
                else:
                    # Show live status
                    print(f"\r  {label:>10s}  [{bar}] {conf:.1%}    ",
                          end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTip: Run 'python demo_mic.py --list-devices' to see available audio devices.")
        print("     Use '--device-id N' to select a specific device.")


if __name__ == '__main__':
    main()
