#!/usr/bin/env python3
"""
NC-SSM C SDK - Live Microphone Streaming Demo
Uses Python for mic capture, C for inference (via ctypes or subprocess).

Since we can't easily load .exe as ctypes on Windows, we use subprocess:
  1. Python captures 1-second audio chunks via sounddevice
  2. Saves as raw float32
  3. Calls test_ncssm.exe with the file
  4. Displays result

Usage:
    python demo_mic_c.py
"""

import sys
import os
import time
import queue
import subprocess
import struct
import tempfile
import numpy as np
import sounddevice as sd

SR = 16000
CHUNK_MS = 200
CHUNK_SAMPLES = int(SR * CHUNK_MS / 1000)
LABELS = ['yes','no','up','down','left','right','on','off','stop','go','silence','unknown']

# Path to C executable
CSDK_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_DEFAULT = os.path.join(CSDK_DIR, 'ncssm_7k.exe')

audio_queue = queue.Queue()
accum = np.zeros(0, dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())


def run_c_inference(audio_1sec, exe_path):
    """Run C SDK on 1-second audio, return (label, confidence, latency_ms)."""
    # Write raw audio to temp file
    tmp = os.path.join(CSDK_DIR, '_mic_audio.raw')
    audio_1sec.astype(np.float32).tofile(tmp)

    # Call C executable
    result = subprocess.run(
        [exe_path, tmp],
        capture_output=True, text=True, timeout=5
    )

    # Parse output
    label = 'unknown'
    conf = 0.0
    latency = 0.0
    for line in result.stdout.split('\n'):
        if 'Prediction:' in line:
            parts = line.split()
            label = parts[1]
            conf = float(parts[2].strip('(%)')) / 100
        if 'Latency:' in line:
            latency = float(line.split()[1])

    return label, conf, latency


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['7k', '20k'], default='7k')
    args = ap.parse_args()
    EXE_PATH = os.path.join(CSDK_DIR, f'ncssm_{args.model}.exe')

    if not os.path.exists(EXE_PATH):
        print(f"Error: {EXE_PATH} not found. Build first:")
        print("  python -m ziglang cc -O2 -o test_ncssm.exe test/test_ncssm.c src/*.c -Iinclude -lm")
        return

    print("=" * 60)
    print("  NC-SSM C SDK - Live Microphone Streaming")
    print(f"  Engine: C ({os.path.basename(EXE_PATH)})")
    print(f"  Chunk: {CHUNK_MS}ms, SR: {SR}Hz")
    print("=" * 60)
    print()
    print("Listening... (say: yes, no, up, down, left, right, on, off, stop, go)")
    print("Press Ctrl+C to stop.")
    print()

    global accum

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                            blocksize=CHUNK_SAMPLES, callback=audio_callback):
            while True:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                accum = np.concatenate([accum, chunk])

                if len(accum) < SR:
                    bar_len = int(len(accum) / SR * 20)
                    bar = '#' * bar_len + '-' * (20 - bar_len)
                    print(f"\r  Buffering [{bar}] {len(accum)/SR*1000:.0f}ms",
                          end='', flush=True)
                    continue

                # Take last 1 second
                audio_1sec = accum[-SR:]

                # Run C inference
                label, conf, c_latency = run_c_inference(audio_1sec, EXE_PATH)

                bar_len = int(conf * 20)
                bar = '|' * bar_len + '.' * (20 - bar_len)

                if conf >= 0.5 and label not in ('silence', 'unknown'):
                    print(f"\r  >>> {label:>10s}  [{bar}] {conf:.1%}  C:{c_latency:.0f}ms  <<<")
                else:
                    print(f"\r  {label:>10s}  [{bar}] {conf:.1%}  C:{c_latency:.0f}ms    ",
                          end='', flush=True)

                # Trim buffer
                if len(accum) > SR + CHUNK_SAMPLES:
                    accum = accum[-(SR + CHUNK_SAMPLES):]

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        tmp = os.path.join(CSDK_DIR, '_mic_audio.raw')
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == '__main__':
    main()
