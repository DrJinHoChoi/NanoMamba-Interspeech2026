#!/usr/bin/env python3
"""
End-to-End Accuracy Verification: C SDK vs Python
Runs actual C executable and Python model on identical inputs, compares outputs.

Usage:
    python test/verify_accuracy.py [--model 7k|20k] [--n-samples 100]
"""

import sys
import os
import subprocess
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2].parent))
from nanomamba import create_nanomamba_nc_20k, create_nanomamba_nc_matched

CSDK_DIR = Path(__file__).parent.parent
CKPT_DIR = Path(__file__).parents[2].parent / 'checkpoints_full'
LABELS = ['yes','no','up','down','left','right','on','off','stop','go','silence','unknown']


def load_model(variant='20k'):
    if variant == '20k':
        model = create_nanomamba_nc_20k()
        ckpt_path = CKPT_DIR / 'NanoMamba-NC-20K' / 'best.pt'
    else:
        model = create_nanomamba_nc_matched()
        ckpt_path = CKPT_DIR / 'NC-SSM' / 'best.pt'

    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def run_c_inference(audio_np, exe_path):
    """Run C SDK on float32 audio, parse logits from stdout."""
    tmp = str(CSDK_DIR / '_verify_audio.raw')
    audio_np.astype(np.float32).tofile(tmp)

    result = subprocess.run(
        [str(exe_path), tmp],
        capture_output=True, text=True, timeout=10
    )
    os.remove(tmp)

    # Parse logits from output
    logits = {}
    latency = 0.0
    for line in result.stdout.split('\n'):
        line = line.strip()
        for label in LABELS:
            if label in line and 'logit=' in line:
                # Parse: "  * yes       : logit= -0.8857  prob=0.0272"
                parts = line.split('logit=')
                if len(parts) >= 2:
                    logit_str = parts[1].split('prob=')[0].strip()
                    logits[label] = float(logit_str)
        if 'Latency:' in line:
            latency = float(line.split()[1])

    logits_arr = np.array([logits.get(l, 0.0) for l in LABELS])
    return logits_arr, latency


def run_python_inference(model, audio_np):
    """Run Python model, return logits."""
    audio = torch.from_numpy(audio_np).unsqueeze(0)
    with torch.no_grad():
        logits = model(audio)
    return logits[0].numpy()


def generate_test_audio(seed, audio_type='noise', amplitude=0.01):
    """Generate various test audio patterns."""
    rng = np.random.RandomState(seed)

    if audio_type == 'noise':
        return (rng.randn(16000) * amplitude).astype(np.float32)
    elif audio_type == 'sine':
        freq = rng.uniform(200, 4000)
        t = np.arange(16000) / 16000.0
        return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)
    elif audio_type == 'chirp':
        t = np.arange(16000) / 16000.0
        freq = np.linspace(200, 4000, 16000)
        return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)
    elif audio_type == 'silence':
        return np.zeros(16000, dtype=np.float32)
    elif audio_type == 'impulse':
        audio = np.zeros(16000, dtype=np.float32)
        audio[8000] = amplitude * 10
        return audio
    elif audio_type == 'speech_like':
        # Simulated speech-like modulated noise
        t = np.arange(16000) / 16000.0
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # ~3Hz AM
        noise = rng.randn(16000)
        return (noise * envelope * amplitude).astype(np.float32)
    else:
        return (rng.randn(16000) * amplitude).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['7k', '20k'], default='20k')
    parser.add_argument('--n-samples', type=int, default=20)
    args = parser.parse_args()

    exe_path = CSDK_DIR / f'ncssm_{args.model}.exe'
    if not exe_path.exists():
        print(f"Error: {exe_path} not found")
        return

    print("=" * 70)
    print(f"  NC-SSM C SDK vs Python Accuracy Verification")
    print(f"  Model: NC-SSM-{args.model.upper()}")
    print(f"  C executable: {exe_path.name}")
    print("=" * 70)

    model = load_model(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Python model: {n_params:,} params\n")

    audio_types = ['noise', 'sine', 'chirp', 'silence', 'impulse', 'speech_like']
    amplitudes = [0.001, 0.01, 0.05, 0.1]

    results = []
    match_count = 0
    total = 0

    for seed in range(args.n_samples):
        atype = audio_types[seed % len(audio_types)]
        amp = amplitudes[seed % len(amplitudes)]
        audio = generate_test_audio(seed, atype, amp)

        # Run both
        py_logits = run_python_inference(model, audio)
        c_logits, c_latency = run_c_inference(audio, exe_path)

        py_pred = LABELS[np.argmax(py_logits)]
        c_pred = LABELS[np.argmax(c_logits)]

        # Logit-level comparison
        logit_diff = np.abs(py_logits - c_logits)
        max_diff = logit_diff.max()
        mean_diff = logit_diff.mean()

        match = (py_pred == c_pred)
        if match:
            match_count += 1
        total += 1

        status = "OK" if match else "MISS"
        print(f"  [{total:3d}] {atype:12s} amp={amp:.3f} | "
              f"Py={py_pred:8s} C={c_pred:8s} {status} | "
              f"dLogit: max={max_diff:.4f} mean={mean_diff:.4f} | "
              f"C:{c_latency:.0f}ms")

        results.append({
            'type': atype, 'amp': amp,
            'py_pred': py_pred, 'c_pred': c_pred,
            'match': match, 'max_diff': max_diff, 'mean_diff': mean_diff,
            'c_latency': c_latency
        })

    # Summary
    print("\n" + "=" * 70)
    print(f"  SUMMARY")
    print(f"  Prediction agreement: {match_count}/{total} ({match_count/total*100:.1f}%)")

    all_max = max(r['max_diff'] for r in results)
    all_mean = np.mean([r['mean_diff'] for r in results])
    avg_latency = np.mean([r['c_latency'] for r in results])

    print(f"  Logit max diff:  {all_max:.6f}")
    print(f"  Logit mean diff: {all_mean:.6f}")
    print(f"  C avg latency:   {avg_latency:.1f} ms")

    if all_max < 0.1:
        print(f"\n  PASS: C SDK matches Python within tolerance (max delta < 0.1)")
    elif all_max < 1.0:
        print(f"\n  WARN: C SDK has moderate deviation (max delta = {all_max:.4f})")
    else:
        print(f"\n  FAIL: C SDK has large deviation (max delta = {all_max:.4f})")

    # Mismatches
    mismatches = [r for r in results if not r['match']]
    if mismatches:
        print(f"\n  Mismatched predictions ({len(mismatches)}):")
        for r in mismatches:
            print(f"    {r['type']:12s} amp={r['amp']:.3f}: Py={r['py_pred']} vs C={r['c_pred']} (d={r['max_diff']:.4f})")

    print("=" * 70)


if __name__ == '__main__':
    main()
