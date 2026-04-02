#!/usr/bin/env python3
"""Verify C vs Python on REAL audio samples (keyword recordings)."""
import sys, os, subprocess, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2].parent))
from nanomamba import create_nanomamba_nc_20k, create_nanomamba_nc_matched

CSDK_DIR = Path(__file__).parent.parent
CKPT_DIR = Path(__file__).parents[2].parent / 'checkpoints_full'
LABELS = ['yes','no','up','down','left','right','on','off','stop','go','silence','unknown']
AUDIO_DIR = Path(__file__).parent

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

def run_c(audio_path, exe_path):
    result = subprocess.run([str(exe_path), str(audio_path)],
                          capture_output=True, text=True, timeout=10)
    label, conf, latency = 'unknown', 0.0, 0.0
    logits = {}
    for line in result.stdout.split('\n'):
        for l in LABELS:
            if l in line and 'logit=' in line:
                parts = line.split('logit=')
                if len(parts) >= 2:
                    logits[l] = float(parts[1].split('prob=')[0].strip())
        if 'Prediction:' in line:
            parts = line.split()
            label = parts[1]
            conf = float(parts[2].strip('(%)')) / 100
        if 'Latency:' in line:
            latency = float(line.split()[1])
    logits_arr = np.array([logits.get(l, 0.0) for l in LABELS])
    return label, conf, logits_arr, latency

def run_py(model, audio_np):
    audio = torch.from_numpy(audio_np).unsqueeze(0)
    with torch.no_grad():
        logits = model(audio)
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    idx = probs.argmax()
    return LABELS[idx], float(probs[idx]), logits[0].numpy()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['7k','20k'], default='20k')
    args = ap.parse_args()

    exe = CSDK_DIR / f'ncssm_{args.model}.exe'
    model = load_model(args.model)

    # Find real audio files
    audio_files = sorted(AUDIO_DIR.glob('audio_*.raw'))
    if not audio_files:
        print("No audio_*.raw test files found!")
        return

    print("=" * 70)
    print(f"  NC-SSM C vs Python - Real Audio Verification (NC-SSM-{args.model.upper()})")
    print("=" * 70)

    match_count = 0
    total = 0
    for af in audio_files:
        expected = af.stem.replace('audio_', '')
        audio_np = np.fromfile(str(af), dtype=np.float32)
        if len(audio_np) < 16000:
            audio_np = np.pad(audio_np, (0, 16000 - len(audio_np)))
        elif len(audio_np) > 16000:
            audio_np = audio_np[:16000]

        py_label, py_conf, py_logits = run_py(model, audio_np)
        c_label, c_conf, c_logits, c_lat = run_c(af, exe)

        diff = np.abs(py_logits - c_logits)
        match = (py_label == c_label)
        if match: match_count += 1
        total += 1

        status = "OK" if match else "MISS"
        correct_py = "v" if py_label == expected else "x"
        correct_c = "v" if c_label == expected else "x"
        print(f"  {expected:>8s} | Py={py_label:>8s}({py_conf:.2f}){correct_py} "
              f"C={c_label:>8s}({c_conf:.2f}){correct_c} {status} | "
              f"logit_max_d={diff.max():.3f} mean_d={diff.mean():.3f} C:{c_lat:.0f}ms")

    print(f"\n  Agreement: {match_count}/{total} ({match_count/total*100:.0f}%)")
    print("=" * 70)

if __name__ == '__main__':
    main()
