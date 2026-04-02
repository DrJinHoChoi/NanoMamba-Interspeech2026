#!/usr/bin/env python3
"""Compare C and Python intermediate outputs side by side."""
import numpy as np
from pathlib import Path

DEBUG = Path(__file__).parent / 'debug_data'

def load(name, shape=None):
    path = DEBUG / f'{name}.raw'
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.float32)
    if shape:
        data = data.reshape(shape)
    return data

def compare(name, py_name=None, c_name=None, shape=None):
    if py_name is None: py_name = name
    if c_name is None: c_name = f'{name}_c'

    py = load(py_name, shape)
    c = load(c_name, shape)

    if py is None:
        print(f"  {name}: Python data not found")
        return
    if c is None:
        print(f"  {name}: C data not found")
        return

    mn = min(len(py.flat), len(c.flat))
    py_flat = py.flat[:mn]
    c_flat = c.flat[:mn]

    diff = np.abs(py_flat - c_flat)
    print(f"  {name:24s}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, "
          f"py_range=[{py_flat.min():.4f},{py_flat.max():.4f}], "
          f"c_range=[{c_flat.min():.4f},{c_flat.max():.4f}]")

    # Find worst element
    idx = diff.argmax()
    print(f"    worst at idx={idx}: py={py_flat[idx]:.6f}, c={c_flat[idx]:.6f}")

print("=== C vs Python Stage Comparison ===\n")
compare('stft_mag')
compare('raw_mel')
compare('snr_mel', 'snr_estimator_output')
compare('after_lsg')
compare('after_pcen')
compare('after_instnorm')
compare('patch_proj')
compare('block0_norm')
compare('block0_ssm_input', 'block0_ssm_input')
compare('block0_ssm_output', 'block0_ssm_output')
compare('block0_gated', 'block0_gated')
compare('block0_full_output', 'block0_full_output')
compare('logits')
