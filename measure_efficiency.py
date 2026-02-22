#!/usr/bin/env python3
"""
Measure MACs, parameters, and latency for all KWS models.

Uses forward hooks for MACs counting (avoids thop compatibility issues
with NanoMamba's custom STFT/padding operations).

Usage:
    python measure_efficiency.py                    # CPU only
    python measure_efficiency.py --device cuda      # GPU
    python measure_efficiency.py --device both      # CPU + GPU
"""

import argparse
import time
import torch
import torch.nn as nn


def count_macs(model, input_tensor):
    """Count MACs using forward hooks on Linear and Conv layers.

    Counts multiply-accumulate operations from nn.Linear, nn.Conv1d, nn.Conv2d.
    Does NOT count element-wise ops (sigmoid, tanh, etc.) or custom ops (STFT).

    Returns:
        int: Total MACs (multiply-accumulate operations)
    """
    total_macs = [0]
    hooks = []

    def linear_hook(module, inp, out):
        x = inp[0]
        seq = x.shape[1] if x.dim() == 3 else 1
        total_macs[0] += seq * module.in_features * module.out_features

    def conv1d_hook(module, inp, out):
        out_len = out.shape[2]
        total_macs[0] += (
            (module.in_channels // module.groups)
            * module.kernel_size[0]
            * module.out_channels
            * out_len
        )

    def conv2d_hook(module, inp, out):
        out_h, out_w = out.shape[2], out.shape[3]
        total_macs[0] += (
            (module.in_channels // module.groups)
            * module.kernel_size[0]
            * module.kernel_size[1]
            * module.out_channels
            * out_h
            * out_w
        )

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return total_macs[0]


def measure_latency(model, input_tensor, n_runs=200, warmup=50):
    """Measure inference latency with warmup.

    Args:
        model: PyTorch model (already on correct device)
        input_tensor: Input tensor (already on correct device)
        n_runs: Number of inference runs for timing
        warmup: Number of warmup runs

    Returns:
        float: Average latency in milliseconds
    """
    device = next(model.parameters()).device
    is_cuda = device.type == 'cuda'

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(input_tensor)
        if is_cuda:
            torch.cuda.synchronize()

        # Timed runs
        start = time.time()
        for _ in range(n_runs):
            model(input_tensor)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - start

    return elapsed / n_runs * 1000  # ms


def get_input_tensor(model, device):
    """Get appropriate input tensor based on model type.

    NanoMamba models: raw audio (B, T) = (1, 16000)
    CNN models:       mel spectrogram (B, n_mels, T) = (1, 40, 98)
    """
    if hasattr(model, 'snr_estimator'):
        # NanoMamba: raw audio input
        return torch.randn(1, 16000, device=device)
    else:
        # CNN models (DS-CNN-S, BC-ResNet, etc.): mel spectrogram
        return torch.randn(1, 40, 98, device=device)


def main():
    parser = argparse.ArgumentParser(description='Measure KWS model efficiency')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'both'],
                        help='Device for latency measurement')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: all 4)')
    parser.add_argument('--n_runs', type=int, default=200,
                        help='Number of inference runs for latency')
    args = parser.parse_args()

    # Import models
    from train_all_models import create_all_models
    all_models = create_all_models()

    # Select models
    if args.models:
        selected = [m.strip() for m in args.models.split(',')]
    else:
        selected = ['NanoMamba-Small', 'NanoMamba-Tiny',
                    'NanoMamba-Small-FF', 'NanoMamba-Tiny-FF',
                    'NanoMamba-Tiny-WS', 'NanoMamba-Tiny-WS-FF',
                    'DS-CNN-S', 'BC-ResNet-1']

    devices = []
    if args.device in ('cpu', 'both'):
        devices.append(torch.device('cpu'))
    if args.device in ('cuda', 'both'):
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        else:
            print("  CUDA not available, skipping GPU measurement")

    # Header
    lat_cols = ' '.join([f'{"CPU(ms)":>10}' if d.type == 'cpu' else f'{"GPU(ms)":>10}'
                         for d in devices])
    print(f"\n  {'Model':<22} {'Params':>8} {'INT8(KB)':>9} {'MACs(M)':>9} {lat_cols}")
    print("  " + "=" * (55 + 11 * len(devices)))

    for name in selected:
        if name not in all_models:
            print(f"  {name:<22} -- not found, skipping")
            continue

        model = all_models[name].eval()
        params = sum(p.numel() for p in model.parameters())
        int8_kb = params / 1024  # 1 byte per param in INT8

        # MACs (always on CPU)
        model_cpu = model.to('cpu')
        inp_cpu = get_input_tensor(model_cpu, torch.device('cpu'))
        macs = count_macs(model_cpu, inp_cpu)

        # Latency per device
        latencies = []
        for device in devices:
            model_dev = model.to(device)
            inp_dev = get_input_tensor(model_dev, device)
            lat = measure_latency(model_dev, inp_dev, n_runs=args.n_runs)
            latencies.append(lat)

        lat_str = ' '.join([f'{lat:>10.2f}' for lat in latencies])
        print(f"  {name:<22} {params/1e3:>7.1f}K {int8_kb:>8.1f} {macs/1e6:>8.2f}M {lat_str}")

    # Input shape info
    print(f"\n  Note: NanoMamba input = raw audio (1, 16000)")
    print(f"        CNN input = mel spectrogram (1, 40, 98)")
    print(f"        MACs count Linear + Conv layers only (excludes STFT, elementwise ops)")


if __name__ == '__main__':
    main()
