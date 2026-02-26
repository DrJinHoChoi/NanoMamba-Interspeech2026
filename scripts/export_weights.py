#!/usr/bin/env python3
"""
NanoMamba Weight Export Script
================================
Exports trained PyTorch model weights to INT8 hex format for FPGA/ASIC.

Weight Memory Map (4,736 bytes total):
  0x0000 - 0x020F : SNR estimator weights (~520 bytes)
  0x0210 - 0x02CF : PCEN Expert 0 params (160 bytes, delta=2.0)
  0x02D0 - 0x038F : PCEN Expert 1 params (160 bytes, delta=0.01)
  0x0390 - 0x0391 : gate_temp (2 bytes, FP16)
  0x0400 - 0x06FF : Block 0 weights (in_proj, conv1d, x_proj, snr_proj)
  0x0700 - 0x09FF : Block 1 weights (same layout, weight sharing → identical)
  0x0A00 - 0x0A4F : patch_proj (40*16 = 640 bytes)
  0x0A50 - 0x0AFF : classifier (16*12 = 192 bytes + 12 bias)
  0x0B00 - 0x127F : reserved / LUT tables

Output format: Verilog $readmemh compatible (hex values, one per line)

Author: Jin Ho Choi, Ph.D.
"""

import torch
import numpy as np
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def quantize_to_int8(tensor, scale=None):
    """Quantize float tensor to INT8 with symmetric quantization"""
    if scale is None:
        abs_max = tensor.abs().max().item()
        if abs_max == 0:
            scale = 1.0
        else:
            scale = 127.0 / abs_max

    quantized = torch.clamp(torch.round(tensor * scale), -128, 127).to(torch.int8)
    return quantized, scale


def float_to_fp16_bytes(val):
    """Convert float to FP16 (2 bytes)"""
    fp16 = np.float16(val)
    return int.from_bytes(fp16.tobytes(), byteorder='little')


def export_weights(checkpoint_path, output_path, format='hex8'):
    """Export model weights to memory file"""

    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Creating dummy weights for RTL simulation...")
        create_dummy_weights(output_path)
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Model keys: {len(state_dict)}")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape}")

    # Memory layout
    WEIGHT_DEPTH = 4736
    mem = np.zeros(WEIGHT_DEPTH, dtype=np.int8)

    # ---- Map PyTorch weights to memory addresses ----
    addr = 0x0000

    # 1. SNR estimator weights (placeholder — depends on model)
    print(f"\n[0x{addr:04X}] SNR estimator weights")
    # Skip for now, fill with zeros

    # 2. PCEN Expert 0 params: s, alpha, delta, r × 40 channels
    addr = 0x0210
    print(f"[0x{addr:04X}] PCEN Expert 0 params (160 bytes)")
    for key in state_dict:
        if 'dual_pcen' in key and 'expert_0' in key:
            data, scale = quantize_to_int8(state_dict[key].flatten())
            n = min(len(data), 160)
            for i in range(n):
                if addr + i < WEIGHT_DEPTH:
                    mem[addr + i] = data[i].item()

    # 3. PCEN Expert 1 params
    addr = 0x02D0
    print(f"[0x{addr:04X}] PCEN Expert 1 params (160 bytes)")
    for key in state_dict:
        if 'dual_pcen' in key and 'expert_1' in key:
            data, scale = quantize_to_int8(state_dict[key].flatten())
            n = min(len(data), 160)
            for i in range(n):
                if addr + i < WEIGHT_DEPTH:
                    mem[addr + i] = data[i].item()

    # 4. gate_temp (FP16)
    addr = 0x0390
    print(f"[0x{addr:04X}] gate_temp (2 bytes FP16)")
    gate_temp = 5.0  # Default
    for key in state_dict:
        if 'gate_temp' in key:
            gate_temp = state_dict[key].item()
    fp16_val = float_to_fp16_bytes(gate_temp)
    mem[addr] = fp16_val & 0xFF
    mem[addr + 1] = (fp16_val >> 8) & 0xFF
    print(f"  gate_temp = {gate_temp} → FP16: 0x{fp16_val:04X}")

    # 5. Block 0 weights
    addr = 0x0400
    print(f"[0x{addr:04X}] Block 0 weights")
    block_keys = ['in_proj', 'conv1d', 'x_proj', 'snr_proj', 'dt_proj', 'out_proj',
                  'A_log', 'D']
    for key in sorted(state_dict.keys()):
        if 'blocks.0' in key or 'block.0' in key:
            data, scale = quantize_to_int8(state_dict[key].flatten())
            n = len(data)
            for i in range(n):
                if addr + i < WEIGHT_DEPTH:
                    mem[addr + i] = data[i].item()
            print(f"  {key}: {n} bytes @ 0x{addr:04X} (scale={scale:.4f})")
            addr += n

    # 6. Block 1 weights (may be same as Block 0 for weight sharing)
    addr = 0x0700
    print(f"[0x{addr:04X}] Block 1 weights")
    for key in sorted(state_dict.keys()):
        if 'blocks.1' in key or 'block.1' in key:
            data, scale = quantize_to_int8(state_dict[key].flatten())
            n = len(data)
            for i in range(n):
                if addr + i < WEIGHT_DEPTH:
                    mem[addr + i] = data[i].item()
            print(f"  {key}: {n} bytes @ 0x{addr:04X}")
            addr += n

    # 7. Patch projection
    addr = 0x0A00
    print(f"[0x{addr:04X}] Patch projection")
    for key in state_dict:
        if 'patch_proj' in key or 'patch_embed' in key:
            data, scale = quantize_to_int8(state_dict[key].flatten())
            n = len(data)
            for i in range(n):
                if addr + i < WEIGHT_DEPTH:
                    mem[addr + i] = data[i].item()
            print(f"  {key}: {n} bytes")

    # 8. Classifier
    addr = 0x0A50
    print(f"[0x{addr:04X}] Classifier")
    for key in sorted(state_dict.keys()):
        if 'classifier' in key or 'head' in key or 'fc' in key:
            data, scale = quantize_to_int8(state_dict[key].flatten())
            n = len(data)
            for i in range(n):
                if addr + i < WEIGHT_DEPTH:
                    mem[addr + i] = data[i].item()
            print(f"  {key}: {n} bytes @ 0x{addr:04X}")
            addr += n

    # Write output
    write_weight_mem(output_path, mem, format)

    # Summary
    nonzero = np.count_nonzero(mem)
    print(f"\n{'='*50}")
    print(f"  Weight export complete")
    print(f"  Total size: {WEIGHT_DEPTH} bytes ({WEIGHT_DEPTH/1024:.1f} KB)")
    print(f"  Non-zero:   {nonzero} bytes")
    print(f"  Output:     {output_path}")
    print(f"{'='*50}")


def create_dummy_weights(output_path):
    """Create dummy weights for simulation testing"""
    WEIGHT_DEPTH = 4736
    mem = np.zeros(WEIGHT_DEPTH, dtype=np.int8)

    # Fill with pseudo-random INT8 values for simulation
    np.random.seed(42)
    mem[:] = np.random.randint(-64, 64, size=WEIGHT_DEPTH, dtype=np.int8)

    # Set known values
    # gate_temp = 5.0 (FP16: 0x4500)
    mem[0x0390] = 0x00
    mem[0x0391] = 0x45

    write_weight_mem(output_path, mem, 'hex8')
    print(f"Dummy weights written to {output_path}")


def write_weight_mem(filepath, data, format='hex8'):
    """Write weight memory to file"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("// NanoMamba INT8 Weight Memory\n")
        f.write(f"// {len(data)} bytes ({len(data)/1024:.1f} KB)\n")
        f.write("// Format: hex, one byte per line\n")
        f.write("// Generated by export_weights.py\n")
        f.write("// Author: Jin Ho Choi, Ph.D.\n\n")

        for i, val in enumerate(data):
            if i % 16 == 0:
                f.write(f"// 0x{i:04X}\n")
            f.write(f"{int(val) & 0xFF:02X}\n")

    print(f"Written {len(data)} bytes to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Export NanoMamba weights to FPGA/ASIC format')
    parser.add_argument('--checkpoint', '-c',
                        default='../checkpoints/nanomamba_tiny_best.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', '-o',
                        default='../rtl/mem/weights.mem',
                        help='Output .mem file path')
    parser.add_argument('--format', '-f', default='hex8',
                        choices=['hex8', 'bin', 'coe'],
                        help='Output format')
    parser.add_argument('--dummy', action='store_true',
                        help='Generate dummy weights for simulation')
    args = parser.parse_args()

    if args.dummy:
        create_dummy_weights(args.output)
    else:
        export_weights(args.checkpoint, args.output, args.format)


if __name__ == '__main__':
    main()
