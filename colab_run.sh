#!/bin/bash
# ============================================================================
# NanoMamba Colab Quick-Run Script
# ============================================================================
# Copy this to Colab cells or run as a shell script.
#
# Prerequisites:
#   1. Mount Google Drive
#   2. Upload nanomamba.py + train_colab.py to /content/drive/MyDrive/NanoMamba/
# ============================================================================

# ----- Cell 1: Setup -----
# Mount Drive & set working directory
# from google.colab import drive
# drive.mount('/content/drive')

WORK_DIR="/content/drive/MyDrive/NanoMamba"
cd "$WORK_DIR"

# Verify files exist
echo "=== Files ==="
ls -la nanomamba.py train_colab.py
echo ""

# Quick test: verify model creation
python -c "
from nanomamba import create_nanomamba_tiny, create_nanomamba_tiny_tc, create_nanomamba_tiny_ws_tc
import torch
import torch.nn.functional as F

for name, fn in [('Tiny', create_nanomamba_tiny),
                  ('Tiny-TC', create_nanomamba_tiny_tc),
                  ('Tiny-WS-TC', create_nanomamba_tiny_ws_tc)]:
    m = fn()
    p = sum(x.numel() for x in m.parameters())
    # Check structural params
    has_delta_floor = any('log_delta_floor' in n for n, _ in m.named_parameters())
    has_epsilon = any('log_epsilon' in n for n, _ in m.named_parameters())
    print(f'  NanoMamba-{name}: {p:,} params | delta_floor={has_delta_floor} | epsilon={has_epsilon}')
print('All models OK!')
"

# ----- Cell 2: Train NanoMamba-Tiny (baseline with structural Δ floor + ε) -----
echo ""
echo "=== Training NanoMamba-Tiny (structural) ==="
python train_colab.py \
    --models NanoMamba-Tiny \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results \
    --epochs 30 \
    --batch_size 128 \
    --lr 3e-3 \
    --noise_types factory,white,babble,street,pink \
    --snr_range -15,-10,-5,0,5,10,15

# ----- Cell 3: Train NanoMamba-Tiny-TC (+ TinyConv2D) -----
echo ""
echo "=== Training NanoMamba-Tiny-TC ==="
python train_colab.py \
    --models NanoMamba-Tiny-TC \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results \
    --epochs 30 \
    --batch_size 128 \
    --lr 3e-3 \
    --noise_types factory,white,babble,street,pink \
    --snr_range -15,-10,-5,0,5,10,15

# ----- Cell 4: Train NanoMamba-Tiny-WS-TC (weight shared + TC) -----
echo ""
echo "=== Training NanoMamba-Tiny-WS-TC ==="
python train_colab.py \
    --models NanoMamba-Tiny-WS-TC \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results \
    --epochs 30 \
    --batch_size 128 \
    --lr 3e-3 \
    --noise_types factory,white,babble,street,pink \
    --snr_range -15,-10,-5,0,5,10,15

echo ""
echo "=== All training complete! ==="
echo "Results saved to: $WORK_DIR/results/final_results.json"
