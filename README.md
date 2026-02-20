# NanoMamba: Noise-Robust Keyword Spotting with Spectral-Aware State Space Models

> **Submitted to Interspeech 2026**

## Overview

NanoMamba is an ultra-compact keyword spotting (KWS) architecture built on **Spectral-Aware State Space Models (SA-SSM)** that dynamically modulates SSM dynamics based on per-band SNR estimates. SA-SSM adjusts the discretization step **Delta** and input matrix **B** in response to estimated noise conditions, enabling inherent noise adaptation without a separate denoising front-end.

## Key Results

| Model | Params | Clean Acc (%) | 0dB Factory (%) |
|-------|--------|---------------|-----------------|
| NanoMamba-Tiny | 4,634 | TBD | TBD |
| NanoMamba-Small | 12,032 | TBD | TBD |
| BC-ResNet-1 | 7,464 | TBD | TBD |
| BC-ResNet-3 | 43,200 | TBD | TBD |
| DS-CNN-S | 23,756 | TBD | TBD |

## Architecture

- **Input**: 1s raw audio (16kHz) -> STFT (512 FFT, 160 hop) -> 40-band log-mel
- **SNR Estimator**: noise floor from first K=5 frames, per-band SNR
- **SA-SSM Blocks**: Delta-modulation + B-gating conditioned on SNR
- **Output**: 12-class classifier (10 keywords + silence + unknown)

## Project Structure

```
NanoMamba-Interspeech2026/
├── paper/                     # Interspeech 2026 LaTeX paper
│   ├── interspeech2026.tex    # Main paper (4p + refs)
│   ├── refs.bib               # References
│   ├── Interspeech.cls        # Official Interspeech 2026 style
│   └── IEEEtran.bst           # Bibliography style
├── src/                       # Source code
│   ├── nanomamba.py           # NanoMamba SA-SSM model
│   ├── train_all_models.py    # Training pipeline (9 models + noise eval)
│   └── model.py               # Baseline models (BC-ResNet, DS-CNN)
├── colab/                     # Google Colab notebooks
│   └── SmartEar_KWS_Colab.ipynb  # GPU training notebook
└── README.md
```

## Quick Start (Colab)

1. Open `colab/SmartEar_KWS_Colab.ipynb` in Google Colab
2. Select GPU runtime (T4 or better)
3. Run Cell 1 (setup + dataset download)
4. Run Cell 2 (train all 9 models + noise evaluation)
5. Run Cell 3 (download results)

## SA-SSM Mechanism

**Delta-modulation**: `Delta_t = softplus(W_delta * x_t + W_s * s_hat_t)`
- High SNR -> larger Delta -> faster state dynamics
- Low SNR -> smaller Delta -> suppresses noise transients

**B-gating**: `B_tilde = B_t * (1 - alpha + alpha * sigma(W_g * s_hat_t))`
- Low SNR -> gate closes -> reduces noisy input contribution
- alpha (learnable, init 0.5) controls gating strength

## Dataset

Google Speech Commands V2 (12-class):
- 10 keywords: yes, no, up, down, left, right, on, off, stop, go
- 2 additional: silence, unknown
- 86,843 train / 10,481 val / 11,005 test

## Training Configuration

- Optimizer: AdamW (beta1=0.9, beta2=0.999)
- LR: 3e-3 (<20K params), 1e-3 (default)
- Scheduler: Cosine annealing
- Label smoothing: 0.1
- Epochs: 30, Batch size: 64
- Augmentation: time shift (+-100ms), volume (+-20%), Gaussian noise (p=0.3)

## Noise Evaluation

- **Types**: Factory, White (Gaussian), Babble (5-9 talkers)
- **SNR levels**: {-15, -10, -5, 0, 5, 10, 15} dB
- Audio-domain mixing with RMS-based scaling

## License

MIT License
