#!/usr/bin/env python3
"""
NanoMamba TASLP — Noise Robustness Results Visualization
Generates publication-quality figures from per-sample noise-aug experiment results.

Usage:
    python plot_results.py          # generates 4 PNG figures in paper/ directory
    python plot_results.py --pdf    # generates PDF versions (for LaTeX)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import sys
import os

# ============================================================================
# Experiment Results (Per-Sample Noise-Aug Trained, GSC V2 12-class)
# Order: [-15dB, -10dB, -5dB, 0dB, 5dB, 10dB, 15dB, clean]
# ============================================================================
RESULTS = {
    'NanoMamba-Tiny-DualPCEN': {
        'params': 4957,
        'label': 'NanoMamba (5.0K)',
        'factory': [60.9, 67.7, 78.5, 85.2, 88.8, 91.4, 91.6, 93.7],
        'white':   [36.2, 60.4, 74.4, 83.7, 88.7, 90.8, 92.3, 93.8],
        'babble':  [70.0, 77.3, 85.4, 89.4, 91.8, 92.7, 92.9, 93.8],
        'street':  [58.4, 62.7, 74.0, 82.1, 87.8, 90.5, 92.2, 93.7],
        'pink':    [24.1, 57.1, 76.3, 84.9, 89.5, 91.9, 93.2, 93.7],
    },
    'DS-CNN-S': {
        'params': 23756,
        'label': 'DS-CNN-S (23.8K)',
        'factory': [64.9, 74.6, 86.5, 91.5, 94.0, 95.1, 95.9, 96.8],
        'white':   [61.4, 66.5, 81.8, 90.0, 93.1, 94.7, 95.6, 96.8],
        'babble':  [77.0, 85.3, 90.7, 94.2, 95.5, 96.1, 96.5, 96.8],
        'street':  [62.1, 69.4, 81.4, 89.6, 92.6, 94.9, 95.7, 96.8],
        'pink':    [60.9, 66.5, 81.9, 90.6, 93.5, 94.8, 95.9, 96.8],
    },
    'BC-ResNet-1': {
        'params': 7464,
        'label': 'BC-ResNet-1 (7.5K)',
        'factory': [65.7, 75.0, 83.6, 89.4, 92.2, 93.0, 93.9, 95.3],
        'white':   [59.3, 64.5, 77.3, 85.7, 90.2, 92.7, 93.8, 95.3],
        'babble':  [76.3, 84.4, 89.1, 91.9, 93.7, 94.7, 94.7, 95.3],
        'street':  [60.6, 64.4, 76.1, 85.3, 90.5, 93.0, 94.1, 95.3],
        'pink':    [59.6, 65.6, 79.7, 88.2, 91.7, 93.6, 94.4, 95.3],
    },
}

SNR_LEVELS = [-15, -10, -5, 0, 5, 10, 15]
SNR_LABELS = ['-15', '-10', '-5', '0', '5', '10', '15', 'Clean']
SNR_TICKS = list(range(8))  # 0..7 for 8 data points
NOISE_TYPES = ['factory', 'white', 'babble', 'street', 'pink']
NOISE_LABELS = {'factory': 'Factory', 'white': 'White',
                'babble': 'Babble', 'street': 'Street', 'pink': 'Pink'}

# Style
MODEL_STYLES = {
    'NanoMamba-Tiny-DualPCEN': dict(color='#2563EB', marker='o', ls='-', lw=2.2, ms=7, zorder=3),
    'DS-CNN-S':               dict(color='#DC2626', marker='^', ls='--', lw=1.8, ms=7, zorder=2),
    'BC-ResNet-1':            dict(color='#16A34A', marker='s', ls='-.', lw=1.8, ms=6, zorder=2),
}

# Output dir
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
USE_PDF = '--pdf' in sys.argv
EXT = 'pdf' if USE_PDF else 'png'


def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


# ============================================================================
# Fig A: SNR vs Accuracy — 5 Noise Types × 3 Models
# ============================================================================
def plot_snr_accuracy():
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, noise in enumerate(NOISE_TYPES):
        ax = axes[idx]
        for model_name, data in RESULTS.items():
            s = MODEL_STYLES[model_name]
            ax.plot(SNR_TICKS, data[noise],
                    color=s['color'], marker=s['marker'],
                    linestyle=s['ls'], linewidth=s['lw'],
                    markersize=s['ms'], zorder=s['zorder'],
                    label=data['label'])

        ax.set_title(NOISE_LABELS[noise], fontweight='bold')
        ax.set_xticks(SNR_TICKS)
        ax.set_xticklabels(SNR_LABELS, rotation=45, ha='right')
        ax.set_xlabel('SNR (dB)')
        if idx == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(20, 100)
        ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.5)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.08), frameon=True,
               fancybox=True, shadow=False)

    out = os.path.join(OUT_DIR, f'fig_snr_accuracy.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig B: Parameter Efficiency — Accuracy per 1K params
# ============================================================================
def plot_param_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # --- Left: Clean accuracy vs Params (bubble chart) ---
    ax = axes[0]
    for model_name, data in RESULTS.items():
        s = MODEL_STYLES[model_name]
        params_k = data['params'] / 1000
        # Average clean across all noise types (should be same)
        clean_acc = np.mean([data[n][-1] for n in NOISE_TYPES])
        ax.scatter(params_k, clean_acc, c=s['color'], marker=s['marker'],
                   s=150, zorder=3, edgecolors='white', linewidth=0.8,
                   label=data['label'])
        ax.annotate(f"{clean_acc:.1f}%", (params_k, clean_acc),
                    textcoords="offset points", xytext=(8, -2),
                    fontsize=8, color=s['color'])
    ax.set_xlabel('Parameters (K)')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('(a) Clean Accuracy vs Model Size', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')

    # --- Right: Accuracy/param at different SNRs ---
    ax = axes[1]
    snr_indices = [0, 3, 7]  # -15dB, 0dB, clean
    snr_names = ['-15dB', '0dB', 'Clean']
    x = np.arange(len(snr_names))
    width = 0.25

    for i, (model_name, data) in enumerate(RESULTS.items()):
        s = MODEL_STYLES[model_name]
        params_k = data['params'] / 1000
        # Average across all noise types at each SNR level
        effs = []
        for si in snr_indices:
            avg_acc = np.mean([data[n][si] for n in NOISE_TYPES])
            effs.append(avg_acc / params_k)

        bars = ax.bar(x + (i - 1) * width, effs, width,
                       color=s['color'], alpha=0.85,
                       label=data['label'], edgecolor='white', linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, effs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=6.5,
                    color=s['color'])

    ax.set_xlabel('SNR Condition')
    ax.set_ylabel('Accuracy / K-params (%/K)')
    ax.set_title('(b) Parameter Efficiency', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_names)
    ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_param_efficiency.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig C: Accuracy Drop Heatmap (Clean → 0dB)
# ============================================================================
def plot_accuracy_drop_heatmap():
    fig, ax = plt.subplots(figsize=(5.5, 2.2))

    model_names = list(RESULTS.keys())
    model_labels = [RESULTS[m]['label'] for m in model_names]
    n_models = len(model_names)
    n_noise = len(NOISE_TYPES)

    # Compute drops: clean_acc - 0dB_acc
    drop_matrix = np.zeros((n_models, n_noise))
    for i, model_name in enumerate(model_names):
        for j, noise in enumerate(NOISE_TYPES):
            clean = RESULTS[model_name][noise][-1]   # last = clean
            zero_db = RESULTS[model_name][noise][3]   # index 3 = 0dB
            drop_matrix[i, j] = clean - zero_db

    im = ax.imshow(drop_matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=15)

    # Labels
    ax.set_xticks(range(n_noise))
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_TYPES])
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels)

    # Annotate values
    for i in range(n_models):
        for j in range(n_noise):
            val = drop_matrix[i, j]
            color = 'white' if val > 10 else 'black'
            ax.text(j, i, f'{val:.1f}%p', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_title('Accuracy Drop: Clean → 0 dB (lower is better)', fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Drop (%p)', fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_accuracy_drop.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig D: Radar/Spider Chart — 0dB performance across noise types
# ============================================================================
def plot_radar():
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    categories = [NOISE_LABELS[n] for n in NOISE_TYPES]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for model_name, data in RESULTS.items():
        s = MODEL_STYLES[model_name]
        # 0dB values (index 3)
        values = [data[n][3] for n in NOISE_TYPES]
        values += values[:1]  # close
        ax.plot(angles, values, color=s['color'], linewidth=s['lw'],
                linestyle=s['ls'], marker=s['marker'], markersize=5,
                label=data['label'])
        ax.fill(angles, values, color=s['color'], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(75, 96)
    ax.set_yticks([80, 85, 90, 95])
    ax.set_yticklabels(['80%', '85%', '90%', '95%'], fontsize=7, color='gray')
    ax.set_title('Accuracy at 0 dB SNR', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7.5)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_radar_0dB.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig E: Extreme SNR (-15, -10, -5 dB) Average — Bar Chart
# ============================================================================
def plot_extreme_bar():
    fig, ax = plt.subplots(figsize=(6, 3))

    x = np.arange(len(NOISE_TYPES))
    width = 0.25

    for i, (model_name, data) in enumerate(RESULTS.items()):
        s = MODEL_STYLES[model_name]
        # Average of -15, -10, -5 dB (indices 0, 1, 2)
        extreme_avgs = [np.mean(data[n][:3]) for n in NOISE_TYPES]
        bars = ax.bar(x + (i - 1) * width, extreme_avgs, width,
                       color=s['color'], alpha=0.85,
                       label=data['label'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, extreme_avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7,
                    color=s['color'], fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Average Accuracy at Extreme SNR (-15, -10, -5 dB)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_TYPES])
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_ylim(0, 95)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_extreme_snr.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig F: Individual Noise Type Plots (large, detailed)
# ============================================================================
def plot_per_noise():
    """Generate one detailed figure per noise type with data annotations."""
    for noise in NOISE_TYPES:
        fig, ax = plt.subplots(figsize=(6, 4))

        for model_name, data in RESULTS.items():
            s = MODEL_STYLES[model_name]
            vals = data[noise]
            ax.plot(SNR_TICKS, vals,
                    color=s['color'], marker=s['marker'],
                    linestyle=s['ls'], linewidth=s['lw'] + 0.3,
                    markersize=s['ms'] + 1, zorder=s['zorder'],
                    label=data['label'])
            # Annotate each point
            for xi, yi in zip(SNR_TICKS, vals):
                offset_y = 2.5 if model_name == 'NanoMamba-Tiny-DualPCEN' else -4.5
                if model_name == 'BC-ResNet-1':
                    offset_y = -4.5 if yi < data[noise][-1] else 2.5
                ax.annotate(f'{yi:.1f}', (xi, yi),
                            textcoords="offset points",
                            xytext=(0, offset_y),
                            ha='center', fontsize=7,
                            color=s['color'], fontweight='bold')

        # Highlight zones
        ax.axhspan(90, 100, color='green', alpha=0.04)
        ax.axhspan(0, 70, color='red', alpha=0.04)
        ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.5)

        # Gap annotation at 0dB
        nm_0db = RESULTS['NanoMamba-Tiny-DualPCEN'][noise][3]
        ds_0db = RESULTS['DS-CNN-S'][noise][3]
        gap = ds_0db - nm_0db
        ax.annotate(f'Gap: {gap:.1f}%p',
                    xy=(3, (nm_0db + ds_0db) / 2),
                    fontsize=8, color='#666666', ha='left',
                    xytext=(3.4, (nm_0db + ds_0db) / 2),
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8))

        ax.set_title(f'{NOISE_LABELS[noise]} Noise - SNR vs Accuracy',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(SNR_TICKS)
        ax.set_xticklabels(SNR_LABELS, fontsize=9)
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_ylim(15 if noise in ['white', 'pink'] else 50, 100)
        ax.legend(fontsize=9, loc='lower right',
                  framealpha=0.9, edgecolor='gray')

        # Parameter info box
        info = "Params: NanoMamba 5.0K | DS-CNN-S 23.8K | BC-ResNet-1 7.5K"
        ax.text(0.5, 0.02, info, transform=ax.transAxes,
                fontsize=7, ha='center', color='gray', style='italic')

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'fig_noise_{noise}.{EXT}')
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    set_style()
    print(f"\nGenerating NanoMamba result figures ({EXT.upper()})...\n")

    plot_snr_accuracy()
    plot_param_efficiency()
    plot_accuracy_drop_heatmap()
    plot_radar()
    plot_extreme_bar()
    plot_per_noise()

    print(f"\nDone! All figures saved to: {OUT_DIR}/")
    print(f"\nKey observations:")
    print(f"  - DS-CNN-S leads in absolute accuracy (4.8x more params)")
    print(f"  - NanoMamba achieves 4.6x better parameter efficiency")
    print(f"  - White/Pink noise: largest gap (broadband, no spectral structure)")
    print(f"  - Babble noise: smallest gap (DualPCEN routing helps)")
