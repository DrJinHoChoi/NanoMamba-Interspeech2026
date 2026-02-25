"""
PCEN Noise Comparison: Factory / White / Babble
All models full SNR curves: PCEN δ=2.0, δ=0.01, DS-CNN-S, BC-ResNet-1, NM-Tiny, NM-Small
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'sans-serif'

snr_levels = [-15, -10, -5, 0, 5, 10, 15]

# ===== All models — full SNR data =====
models = {
    'NM-Tiny-PCEN δ=2.0 (4.8K)': {
        'factory': [11.4, 30.9, 56.8, 69.9, 79.8, 84.7, 87.3],
        'white':   [12.9, 45.3, 65.8, 78.6, 85.8, 89.9, 92.9],
        'babble':  [65.0, 68.6, 74.5, 81.0, 86.8, 90.9, 93.0],
        'clean': 95.1,
        'color': '#9B2335', 'marker': '*', 'ls': '-', 'lw': 3.0, 'ms': 13, 'zorder': 16,
    },
    'NM-Tiny-PCEN δ=0.01 (4.8K)': {
        'factory': [31.0, 50.3, 67.4, 74.0, 79.7, 84.9, 87.7],
        'white':   [18.1, 44.3, 64.3, 77.1, 84.7, 89.6, 91.8],
        'babble':  [62.8, 65.6, 68.0, 74.4, 80.5, 86.2, 90.0],
        'clean': 94.6,
        'color': '#E76F51', 'marker': 'P', 'ls': '-', 'lw': 3.0, 'ms': 11, 'zorder': 15,
    },
    'NM-Tiny (4.6K)': {
        'factory': [38.4, 56.1, 70.1, 77.6, 83.2, 85.1, 86.6],
        'white':   [20.2, 51.6, 69.3, 79.8, 86.2, 90.1, 91.8],
        'babble':  [58.6, 60.4, 65.0, 69.6, 77.3, 84.1, 87.4],
        'clean': 92.9,
        'color': '#F4845F', 'marker': 'D', 'ls': '--', 'lw': 2.0, 'ms': 7, 'zorder': 9,
    },
    'NM-Small (12K)': {
        'factory': [22.7, 47.9, 69.2, 78.0, 83.9, 87.0, 89.4],
        'white':   [17.6, 52.6, 73.9, 83.9, 90.2, 93.0, 94.2],
        'babble':  [61.5, 67.8, 73.8, 79.2, 85.9, 89.7, 92.1],
        'clean': 95.2,
        'color': '#E63946', 'marker': 's', 'ls': '--', 'lw': 2.0, 'ms': 7, 'zorder': 10,
    },
    'DS-CNN-S (23.7K)': {
        'factory': [59.2, 62.6, 66.4, 75.6, 83.9, 90.7, 93.3],
        'white':   [11.1, 12.0, 11.3, 13.9, 30.0, 55.6, 75.3],
        'babble':  [34.9, 45.7, 55.4, 70.1, 81.0, 88.8, 92.8],
        'clean': 96.6,
        'color': '#457B9D', 'marker': 'o', 'ls': '-.', 'lw': 2.0, 'ms': 7, 'zorder': 8,
    },
    'BC-ResNet-1 (7.5K)': {
        'factory': [57.1, 61.5, 65.5, 71.6, 78.3, 83.8, 87.7],
        'white':   [22.0, 25.0, 37.8, 54.7, 66.1, 75.5, 84.4],
        'babble':  [37.9, 46.6, 58.0, 73.7, 85.0, 91.5, 94.1],
        'clean': 96.0,
        'color': '#2A9D8F', 'marker': '^', 'ls': ':', 'lw': 2.0, 'ms': 7, 'zorder': 8,
    },
}

# ===== Create figure =====
fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), sharey=True)
noise_types = ['factory', 'white', 'babble']
titles = ['(a) Factory Noise (Stationary)', '(b) White Noise (Broadband)', '(c) Babble Noise (Non-stationary)']

for idx, (noise, title) in enumerate(zip(noise_types, titles)):
    ax = axes[idx]

    for name, d in models.items():
        ax.plot(snr_levels, d[noise],
                color=d['color'], marker=d['marker'], ls=d['ls'],
                lw=d['lw'], markersize=d['ms'], label=name, zorder=d['zorder'])

    # --- Key annotations ---
    if noise == 'factory':
        # DS-CNN-S dominates factory at low SNR
        ax.annotate('CNN: factory\nstructural advantage',
                     xy=(-15, 59.2), xytext=(-7, 42),
                     fontsize=9, color='#457B9D', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#457B9D', lw=1.5),
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#457B9D', alpha=0.08))
        # PCEN δ=0.01 improvement
        ax.annotate('δ=0.01: factory\n+19.6%p vs δ=2.0',
                     xy=(-15, 31.0), xytext=(-7, 15),
                     fontsize=9, color='#E76F51', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#E76F51', lw=1.5),
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#E76F51', alpha=0.08))

    elif noise == 'white':
        # DS-CNN-S collapse
        ax.annotate('DS-CNN-S\nCollapse!',
                     xy=(0, 13.9), xytext=(5, 28),
                     fontsize=9, color='#457B9D', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#457B9D', lw=1.5),
                     ha='center')

    elif noise == 'babble':
        # PCEN δ=2.0 beats all at 0dB
        ax.annotate('δ=2.0: 81.0%\nBEST of ALL',
                     xy=(0, 81.0), xytext=(-7, 44),
                     fontsize=9, color='#9B2335', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#9B2335', lw=1.5),
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#9B2335', alpha=0.08))

    # --- Formatting ---
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xticks(snr_levels)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.axvspan(-15, -5, alpha=0.05, color='red')
    ax.axvspan(-5, 15, alpha=0.03, color='green')

# Single shared legend at bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6,
           fontsize=10, bbox_to_anchor=(0.5, -0.04),
           frameon=True, fancybox=True, shadow=True,
           columnspacing=1.5)

plt.suptitle('NanoMamba-Tiny-PCEN vs All Baselines: Noise-Type Comparison',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pcen_noise_comparison.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: pcen_noise_comparison.png")
plt.show()
