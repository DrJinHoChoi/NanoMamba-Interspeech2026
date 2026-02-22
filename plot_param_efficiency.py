"""
Parameter Efficiency Visualization
X: Parameters, Y: Accuracy (Clean vs Noisy)
Shows that NanoMamba achieves better noise robustness with fewer parameters
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'sans-serif'

# Model data
models = {
    'NanoMamba-Small': {
        'params': 12032,
        'clean': 95.11,   # Test accuracy
        'factory_0dB': 78.0,
        'white_0dB': 83.9,
        'babble_0dB': 79.2,
        'avg_0dB': 80.4,
        'white_neg5': 73.9,
        'babble_neg5': 73.8,
        'color': '#E63946', 'marker': 's',
    },
    'NanoMamba-Tiny': {
        'params': 4634,
        'clean': 92.31,
        'factory_0dB': 77.1,
        'white_0dB': 80.1,
        'babble_0dB': 70.8,
        'avg_0dB': 76.0,
        'white_neg5': 71.1,
        'babble_neg5': 65.4,
        'color': '#F4845F', 'marker': 'D',
    },
    'DS-CNN-S': {
        'params': 23756,
        'clean': 96.39,
        'factory_0dB': 75.6,
        'white_0dB': 13.9,
        'babble_0dB': 70.1,
        'avg_0dB': 53.2,
        'white_neg5': 11.3,
        'babble_neg5': 55.4,
        'color': '#457B9D', 'marker': 'o',
    },
    'BC-ResNet-1': {
        'params': 7464,
        'clean': 96.05,
        'factory_0dB': 71.6,
        'white_0dB': 54.7,
        'babble_0dB': 73.7,
        'avg_0dB': 66.7,
        'white_neg5': 37.8,
        'babble_neg5': 58.0,
        'color': '#2A9D8F', 'marker': '^',
    },
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ---- Plot 1: Clean vs Avg Noisy (0dB) ----
ax = axes[0]
for name, d in models.items():
    # Clean accuracy (hollow marker)
    ax.scatter(d['params']/1e3, d['clean'], color=d['color'],
               marker=d['marker'], s=150, edgecolors=d['color'],
               facecolors='white', linewidths=2, zorder=5)
    # Noisy avg 0dB (filled marker)
    ax.scatter(d['params']/1e3, d['avg_0dB'], color=d['color'],
               marker=d['marker'], s=150, zorder=5)
    # Arrow from clean to noisy
    ax.annotate('', xy=(d['params']/1e3, d['avg_0dB']),
                xytext=(d['params']/1e3, d['clean']),
                arrowprops=dict(arrowstyle='->', color=d['color'],
                               lw=1.5, alpha=0.5))
    # Label
    short = name.replace('NanoMamba-', 'NM-')
    ax.annotate(short, (d['params']/1e3, d['avg_0dB']),
                textcoords='offset points', xytext=(8, -5),
                fontsize=9, color=d['color'], fontweight='bold')

ax.set_xlabel('Parameters (K)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Clean (○) vs Avg Noisy 0dB (●)', fontsize=13, fontweight='bold')
ax.set_ylim(40, 100)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks([5, 10, 25])
ax.set_xticklabels(['5K', '10K', '25K'])

# ---- Plot 2: White Noise -5dB (worst case for CNN) ----
ax = axes[1]
for name, d in models.items():
    ax.scatter(d['params']/1e3, d['white_neg5'], color=d['color'],
               marker=d['marker'], s=200, zorder=5)
    short = name.replace('NanoMamba-', 'NM-')
    offset_y = 5 if 'Nano' in name else -8
    ax.annotate(f"{short}\n{d['white_neg5']:.1f}%",
                (d['params']/1e3, d['white_neg5']),
                textcoords='offset points', xytext=(10, offset_y),
                fontsize=9, color=d['color'], fontweight='bold')

ax.axhline(y=8.33, color='gray', ls='--', alpha=0.5, label='Random (8.3%)')
ax.set_xlabel('Parameters (K)', fontsize=12)
ax.set_title('White Noise @ -5dB', fontsize=13, fontweight='bold')
ax.set_ylim(0, 85)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks([5, 10, 25])
ax.set_xticklabels(['5K', '10K', '25K'])
ax.legend(fontsize=9)

# ---- Plot 3: Noise Robustness Score (avg across all conditions) ----
ax = axes[2]
# Calculate "Robustness Score" = avg accuracy across all noise conditions at 0dB
# normalized by clean accuracy (higher = more robust)
for name, d in models.items():
    robustness = d['avg_0dB'] / d['clean'] * 100  # retention %
    ax.barh(name.split('\n')[0] if '\n' in name else name,
            robustness, color=d['color'], alpha=0.8, height=0.5)
    ax.text(robustness + 1, list(models.keys()).index(name),
            f"{robustness:.1f}%", va='center', fontweight='bold',
            color=d['color'], fontsize=11)

ax.set_xlabel('Accuracy Retention at 0dB (%)', fontsize=12)
ax.set_title('Noise Robustness Score\n(Avg 0dB / Clean)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

fig.suptitle('Parameter Efficiency: NanoMamba vs CNN Baselines',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('param_efficiency.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('param_efficiency.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: param_efficiency.png, param_efficiency.pdf")
