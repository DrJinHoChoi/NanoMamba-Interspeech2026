"""
Babble Noise Comparison: NanoMamba-Tiny-TC vs All Models
Shows TC's advantage on non-stationary noise (babble)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'sans-serif'

snr_levels = [-15, -10, -5, 0, 5, 10, 15]

# All models - babble noise data
models = {
    'DS-CNN-S (23.7K)': {
        'babble': [34.9, 45.7, 55.4, 70.1, 81.0, 88.8, 92.8],
        'clean': 96.6,
        'color': '#457B9D', 'marker': 'o', 'ls': '-.', 'lw': 2.0,
    },
    'BC-ResNet-1 (7.5K)': {
        'babble': [37.9, 46.6, 58.0, 73.7, 85.0, 91.5, 94.1],
        'clean': 96.0,
        'color': '#2A9D8F', 'marker': '^', 'ls': ':', 'lw': 2.0,
    },
    'NM-Small (12K)': {
        'babble': [61.5, 67.8, 73.8, 79.2, 85.9, 89.7, 92.1],
        'clean': 95.2,
        'color': '#E63946', 'marker': 's', 'ls': '-', 'lw': 2.0,
    },
    'NM-Tiny (4.6K)': {
        'babble': [58.6, 60.4, 65.0, 69.6, 77.3, 84.1, 87.4],
        'clean': 92.9,
        'color': '#F4845F', 'marker': 'D', 'ls': '--', 'lw': 2.0,
    },
    'NM-Tiny-TC (4.6K)': {
        'babble': [63.3, 65.2, 68.0, 73.2, 78.7, 84.6, 88.4],
        'clean': 92.6,
        'color': '#9B2335', 'marker': '*', 'ls': '-', 'lw': 3.0,
    },
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [2, 1]})

# ===== Left: Line plot - Babble noise across SNR =====
for name, d in models.items():
    zorder = 15 if 'TC' in name else 8
    ms = 12 if 'TC' in name else 7
    ax1.plot(snr_levels, d['babble'],
             color=d['color'], marker=d['marker'], ls=d['ls'],
             lw=d['lw'], markersize=ms, label=name, zorder=zorder)
    # Clean accuracy dashed line
    ax1.axhline(y=d['clean'], color=d['color'], ls=':', alpha=0.2, lw=1)

# Highlight TC improvement zone
for i, snr in enumerate(snr_levels):
    tiny_val = models['NM-Tiny (4.6K)']['babble'][i]
    tc_val = models['NM-Tiny-TC (4.6K)']['babble'][i]
    diff = tc_val - tiny_val
    if diff > 0:
        ax1.annotate(f'+{diff:.1f}',
                     xy=(snr, tc_val), xytext=(snr+0.3, tc_val+3),
                     fontsize=8, color='#9B2335', fontweight='bold',
                     ha='left')

# Annotations
ax1.annotate('TC: Non-stationary\nnoise invariance',
             xy=(-15, 63.3), xytext=(-8, 45),
             fontsize=10, color='#9B2335', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#9B2335', lw=1.5),
             ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#9B2335', alpha=0.1))

ax1.annotate('CNN: weak at\nlow SNR babble',
             xy=(-15, 34.9), xytext=(-8, 20),
             fontsize=9, color='#457B9D',
             arrowprops=dict(arrowstyle='->', color='#457B9D', lw=1.2),
             ha='center')

ax1.set_title('Babble Noise (Non-stationary) — Accuracy vs SNR',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('SNR (dB)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_xticks(snr_levels)
ax1.set_ylim(20, 100)
ax1.grid(True, alpha=0.3)
ax1.axvspan(-15, -5, alpha=0.06, color='red')
ax1.axvspan(-5, 15, alpha=0.04, color='green')
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)

# ===== Right: Bar chart - 0dB and -15dB comparison =====
model_names = ['DS-CNN-S\n(23.7K)', 'BC-ResNet-1\n(7.5K)', 'NM-Small\n(12K)',
               'NM-Tiny\n(4.6K)', 'NM-Tiny-TC\n(4.6K)']
colors = ['#457B9D', '#2A9D8F', '#E63946', '#F4845F', '#9B2335']
vals_0dB = [70.1, 73.7, 79.2, 69.6, 73.2]
vals_m15dB = [34.9, 37.9, 61.5, 58.6, 63.3]

x = np.arange(len(model_names))
w = 0.35

bars1 = ax2.bar(x - w/2, vals_m15dB, w, label='-15 dB', color=colors, alpha=0.5,
                edgecolor=colors, linewidth=1.5)
bars2 = ax2.bar(x + w/2, vals_0dB, w, label='0 dB', color=colors, alpha=0.9,
                edgecolor='black', linewidth=0.5)

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 1,
             f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 1,
             f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Highlight TC bar
bars1[-1].set_edgecolor('#9B2335')
bars1[-1].set_linewidth(3)
bars2[-1].set_edgecolor('#9B2335')
bars2[-1].set_linewidth(3)

ax2.set_title('Babble @ -15dB & 0dB', fontsize=14, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=9)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2, axis='y')

# Add retention rate text
for i, (m15, zero) in enumerate(zip(vals_m15dB, vals_0dB)):
    ret = m15 / zero * 100
    ax2.text(x[i], 5, f'Ret:{ret:.0f}%', ha='center', fontsize=7,
             fontweight='bold', color=colors[i])

plt.suptitle('NanoMamba-Tiny-TC: Structural Noise Invariance for Non-stationary Noise',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('babble_comparison.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: babble_comparison.png")
plt.show()
