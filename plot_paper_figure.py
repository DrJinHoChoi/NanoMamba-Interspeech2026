"""
Interspeech 2026 - Noise Robustness Analysis Figure
Publication-quality: 2 rows
  Row 1: Factory / White / Babble  (3 main noise types, all 4 models)
  Row 2: Street / Pink / Summary bar chart at -15dB
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# ============================================================
# Data
# ============================================================
snr_levels = [-15, -10, -5, 0, 5, 10, 15]

models = {
    'NanoMamba-Tiny\n(4.6K)': {
        'factory': [38.4, 56.1, 70.1, 77.6, 83.2, 85.1, 86.6],
        'white':   [20.2, 51.6, 69.3, 79.8, 86.2, 90.1, 91.8],
        'babble':  [58.6, 60.4, 65.0, 69.6, 77.3, 84.1, 87.4],
        'street':  [46.8, 58.9, 71.1, 78.8, 85.9, 89.0, 91.7],
        'pink':    [ 9.9, 38.3, 69.4, 81.9, 88.6, 91.3, 92.5],
        'clean': 92.9,
    },
    'NanoMamba-Small\n(12K)': {
        'factory': [22.7, 47.9, 69.2, 78.0, 83.9, 87.0, 89.4],
        'white':   [17.6, 52.6, 73.9, 83.9, 90.2, 93.0, 94.2],
        'babble':  [61.5, 67.8, 73.8, 79.2, 85.9, 89.7, 92.1],
        'street':  None,
        'pink':    None,
        'clean': 95.2,
    },
    'DS-CNN-S\n(23.7K)': {
        'factory': [59.2, 62.6, 66.4, 75.6, 83.9, 90.7, 93.3],
        'white':   [11.1, 12.0, 11.3, 13.9, 30.0, 55.6, 75.3],
        'babble':  [34.9, 45.7, 55.4, 70.1, 81.0, 88.8, 92.8],
        'street':  None,
        'pink':    None,
        'clean': 96.6,
    },
    'BC-ResNet-1\n(7.5K)': {
        'factory': [57.1, 61.5, 65.5, 71.6, 78.3, 83.8, 87.7],
        'white':   [22.0, 25.0, 37.8, 54.7, 66.1, 75.5, 84.4],
        'babble':  [37.9, 46.6, 58.0, 73.7, 85.0, 91.5, 94.1],
        'street':  None,
        'pink':    None,
        'clean': 96.0,
    },
}

colors = {
    'NanoMamba-Tiny\n(4.6K)':  '#E63946',
    'NanoMamba-Small\n(12K)':  '#F4845F',
    'DS-CNN-S\n(23.7K)':      '#457B9D',
    'BC-ResNet-1\n(7.5K)':    '#2A9D8F',
}

markers = {
    'NanoMamba-Tiny\n(4.6K)':  'D',
    'NanoMamba-Small\n(12K)':  's',
    'DS-CNN-S\n(23.7K)':      'o',
    'BC-ResNet-1\n(7.5K)':    '^',
}

linestyles = {
    'NanoMamba-Tiny\n(4.6K)':  '-',
    'NanoMamba-Small\n(12K)':  '--',
    'DS-CNN-S\n(23.7K)':      '-.',
    'BC-ResNet-1\n(7.5K)':    ':',
}

# ============================================================
# Figure: 2x3 layout
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

noise_map = [
    # Row 1
    ('factory', 'Factory Noise (Stationary)'),
    ('white',   'White Noise (Broadband)'),
    ('babble',  'Babble Noise (Non-stationary)'),
    # Row 2
    ('street',  'Street Noise (Mixed)'),
    ('pink',    'Pink Noise (1/f)'),
    (None,      None),  # Summary bar chart
]

# --- Row 1 & Row 2 line plots ---
for idx, (noise, title) in enumerate(noise_map[:5]):
    row, col = idx // 3, idx % 3
    ax = axes[row][col]

    for mname, mdata in models.items():
        vals = mdata[noise]
        if vals is None:
            continue
        ax.plot(snr_levels, vals,
                color=colors[mname], marker=markers[mname],
                ls=linestyles[mname], lw=2.0, markersize=6,
                label=mname, zorder=10)
        # Clean accuracy reference line
        ax.axhline(y=mdata['clean'], color=colors[mname],
                   ls=':', alpha=0.25, lw=1)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('SNR (dB)')
    if col == 0:
        ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(snr_levels)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)

    # Shade zones
    ax.axvspan(-15, -5, alpha=0.04, color='red')
    ax.axvspan(-5, 15, alpha=0.04, color='green')

    # Annotations
    if noise == 'white':
        ax.annotate('CNN\nCollapse', xy=(0, 13.9), xytext=(5, 28),
                    fontsize=9, color='#457B9D', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#457B9D', lw=1.5),
                    ha='center', zorder=20)

    if noise == 'factory':
        ax.annotate('SSM\nGap', xy=(-15, 38.4), xytext=(-10, 20),
                    fontsize=9, color='#E63946', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
                    ha='center', zorder=20)

    if noise == 'pink':
        ax.annotate('1/f\noverlap', xy=(-15, 9.9), xytext=(-10, 28),
                    fontsize=9, color='#E63946', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
                    ha='center', zorder=20)

    # Panel label (a), (b), (c)...
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    ax.text(0.02, 0.96, panel_labels[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

# --- Row 2, Col 2: Summary bar chart at -15dB ---
ax_bar = axes[1][2]

bar_noises = ['factory', 'white', 'babble']
bar_labels = ['Factory', 'White', 'Babble']
x = np.arange(len(bar_noises))
width = 0.18

bar_models = ['NanoMamba-Tiny\n(4.6K)', 'NanoMamba-Small\n(12K)',
              'DS-CNN-S\n(23.7K)', 'BC-ResNet-1\n(7.5K)']
bar_names = ['NM-Tiny', 'NM-Small', 'DS-CNN-S', 'BC-ResNet-1']

for i, (mname, bname) in enumerate(zip(bar_models, bar_names)):
    vals = []
    for noise in bar_noises:
        v = models[mname][noise]
        vals.append(v[0] if v else 0)  # -15dB = index 0

    bars = ax_bar.bar(x + i * width - 1.5 * width, vals, width,
                      color=colors[mname], label=bname, edgecolor='white',
                      linewidth=0.5, zorder=5)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=7,
                       fontweight='bold', color=colors[mname])

ax_bar.set_title('(f) Comparison at SNR = -15 dB', fontsize=12, fontweight='bold', pad=8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(bar_labels)
ax_bar.set_ylabel('Accuracy (%)')
ax_bar.set_ylim(0, 75)
ax_bar.grid(True, axis='y', alpha=0.25)
ax_bar.legend(fontsize=8, loc='upper right', framealpha=0.9)

# SSM advantage region
ax_bar.annotate('SSM\nAdvantage', xy=(1, 20), fontsize=9,
                color='#E63946', fontweight='bold', ha='center', alpha=0.7)
# CNN advantage region
ax_bar.annotate('CNN\nAdvantage', xy=(0, 62), fontsize=9,
                color='#457B9D', fontweight='bold', ha='center', alpha=0.7)

# --- Global legend ---
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           fontsize=10, bbox_to_anchor=(0.5, -0.01),
           frameon=True, fancybox=True, shadow=True,
           columnspacing=2.0)

fig.suptitle('Noise Robustness Analysis: NanoMamba SA-SSM vs CNN Baselines\n'
             '(All models trained on clean data only — unseen noise evaluation)',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout(h_pad=2.5, w_pad=1.5)
plt.savefig('noise_analysis_paper.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('noise_analysis_paper.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: noise_analysis_paper.png, noise_analysis_paper.pdf")
plt.show()
