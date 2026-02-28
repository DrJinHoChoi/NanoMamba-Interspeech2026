"""
NanoMamba vs CNN Baselines — Noise Robustness Comparison
=========================================================
DS-CNN-S (23.7K), BC-ResNet-1 (7.5K) vs NanoMamba Full System (4,957)
Factory / White / Babble noise, SNR -15 ~ +15 dB

Data from Colab experiments (Feb 2026).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ============================================================
# Data
# ============================================================
snr_levels = [-15, -10, -5, 0, 5, 10, 15]

models = {
    'DS-CNN-S\n(23,700 params)': {
        'factory': [59.2, 62.6, 66.4, 75.6, 83.9, 90.7, 93.3],
        'white':   [11.1, 12.0, 11.3, 13.9, 30.0, 55.6, 75.3],
        'babble':  [34.9, 45.7, 55.4, 70.1, 81.0, 88.8, 92.8],
        'clean': 96.6,
        'params': 23700,
        'color': '#457B9D',
        'marker': 'o',
        'ls': '-.',
        'lw': 2.0,
    },
    'BC-ResNet-1\n(7,500 params)': {
        'factory': [57.1, 61.5, 65.5, 71.6, 78.3, 83.8, 87.7],
        'white':   [22.0, 25.0, 37.8, 54.7, 66.1, 75.5, 84.4],
        'babble':  [37.9, 46.6, 58.0, 73.7, 85.0, 91.5, 94.1],
        'clean': 96.0,
        'params': 7500,
        'color': '#2A9D8F',
        'marker': '^',
        'ls': '--',
        'lw': 2.0,
    },
    'NanoMamba Baseline\n(4,636 params)': {
        'factory': [38.4, 56.1, 70.1, 77.6, 83.2, 85.1, 86.6],
        'white':   [20.2, 51.6, 69.3, 79.8, 86.2, 90.1, 91.8],
        'babble':  [58.6, 60.4, 65.0, 69.6, 77.3, 84.1, 87.4],
        'clean': 92.9,
        'params': 4636,
        'color': '#F4845F',
        'marker': 'D',
        'ls': ':',
        'lw': 1.8,
    },
    'NanoMamba Full System\n(4,957 params, +0 extra)': {
        'factory': [45.2, 58.7, 70.8, 78.3, 84.0, 88.2, 91.5],
        'white':   [47.5, 59.3, 71.0, 80.5, 87.1, 91.0, 93.2],
        'babble':  [55.2, 60.8, 67.5, 74.2, 80.6, 86.5, 90.8],
        'clean': 94.6,
        'params': 4957,
        'color': '#E63946',
        'marker': '*',
        'ls': '-',
        'lw': 3.0,
    },
}

noise_types = ['factory', 'white', 'babble']
noise_titles = {
    'factory': 'Factory Noise',
    'white':   'White Noise (Broadband)',
    'babble':  'Babble Noise (Non-stationary)',
}

# ============================================================
# Figure: 2 rows — top: SNR curves, bottom: bar chart
# ============================================================
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.22,
                      height_ratios=[1.2, 1])

# ============================================================
# Row 1: SNR Curves (3 subplots)
# ============================================================
for idx, noise in enumerate(noise_types):
    ax = fig.add_subplot(gs[0, idx])

    for model_name, d in models.items():
        vals = d[noise]
        ax.plot(snr_levels, vals,
                color=d['color'], marker=d['marker'], ls=d['ls'],
                lw=d['lw'], markersize=8 if '*' in d['marker'] else 6,
                label=model_name, zorder=10 if 'Full' in model_name else 5)
        # Clean reference line
        ax.axhline(y=d['clean'], color=d['color'], ls=':', alpha=0.25, lw=1)

    ax.set_title(noise_titles[noise], fontsize=15, fontweight='bold')
    ax.set_xlabel('SNR (dB)')
    if idx == 0:
        ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(snr_levels)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.axvspan(-15, -5, alpha=0.05, color='red')
    ax.axvspan(5, 15, alpha=0.03, color='green')

    # --- Annotations ---
    if noise == 'white':
        # DS-CNN-S White collapse
        ax.annotate('DS-CNN-S\nCollapse!\n11.1%',
                    xy=(-15, 11.1), xytext=(-8, 8),
                    fontsize=9, color='#457B9D', fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#457B9D', lw=1.5))
        # NanoMamba Full advantage
        ax.annotate('Full System\n47.5%',
                    xy=(-15, 47.5), xytext=(-8, 58),
                    fontsize=10, color='#E63946', fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5))
        # Improvement arrow
        ax.annotate('', xy=(-14.5, 47.5), xytext=(-14.5, 11.1),
                    arrowprops=dict(arrowstyle='<->', color='#9B59B6', lw=2))
        ax.text(-13.5, 28, '+36.4%p\nvs DS-CNN',
                fontsize=9, color='#9B59B6', fontweight='bold')

    elif noise == 'babble':
        # NanoMamba Babble advantage
        ax.annotate('NanoMamba\n55.2%',
                    xy=(-15, 55.2), xytext=(-8, 65),
                    fontsize=10, color='#E63946', fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5))
        ax.annotate('DS-CNN-S\n34.9%',
                    xy=(-15, 34.9), xytext=(-8, 22),
                    fontsize=9, color='#457B9D', fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#457B9D', lw=1.5))
        ax.annotate('', xy=(-14.5, 55.2), xytext=(-14.5, 34.9),
                    arrowprops=dict(arrowstyle='<->', color='#9B59B6', lw=2))
        ax.text(-13.5, 43, '+20.3%p',
                fontsize=9, color='#9B59B6', fontweight='bold')

    elif noise == 'factory':
        # Factory: CNN has advantage at -15dB but NanoMamba catches up
        ax.annotate('DS-CNN\n59.2%',
                    xy=(-15, 59.2), xytext=(-8, 68),
                    fontsize=9, color='#457B9D', fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#457B9D', lw=1.5))
        ax.annotate('NanoMamba\n45.2%',
                    xy=(-15, 45.2), xytext=(-8, 33),
                    fontsize=10, color='#E63946', fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5))
        # Show params comparison
        ax.text(8, 15, 'DS-CNN: 23.7K params\nNanoMamba: 4.9K params\n(4.8x smaller)',
                fontsize=9, color='gray', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA',
                         edgecolor='#DEE2E6', alpha=0.9))

# Legend (once, for top row)
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4,
           fontsize=11, bbox_to_anchor=(0.5, 0.52),
           frameon=True, fancybox=True, shadow=True,
           borderpad=0.8, columnspacing=1.5)

# ============================================================
# Row 2, Left: Bar Chart — Accuracy at -15dB
# ============================================================
ax_bar = fig.add_subplot(gs[1, 0:2])

model_short = ['DS-CNN-S\n(23.7K)', 'BC-ResNet-1\n(7.5K)',
               'NanoMamba\nBaseline (4.6K)', 'NanoMamba\nFull (4.9K)']
colors_bar = ['#457B9D', '#2A9D8F', '#F4845F', '#E63946']
hatches = ['///', '\\\\\\', '...', '']

factory_15 = [59.2, 57.1, 38.4, 45.2]
white_15   = [11.1, 22.0, 20.2, 47.5]
babble_15  = [34.9, 37.9, 58.6, 55.2]

x = np.arange(len(model_short))
w = 0.22

bars_f = ax_bar.bar(x - w, factory_15, w, color='#E74C3C', alpha=0.8,
                    label='Factory -15dB', edgecolor='white', zorder=3)
bars_w = ax_bar.bar(x,     white_15,   w, color='#3498DB', alpha=0.8,
                    label='White -15dB', edgecolor='white', zorder=3)
bars_b = ax_bar.bar(x + w, babble_15,  w, color='#2ECC71', alpha=0.8,
                    label='Babble -15dB', edgecolor='white', zorder=3)

# Value labels
for bars in [bars_f, bars_w, bars_b]:
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 1,
                   f'{h:.1f}', ha='center', va='bottom', fontsize=9,
                   fontweight='bold')

# Highlight best per noise
# White: NanoMamba Full is best (47.5)
ax_bar.annotate('BEST',
               xy=(3, 47.5 + 4), fontsize=8, color='#3498DB',
               fontweight='bold', ha='center')

# Babble: NanoMamba Baseline is best (58.6)
ax_bar.annotate('BEST',
               xy=(2 + w, 58.6 + 4), fontsize=8, color='#2ECC71',
               fontweight='bold', ha='center')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(model_short, fontsize=11)
ax_bar.set_ylabel('Accuracy at -15dB (%)')
ax_bar.set_title('Accuracy Comparison at -15dB SNR (Hardest Condition)',
                fontweight='bold', fontsize=13)
ax_bar.legend(loc='upper left', fontsize=10, ncol=3)
ax_bar.set_ylim(0, 75)
ax_bar.grid(axis='y', alpha=0.3, zorder=0)

# Model size indicators below bars
for i, (name, params) in enumerate(zip(model_short,
    [23700, 7500, 4636, 4957])):
    ax_bar.text(i, -6, f'{params:,}', ha='center', fontsize=9,
               color='gray', style='italic')
ax_bar.text(-0.5, -6, 'Params:', ha='right', fontsize=9,
           color='gray', fontweight='bold')

# ============================================================
# Row 2, Right: Efficiency Summary
# ============================================================
ax_eff = fig.add_subplot(gs[1, 2])

# Scatter: params (x, log) vs avg noise acc at -15dB (y)
# Bubble size = clean accuracy
algo_data = [
    ('DS-CNN-S',        23700, np.mean([59.2, 11.1, 34.9]), 96.6, '#457B9D', 'o'),
    ('BC-ResNet-1',      7500, np.mean([57.1, 22.0, 37.9]), 96.0, '#2A9D8F', '^'),
    ('NanoMamba\nBaseline', 4636, np.mean([38.4, 20.2, 58.6]), 92.9, '#F4845F', 'D'),
    ('NanoMamba\nFull System', 4957, np.mean([45.2, 47.5, 55.2]), 94.6, '#E63946', '*'),
]

for name, params, noise_avg, clean, color, marker in algo_data:
    size = 250 if marker == '*' else 150
    ax_eff.scatter(params, noise_avg, s=size, c=color, marker=marker,
                  edgecolors='black', linewidth=0.8, zorder=5, alpha=0.9)
    # Label
    if 'Full' in name:
        ax_eff.annotate(f'{name}\n({noise_avg:.1f}%)',
                       xy=(params, noise_avg),
                       xytext=(params * 1.8, noise_avg + 3),
                       fontsize=9, fontweight='bold', color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    elif 'Baseline' in name:
        ax_eff.annotate(f'{name}\n({noise_avg:.1f}%)',
                       xy=(params, noise_avg),
                       xytext=(params * 1.8, noise_avg - 8),
                       fontsize=9, color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
    elif 'DS-CNN' in name:
        ax_eff.annotate(f'{name}\n({noise_avg:.1f}%)',
                       xy=(params, noise_avg),
                       xytext=(params * 0.5, noise_avg + 6),
                       fontsize=9, color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
    else:
        ax_eff.annotate(f'{name}\n({noise_avg:.1f}%)',
                       xy=(params, noise_avg),
                       xytext=(params * 0.4, noise_avg - 6),
                       fontsize=9, color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

# Highlight NanoMamba advantage zone
rect = mpatches.FancyBboxPatch((3500, 43), 4000, 12,
                                boxstyle="round,pad=0.3",
                                facecolor='#E63946', alpha=0.08,
                                edgecolor='#E63946', ls='--', lw=1.5)
ax_eff.add_patch(rect)
ax_eff.text(5500, 57, 'Smallest + Best\nNoise Robustness',
           fontsize=10, color='#E63946', ha='center',
           fontweight='bold', style='italic')

ax_eff.set_xscale('log')
ax_eff.set_xlabel('Parameters (log scale)')
ax_eff.set_ylabel('Avg Accuracy at -15dB (%)')
ax_eff.set_title('Efficiency: Params vs Noise Robustness',
                fontweight='bold', fontsize=13)
ax_eff.set_xlim(3000, 40000)
ax_eff.set_ylim(25, 60)
ax_eff.grid(True, alpha=0.3)

# ============================================================
# Super title
# ============================================================
fig.suptitle('NanoMamba vs CNN Baselines: Noise Robustness Comparison\n'
             '(NanoMamba: 4,957 params | DS-CNN-S: 23,700 params | BC-ResNet-1: 7,500 params)',
             fontsize=16, fontweight='bold', y=1.02)

plt.savefig('C:/Users/jinho/Downloads/NanoMamba-Interspeech2026/baseline_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: baseline_comparison.png")
