"""
Noise Robustness Visualization for NanoMamba Interspeech 2026
Generates 5 subplots (Factory, White, Babble, Street, Pink) showing accuracy vs SNR
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'sans-serif'

# ============================================================
# Data from noise evaluation
# ============================================================
snr_levels = [-15, -10, -5, 0, 5, 10, 15]
snr_labels = ['-15', '-10', '-5', '0', '5', '10', '15']

data = {
    'NanoMamba-Small\n(12K params)': {
        'factory': [22.7, 47.9, 69.2, 78.0, 83.9, 87.0, 89.4],
        'white':   [17.6, 52.6, 73.9, 83.9, 90.2, 93.0, 94.2],
        'babble':  [61.5, 67.8, 73.8, 79.2, 85.9, 89.7, 92.1],
        'street':  None,  # TBD
        'pink':    None,  # TBD
        'clean': 95.2,
    },
    'NanoMamba-Tiny\n(4.6K params)': {
        'factory': [38.4, 56.1, 70.1, 77.6, 83.2, 85.1, 86.6],
        'white':   [20.2, 51.6, 69.3, 79.8, 86.2, 90.1, 91.8],
        'babble':  [58.6, 60.4, 65.0, 69.6, 77.3, 84.1, 87.4],
        'street':  [46.8, 58.9, 71.1, 78.8, 85.9, 89.0, 91.7],
        'pink':    [ 9.9, 38.3, 69.4, 81.9, 88.6, 91.3, 92.5],
        'clean': 92.9,
    },
    'DS-CNN-S\n(23.7K params)': {
        'factory': [59.2, 62.6, 66.4, 75.6, 83.9, 90.7, 93.3],
        'white':   [11.1, 12.0, 11.3, 13.9, 30.0, 55.6, 75.3],
        'babble':  [34.9, 45.7, 55.4, 70.1, 81.0, 88.8, 92.8],
        'street':  None,  # TBD
        'pink':    None,  # TBD
        'clean': 96.6,
    },
    'BC-ResNet-1\n(7.5K params)': {
        'factory': [57.1, 61.5, 65.5, 71.6, 78.3, 83.8, 87.7],
        'white':   [22.0, 25.0, 37.8, 54.7, 66.1, 75.5, 84.4],
        'babble':  [37.9, 46.6, 58.0, 73.7, 85.0, 91.5, 94.1],
        'street':  None,  # TBD
        'pink':    None,  # TBD
        'clean': 96.0,
    },
}

# Colors and styles
styles = {
    'NanoMamba-Small\n(12K params)':  {'color': '#E63946', 'marker': 's', 'ls': '-',  'lw': 2.5, 'zorder': 10},
    'NanoMamba-Tiny\n(4.6K params)':  {'color': '#F4845F', 'marker': 'D', 'ls': '--', 'lw': 2.0, 'zorder': 9},
    'DS-CNN-S\n(23.7K params)':       {'color': '#457B9D', 'marker': 'o', 'ls': '-.',  'lw': 2.0, 'zorder': 8},
    'BC-ResNet-1\n(7.5K params)':     {'color': '#2A9D8F', 'marker': '^', 'ls': ':',  'lw': 2.0, 'zorder': 8},
}

noise_types = ['factory', 'white', 'babble', 'street', 'pink']
noise_titles = {
    'factory': 'Factory Noise',
    'white':   'White Noise (Broadband)',
    'babble':  'Babble Noise (Non-stationary)',
    'street':  'Street Noise',
    'pink':    'Pink Noise (1/f)',
}

fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=True)

for idx, noise in enumerate(noise_types):
    ax = axes[idx]

    for model_name, model_data in data.items():
        s = styles[model_name]
        values = model_data[noise]

        if values is None:
            continue  # skip models without data for this noise

        ax.plot(snr_levels, values,
                color=s['color'], marker=s['marker'], ls=s['ls'],
                lw=s['lw'], markersize=7, label=model_name,
                zorder=s['zorder'])

        # Add clean accuracy as dashed horizontal line
        ax.axhline(y=model_data['clean'], color=s['color'],
                   ls=':', alpha=0.3, lw=1)

    ax.set_title(noise_titles[noise], fontsize=14, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(snr_levels)
    ax.set_xticklabels(snr_labels)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.axvspan(-15, -5, alpha=0.05, color='red', label='_')  # extreme zone
    ax.axvspan(-5, 15, alpha=0.05, color='green', label='_')  # practical zone

    # Highlight DS-CNN-S collapse on white noise
    if noise == 'white':
        ax.annotate('CNN Collapse!', xy=(0, 13.9), xytext=(3, 25),
                    fontsize=9, color='#457B9D', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#457B9D'),
                    ha='center')

    # Highlight pink noise severity
    if noise == 'pink':
        ax.annotate('1/f formant\noverlap', xy=(-15, 9.9), xytext=(-10, 25),
                    fontsize=9, color='#F4845F', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#F4845F'),
                    ha='center')

# Single legend at bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           fontsize=11, bbox_to_anchor=(0.5, -0.02),
           frameon=True, fancybox=True, shadow=True)

fig.suptitle('Noise Robustness: NanoMamba SA-SSM vs CNN Baselines',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('noise_robustness.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('noise_robustness.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: noise_robustness.png, noise_robustness.pdf")
plt.show()
