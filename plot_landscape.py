import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(12, 7))

# ===== Models data =====
models = {
    'NanoMamba-Tiny':  {'params': 4634,    'acc': 92.94, 'color': '#FF0000', 'marker': '*',  'ms': 450, 'ours': True},
    'NanoMamba-Small': {'params': 12032,   'acc': 95.2,  'color': '#FF0000', 'marker': '*',  'ms': 550, 'ours': True},
    'BC-ResNet-1':     {'params': 7464,    'acc': 96.6,  'color': '#2E8B57', 'marker': 'D',  'ms': 140, 'ours': False},
    'BC-ResNet-2':     {'params': 21860,   'acc': 97.2,  'color': '#228B22', 'marker': 'D',  'ms': 160, 'ours': False},
    'BC-ResNet-3':     {'params': 43200,   'acc': 97.5,  'color': '#006400', 'marker': 'D',  'ms': 180, 'ours': False},
    'BC-ResNet-6':     {'params': 82500,   'acc': 97.8,  'color': '#004400', 'marker': 'D',  'ms': 150, 'ours': False},
    'DS-CNN-S':        {'params': 23756,   'acc': 94.4,  'color': '#4169E1', 'marker': 's',  'ms': 150, 'ours': False},
    'DS-CNN-M':        {'params': 55636,   'acc': 95.4,  'color': '#2850B0', 'marker': 's',  'ms': 130, 'ours': False},
    'DS-CNN-L':        {'params': 218676,  'acc': 95.4,  'color': '#1E3A80', 'marker': 's',  'ms': 130, 'ours': False},
    'KWT-1':           {'params': 607000,  'acc': 97.7,  'color': '#999999', 'marker': '^',  'ms': 100, 'ours': False},
    'KW-Mamba':        {'params': 3400000, 'acc': 97.5,  'color': '#BBBBBB', 'marker': 'v',  'ms': 100, 'ours': False},
}

for name, d in models.items():
    zorder = 10 if d['ours'] else 4
    edge = 'black' if d['ours'] else '#555555'
    lw = 2 if d['ours'] else 0.8
    ax.scatter(d['params'], d['acc'], c=d['color'], marker=d['marker'],
              s=d['ms'], zorder=zorder, edgecolors=edge, linewidth=lw)

# Labels - Ours
ax.annotate('NanoMamba-Tiny\n4.6K  |  92.9%', xy=(4634, 92.94),
           xytext=(1800, 91.5), fontsize=10, fontweight='bold', color='#CC0000',
           arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.5))

ax.annotate('NanoMamba-Small\n12K  |  95.2%', xy=(12032, 95.2),
           xytext=(2500, 96.2), fontsize=11, fontweight='bold', color='#CC0000',
           arrowprops=dict(arrowstyle='->', color='#CC0000', lw=2),
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE', edgecolor='#CC0000', alpha=0.8))

# Labels - BC-ResNet
ax.annotate('BC-ResNet-1\n7.5K | 96.6%', xy=(7464, 96.6),
           xytext=(3000, 97.8), fontsize=8.5, color='#2E8B57',
           arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=1))
ax.annotate('BC-ResNet-2\n21.9K | 97.2%', xy=(21860, 97.2),
           xytext=(50000, 98.2), fontsize=8.5, color='#228B22',
           arrowprops=dict(arrowstyle='->', color='#228B22', lw=1))
ax.annotate('BC-ResNet-3\n43.2K | 97.5%', xy=(43200, 97.5),
           xytext=(100000, 96.9), fontsize=8.5, color='#006400',
           arrowprops=dict(arrowstyle='->', color='#006400', lw=1))
ax.annotate('BC-ResNet-6\n82.5K | 97.8%', xy=(82500, 97.8),
           xytext=(150000, 98.5), fontsize=8, color='#004400',
           arrowprops=dict(arrowstyle='->', color='#004400', lw=1))

# Labels - DS-CNN
ax.annotate('DS-CNN-S\n23.7K | 94.4%', xy=(23756, 94.4),
           xytext=(55000, 93.5), fontsize=8.5, color='#4169E1',
           arrowprops=dict(arrowstyle='->', color='#4169E1', lw=1))
ax.annotate('DS-CNN-M', xy=(55636, 95.4), xytext=(90000, 95.9),
           fontsize=7.5, color='#2850B0', arrowprops=dict(arrowstyle='->', color='#2850B0', lw=0.8))
ax.annotate('DS-CNN-L', xy=(218676, 95.4), xytext=(350000, 94.5),
           fontsize=7.5, color='#1E3A80', arrowprops=dict(arrowstyle='->', color='#1E3A80', lw=0.8))

# Labels - Large
ax.annotate('KWT-1\n607K', xy=(607000, 97.7), xytext=(800000, 96.8),
           fontsize=7, color='gray', arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
ax.annotate('KW-Mamba\n3.4M', xy=(3400000, 97.5), xytext=(3500000, 96.3),
           fontsize=7, color='gray', arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))

# Ultra-compact zone
rect1 = Rectangle((2000, 91), 13000, 8.5, facecolor='red', alpha=0.04, edgecolor='red',
                   linestyle='--', linewidth=1.2)
ax.add_patch(rect1)
ax.text(2200, 98.8, '< 15K params (Edge/MCU)', fontsize=9, color='red', alpha=0.5, style='italic')

# Comparison arrows
ax.annotate('', xy=(12032, 95.2), xytext=(23756, 94.4),
           arrowprops=dict(arrowstyle='<->', color='orange', lw=2, linestyle='--'))
ax.text(16500, 94.1, '2x smaller\n+0.8%', fontsize=8, fontweight='bold', color='darkorange', ha='center',
       bbox=dict(boxstyle='round', facecolor='#FFF8E8', edgecolor='orange', alpha=0.9))

ax.annotate('', xy=(12032, 95.2), xytext=(43200, 97.5),
           arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5, linestyle='--'))
ax.text(28000, 96.8, '3.6x smaller\n-2.3%', fontsize=8, color='purple', ha='center',
       bbox=dict(boxstyle='round', facecolor='#F8F0FF', edgecolor='purple', alpha=0.9))

ax.set_xscale('log')
ax.set_xlabel('Parameters (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('Clean Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('KWS Model Landscape - GSC V2 (12-class)\nNanoMamba vs State-of-the-Art',
            fontsize=14, fontweight='bold')
ax.set_xlim(1500, 6000000)
ax.set_ylim(91, 99)
ax.grid(True, alpha=0.25, which='both')

legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#FF0000', markersize=18,
           markeredgecolor='black', markeredgewidth=1.5, label='NanoMamba (Ours)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#228B22', markersize=10,
           markeredgecolor='#555', label='BC-ResNet'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#4169E1', markersize=10,
           markeredgecolor='#555', label='DS-CNN'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#999999', markersize=10,
           markeredgecolor='#555', label='Large Models'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(r'C:\Users\jinho\Downloads\NanoMamba-Interspeech2026\model_landscape.png', dpi=150, bbox_inches='tight')
print("Saved: model_landscape.png")
