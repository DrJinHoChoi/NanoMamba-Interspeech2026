"""
Structural Noise Robustness Comparison:
NanoMamba-Tiny vs Small vs Tiny-DualPCEN (with GTCRN enhancer)
Full SNR curves for all 5 noise types + Clean baseline comparison

Also compares with previous results (without GTCRN/structural changes)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'sans-serif'

snr_levels = [-15, -10, -5, 0, 5, 10, 15]

# ===== NEW results (5 structural changes + GTCRN enhancer) =====
new_data = {
    'NanoMamba-Tiny\n(4.6K, +GTCRN)': {
        'factory': [60.3, 63.1, 66.3, 68.7, 71.0, 73.0, 74.1],
        'white':   [59.7, 60.8, 63.3, 66.5, 69.1, 70.9, 72.5],
        'babble':  [43.9, 47.3, 54.8, 59.3, 65.5, 70.4, 73.2],
        'street':  [59.5, 59.9, 61.6, 63.6, 66.6, 69.5, 71.7],
        'pink':    [59.4, 60.6, 63.9, 67.5, 70.0, 71.7, 73.1],
        'clean': 79.6,
        'color': '#F4845F', 'marker': 'D', 'ls': '--', 'lw': 2.0,
    },
    'NanoMamba-Small\n(12K, +GTCRN)': {
        'factory': [60.9, 70.0, 75.5, 78.0, 79.5, 80.6, 81.7],
        'white':   [55.6, 60.4, 64.9, 70.1, 74.6, 78.2, 80.6],
        'babble':  [61.0, 63.3, 67.4, 71.5, 74.4, 77.2, 79.5],
        'street':  [57.6, 59.6, 62.6, 65.9, 71.9, 76.0, 79.7],
        'pink':    [54.1, 60.9, 68.5, 74.3, 78.3, 80.3, 81.8],
        'clean': 87.7,
        'color': '#E63946', 'marker': 's', 'ls': '-', 'lw': 2.5,
    },
    'NanoMamba-Tiny-DualPCEN\n(4.9K, +GTCRN)': {
        'factory': [54.9, 63.7, 69.6, 76.5, 79.4, 80.7, 81.5],
        'white':   [59.1, 60.4, 63.9, 68.0, 72.3, 75.4, 78.5],
        'babble':  [60.4, 65.4, 70.8, 77.0, 79.6, 81.7, 83.2],
        'street':  [57.3, 59.5, 63.1, 66.2, 72.0, 74.7, 79.0],
        'pink':    [54.4, 60.1, 67.2, 71.9, 77.9, 80.4, 82.0],
        'clean': 87.2,
        'color': '#9B2335', 'marker': '*', 'ls': '-', 'lw': 3.0,
    },
}

# ===== PREVIOUS results (no GTCRN, no structural changes) =====
prev_data = {
    'NanoMamba-Tiny (prev)': {
        'factory': [38.4, 56.1, 70.1, 77.6, 83.2, 85.1, 86.6],
        'white':   [20.2, 51.6, 69.3, 79.8, 86.2, 90.1, 91.8],
        'babble':  [58.6, 60.4, 65.0, 69.6, 77.3, 84.1, 87.4],
        'street':  [46.8, 58.9, 71.1, 78.8, 85.9, 89.0, 91.7],
        'pink':    [ 9.9, 38.3, 69.4, 81.9, 88.6, 91.3, 92.5],
        'clean': 92.9,
        'color': '#F4845F', 'marker': 'o', 'ls': ':', 'lw': 1.5,
    },
    'NanoMamba-Small (prev)': {
        'factory': [22.7, 47.9, 69.2, 78.0, 83.9, 87.0, 89.4],
        'white':   [17.6, 52.6, 73.9, 83.9, 90.2, 93.0, 94.2],
        'babble':  [61.5, 67.8, 73.8, 79.2, 85.9, 89.7, 92.1],
        'street':  None,
        'pink':    None,
        'clean': 95.2,
        'color': '#E63946', 'marker': 'o', 'ls': ':', 'lw': 1.5,
    },
}

# ===== Plot: 5 noise types =====
noise_types = ['factory', 'white', 'babble', 'street', 'pink']
noise_titles = {
    'factory': 'Factory Noise',
    'white': 'White Noise',
    'babble': 'Babble Noise',
    'street': 'Street Noise',
    'pink': 'Pink Noise (1/f)',
}

fig, axes = plt.subplots(1, 5, figsize=(32, 7), sharey=True)

for idx, noise in enumerate(noise_types):
    ax = axes[idx]

    # Plot previous results (dashed thin)
    for name, d in prev_data.items():
        vals = d[noise]
        if vals is None:
            continue
        ax.plot(snr_levels, vals,
                color=d['color'], marker=d['marker'], ls=d['ls'],
                lw=d['lw'], markersize=5, label=name, alpha=0.5,
                zorder=5)
        ax.axhline(y=d['clean'], color=d['color'], ls=':', alpha=0.2, lw=1)

    # Plot new results (solid thick)
    for name, d in new_data.items():
        vals = d[noise]
        ax.plot(snr_levels, vals,
                color=d['color'], marker=d['marker'], ls=d['ls'],
                lw=d['lw'], markersize=8, label=name, zorder=10)
        ax.axhline(y=d['clean'], color=d['color'], ls='--', alpha=0.4, lw=1)

    ax.set_title(noise_titles[noise], fontsize=14, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(snr_levels)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.axvspan(-15, -5, alpha=0.05, color='red')
    ax.axvspan(-5, 15, alpha=0.03, color='green')

    # Highlight key improvements at -15dB
    if noise == 'white':
        ax.annotate('prev: 20.2%', xy=(-15, 20.2), xytext=(-10, 15),
                    fontsize=8, color='#F4845F', alpha=0.7,
                    arrowprops=dict(arrowstyle='->', color='#F4845F', alpha=0.5))
        ax.annotate('+GTCRN: 59.7%', xy=(-15, 59.7), xytext=(-8, 52),
                    fontsize=8, color='#F4845F', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#F4845F'))
    elif noise == 'pink':
        ax.annotate('prev: 9.9%', xy=(-15, 9.9), xytext=(-10, 5),
                    fontsize=8, color='#F4845F', alpha=0.7,
                    arrowprops=dict(arrowstyle='->', color='#F4845F', alpha=0.5))
        ax.annotate('+GTCRN: 59.4%', xy=(-15, 59.4), xytext=(-8, 48),
                    fontsize=8, color='#F4845F', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#F4845F'))

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.04),
           frameon=True, fancybox=True, shadow=True)

fig.suptitle('NanoMamba Noise Robustness: GTCRN Enhanced vs Previous (no enhancer)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('structural_comparison.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: structural_comparison.png")


# ===== Summary Table =====
print("\n" + "=" * 90)
print("COMPARISON TABLE: New (GTCRN) vs Previous (no enhancer)")
print("=" * 90)
print(f"{'Noise':<10} | {'Model':<25} | {'Clean':>7} | {'-15dB':>7} | {'0dB':>7} | {'15dB':>7}")
print("-" * 90)

for noise in noise_types:
    # Previous Tiny
    prev_tiny = prev_data['NanoMamba-Tiny (prev)']
    pv = prev_tiny[noise]
    if pv:
        print(f"{noise:<10} | {'NanoMamba-Tiny (prev)':<25} | {prev_tiny['clean']:>6.1f}% | {pv[0]:>6.1f}% | {pv[3]:>6.1f}% | {pv[6]:>6.1f}%")

    # New Tiny
    new_tiny = new_data['NanoMamba-Tiny\n(4.6K, +GTCRN)']
    nv = new_tiny[noise]
    print(f"{'':<10} | {'NanoMamba-Tiny (+GTCRN)':<25} | {new_tiny['clean']:>6.1f}% | {nv[0]:>6.1f}% | {nv[3]:>6.1f}% | {nv[6]:>6.1f}%")

    # Delta
    if pv:
        d_clean = new_tiny['clean'] - prev_tiny['clean']
        d_15 = nv[0] - pv[0]
        d_0 = nv[3] - pv[3]
        d15 = nv[6] - pv[6]
        print(f"{'':<10} | {'   Δ':<25} | {d_clean:>+6.1f}  | {d_15:>+6.1f}  | {d_0:>+6.1f}  | {d15:>+6.1f} ")

    # New DualPCEN
    new_dp = new_data['NanoMamba-Tiny-DualPCEN\n(4.9K, +GTCRN)']
    dv = new_dp[noise]
    print(f"{'':<10} | {'Tiny-DualPCEN (+GTCRN)':<25} | {new_dp['clean']:>6.1f}% | {dv[0]:>6.1f}% | {dv[3]:>6.1f}% | {dv[6]:>6.1f}%")
    print("-" * 90)

# Key observations
print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)
print("1. ALL models show Clean degradation with GTCRN:")
print(f"   - Tiny:  92.9% → 79.6% (Δ = -13.3%p)")
print(f"   - Small: 95.2% → 87.7% (Δ = -7.5%p)")
print(f"   - DualPCEN: ~94.8% → 87.2% (Δ = ~-7.6%p)")
print()
print("2. GTCRN dramatic improvement at -15dB:")
print(f"   - White:  20.2% → 59.7% (Δ = +39.5%p)")
print(f"   - Pink:   9.9% → 59.4% (Δ = +49.5%p)")
print(f"   - Factory: 38.4% → 60.3% (Δ = +21.9%p)")
print()
print("3. DualPCEN outperforms Tiny at 0dB+ (better routing)")
print(f"   - Babble 0dB: 59.3% vs 77.0% (DualPCEN +17.7%p)")
print(f"   - Factory 0dB: 68.7% vs 76.5% (DualPCEN +7.8%p)")
print()
print("4. All -15dB results converge to ~55-60% regardless of noise type")
print("   → This is likely GTCRN's enhancement ceiling")
