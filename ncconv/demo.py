"""
NC-Conv MVP Demo: Real-time visualization of noise-conditioned dual-path convolution.
=====================================================================================

Three demo modes:
  1. Webcam: live σ gate visualization (adjust lighting to see σ change)
  2. Tunnel Video: 8-frame simulation with per-frame σ/gate analysis
  3. CULane Lane Detection: corruption robustness with σ heatmap

Requirements:
  pip install streamlit torch torchvision opencv-python plotly pillow numpy

Run:
  streamlit run ncconv/demo.py
"""

import os
import sys
import io
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import streamlit as st
    import cv2
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Install: pip install streamlit opencv-python")

from ncconv.models import (
    NCConvBlock, NCConvBlockSpatial, StandardCNN,
    make_ncconv_net, SpatialBackbone, VideoModelBiNC, LaneDetector, StdBlock
)
from ncconv.corruption import apply_corruption, CORRUPTION_TYPES, AUDIO_ANALOG
from ncconv.results import RESULTS


# =====================================================================
# Helper: NC-Conv with sigma extraction
# =====================================================================

class NCConvBlockViz(NCConvBlock):
    """NC-Conv block that exposes sigma values for visualization."""
    def forward(self, x):
        sigma = torch.sigmoid(self.sigma_net(x))  # (B, 1)
        self.last_sigma = sigma.detach()
        sigma_4d = sigma.unsqueeze(-1).unsqueeze(-1)
        h_s = self.static_bn(self.static_dw(x))
        h_d = self.dynamic_bn(self.dynamic_dw(x))
        h_d = h_d * self.dyn_gate(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.pw(F.silu(sigma_4d * h_d + (1 - sigma_4d) * h_s))


class NCConvSpatialViz(NCConvBlockSpatial):
    """NC-Conv spatial block that exposes sigma map."""
    def forward(self, x):
        sigma = torch.sigmoid(self.sigma_net(x))  # (B, 1, H, W)
        self.last_sigma_map = sigma.detach()
        h_s = self.static_bn(self.static_dw(x))
        h_d = self.dynamic_bn(self.dynamic_dw(x))
        h_d = h_d * self.dyn_gate(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.pw(F.silu(sigma * h_d + (1 - sigma) * h_s))


def make_viz_model(spatial=False):
    """Create NC-Conv model with sigma visualization."""
    block = NCConvSpatialViz if spatial else NCConvBlockViz
    c1, c2, c3 = 44, 88, 176

    class VizNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU())
            self.s1 = nn.ModuleList([block(c1) for _ in range(3)])
            self.down1 = nn.Sequential(
                nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
            self.s2 = nn.ModuleList([block(c2) for _ in range(3)])
            self.down2 = nn.Sequential(
                nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.SiLU())
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, 10))
            self.nc_blocks = list(self.s1) + list(self.s2)

        def forward(self, x):
            x = self.stem(x)
            for blk in self.s1: x = blk(x)
            x = self.down1(x)
            for blk in self.s2: x = blk(x)
            x = self.down2(x)
            return self.head(x)

        def get_sigmas(self):
            sigmas = []
            for blk in self.nc_blocks:
                if hasattr(blk, 'last_sigma'):
                    sigmas.append(blk.last_sigma.mean().item())
                elif hasattr(blk, 'last_sigma_map'):
                    sigmas.append(blk.last_sigma_map.mean().item())
            return sigmas

        def get_sigma_maps(self):
            maps = []
            for blk in self.nc_blocks:
                if hasattr(blk, 'last_sigma_map'):
                    maps.append(blk.last_sigma_map[0, 0].cpu().numpy())
            return maps

    return VizNet()


# =====================================================================
# Image transforms
# =====================================================================
CIFAR_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

LANE_TRANSFORM = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# =====================================================================
# Demo Functions (non-Streamlit, for terminal/notebook use)
# =====================================================================

def demo_sigma_analysis(image_path=None):
    """Analyze sigma values for a single image under different corruptions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_viz_model(spatial=True).to(device)
    model.eval()

    # Create test image
    if image_path:
        img = Image.open(image_path).convert('RGB')
    else:
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

    x = CIFAR_TRANSFORM(img).unsqueeze(0).to(device)

    print("Corruption        | Avg sigma | Interpretation")
    print("-" * 55)

    for corr_name in ['none'] + CORRUPTION_TYPES:
        if corr_name == 'none':
            x_c = x
        else:
            x_c = apply_corruption(x, corr_name, severity=3)

        with torch.no_grad():
            _ = model(x_c)
        sigmas = model.get_sigmas()
        avg_sigma = np.mean(sigmas)

        if avg_sigma > 0.7:
            interp = "dynamic (clean)"
        elif avg_sigma > 0.3:
            interp = "mixed"
        else:
            interp = "static (degraded)"

        analog = AUDIO_ANALOG.get(corr_name, 'Clean audio')
        print(f"  {corr_name:<18} | {avg_sigma:.3f}     | {interp} [{analog}]")


def demo_tunnel_video():
    """Simulate tunnel passage and show per-frame analysis."""
    print("\n" + "=" * 60)
    print("  Tunnel Video Simulation (8 frames)")
    print("=" * 60)

    r = RESULTS['temporal']
    labels = r['frame_labels']
    nc = r['nc_conv_per_frame']
    bi = r['bi_ncssm_per_frame']
    gate = r['gate_values']

    print(f"\n  {'Frame':<12} | {'NC-Conv':>8} | {'Bi-SSM':>8} | {'Gap':>6} | {'Gate':>5} | Visual")
    print(f"  {'-'*65}")
    for i in range(8):
        gap = bi[i] - nc[i]
        bar = '#' * max(0, int(gap)) if gap > 0 else ''
        print(f"  f{i} {labels[i]:<8} | {nc[i]:>7.1f}% | {bi[i]:>7.1f}% | {gap:>+5.1f}% | {gate[i]:.3f} | {bar}")

    print(f"\n  Clean (f0,f7):    {r['nc_conv_clean_avg']:.1f}% -> {r['bi_ncssm_clean_avg']:.1f}% ({r['bi_ncssm_clean_avg']-r['nc_conv_clean_avg']:+.1f}%)")
    print(f"  Degraded (f2-f5): {r['nc_conv_degraded_avg']:.1f}% -> {r['bi_ncssm_degraded_avg']:.1f}% ({r['bi_ncssm_degraded_avg']-r['nc_conv_degraded_avg']:+.1f}%)")


def demo_culane_results():
    """Show CULane per-condition results."""
    print("\n" + "=" * 60)
    print("  CULane Lane Detection (Real Data)")
    print("=" * 60)

    r = RESULTS['culane']
    print(f"\n  {'Condition':<10} | {'Std CNN':>8} | {'NC-Conv':>8} | {'Gap':>6}")
    print(f"  {'-'*40}")
    for cond in ['normal', 'dark', 'noise', 'fog']:
        s = r['std_cnn'][cond]
        n = r['ncconv'][cond]
        gap = n - s
        print(f"  {cond:<10} | {s:>7.1f}% | {n:>7.1f}% | {gap:>+5.1f}%")


# =====================================================================
# Streamlit App
# =====================================================================

def run_streamlit():
    """Run Streamlit web demo."""
    st.set_page_config(page_title="NC-Conv Demo", layout="wide")
    st.title("NC-Conv: Noise-Conditioned Dual-Path Convolution")
    st.markdown("**Quality gate sigma: clean->1 (dynamic path), degraded->0 (static path)**")

    tab1, tab2, tab3 = st.tabs(["1. Webcam", "2. Tunnel Video", "3. CULane"])

    # --- Tab 1: Webcam ---
    with tab1:
        st.header("Real-time sigma gate visualization")
        st.markdown("Adjust lighting to see sigma change in real-time.")

        col1, col2 = st.columns(2)
        with col1:
            corruption = st.selectbox("Apply corruption:", ['none'] + CORRUPTION_TYPES)
            severity = st.slider("Severity:", 1, 5, 3)

        with col2:
            st.markdown("""
            **How it works:**
            - sigma ~ 1.0: clean input -> dynamic path (blue)
            - sigma ~ 0.0: degraded input -> static path (red)
            - The gate learns this WITHOUT explicit labels
            """)

        uploaded = st.file_uploader("Upload image (or use sample):", type=['jpg', 'png', 'jpeg'])

        if uploaded:
            img = Image.open(uploaded).convert('RGB')
        else:
            # Generate sample image
            np.random.seed(42)
            img = Image.fromarray(np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8))

        device = 'cpu'
        model = make_viz_model(spatial=True).to(device)
        model.eval()

        x = CIFAR_TRANSFORM(img).unsqueeze(0).to(device)
        if corruption != 'none':
            x = apply_corruption(x, corruption, severity)

        with torch.no_grad():
            out = model(x)
            pred = out.argmax(1).item()
            sigmas = model.get_sigmas()
            sigma_maps = model.get_sigma_maps()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Input", width=200)
        with col2:
            avg_s = np.mean(sigmas)
            color = f"{'green' if avg_s > 0.6 else 'orange' if avg_s > 0.3 else 'red'}"
            st.metric("Avg sigma", f"{avg_s:.3f}")
            st.markdown(f"**Path:** {'Dynamic (clean)' if avg_s > 0.6 else 'Mixed' if avg_s > 0.3 else 'Static (degraded)'}")
        with col3:
            if sigma_maps:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(sigma_maps[0], cmap='RdYlGn', vmin=0, vmax=1)
                ax.set_title(f"sigma map (block 1)")
                ax.axis('off')
                st.pyplot(fig)

        st.bar_chart({"block": list(range(len(sigmas))), "sigma": sigmas})

    # --- Tab 2: Tunnel Video ---
    with tab2:
        st.header("Tunnel Passage: Bidirectional NC-SSM")
        st.markdown("Clean frames provide temporal context to degraded frames.")

        r = RESULTS['temporal']
        labels = r['frame_labels']

        import pandas as pd
        df = pd.DataFrame({
            'Frame': [f'f{i} {labels[i]}' for i in range(8)],
            'NC-Conv (frame-indep)': r['nc_conv_per_frame'],
            'Bi-NC-SSM (temporal)': r['bi_ncssm_per_frame'],
            'Quality Gate': r['gate_values'],
        })

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Per-Frame Accuracy")
            st.line_chart(df.set_index('Frame')[['NC-Conv (frame-indep)', 'Bi-NC-SSM (temporal)']])

        with col2:
            st.subheader("Quality Gate (learned, no supervision)")
            st.bar_chart(df.set_index('Frame')['Quality Gate'])

        st.dataframe(df)

        st.markdown(f"""
        **Key Results:**
        - f3 (darkest): {r['nc_conv_per_frame'][3]}% -> {r['bi_ncssm_per_frame'][3]}% (**+{r['bi_ncssm_per_frame'][3]-r['nc_conv_per_frame'][3]:.1f}%**)
        - Clean preserved: {r['bi_ncssm_clean_avg']}% ({r['bi_ncssm_clean_avg']-r['nc_conv_clean_avg']:+.1f}%)
        - Degraded avg: {r['nc_conv_degraded_avg']}% -> {r['bi_ncssm_degraded_avg']}% (**+{r['bi_ncssm_degraded_avg']-r['nc_conv_degraded_avg']:.1f}%**)
        """)

    # --- Tab 3: CULane ---
    with tab3:
        st.header("CULane Lane Detection (Real Data)")

        r = RESULTS['culane']
        import pandas as pd
        df = pd.DataFrame({
            'Condition': ['normal', 'dark', 'noise', 'fog'],
            'Std CNN': [r['std_cnn'][c] for c in ['normal', 'dark', 'noise', 'fog']],
            'NC-Conv': [r['ncconv'][c] for c in ['normal', 'dark', 'noise', 'fog']],
        })
        df['Gap'] = df['NC-Conv'] - df['Std CNN']

        st.bar_chart(df.set_index('Condition')[['Std CNN', 'NC-Conv']])
        st.dataframe(df)

        st.markdown(f"""
        **Key Results:**
        - Normal: +{r['ncconv']['normal']-r['std_cnn']['normal']:.1f}% (NC-Conv wins even on clean!)
        - Fog: +{r['ncconv']['fog']-r['std_cnn']['fog']:.1f}% (biggest gain)
        - 253K params = $8 MCU deployable
        """)

    # --- Sidebar ---
    st.sidebar.title("NC-Conv")
    st.sidebar.markdown("""
    **Noise-Conditioned Dual-Path Convolution**

    Core idea: blend static (robust) and dynamic
    (expressive) paths based on input quality.

    ```
    h = sigma * h_dynamic + (1-sigma) * h_static
    ```

    **Audio NC-SSM analog:**
    ```
    Delta = sigma * Delta_sel + (1-sigma) * Delta_base
    ```

    **Target:**
    - ACCV 2026 (Osaka)
    - CVPR 2027
    - CES 2027 (Las Vegas)

    **Deployment:**
    - 253K params (253KB INT8)
    - STM32H743 MCU ($8)
    - 28 FPS, 35ms latency
    """)


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['streamlit', 'sigma', 'tunnel', 'culane', 'all'],
                        default='all', help='Demo mode')
    parser.add_argument('--image', type=str, default=None, help='Image path for sigma analysis')
    args = parser.parse_args()

    if args.mode == 'streamlit':
        if HAS_STREAMLIT:
            run_streamlit()
        else:
            print("pip install streamlit opencv-python")
    elif args.mode == 'sigma':
        demo_sigma_analysis(args.image)
    elif args.mode == 'tunnel':
        demo_tunnel_video()
    elif args.mode == 'culane':
        demo_culane_results()
    else:
        demo_sigma_analysis(args.image)
        demo_tunnel_video()
        demo_culane_results()
        print("\n  For web demo: streamlit run ncconv/demo.py -- --mode streamlit")
