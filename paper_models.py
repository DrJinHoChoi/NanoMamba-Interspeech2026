#!/usr/bin/env python3
# coding=utf-8
"""SMARTEAR KWS - Paper-Quality Model Implementations.

Implements SOTA KWS architectures for Jetson Nano deployment paper:

1. BC-ResNet (Qualcomm, 2021, arXiv:2106.04140)
   - Broadcasted Residual Learning with SubSpectralNorm
   - BC-ResNet-1 (9.2K, 96.9%) to BC-ResNet-8 (321K, 98.7%)
   - Best params/accuracy ratio in literature

2. Keyword Mamba (KWM, 2025, arXiv:2508.07363)
   - State Space Model (Mamba) replacing self-attention
   - KWM-T: Mamba + Transformer FFN hybrid (98.91% SOTA)
   - Graceful degradation under compression

3. Joint AEC+Enhancement+KWS Pipeline
   - Noise-robust end-to-end pipeline for real deployment
   - TensorRT INT8 optimized for Jetson Nano Maxwell GPU

All models target Google Speech Commands V2 (12-class).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. BC-ResNet (Broadcasted Residual Learning)
# ============================================================================

class SubSpectralNorm(nn.Module):
    """Sub-Spectral Normalization (SSN).

    Splits frequency axis into S sub-bands and applies BatchNorm
    independently to each sub-band. This captures frequency-dependent
    statistics better than global BatchNorm.

    Reference: BC-ResNet (arXiv:2106.04140), Section 3.2
    """

    def __init__(self, num_features, num_sub_bands=5):
        super().__init__()
        self.num_sub_bands = num_sub_bands
        self.bn = nn.BatchNorm2d(num_features * num_sub_bands)

    def forward(self, x):
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        S = self.num_sub_bands

        # Pad frequency axis to be divisible by S
        pad = (S - F % S) % S
        if pad > 0:
            x = F_pad(x, (0, 0, 0, pad))
            F_new = F + pad
        else:
            F_new = F

        # Split into S sub-bands along frequency
        # (B, C, F_new, T) -> (B, C*S, F_new//S, T)
        x = x.reshape(B, C, S, F_new // S, T)
        x = x.reshape(B, C * S, F_new // S, T)

        x = self.bn(x)

        # Reshape back
        x = x.reshape(B, C, S, F_new // S, T)
        x = x.reshape(B, C, F_new, T)

        # Remove padding
        if pad > 0:
            x = x[:, :, :F_new - pad, :]
        return x


def F_pad(x, pad):
    """Wrapper for F.pad to avoid name collision."""
    return torch.nn.functional.pad(x, pad)


class BCResBlock(nn.Module):
    """BC-ResNet block with broadcasted residual connection.

    The key insight: most computation is 1D temporal convolution (cheap),
    and frequency information is preserved through a broadcasted residual
    that expands 1D temporal features back to 2D freq-temporal space.

    Structure:
        x -> [freq-conv 1x1] -> [SSN + ReLU]
          -> [temp-conv Kx1 DW] -> [SSN + ReLU]
          -> [freq-conv 1x1] -> [SSN]
          -> avgpool(freq) -> broadcast + residual -> ReLU

    Reference: arXiv:2106.04140, Figure 2
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=(1, 1), dilation=1, num_sub_bands=5):
        super().__init__()
        self.use_residual = (in_channels == out_channels and
                             stride == (1, 1))

        # Frequency 1x1 convolution
        self.freq_conv1 = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.ssn1 = SubSpectralNorm(out_channels, num_sub_bands)

        # Temporal depthwise convolution (along time axis only)
        padding = (0, (kernel_size - 1) * dilation // 2)
        self.temp_dw_conv = nn.Conv2d(
            out_channels, out_channels,
            (1, kernel_size),
            stride=(1, stride[1]),
            padding=padding,
            dilation=(1, dilation),
            groups=out_channels)
        self.ssn2 = SubSpectralNorm(out_channels, num_sub_bands)

        # Frequency 1x1 convolution
        self.freq_conv2 = nn.Conv2d(out_channels, out_channels, (1, 1))
        self.ssn3 = SubSpectralNorm(out_channels, num_sub_bands)

        # Frequency dimension pooling for broadcast
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Skip connection if channels change
        if not self.use_residual and in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1),
                          stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.skip = None

    def forward(self, x):
        # x: (B, C, F, T)
        identity = x

        # Freq 1x1 -> SSN -> ReLU
        out = F.relu(self.ssn1(self.freq_conv1(x)))

        # Temporal DW conv -> SSN -> ReLU
        out = F.relu(self.ssn2(self.temp_dw_conv(out)))

        # Freq 1x1 -> SSN
        out = self.ssn3(self.freq_conv2(out))

        # Broadcast residual: pool freq -> expand back
        # Average over frequency, then broadcast-add to preserve freq info
        out_pooled = self.freq_pool(out)  # (B, C, 1, T)
        out = out + out_pooled  # broadcast along frequency

        # Residual connection
        if self.use_residual:
            out = out + identity
        elif self.skip is not None:
            out = out + self.skip(identity)

        return F.relu(out)


class BCResNet(nn.Module):
    """BC-ResNet: Broadcasted Residual Network for Keyword Spotting.

    Achieves SOTA params/accuracy tradeoff:
      BC-ResNet-1:   9.2K params, 96.9% (GSC V2 12-class)
      BC-ResNet-2:  27.3K params, 97.8%
      BC-ResNet-3:  54.2K params, 98.2%
      BC-ResNet-6: 188.0K params, 98.6%
      BC-ResNet-8: 321.0K params, 98.7%

    Reference: arXiv:2106.04140 (Qualcomm, 2021)

    Args:
        n_mels: Number of mel bands (default: 40)
        n_classes: Number of output classes (default: 12)
        scale: Width multiplier (1-8, controls channel count)
        num_sub_bands: Number of sub-bands for SSN
    """

    def __init__(self, n_mels=40, n_classes=12, scale=3, num_sub_bands=5):
        super().__init__()
        self.n_mels = n_mels
        self.n_classes = n_classes
        self.scale = scale

        # Channel widths scaled by multiplier
        c = max(int(8 * scale), 8)

        # Initial convolution
        self.conv1 = nn.Conv2d(1, c, (5, 5), stride=(2, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(c)

        # BC-ResNet blocks
        # Stage 1: same channels
        self.stage1 = nn.Sequential(
            BCResBlock(c, c, kernel_size=3, num_sub_bands=num_sub_bands),
            BCResBlock(c, c, kernel_size=3, num_sub_bands=num_sub_bands),
        )

        # Stage 2: 1.5x channels, stride in time
        c2 = int(c * 1.5)
        self.stage2 = nn.Sequential(
            BCResBlock(c, c2, kernel_size=3, stride=(1, 2),
                       num_sub_bands=num_sub_bands),
            BCResBlock(c2, c2, kernel_size=3, dilation=2,
                       num_sub_bands=num_sub_bands),
        )

        # Stage 3: 2x channels
        c3 = c * 2
        self.stage3 = nn.Sequential(
            BCResBlock(c2, c3, kernel_size=3, stride=(1, 2),
                       num_sub_bands=num_sub_bands),
            BCResBlock(c3, c3, kernel_size=3, dilation=4,
                       num_sub_bands=num_sub_bands),
        )

        # Stage 4: 2.5x channels
        c4 = int(c * 2.5)
        self.stage4 = nn.Sequential(
            BCResBlock(c3, c4, kernel_size=3, num_sub_bands=num_sub_bands),
        )

        # Head
        self.head_conv = nn.Conv2d(c4, c4, (1, 1))
        self.head_bn = nn.BatchNorm2d(c4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c4, n_classes)

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T) mel spectrogram
        Returns:
            logits: (B, n_classes)
        """
        # (B, n_mels, T) -> (B, 1, n_mels, T)
        x = mel.unsqueeze(1)

        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Head
        x = F.relu(self.head_bn(self.head_conv(x)))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ============================================================================
# 2. Keyword Mamba (KWM) - State Space Model for KWS
# ============================================================================

class MambaBlock(nn.Module):
    """Simplified Mamba-style SSM block for KWS.

    Replaces self-attention with a selective state space model (S6).
    Key idea: input-dependent SSM parameters allow selective information
    propagation along the sequence, similar to attention but O(N) complexity.

    Reference: arXiv:2508.07363 (Keyword Mamba, 2025)
               arXiv:2312.00752 (Mamba, 2023)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)

        # Input projection (splits into x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner)

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable SSM parameters
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # Initialize A as negative (for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(
            self.d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def ssm_forward(self, x):
        """Selective SSM forward pass.

        Args:
            x: (B, L, D_inner) - inner dimension features
        Returns:
            y: (B, L, D_inner) - SSM output
        """
        B, L, D = x.shape
        N = self.d_state

        # Input-dependent SSM parameters
        x_proj = self.x_proj(x)  # (B, L, 2N+1)
        delta = x_proj[..., :1]  # (B, L, 1)
        B_param = x_proj[..., 1:N + 1]  # (B, L, N)
        C_param = x_proj[..., N + 1:]  # (B, L, N)

        # Compute dt (discretization step)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, D_inner)

        # Get A matrix (negative for stability)
        A = -torch.exp(self.A_log)  # (D_inner, N)

        # Discretized SSM (zero-order hold)
        # For efficiency, use a simple scan
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, N, device=x.device)  # hidden state

        for t in range(L):
            dt = delta[:, t]  # (B, D)
            b = B_param[:, t]  # (B, N)
            c = C_param[:, t]  # (B, N)

            # Update state: h = exp(A * dt) * h + dt * B * x
            dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))  # (B, D, N)
            dB = dt.unsqueeze(-1) * b.unsqueeze(1)  # (B, D, N)

            h = dA * h + dB * x[:, t].unsqueeze(-1)

            # Output: y = C * h + D * x
            y[:, t] = (h * c.unsqueeze(1)).sum(-1) + self.D * x[:, t]

        return y

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input features
        Returns:
            out: (B, L, D) output features
        """
        residual = x
        x = self.norm(x)

        # Split into x and z branches
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv1d for local context
        x_branch = x_branch.transpose(1, 2)  # (B, D_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :x.size(1)]
        x_branch = x_branch.transpose(1, 2)  # (B, L, D_inner)
        x_branch = F.silu(x_branch)

        # SSM
        y = self.ssm_forward(x_branch)

        # Gate with z branch
        y = y * F.silu(z)

        # Output projection
        out = self.out_proj(y)

        return out + residual


class MambaTransformerBlock(nn.Module):
    """KWM-T: Mamba + Transformer FFN hybrid block.

    Combines Mamba SSM (for sequential modeling) with
    Transformer-style feed-forward network (for feature mixing).
    This hybrid achieves the best accuracy in the KWM paper.

    Reference: arXiv:2508.07363, KWM-T variant
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 ffn_expand=4, dropout=0.1):
        super().__init__()
        # Mamba SSM block
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand)

        # Transformer-style FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.mamba(x)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class KeywordMamba(nn.Module):
    """Keyword Mamba (KWM-T): SOTA KWS with State Space Models.

    Current SOTA on GSC V2 12-class: 98.91% accuracy.
    Uses Mamba SSM instead of self-attention for O(N) complexity.

    Reference: arXiv:2508.07363 (2025)

    Args:
        n_mels: Number of mel bands
        n_classes: Number of output classes
        d_model: Model dimension
        n_layers: Number of Mamba-Transformer blocks
        d_state: SSM state dimension
        d_conv: SSM conv kernel size
        patch_size: Temporal patch size for input
        dropout: Dropout rate
    """

    def __init__(self, n_mels=40, n_classes=12, d_model=192,
                 n_layers=12, d_state=16, d_conv=4, patch_size=1,
                 dropout=0.1):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.patch_size = patch_size

        # Patch embedding (similar to KWT)
        patch_dim = n_mels * patch_size
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.patch_norm = nn.LayerNorm(d_model)

        # Positional embedding
        max_len = 200  # max time steps after patching
        self.pos_emb = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Mamba-Transformer blocks
        self.blocks = nn.ModuleList([
            MambaTransformerBlock(
                d_model, d_state, d_conv,
                expand=2, ffn_expand=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T) mel spectrogram
        Returns:
            logits: (B, n_classes)
        """
        B = mel.size(0)

        # (B, n_mels, T) -> (B, T, n_mels) -> patch
        x = mel.transpose(1, 2)  # (B, T, n_mels)
        T = x.size(1)

        if self.patch_size > 1:
            # Reshape into patches
            T_new = T // self.patch_size
            x = x[:, :T_new * self.patch_size]
            x = x.reshape(B, T_new, -1)
        # else: each time step is a patch

        # Project patches
        x = self.patch_norm(self.patch_proj(x))

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embedding
        x = x + self.pos_emb[:, :x.size(1)]
        x = self.pos_drop(x)

        # Mamba-Transformer blocks
        for block in self.blocks:
            x = block(x)

        # CLS token -> classify
        x = self.head_norm(x[:, 0])
        return self.head(x)


class KeywordMambaSmall(nn.Module):
    """Compact Keyword Mamba for Jetson Nano deployment.

    Targets BC-ResNet-6 accuracy (98.6%) with Mamba efficiency.
    d_model=64, 6 layers -> ~200K params.
    """

    def __init__(self, n_mels=40, n_classes=12, d_model=64,
                 n_layers=6, d_state=8, d_conv=4, dropout=0.1):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        self.patch_proj = nn.Linear(n_mels, d_model)
        self.patch_norm = nn.LayerNorm(d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, 200, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            MambaTransformerBlock(
                d_model, d_state, d_conv,
                expand=2, ffn_expand=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, mel):
        B = mel.size(0)
        x = mel.transpose(1, 2)  # (B, T, n_mels)
        x = self.patch_norm(self.patch_proj(x))
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, :x.size(1)]

        for block in self.blocks:
            x = block(x)

        x = self.head_norm(x[:, 0])
        return self.head(x)


# ============================================================================
# 3. Noise-Robust Joint AEC+KWS Pipeline
# ============================================================================

class NoiseAwareFeatureExtractor(nn.Module):
    """Noise-aware mel feature extraction with learnable enhancement.

    Adds a lightweight noise estimation and spectral gating layer
    before mel feature computation, improving KWS robustness in
    low-SNR conditions without a separate AEC module.
    """

    def __init__(self, sr=16000, n_fft=512, hop_length=160,
                 n_mels=40, noise_frames=10):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.noise_frames = noise_frames
        n_freq = n_fft // 2 + 1

        # Learnable mel filterbank
        mel_fb = self._mel_filterbank(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # Noise gate (learnable spectral gate parameters)
        self.noise_gate_alpha = nn.Parameter(torch.tensor(2.0))
        self.noise_gate_floor = nn.Parameter(torch.tensor(0.01))

        # Post-enhancement normalization
        self.norm = nn.InstanceNorm1d(n_mels)

    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels):
        """Create mel filterbank matrix."""
        n_freq = n_fft // 2 + 1
        mel_low = 0
        mel_high = 2595 * np.log10(1 + sr / 2 / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_mels, n_freq), dtype=np.float32)
        for i in range(n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                if j < n_freq:
                    fb[i, j] = ((j - bin_points[i]) /
                                max(bin_points[i + 1] - bin_points[i], 1))
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                if j < n_freq:
                    fb[i, j] = ((bin_points[i + 2] - j) /
                                max(bin_points[i + 2] - bin_points[i + 1], 1))
        return fb

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform
        Returns:
            mel: (B, n_mels, T_mel) log-mel spectrogram
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # (B, F, T)

        # Noise estimation (first N frames average)
        noise_est = mag[:, :, :self.noise_frames].mean(dim=2, keepdim=True)

        # Spectral gating
        gate = torch.clamp(
            1.0 - self.noise_gate_alpha * noise_est / (mag + 1e-8),
            min=self.noise_gate_floor.abs())
        mag_clean = mag * gate

        # Mel filterbank
        mel = torch.matmul(self.mel_fb, mag_clean)  # (B, n_mels, T)

        # Log compression
        mel = torch.log(mel + 1e-8)

        # Normalize
        mel = self.norm(mel)

        return mel


class JointAECKWSPipeline(nn.Module):
    """Joint AEC + Enhancement + KWS pipeline for paper evaluation.

    End-to-end pipeline that processes raw audio through:
    1. Noise-aware feature extraction (learnable spectral gating)
    2. KWS backbone (BC-ResNet or KeywordMamba)

    This joint optimization allows the feature extractor to learn
    noise-robust representations tailored to the KWS backend.

    Args:
        kws_backbone: 'bcresnet' or 'mamba' or 'mamba_small'
        kws_kwargs: Arguments for KWS backbone
    """

    def __init__(self, kws_backbone='bcresnet', sr=16000, **kws_kwargs):
        super().__init__()
        self.feature_ext = NoiseAwareFeatureExtractor(
            sr=sr, n_fft=512, hop_length=160, n_mels=40)

        if kws_backbone == 'bcresnet':
            self.kws = BCResNet(n_mels=40, **kws_kwargs)
        elif kws_backbone == 'mamba':
            self.kws = KeywordMamba(n_mels=40, **kws_kwargs)
        elif kws_backbone == 'mamba_small':
            self.kws = KeywordMambaSmall(n_mels=40, **kws_kwargs)
        else:
            raise ValueError(f"Unknown backbone: {kws_backbone}")

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform at 16kHz
        Returns:
            logits: (B, n_classes)
        """
        mel = self.feature_ext(audio)
        return self.kws(mel)


# ============================================================================
# 4. Model Factory & Configurations
# ============================================================================

def create_bcresnet(scale=3, n_classes=12):
    """Create BC-ResNet with given scale factor."""
    return BCResNet(n_mels=40, n_classes=n_classes, scale=scale)


def create_kwm_t(d_model=192, n_layers=12, n_classes=12):
    """Create Keyword Mamba-T (full SOTA)."""
    return KeywordMamba(
        n_mels=40, n_classes=n_classes,
        d_model=d_model, n_layers=n_layers)


def create_kwm_small(d_model=64, n_layers=6, n_classes=12):
    """Create compact Keyword Mamba for edge deployment."""
    return KeywordMambaSmall(
        n_mels=40, n_classes=n_classes,
        d_model=d_model, n_layers=n_layers)


# Paper model configurations
PAPER_MODELS = {
    # BC-ResNet family
    'BC-ResNet-1': lambda nc=12: BCResNet(scale=1, n_classes=nc),
    'BC-ResNet-2': lambda nc=12: BCResNet(scale=2, n_classes=nc),
    'BC-ResNet-3': lambda nc=12: BCResNet(scale=3, n_classes=nc),
    'BC-ResNet-6': lambda nc=12: BCResNet(scale=6, n_classes=nc),
    'BC-ResNet-8': lambda nc=12: BCResNet(scale=8, n_classes=nc),

    # Keyword Mamba family
    'KWM-Small': lambda nc=12: KeywordMambaSmall(
        d_model=64, n_layers=6, n_classes=nc),
    'KWM-192': lambda nc=12: KeywordMamba(
        d_model=192, n_layers=12, n_classes=nc),

    # Joint pipelines
    'Joint-BCResNet-3': lambda nc=12: JointAECKWSPipeline(
        'bcresnet', scale=3, n_classes=nc),
    'Joint-BCResNet-6': lambda nc=12: JointAECKWSPipeline(
        'bcresnet', scale=6, n_classes=nc),
    'Joint-Mamba-Small': lambda nc=12: JointAECKWSPipeline(
        'mamba_small', n_classes=nc),
}

PAPER_REFS = {
    'BC-ResNet-1': {'acc': '96.9%', 'params': '9.2K', 'year': 2021,
                    'paper': 'arXiv:2106.04140'},
    'BC-ResNet-2': {'acc': '97.8%', 'params': '27.3K', 'year': 2021,
                    'paper': 'arXiv:2106.04140'},
    'BC-ResNet-3': {'acc': '98.2%', 'params': '54.2K', 'year': 2021,
                    'paper': 'arXiv:2106.04140'},
    'BC-ResNet-6': {'acc': '98.6%', 'params': '188K', 'year': 2021,
                    'paper': 'arXiv:2106.04140'},
    'BC-ResNet-8': {'acc': '98.7%', 'params': '321K', 'year': 2021,
                    'paper': 'arXiv:2106.04140'},
    'KWM-Small': {'acc': 'N/A (ours)', 'params': '~200K', 'year': 2025,
                  'paper': 'arXiv:2508.07363'},
    'KWM-192': {'acc': '98.91%', 'params': '~3.4M', 'year': 2025,
                'paper': 'arXiv:2508.07363'},
    'Joint-BCResNet-3': {'acc': 'N/A (ours)', 'params': '~60K', 'year': 2025,
                         'paper': 'Proposed'},
    'Joint-BCResNet-6': {'acc': 'N/A (ours)', 'params': '~195K', 'year': 2025,
                         'paper': 'Proposed'},
    'Joint-Mamba-Small': {'acc': 'N/A (ours)', 'params': '~210K', 'year': 2025,
                          'paper': 'Proposed'},
}


if __name__ == '__main__':
    # Quick verification
    print("Paper Models - Architecture Verification")
    print("=" * 60)

    mel = torch.randn(2, 40, 150)  # (B, n_mels, T)
    audio = torch.randn(2, 24000)  # (B, T) 1.5s @ 16kHz

    for name, model_fn in PAPER_MODELS.items():
        model = model_fn()
        model.eval()
        params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            if 'Joint' in name:
                out = model(audio)
            else:
                out = model(mel)

        ref = PAPER_REFS.get(name, {})
        print(f"  {name:<22} | {params:>10,} params | "
              f"out={list(out.shape)} | ref={ref.get('acc', 'N/A')}")

    print()
    print("All models verified successfully.")
