#!/usr/bin/env python3
# coding=utf-8
"""
NanoMamba - Spectral-Aware Selective State Space Model for Noise-Robust KWS
============================================================================

Core Novelty: Spectral-Aware SSM (SA-SSM)
  Standard Mamba's selection function (dt, B, C) is noise-agnostic -- the SSM
  parameters are projected only from temporal features. SA-SSM injects per-band
  SNR estimates directly into the selection mechanism:

    dt_t = softplus(W_dt * x_t  +  W_snr * s_t  +  b_dt)   # SNR-modulated step
    B_t  = W_B * x_t  +  alpha * diag(sigma(s_t)) * W_Bs * x_t  # SNR-gated input

  High-SNR frames -> large dt -> propagate information
  Low-SNR frames  -> small dt -> suppress noise

  This eliminates the need for a separate AEC/enhancement module.

Architecture:
  Raw Audio -> STFT -> SNR Estimator -> Mel -> Patch Proj -> N x SA-SSM -> GAP -> Classify

Variants:
  NanoMamba-Tiny:  d=16, layers=2, ~3.5K params
  NanoMamba-Small: d=24, layers=3, ~8.5K params
  NanoMamba-Base:  d=40, layers=4, ~28K params

Paper: Interspeech 2026
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SNR Estimator
# ============================================================================

class SNREstimator(nn.Module):
    """Per-frequency-band SNR estimator from magnitude spectrogram.

    Estimates noise floor from initial frames, then computes per-band SNR.
    Projects SNR to mel-scale for compact representation.

    Parameters: ~520 (2*n_freq + 2 for gate, reuses mel_fb from parent)
    """

    def __init__(self, n_freq=257, noise_frames=5):
        super().__init__()
        self.noise_frames = noise_frames

        # Learnable noise floor parameters
        self.noise_scale = nn.Parameter(torch.tensor(1.5))
        self.floor = nn.Parameter(torch.tensor(0.02))

    def forward(self, mag, mel_fb):
        """
        Args:
            mag: (B, F, T) magnitude spectrogram
            mel_fb: (n_mels, F) mel filterbank matrix
        Returns:
            snr_mel: (B, n_mels, T) per-mel-band SNR estimate
        """
        # Estimate noise floor from first N frames
        noise_est = mag[:, :, :self.noise_frames].mean(dim=2, keepdim=True)

        # Per-band SNR (linear scale)
        snr = mag / (self.noise_scale.abs() * noise_est + 1e-8)

        # Project to mel bands for compact representation
        # mel_fb: (n_mels, F), snr: (B, F, T) -> (B, n_mels, T)
        snr_mel = torch.matmul(mel_fb, snr)

        # Normalize to [0, 1] range with soft saturation
        snr_mel = torch.tanh(snr_mel / 10.0)

        return snr_mel


# ============================================================================
# Learnable Frequency Filter (Plug-in)
# ============================================================================

class FrequencyFilter(nn.Module):
    """Learnable frequency-bin mask applied to STFT magnitude.

    A lightweight plug-in module that learns to attenuate or preserve
    individual frequency bins in the magnitude spectrogram. This enables
    frequency-selective noise suppression (e.g., suppressing machine hum
    harmonics at 50/100/150/200/250 Hz) that sequential SSM processing
    cannot achieve directly.

    Initialized near-identity: sigmoid(3.0) ≈ 0.953, so training starts
    from pass-through behavior.

    Parameters: n_freq (default 257) scalar weights.
    """

    def __init__(self, n_freq=257):
        super().__init__()
        # Initialize at 3.0 so sigmoid(3.0) ≈ 0.953 (near pass-through)
        self.freq_mask = nn.Parameter(torch.ones(n_freq) * 3.0)

    def forward(self, mag):
        """Apply learnable frequency mask to magnitude spectrogram.

        Args:
            mag: (B, F, T) magnitude spectrogram from STFT
        Returns:
            filtered_mag: (B, F, T) frequency-filtered magnitude
        """
        mask = torch.sigmoid(self.freq_mask).unsqueeze(0).unsqueeze(-1)
        return mag * mask


# ============================================================================
# Frequency Convolution (Input-Dependent Spectral Filter)
# ============================================================================

class FreqConv(nn.Module):
    """Input-dependent frequency filter via 1D convolution on frequency axis.

    Unlike FrequencyFilter (static mask), this module applies a convolution
    ACROSS frequency bins for each time frame, producing an input-dependent
    mask. This transplants CNN's core advantage — local frequency selectivity
    — into the SSM pipeline with minimal parameters.

    At -15dB factory noise, the local frequency neighborhood reveals whether
    a bin is dominated by machinery harmonics or speech energy, enabling
    adaptive suppression that a static mask cannot achieve.

    Parameters: kernel_size weights + 1 bias (e.g., 5+1 = 6 params).
    """

    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2,
                              bias=True)
        # Initialize near-identity: small weights, bias=1.5 so sigmoid≈0.82
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv.bias, 1.5)

    def forward(self, mag):
        """Apply input-dependent frequency mask.

        Args:
            mag: (B, F, T) magnitude spectrogram from STFT
        Returns:
            filtered_mag: (B, F, T) frequency-filtered magnitude
        """
        B, F, T = mag.shape
        # Reshape: treat each time frame independently
        x = mag.permute(0, 2, 1).reshape(B * T, 1, F)  # (B*T, 1, F)
        mask = torch.sigmoid(self.conv(x))  # (B*T, 1, F) input-dependent!
        x = mag.permute(0, 2, 1).reshape(B * T, 1, F) * mask
        return x.reshape(B, T, F).permute(0, 2, 1)  # (B, F, T)


# ============================================================================
# Spectral-Aware SSM (SA-SSM)
# ============================================================================

class SpectralAwareSSM(nn.Module):
    """Spectral-Aware Selective State Space Model (SA-SSM).

    Modified Mamba S6 SSM where the selection parameters (dt, B) are modulated
    by per-band SNR estimates, enabling noise-aware temporal dynamics.

    Key equations:
      dt_t = softplus(W_dt * x_proj_dt + W_snr * s_t + b_dt)
      B_t  = W_B * x_t + alpha * diag(sigma(s_t)) * W_Bs * x_t
      h_t  = exp(A * dt_t) * h_{t-1} + dt_t * B_t * x_t
      y_t  = C_t * h_t + D * x_t
    """

    def __init__(self, d_inner, d_state, n_mels=40, mode='full'):
        """
        Args:
            d_inner: inner dimension of SSM
            d_state: state dimension N
            n_mels: number of mel bands for SNR input
            mode: ablation mode
                'full'     - both dt modulation + B gating (proposed)
                'dt_only'  - only dt modulation, no B gating
                'b_only'   - only B gating, no dt modulation
                'standard' - standard Mamba (no SNR modulation at all)
        """
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_mels = n_mels
        self.mode = mode

        # Standard SSM projections: x -> (dt_raw, B, C)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)

        # [NOVEL] SNR modulation projection: snr_mel -> (dt_mod, B_gate)
        # dt_mod: 1 value to shift dt, B_gate: d_state values to gate B
        self.snr_proj = nn.Linear(n_mels, d_state + 1, bias=True)

        # dt projection to expand dt to d_inner
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Initialize A as negative (for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(d_inner))

        # SNR gating strength (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # [NOVEL] Gate floor: minimum B-gate value to prevent over-suppression
        # at extreme low SNR (e.g., factory noise at -15dB)
        self.gate_floor = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, snr_mel):
        """
        Args:
            x: (B, L, d_inner) - feature sequence after conv1d + SiLU
            snr_mel: (B, L, n_mels) - per-mel-band SNR for each frame
        Returns:
            y: (B, L, d_inner) - SSM output
        """
        B, L, D = x.shape
        N = self.d_state

        # Standard projections from x
        x_proj = self.x_proj(x)  # (B, L, 2N+1)
        dt_raw = x_proj[..., :1]  # (B, L, 1)
        B_param = x_proj[..., 1:N + 1]  # (B, L, N)
        C_param = x_proj[..., N + 1:]  # (B, L, N)

        # [NOVEL] SNR modulation of selection parameters
        # Ablation modes control which components are active
        snr_mod = self.snr_proj(snr_mel)  # (B, L, N+1)

        if self.mode in ('full', 'dt_only'):
            dt_snr_shift = snr_mod[..., :1]  # (B, L, 1) - additive dt shift
        else:
            dt_snr_shift = torch.zeros_like(dt_raw)  # no dt modulation

        if self.mode in ('full', 'b_only'):
            raw_gate = torch.sigmoid(snr_mod[..., 1:])  # (B, L, N)
            # Gate floor prevents complete signal suppression at extreme low SNR
            B_gate = self.gate_floor + (1.0 - self.gate_floor) * raw_gate
        else:
            B_gate = torch.ones_like(B_param)  # no B gating

        # Compute dt with SNR modulation:
        # High SNR -> larger dt_snr_shift -> larger step -> propagate info
        # Low SNR  -> smaller dt_snr_shift -> smaller step -> suppress noise
        delta = F.softplus(
            self.dt_proj(dt_raw + dt_snr_shift)
        )  # (B, L, D_inner)

        # [NOVEL] SNR-gated B: B = B_standard * (1 - alpha + alpha * snr_gate)
        # When alpha=0, reduces to standard Mamba; when alpha=1, full gating
        if self.mode != 'standard':
            B_param = B_param * (1.0 - self.alpha + self.alpha * B_gate)
        # else: standard mode, B_param unchanged (no SNR gating)

        # Get A matrix (negative for stability)
        A = -torch.exp(self.A_log)  # (D_inner, N)

        # Precompute discretized A and B for all timesteps (vectorized)
        # A: (D, N), delta: (B, L, D)
        dA = torch.exp(
            A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1)
        )  # (B, L, D, N)
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, D, N)
        dBx = dB * x.unsqueeze(-1)  # (B, L, D, N) - input contribution

        # Sequential SSM scan (optimized: precomputed dA, dBx)
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, N, device=x.device)

        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        return y


# ============================================================================
# NanoMamba Block
# ============================================================================

class NanoMambaBlock(nn.Module):
    """Single NanoMamba block: LayerNorm -> in_proj -> DWConv -> SA-SSM -> Gate -> out_proj + Residual."""

    def __init__(self, d_model, d_state=4, d_conv=3, expand=1.5, n_mels=40,
                 ssm_mode='full'):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)

        self.norm = nn.LayerNorm(d_model)

        # Input projection: (d_model) -> (2 * d_inner) for [x_branch, z_gate]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner)

        # Spectral-Aware SSM
        self.sa_ssm = SpectralAwareSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            n_mels=n_mels,
            mode=ssm_mode)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, snr_mel):
        """
        Args:
            x: (B, L, d_model) - input sequence
            snr_mel: (B, L, n_mels) - per-mel-band SNR per frame
        Returns:
            out: (B, L, d_model) - output with residual
        """
        residual = x
        x = self.norm(x)

        # Project and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)

        # Local context via depthwise conv
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :x.size(1)]
        x_branch = x_branch.transpose(1, 2)  # (B, L, d_inner)
        x_branch = F.silu(x_branch)

        # Spectral-Aware SSM
        y = self.sa_ssm(x_branch, snr_mel)

        # Gate with z branch
        y = y * F.silu(z)

        # Output projection + residual
        return self.out_proj(y) + residual


# ============================================================================
# NanoMamba Model
# ============================================================================

class NanoMamba(nn.Module):
    """NanoMamba: Spectral-Aware SSM for Noise-Robust Keyword Spotting.

    End-to-end pipeline:
      Raw Audio -> STFT -> SNR Estimator -> Mel Features -> Patch Projection
      -> N x SA-SSM Blocks -> Global Average Pooling -> Classifier

    The SA-SSM blocks receive both the projected features AND per-mel-band SNR,
    enabling noise-aware temporal modeling without a separate enhancement module.

    Args:
        n_mels: Number of mel bands
        n_classes: Output classes (12 for GSC)
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Depthwise conv kernel size
        expand: Inner dimension expansion factor
        n_layers: Number of SA-SSM blocks
        sr: Sample rate
        n_fft: FFT size
        hop_length: STFT hop length
    """

    def __init__(self, n_mels=40, n_classes=12,
                 d_model=16, d_state=4, d_conv=3, expand=1.5,
                 n_layers=2, sr=16000, n_fft=512, hop_length=160,
                 ssm_mode='full', use_freq_filter=False,
                 use_freq_conv=False, freq_conv_ks=5,
                 weight_sharing=False, n_repeats=3):
        """
        Args:
            ssm_mode: SA-SSM ablation mode
                'full'     - proposed (dt + B modulation)
                'dt_only'  - only dt modulation
                'b_only'   - only B gating
                'standard' - standard Mamba (no SNR modulation)
            use_freq_filter: if True, apply learnable frequency mask on
                STFT magnitude before mel projection and SNR estimation.
                Adds n_freq (257) parameters.
            use_freq_conv: if True, apply input-dependent 1D convolution
                on frequency axis. Transplants CNN's local frequency
                selectivity into SSM. Adds ~6 parameters.
            freq_conv_ks: kernel size for FreqConv (default 5).
            weight_sharing: if True, use a single SA-SSM block repeated
                n_repeats times (depth of n_repeats, params of 1 block).
            n_repeats: number of times to repeat the shared block.
                Only used when weight_sharing=True.
        """
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.d_model = d_model
        self.ssm_mode = ssm_mode
        n_freq = n_fft // 2 + 1
        self.use_freq_filter = use_freq_filter
        self.use_freq_conv = use_freq_conv

        # 0. Frequency processing plug-in (optional)
        if use_freq_filter:
            self.freq_filter = FrequencyFilter(n_freq=n_freq)
        if use_freq_conv:
            self.freq_conv = FreqConv(kernel_size=freq_conv_ks)

        # 1. SNR Estimator
        self.snr_estimator = SNREstimator(n_freq=n_freq)

        # 2. Mel filterbank (fixed)
        mel_fb = self._create_mel_fb(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # 3. Instance normalization
        self.input_norm = nn.InstanceNorm1d(n_mels)

        # 4. Patch projection: mel bands -> d_model
        self.patch_proj = nn.Linear(n_mels, d_model)

        # 5. SA-SSM Blocks
        self.weight_sharing = weight_sharing
        if weight_sharing:
            # Single shared block, repeated n_repeats times
            # Depth = n_repeats, unique params = 1 block
            shared_block = NanoMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                n_mels=n_mels,
                ssm_mode=ssm_mode)
            self.blocks = nn.ModuleList([shared_block])
            self.n_repeats = n_repeats
        else:
            self.blocks = nn.ModuleList([
                NanoMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    n_mels=n_mels,
                    ssm_mode=ssm_mode)
                for _ in range(n_layers)
            ])
            self.n_repeats = n_layers

        # 6. Final norm
        self.final_norm = nn.LayerNorm(d_model)

        # 7. Classifier
        self.classifier = nn.Linear(d_model, n_classes)

    @staticmethod
    def _create_mel_fb(sr, n_fft, n_mels):
        """Create mel filterbank (same as NanoKWS for consistency)."""
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
                    fb[i, j] = (j - bin_points[i]) / max(
                        bin_points[i + 1] - bin_points[i], 1)
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                if j < n_freq:
                    fb[i, j] = (bin_points[i + 2] - j) / max(
                        bin_points[i + 2] - bin_points[i + 1], 1)
        return fb

    def extract_features(self, audio):
        """Extract mel features and SNR from raw audio.

        Args:
            audio: (B, T) raw waveform
        Returns:
            mel: (B, n_mels, T_frames) log-mel spectrogram
            snr_mel: (B, n_mels, T_frames) per-mel-band SNR
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # (B, F, T)

        # [NOVEL] Frequency-domain plug-in (before SNR estimation & mel)
        if self.use_freq_filter:
            mag = self.freq_filter(mag)
        if self.use_freq_conv:
            mag = self.freq_conv(mag)

        # SNR estimation (before mel projection)
        snr_mel = self.snr_estimator(mag, self.mel_fb)  # (B, n_mels, T)

        # Mel features
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T)
        mel = torch.log(mel + 1e-8)
        mel = self.input_norm(mel)

        return mel, snr_mel

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform at 16kHz
        Returns:
            logits: (B, n_classes)
        """
        # Extract features + SNR
        mel, snr_mel = self.extract_features(audio)
        # mel: (B, n_mels, T), snr_mel: (B, n_mels, T)

        # Transpose to (B, T, n_mels) for sequence processing
        x = mel.transpose(1, 2)  # (B, T, n_mels)
        snr = snr_mel.transpose(1, 2)  # (B, T, n_mels)

        # Patch projection
        x = self.patch_proj(x)  # (B, T, d_model)

        # SA-SSM blocks (each receives SNR as side information)
        if self.weight_sharing:
            # Repeat single shared block n_repeats times
            for _ in range(self.n_repeats):
                x = self.blocks[0](x, snr)
        else:
            for block in self.blocks:
                x = block(x, snr)

        # Final norm + global average pooling
        x = self.final_norm(x)  # (B, T, d_model)
        x = x.mean(dim=1)  # (B, d_model)

        # Classify
        return self.classifier(x)


# ============================================================================
# Factory Functions
# ============================================================================

def create_nanomamba_tiny(n_classes=12):
    """NanoMamba-Tiny: ~3.5-5.5K params, sub-4KB INT8."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2)


def create_nanomamba_small(n_classes=12):
    """NanoMamba-Small: ~8-10K params, sub-10KB INT8."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=24, d_state=4, d_conv=3, expand=1.5,
        n_layers=3)


def create_nanomamba_base(n_classes=12):
    """NanoMamba-Base: ~25-30K params, sub-30KB INT8."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=40, d_state=8, d_conv=4, expand=1.5,
        n_layers=4)


# ============================================================================
# Frequency Filter Variants
# ============================================================================

def create_nanomamba_tiny_ff(n_classes=12):
    """NanoMamba-Tiny + Frequency Filter: ~4,893 params.

    Adds learnable frequency-bin mask (257 params) to suppress
    noise-dominated frequency bands before mel projection.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_freq_filter=True)


def create_nanomamba_small_ff(n_classes=12):
    """NanoMamba-Small + Frequency Filter: ~12,292 params.

    Adds learnable frequency-bin mask (257 params) to suppress
    noise-dominated frequency bands before mel projection.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=24, d_state=4, d_conv=3, expand=1.5,
        n_layers=3, use_freq_filter=True)


# ============================================================================
# FreqConv Variants (CNN frequency selectivity transplant)
# ============================================================================

def create_nanomamba_tiny_fc(n_classes=12):
    """NanoMamba-Tiny + FreqConv: ~4,642 params (+6 from baseline).

    Transplants CNN's local frequency selectivity via 1D conv on freq axis.
    Input-dependent mask enables adaptive noise suppression.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_freq_conv=True)


def create_nanomamba_small_fc(n_classes=12):
    """NanoMamba-Small + FreqConv: ~12,041 params (+6 from baseline)."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=24, d_state=4, d_conv=3, expand=1.5,
        n_layers=3, use_freq_conv=True)


# ============================================================================
# Weight Sharing Variants (Journal Extension)
# ============================================================================

def create_nanomamba_tiny_ws(n_classes=12):
    """NanoMamba-Tiny-WS: Weight-Shared, d=20, 1 block × 3 repeats.

    Depth = 3 layers, unique params ≈ 1 block.
    Target: Small-level accuracy with Tiny-level params.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=4, d_conv=3, expand=1.5,
        n_layers=1, weight_sharing=True, n_repeats=3)


def create_nanomamba_tiny_ws_ff(n_classes=12):
    """NanoMamba-Tiny-WS-FF: Weight-Shared + FreqFilter.

    Ultimate efficiency: ~4.8K params, depth=3, frequency-selective.
    Target: Beat BC-ResNet-1 (7.5K) in all metrics.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=4, d_conv=3, expand=1.5,
        n_layers=1, weight_sharing=True, n_repeats=3,
        use_freq_filter=True)


# ============================================================================
# Ablation Factory Functions
# ============================================================================

def create_nanomamba_tiny_ablation(n_classes=12, mode='standard'):
    """Create NanoMamba-Tiny with specified ablation mode.

    Args:
        mode: 'full', 'dt_only', 'b_only', 'standard'
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, ssm_mode=mode)


def create_ablation_models(n_classes=12):
    """Create all ablation variants of NanoMamba-Tiny.

    Returns dict of {name: model} for ablation study.
    """
    modes = {
        'NanoMamba-Tiny-Full': 'full',
        'NanoMamba-Tiny-dtOnly': 'dt_only',
        'NanoMamba-Tiny-bOnly': 'b_only',
        'NanoMamba-Tiny-Standard': 'standard',
    }
    return {name: create_nanomamba_tiny_ablation(n_classes, mode)
            for name, mode in modes.items()}


# ============================================================================
# Verification
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  NanoMamba - Spectral-Aware SSM for Noise-Robust KWS")
    print("=" * 70)

    audio = torch.randn(2, 16000)  # 1s @ 16kHz

    configs = {
        'NanoMamba-Tiny': create_nanomamba_tiny,
        'NanoMamba-Small': create_nanomamba_small,
        'NanoMamba-Base': create_nanomamba_base,
        'NanoMamba-Tiny-FF': create_nanomamba_tiny_ff,
        'NanoMamba-Small-FF': create_nanomamba_small_ff,
        'NanoMamba-Tiny-FC': create_nanomamba_tiny_fc,
        'NanoMamba-Small-FC': create_nanomamba_small_fc,
        'NanoMamba-Tiny-WS': create_nanomamba_tiny_ws,
        'NanoMamba-Tiny-WS-FF': create_nanomamba_tiny_ws_ff,
    }

    print(f"\n  {'Model':<22} | {'Params':>8} | {'FP32 KB':>8} | {'INT8 KB':>8} | Output")
    print("  " + "-" * 75)

    for name, create_fn in configs.items():
        model = create_fn()
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        fp32_kb = sum(p.numel() * 4 for p in model.parameters()) / 1024
        int8_kb = sum(p.numel() * 1 for p in model.parameters()) / 1024

        with torch.no_grad():
            out = model(audio)

        print(f"  {name:<22} | {params:>8,} | {fp32_kb:>7.1f} | {int8_kb:>7.1f} | "
              f"{list(out.shape)}")

    print("\n  SA-SSM Novelty:")
    print("  - dt modulated by per-band SNR -> noise-aware step size")
    print("  - B gated by SNR -> noise-aware input selection")
    print("  - No separate AEC module needed")
    print("  - Graceful noise degradation built into SSM dynamics")

    # Detailed breakdown for Tiny
    print("\n  Parameter breakdown (NanoMamba-Tiny):")
    m = create_nanomamba_tiny()
    for name, p in m.named_parameters():
        print(f"    {name:<45} {p.numel():>6}  {list(p.shape)}")
    total = sum(p.numel() for p in m.parameters())
    print(f"    {'TOTAL':<45} {total:>6}")
