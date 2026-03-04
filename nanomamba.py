#!/usr/bin/env python3
# coding=utf-8
# NanoMamba: Noise-Robust State Space Models for Keyword Spotting
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.
# Dual License: Free for academic/research use. Commercial use requires license.
# See LICENSE file. Contact: jinhochoi@smartear.co.kr for commercial licensing.
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

    def __init__(self, n_freq=257, noise_frames=5, use_running_ema=False):
        super().__init__()
        self.noise_frames = noise_frames
        self.use_running_ema = use_running_ema

        # Learnable noise floor parameters
        self.noise_scale = nn.Parameter(torch.tensor(1.5))
        self.floor = nn.Parameter(torch.tensor(0.02))

        # Running EMA parameters for adaptive noise tracking
        # Asymmetric: slow rise (speech/impact), faster fall (true noise floor)
        if use_running_ema:
            # sigmoid(-2.2) ≈ 0.10: when frame < noise_floor, update 10%
            self.raw_beta = nn.Parameter(torch.tensor(-2.2))
            # sigmoid(-3.0) ≈ 0.05: when frame > noise_floor, update 5%
            self.raw_gamma = nn.Parameter(torch.tensor(-3.0))

    def forward(self, mag, mel_fb):
        """
        Args:
            mag: (B, F, T) magnitude spectrogram
            mel_fb: (n_mels, F) mel filterbank matrix
        Returns:
            snr_mel: (B, n_mels, T) per-mel-band SNR estimate
        """
        # Phase 1: Initial estimate from first N frames
        init_noise = mag[:, :, :self.noise_frames].mean(dim=2, keepdim=True)

        if self.use_running_ema:
            # Phase 2: Running EMA noise floor tracking
            beta = torch.sigmoid(self.raw_beta)    # ~0.10
            gamma = torch.sigmoid(self.raw_gamma)  # ~0.05

            B, F, T = mag.shape
            noise_floor = init_noise.clone()  # (B, F, 1)
            noise_estimates = []

            for t in range(T):
                frame = mag[:, :, t:t+1]  # (B, F, 1)
                # Asymmetric: slow rise for speech/impacts, faster for noise
                is_above = (frame > noise_floor).float()
                alpha_t = gamma * is_above + beta * (1 - is_above)
                noise_floor = (1 - alpha_t) * noise_floor + alpha_t * frame
                noise_estimates.append(noise_floor)

            running_noise = torch.cat(noise_estimates, dim=-1)  # (B, F, T)

            # Safety: never underestimate below half of initial estimate
            effective_noise = torch.maximum(
                running_noise,
                init_noise.expand_as(running_noise) * 0.5
            )

            # Per-band SNR (linear scale)
            snr = mag / (self.noise_scale.abs() * effective_noise + 1e-8)
        else:
            # Original: static noise estimate
            snr = mag / (self.noise_scale.abs() * init_noise + 1e-8)

        # Project to mel bands
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
# PCEN: Per-Channel Energy Normalization (Structural Noise Suppression)
# ============================================================================

class PCEN(nn.Module):
    """Per-Channel Energy Normalization — structural noise suppression.

    Replaces log(mel) with adaptive AGC + dynamic range compression.
    The AGC tracks the local energy envelope per channel and normalizes
    by it, inherently suppressing stationary/slowly-varying noise (factory
    hum, pink noise) without any noise-augmented training.

    At -15dB factory noise, log(signal+noise) ≈ log(noise) — speech info
    is destroyed. PCEN instead computes mel * (eps + smoother)^{-alpha},
    dividing by the noise envelope and recovering relative speech structure.

    Parameters: 4 * n_mels = 160 (for n_mels=40)

    Reference: Wang et al., "Trainable Frontend For Robust and Far-Field
    Keyword Spotting", ICASSP 2017.
    """

    def __init__(self, n_mels=40, s_init=0.15, alpha_init=0.99,
                 delta_init=0.01, r_init=0.1, eps=1e-6, trainable=True,
                 delta_clamp=(0.001, 0.1)):
        super().__init__()
        self.eps = eps
        self.n_mels = n_mels
        self.delta_clamp = delta_clamp

        if trainable:
            # Per-channel learnable params (sigmoid/exp constrained)
            self.log_s = nn.Parameter(
                torch.full((n_mels,), math.log(s_init / (1 - s_init))))
            self.log_alpha = nn.Parameter(
                torch.full((n_mels,), math.log(alpha_init / (1 - alpha_init))))
            self.log_delta = nn.Parameter(
                torch.full((n_mels,), math.log(delta_init)))
            self.log_r = nn.Parameter(
                torch.full((n_mels,), math.log(r_init / (1 - r_init))))
        else:
            self.register_buffer('log_s',
                torch.full((n_mels,), math.log(s_init / (1 - s_init))))
            self.register_buffer('log_alpha',
                torch.full((n_mels,), math.log(alpha_init / (1 - alpha_init))))
            self.register_buffer('log_delta',
                torch.full((n_mels,), math.log(delta_init)))
            self.register_buffer('log_r',
                torch.full((n_mels,), math.log(r_init / (1 - r_init))))

    def forward(self, mel, snr_mel=None):
        """
        Args:
            mel: (B, n_mels, T) LINEAR mel energy (before log!)
            snr_mel: (B, n_mels, T) optional per-mel-band SNR for adaptive compression.
                     When provided, compression exponent r is boosted at low SNR to
                     amplify weak speech signals. Backward-compatible: None = original.
        Returns:
            pcen_out: (B, n_mels, T) PCEN-normalized features
        """
        # Constrained parameters (noise-biased clamping prevents clean drift)
        s = torch.sigmoid(self.log_s).clamp(0.05, 0.3).unsqueeze(0).unsqueeze(-1)       # (1, M, 1)
        alpha = torch.sigmoid(self.log_alpha).clamp(0.9, 0.999).unsqueeze(0).unsqueeze(-1)
        delta = torch.exp(self.log_delta).clamp(*self.delta_clamp).unsqueeze(0).unsqueeze(-1)
        r = torch.sigmoid(self.log_r).clamp(0.05, 0.25).unsqueeze(0).unsqueeze(-1)

        if snr_mel is not None:
            # [NOVEL] SNR-Adaptive Compression Exponent (per-band):
            # At low SNR, speech is 31.6× weaker than noise (-15dB). More aggressive
            # compression (higher r) narrows dynamic range, amplifying weak speech.
            # At high SNR, keep original r to preserve clean-speech quality.
            # Per-band: low-freq bands under factory hum get more compression,
            # while high-freq bands with higher SNR are preserved.
            # snr_mel: (B, M, T) ∈ [0,1] per mel band — use directly
            # r: (1, M, 1) broadcasts with snr_mel (B, M, T) → (B, M, T)
            # Low SNR (snr_mel→0): r_eff = r × 1.5 (50% more compression)
            # High SNR (snr_mel→1): r_eff = r × 1.0 (unchanged)
            r = (r * (1.0 + 0.5 * (1.0 - snr_mel))).clamp(0.05, 0.40)

            # [NOVEL] SNR-Adaptive AGC Speed (per-band):
            # At low SNR, noise envelope changes faster than speech —
            # PCEN's IIR smoother needs to track more aggressively to
            # follow rapid noise fluctuations and extract speech modulation.
            # At high SNR, slow tracking preserves clean speech quality.
            # s: (1, M, 1) broadcasts with snr_mel (B, M, T) → (B, M, T)
            # Low SNR (snr_mel→0): s_eff = s × 1.3 (30% faster tracking)
            # High SNR (snr_mel→1): s_eff = s × 1.0 (unchanged)
            s = (s * (1.0 + 0.3 * (1.0 - snr_mel))).clamp(0.05, 0.40)

        # IIR smoothing of energy envelope (AGC)
        # s may be (1, M, 1) [no snr_mel] or (B, M, T) [with snr_mel]
        B, M, T = mel.shape
        smoother = mel[:, :, :1]  # Initialize with first frame
        per_frame_s = (s.dim() == 3 and s.size(-1) > 1)

        smoothed_frames = []
        for t in range(T):
            s_t = s[:, :, t:t+1] if per_frame_s else s
            smoother = (1 - s_t) * smoother + s_t * mel[:, :, t:t+1]
            smoothed_frames.append(smoother)

        smoothed = torch.cat(smoothed_frames, dim=-1)  # (B, M, T)

        # AGC + dynamic range compression
        gain = (self.eps + smoothed) ** (-alpha)
        pcen_out = (mel * gain + delta) ** r - delta ** r

        return pcen_out


# ============================================================================
# Dual-PCEN: Noise-Adaptive Routing for ALL Noise Types
# ============================================================================

class DualPCEN(nn.Module):
    """Dual-PCEN with Multi-Dimensional Routing.

    Structural robustness to ALL noise types in a single module.

    Insight: No single PCEN parameterization dominates all noise types.
      - High δ (2.0):  Kills AGC → offset-dominant → babble champion
      - Low δ (0.01):  Pure AGC tracking → stationary noise champion

    Solution: Two complementary PCEN front-ends + multi-dimensional routing.

    [NOVEL] Routing Signal — Spectral Flatness + Spectral Tilt (0 learnable params):
      SF = exp(mean(log(mel))) / mean(mel)    ∈ [0, 1]
      Tilt = low_freq_energy / (low + high + eps)  ∈ [0, 1]
      SF alone misroutes pink noise (SF=0.3, but stationary) to babble expert.
      Tilt correction: pink has tilt≈0.85 (low-freq concentrated) → boost SF.

    Extra params: 160 (2nd PCEN) + 1 (gate temperature) = 161
    Total added to NanoMamba-Tiny: 4.6K + 161 = 4.8K

    Reference:
      - PCEN: Wang et al., "Trainable Frontend", ICASSP 2017
      - Spectral Flatness: Johnston, "Transform Coding of Audio", 1988
    """

    def __init__(self, n_mels=40):
        super().__init__()

        # Expert 1: Non-stationary noise (babble) — high δ kills AGC
        # Offset-dominant mode: preserves relative speech structure in babble
        self.pcen_nonstat = PCEN(
            n_mels=n_mels,
            s_init=0.025,      # slow smoothing → stable envelope
            alpha_init=0.99,
            delta_init=2.0,    # HIGH δ → AGC negligible, offset dominates
            r_init=0.5,
            delta_clamp=(0.5, 5.0))   # wide range: allow large δ

        # Expert 2: Stationary noise (factory, white, pink) — low δ enables AGC
        # AGC-dominant mode: adaptive gain control tracks slowly-varying noise
        self.pcen_stat = PCEN(
            n_mels=n_mels,
            s_init=0.15,       # fast smoothing → quick noise tracking
            alpha_init=0.99,
            delta_init=0.01,   # LOW δ → pure AGC, divides out noise floor
            r_init=0.1,
            delta_clamp=(0.001, 0.1))  # narrow range: keep δ small

        # Gate temperature: controls routing sharpness (1 learnable param)
        # Positive → sharper switching, negative → softer blending
        self.gate_temp = nn.Parameter(torch.tensor(5.0))

    def forward(self, mel_linear):
        """
        Args:
            mel_linear: (B, n_mels, T) LINEAR mel energy (before any normalization)
        Returns:
            pcen_out: (B, n_mels, T) noise-adaptively routed PCEN output
        """
        # Both experts process the same input
        out_nonstat = self.pcen_nonstat(mel_linear)  # babble expert
        out_stat = self.pcen_stat(mel_linear)        # factory/white expert

        # Spectral Flatness — per-frame noise stationarity measure (0 params)
        # SF = geometric_mean(mel) / arithmetic_mean(mel)
        # Computed across mel bands for each time frame
        log_mel = torch.log(mel_linear + 1e-8)                        # (B, M, T)
        geo_mean = torch.exp(log_mel.mean(dim=1, keepdim=True))       # (B, 1, T)
        arith_mean = mel_linear.mean(dim=1, keepdim=True) + 1e-8      # (B, 1, T)
        sf = (geo_mean / arith_mean).clamp(0, 1)                      # (B, 1, T)

        # [NOVEL] Spectral Tilt: low-frequency energy concentration (0 params)
        # Distinguishes colored stationary noise (pink: tilt≈0.85) from
        # non-stationary noise (babble: tilt≈0.55). SF alone misroutes pink
        # noise (SF=0.3, peaked spectrum) to babble expert — tilt corrects this.
        n_mels = mel_linear.size(1)
        low_energy = mel_linear[:, :n_mels // 3, :].mean(dim=1, keepdim=True)
        high_energy = mel_linear[:, 2 * n_mels // 3:, :].mean(dim=1, keepdim=True)
        spectral_tilt = (low_energy / (low_energy + high_energy + 1e-8)).clamp(0, 1)

        # [NOVEL] Multi-dimensional routing: SF + Tilt correction
        # When SF is low BUT tilt is high → colored stationary (pink) → boost SF
        # Pink:   sf=0.3, tilt=0.85 → sf_adj=0.3+0.7*0.25=0.475 → gate≈0.44
        # Babble: sf=0.4, tilt=0.55 → sf_adj=0.4+0.6*0.0=0.4    → gate≈0.27
        # White:  sf=0.95, tilt=0.50 → sf_adj=0.95+0.05*0.0=0.95 → gate≈0.92
        sf_adjusted = sf + (1.0 - sf) * torch.relu(spectral_tilt - 0.6)

        # Route: high SF → stationary expert, low SF → non-stationary expert
        gate = torch.sigmoid(self.gate_temp * (sf_adjusted - 0.5))    # (B, 1, T)

        # Weighted blend (broadcasts across mel bands)
        pcen_out = gate * out_stat + (1 - gate) * out_nonstat

        return pcen_out


# ============================================================================
# DualPCEN v2: Enhanced Routing (TMI + SNR-Conditioned + Temporal Smoothing)
# ============================================================================

class DualPCEN_v2(nn.Module):
    """Enhanced Dual-PCEN with Temporal + SNR-Conditioned Routing.

    Four improvements over DualPCEN, all at 0 extra inference parameters:

    1. TMI (Temporal Modulation Index): time-domain stationarity signal.
       SF measures frequency flatness, TMI measures temporal energy variance.
       Stationary noise (white/factory) → low TMI, non-stationary (babble) → high TMI.
       Orthogonal to SF: resolves ambiguous cases where spectrum shape is similar.

    2. SNR-Conditioned Gate Temperature: at low SNR noise dominates and noise
       type is clear from acoustics → sharper routing. At high SNR, speech dominates
       and routing matters less → softer blending. Uses already-computed snr_mel.

    3. Temporal Smoothing: per-frame SF is noisy at low SNR. Causal moving average
       (K=7, ~70ms) stabilizes routing decisions. GPU-friendly via conv1d.

    4. Auxiliary Routing Loss support: stores gate values for training-time
       supervision with known noise type labels. 0 inference overhead.

    Extra params vs DualPCEN: 0 (identical parameter count).
    """

    def __init__(self, n_mels=40, smooth_window=7, snr_temp_scale=2.0):
        super().__init__()
        self.n_mels_cfg = n_mels

        # Expert 1: Non-stationary noise (babble) — high δ kills AGC
        self.pcen_nonstat = PCEN(
            n_mels=n_mels,
            s_init=0.025, alpha_init=0.99,
            delta_init=2.0, r_init=0.5,
            delta_clamp=(0.5, 5.0))

        # Expert 2: Stationary noise (factory, white, pink) — low δ enables AGC
        self.pcen_stat = PCEN(
            n_mels=n_mels,
            s_init=0.15, alpha_init=0.99,
            delta_init=0.01, r_init=0.1,
            delta_clamp=(0.001, 0.1))

        # Gate temperature (1 learnable param, same as DualPCEN)
        self.gate_temp = nn.Parameter(torch.tensor(5.0))

        # Smoothing config (0 learnable params)
        self.smooth_window = smooth_window
        self.snr_temp_scale = snr_temp_scale

        # Pre-register smoothing kernel as buffer (avoids re-creation per forward)
        if smooth_window > 1:
            kernel = torch.ones(1, 1, smooth_window) / smooth_window
            self.register_buffer('smooth_kernel', kernel)

        # Storage for auxiliary routing loss (training-time only)
        self._last_gate = None

    def _causal_smooth(self, x):
        """Causal moving average. 0 params, GPU-friendly via conv1d.

        Args:
            x: (B, 1, T) signal to smooth
        Returns:
            smoothed: (B, 1, T) causal-smoothed signal
        """
        K = self.smooth_window
        if K <= 1:
            return x
        return F.conv1d(F.pad(x, (K - 1, 0)), self.smooth_kernel)

    def forward(self, mel_linear, snr_mel=None):
        """
        Args:
            mel_linear: (B, n_mels, T) LINEAR mel energy (before normalization)
            snr_mel: (B, n_mels, T) per-mel-band SNR in [0,1] from SNREstimator
        Returns:
            pcen_out: (B, n_mels, T) noise-adaptively routed PCEN output
        """
        # Both experts process the same input
        # [v2] Pass snr_mel to experts for SNR-adaptive compression exponent
        out_nonstat = self.pcen_nonstat(mel_linear, snr_mel=snr_mel)
        out_stat = self.pcen_stat(mel_linear, snr_mel=snr_mel)

        # === Spectral Flatness (0 params) ===
        log_mel = torch.log(mel_linear + 1e-8)
        geo_mean = torch.exp(log_mel.mean(dim=1, keepdim=True))
        arith_mean = mel_linear.mean(dim=1, keepdim=True) + 1e-8
        sf_raw = (geo_mean / arith_mean).clamp(0, 1)  # (B, 1, T)

        # [v2] Temporal smoothing of SF (0 params)
        sf = self._causal_smooth(sf_raw)

        # === Spectral Tilt (0 params) ===
        n_mels = mel_linear.size(1)
        low_energy = mel_linear[:, :n_mels // 3, :].mean(dim=1, keepdim=True)
        high_energy = mel_linear[:, 2 * n_mels // 3:, :].mean(dim=1, keepdim=True)
        spectral_tilt = (low_energy / (low_energy + high_energy + 1e-8)).clamp(0, 1)

        # SF + Tilt correction (same as DualPCEN)
        sf_adjusted = sf + (1.0 - sf) * torch.relu(spectral_tilt - 0.6)

        # === [v2] TMI: Temporal Modulation Index (0 params) ===
        # Coefficient of variation of frame energy over causal window.
        # Stationary noise → low TMI, non-stationary → high TMI.
        frame_energy = mel_linear.mean(dim=1, keepdim=True)  # (B, 1, T)
        ema_E = self._causal_smooth(frame_energy)
        ema_E2 = self._causal_smooth(frame_energy ** 2)
        variance = (ema_E2 - ema_E ** 2).clamp(min=0)
        tmi = variance.sqrt() / (ema_E + 1e-8)  # CV coefficient
        tmi = self._causal_smooth(tmi.clamp(0, 2.0) / 2.0)  # normalize to [0,1]

        # TMI correction: low TMI (temporally stationary) → boost toward stat expert
        tmi_boost = torch.relu(0.5 - tmi) * 0.5
        routing_signal = sf_adjusted + (1.0 - sf_adjusted) * tmi_boost

        # === [v2] SNR-conditioned temperature (0 params) ===
        # Low SNR → noise dominates → noise type is acoustically clear → sharper gate
        # High SNR → speech dominates → routing less critical → softer blending
        if snr_mel is not None:
            # snr_mel: tanh(snr/10) ∈ [0,1], 0=noise-dominated, 1=clean
            snr_global = snr_mel.mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            snr_scale = 1.0 + self.snr_temp_scale * (1.0 - snr_global)
            effective_temp = self.gate_temp * snr_scale
        else:
            effective_temp = self.gate_temp

        # Gate computation
        gate = torch.sigmoid(effective_temp * (routing_signal - 0.5))

        # Weighted blend
        pcen_out = gate * out_stat + (1 - gate) * out_nonstat

        # Store gate for auxiliary routing loss (training-time only)
        self._last_gate = gate.mean(dim=(1, 2))  # (B,) for aux loss
        # Store per-frame gate for SA-SSM v2 per-timestep conditioning
        # gate: (B, 1, T) → (B, T) for indexing inside SSM scan loop
        self._last_gate_per_frame = gate.squeeze(1)  # (B, T)

        return pcen_out


# ============================================================================
# Multi-PCEN: N-Expert PCEN with Hierarchical Routing
# ============================================================================

class MultiPCEN(nn.Module):
    """N-Expert PCEN with Hierarchical Routing.

    Generalizes DualPCEN to N experts with hierarchical signal-based routing.

    Insight: DualPCEN's 2-expert split (babble vs stationary) leaves
    factory/street noise in a 50:50 blend zone (gate≈0.5-0.6).
    A 3rd expert with medium δ=0.1 captures colored/structured noise
    characteristics that neither extreme δ handles well.

    Expert Configuration:
      Expert 0: Non-stationary (babble) — δ=2.0, s=0.025 (AGC off, offset mode)
      Expert 1: Broadband stationary (white/pink) — δ=0.01, s=0.15 (pure AGC)
      Expert 2: Colored stationary (factory/street) — δ=0.1, s=0.08 (medium AGC)

    Hierarchical Routing (signal-based, 2 learnable temps only):
      Level 1: SF+Tilt → stationary vs non-stationary (from DualPCEN)
      Level 2: SF alone → broadband(high SF) vs colored(low SF) within stationary

    Extra params vs DualPCEN: +160 (3rd PCEN) + 1 (gate_temp2) = +161

    Reference:
      - DualPCEN: Choi, "NanoMamba", TASLP 2026
      - PCEN: Wang et al., "Trainable Frontend", ICASSP 2017
    """

    # Default expert configurations
    EXPERT_CONFIGS = [
        # Expert 0: Non-stationary noise (babble, speech interference)
        # High δ kills AGC → offset-dominant → preserves relative speech structure
        dict(s_init=0.025, alpha_init=0.99, delta_init=2.0,
             r_init=0.5, delta_clamp=(0.5, 5.0)),
        # Expert 1: Broadband stationary (white, pink)
        # Low δ enables pure AGC → divides out flat noise floor
        dict(s_init=0.15, alpha_init=0.99, delta_init=0.01,
             r_init=0.1, delta_clamp=(0.001, 0.1)),
        # Expert 2: Colored/structured stationary (factory, street)
        # Medium δ → moderate AGC that preserves harmonic structure
        dict(s_init=0.08, alpha_init=0.98, delta_init=0.1,
             r_init=0.3, delta_clamp=(0.05, 1.0)),
    ]

    def __init__(self, n_mels=40, n_experts=3):
        super().__init__()
        self.n_experts = n_experts
        self.n_mels = n_mels

        # Create expert PCEN modules
        configs = self.EXPERT_CONFIGS[:n_experts]
        self.experts = nn.ModuleList([
            PCEN(n_mels=n_mels, **cfg) for cfg in configs
        ])

        # Gate temperatures (learnable routing sharpness)
        # Level 1: stationary vs non-stationary (same as DualPCEN)
        self.gate_temp = nn.Parameter(torch.tensor(5.0))
        # Level 2: broadband vs colored (within stationary)
        if n_experts >= 3:
            self.gate_temp2 = nn.Parameter(torch.tensor(5.0))

    def forward(self, mel_linear):
        """
        Args:
            mel_linear: (B, n_mels, T) LINEAR mel energy (before any normalization)
        Returns:
            pcen_out: (B, n_mels, T) noise-adaptively routed PCEN output
        """
        # All experts process the same input
        outputs = [expert(mel_linear) for expert in self.experts]

        # === Spectral Flatness (0 params) — same as DualPCEN ===
        log_mel = torch.log(mel_linear + 1e-8)
        geo_mean = torch.exp(log_mel.mean(dim=1, keepdim=True))
        arith_mean = mel_linear.mean(dim=1, keepdim=True) + 1e-8
        sf = (geo_mean / arith_mean).clamp(0, 1)  # (B, 1, T)

        # === Spectral Tilt (0 params) — same as DualPCEN ===
        n_mels = mel_linear.size(1)
        low_energy = mel_linear[:, :n_mels // 3, :].mean(dim=1, keepdim=True)
        high_energy = mel_linear[:, 2 * n_mels // 3:, :].mean(dim=1, keepdim=True)
        spectral_tilt = (low_energy / (low_energy + high_energy + 1e-8)).clamp(0, 1)

        # === Multi-dimensional routing: SF + Tilt correction ===
        sf_adjusted = sf + (1.0 - sf) * torch.relu(spectral_tilt - 0.6)

        # === Routing Level 1: Stationary vs Non-stationary ===
        p_stat = torch.sigmoid(self.gate_temp * (sf_adjusted - 0.5))  # (B, 1, T)

        if self.n_experts == 2:
            # Fallback to DualPCEN behavior
            pcen_out = p_stat * outputs[1] + (1 - p_stat) * outputs[0]
        elif self.n_experts >= 3:
            # === Routing Level 2: Broadband vs Colored (within stationary) ===
            # High SF → broadband (white/pink → Expert 1)
            # Low SF → colored (factory/street → Expert 2)
            p_broad = torch.sigmoid(self.gate_temp2 * (sf - 0.7))  # (B, 1, T)

            # Final expert weights
            w_nonstat = 1 - p_stat                  # Expert 0 (babble)
            w_broad = p_stat * p_broad              # Expert 1 (white/pink)
            w_colored = p_stat * (1 - p_broad)      # Expert 2 (factory/street)

            pcen_out = (w_nonstat * outputs[0] +
                        w_broad * outputs[1] +
                        w_colored * outputs[2])

        return pcen_out


# ============================================================================
# MultiPCEN v2: Enhanced N-Expert Routing (TMI + SNR-Conditioned)
# ============================================================================

class MultiPCEN_v2(nn.Module):
    """Enhanced N-Expert PCEN with TMI + SNR-Conditioned Hierarchical Routing.

    Same improvements as DualPCEN_v2, applied to both routing levels:
    1. TMI (Temporal Modulation Index) for time-domain stationarity
    2. SNR-conditioned gate temperatures (sharper at low SNR)
    3. Temporal smoothing of SF and TMI signals
    4. Auxiliary routing loss support (_last_gate_l1, _last_gate_l2)

    Extra params vs MultiPCEN: 0 (identical parameter count).
    """

    EXPERT_CONFIGS = MultiPCEN.EXPERT_CONFIGS  # reuse same configs

    def __init__(self, n_mels=40, n_experts=3, smooth_window=7,
                 snr_temp_scale=2.0):
        super().__init__()
        self.n_experts = n_experts
        self.n_mels_cfg = n_mels

        configs = self.EXPERT_CONFIGS[:n_experts]
        self.experts = nn.ModuleList([
            PCEN(n_mels=n_mels, **cfg) for cfg in configs
        ])

        self.gate_temp = nn.Parameter(torch.tensor(5.0))
        if n_experts >= 3:
            self.gate_temp2 = nn.Parameter(torch.tensor(5.0))

        self.smooth_window = smooth_window
        self.snr_temp_scale = snr_temp_scale

        if smooth_window > 1:
            kernel = torch.ones(1, 1, smooth_window) / smooth_window
            self.register_buffer('smooth_kernel', kernel)

        self._last_gate_l1 = None
        self._last_gate_l2 = None

    def _causal_smooth(self, x):
        K = self.smooth_window
        if K <= 1:
            return x
        return F.conv1d(F.pad(x, (K - 1, 0)), self.smooth_kernel)

    def forward(self, mel_linear, snr_mel=None):
        # [v2] Pass snr_mel to experts for SNR-adaptive compression exponent
        outputs = [expert(mel_linear, snr_mel=snr_mel) for expert in self.experts]

        # Spectral Flatness + temporal smoothing
        log_mel = torch.log(mel_linear + 1e-8)
        geo_mean = torch.exp(log_mel.mean(dim=1, keepdim=True))
        arith_mean = mel_linear.mean(dim=1, keepdim=True) + 1e-8
        sf_raw = (geo_mean / arith_mean).clamp(0, 1)
        sf = self._causal_smooth(sf_raw)

        # Spectral Tilt
        n_mels = mel_linear.size(1)
        low_energy = mel_linear[:, :n_mels // 3, :].mean(dim=1, keepdim=True)
        high_energy = mel_linear[:, 2 * n_mels // 3:, :].mean(dim=1, keepdim=True)
        spectral_tilt = (low_energy / (low_energy + high_energy + 1e-8)).clamp(0, 1)

        sf_adjusted = sf + (1.0 - sf) * torch.relu(spectral_tilt - 0.6)

        # [v2] TMI: Temporal Modulation Index
        frame_energy = mel_linear.mean(dim=1, keepdim=True)
        ema_E = self._causal_smooth(frame_energy)
        ema_E2 = self._causal_smooth(frame_energy ** 2)
        variance = (ema_E2 - ema_E ** 2).clamp(min=0)
        tmi = variance.sqrt() / (ema_E + 1e-8)
        tmi = self._causal_smooth(tmi.clamp(0, 2.0) / 2.0)

        tmi_boost = torch.relu(0.5 - tmi) * 0.5
        routing_signal = sf_adjusted + (1.0 - sf_adjusted) * tmi_boost

        # [v2] SNR-conditioned temperatures
        if snr_mel is not None:
            snr_global = snr_mel.mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            snr_scale = 1.0 + self.snr_temp_scale * (1.0 - snr_global)
        else:
            snr_scale = 1.0

        # Level 1: Stationary vs Non-stationary
        eff_temp1 = self.gate_temp * snr_scale
        p_stat = torch.sigmoid(eff_temp1 * (routing_signal - 0.5))
        self._last_gate_l1 = p_stat.mean(dim=(1, 2))  # (B,) for aux loss
        # Per-frame gate for SA-SSM v2 per-timestep conditioning
        self._last_gate_l1_per_frame = p_stat.squeeze(1)  # (B, T)

        if self.n_experts == 2:
            pcen_out = p_stat * outputs[1] + (1 - p_stat) * outputs[0]
        elif self.n_experts >= 3:
            # Level 2: Broadband vs Colored (within stationary)
            # Use smoothed raw SF (not TMI-adjusted) for sub-stationary routing
            eff_temp2 = self.gate_temp2 * snr_scale
            p_broad = torch.sigmoid(eff_temp2 * (sf - 0.7))
            self._last_gate_l2 = p_broad.mean(dim=(1, 2))

            w_nonstat = 1 - p_stat
            w_broad = p_stat * p_broad
            w_colored = p_stat * (1 - p_broad)

            pcen_out = (w_nonstat * outputs[0] +
                        w_broad * outputs[1] +
                        w_colored * outputs[2])

        return pcen_out


# ============================================================================
# Frequency-Dependent Floor (Low-Freq Structural Protection)
# ============================================================================

class FrequencyDependentFloor(nn.Module):
    """Frequency-dependent minimum energy floor for mel features.

    Factory/pink noise concentrates in low mel bands (0-12, ~0-800Hz).
    This module adds a frequency-dependent minimum energy level,
    ensuring low-frequency bands always retain a minimum signal level
    that prevents complete information loss at extreme negative SNR.

    Parameters: 0 (non-learnable register_buffer)
    """

    def __init__(self, n_mels=40):
        super().__init__()
        floor = torch.zeros(n_mels)
        for i in range(n_mels):
            ratio = 1.0 - (i / (n_mels - 1))  # 1.0 at band 0, 0.0 at band 39
            floor[i] = 0.05 * math.exp(-3.0 * (1.0 - ratio))
        self.register_buffer('freq_floor',
                             floor.unsqueeze(0).unsqueeze(-1))  # (1, M, 1)

    def forward(self, mel_linear):
        """Apply frequency-dependent floor to linear mel energy.

        Args:
            mel_linear: (B, n_mels, T) linear mel energy (before PCEN/log)
        Returns:
            mel_floored: (B, n_mels, T) with frequency-dependent minimum
        """
        return torch.maximum(mel_linear, self.freq_floor.expand_as(mel_linear))


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
# MoE-Freq: SNR-Conditioned Mixture of Experts Frequency Filter
# ============================================================================

class MoEFreq(nn.Module):
    """Mixture-of-Experts Frequency Filter — 지피지기 백전불패.

    Uses the SNR profile (already computed by SNREstimator) as a "noise
    fingerprint" to route between frequency-processing experts:
      Expert 1 (narrow, k=3): tonal noise (factory harmonics at 50/100/200Hz)
      Expert 2 (wide, k=7):   broadband noise (white/fan/HVAC)
      Expert 3 (identity):    clean pass-through (no filtering needed)

    The gating network uses SNR statistics (mean, std) to determine the
    noise environment and select the optimal expert combination.

    Total parameters: 4 + 8 + 9 = 21 params.
    """

    def __init__(self):
        super().__init__()
        # Expert 1: narrow-band tonal noise suppression (4 params)
        self.expert_narrow = nn.Conv1d(1, 1, 3, padding=1, bias=True)
        # Expert 2: wide-band broadband noise suppression (8 params)
        self.expert_wide = nn.Conv1d(1, 1, 7, padding=3, bias=True)
        # Expert 3: identity (0 params) — implicit, no module needed

        # Gating: SNR mean + std → 3 expert weights (9 params)
        self.gate = nn.Linear(2, 3)

        # Initialize experts near-identity (sigmoid(1.5) ≈ 0.82)
        nn.init.normal_(self.expert_narrow.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.expert_narrow.bias, 1.5)
        nn.init.normal_(self.expert_wide.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.expert_wide.bias, 1.5)

        # Initialize gating to prefer identity (clean pass-through)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))

    def forward(self, mag, snr_profile):
        """Apply SNR-conditioned frequency filtering.

        Args:
            mag: (B, F, T) STFT magnitude
            snr_profile: (B, n_mels, T) per-band SNR from SNREstimator
        Returns:
            filtered_mag: (B, F, T)
        """
        B, Freq, T = mag.shape

        # Extract noise fingerprint from SNR profile
        snr_mean = snr_profile.mean(dim=(1, 2))  # (B,)  overall noise level
        snr_std = snr_profile.std(dim=(1, 2))     # (B,)  frequency selectivity
        snr_stats = torch.stack([snr_mean, snr_std], dim=1)  # (B, 2)

        # Gating: decide which expert(s) to use
        weights = torch.softmax(self.gate(snr_stats), dim=1)  # (B, 3)

        # Expert processing (per time-frame)
        x = mag.permute(0, 2, 1).reshape(B * T, 1, Freq)  # (B*T, 1, Freq)

        mask_narrow = torch.sigmoid(self.expert_narrow(x))  # (B*T, 1, Freq)
        mask_wide = torch.sigmoid(self.expert_wide(x))      # (B*T, 1, Freq)

        mask_narrow = mask_narrow.reshape(B, T, Freq).permute(0, 2, 1)  # (B, F, T)
        mask_wide = mask_wide.reshape(B, T, Freq).permute(0, 2, 1)      # (B, F, T)

        # Weighted expert combination
        w = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
        filtered = (w[:, 0] * (mag * mask_narrow) +   # Expert 1: narrow
                    w[:, 1] * (mag * mask_wide) +      # Expert 2: wide
                    w[:, 2] * mag)                      # Expert 3: identity

        return filtered


# ============================================================================
# TinyConv2D: CNN structural noise robustness transplant (Hybrid CNN-SSM)
# ============================================================================

class TinyConv2D(nn.Module):
    """Minimal 2D CNN on mel spectrogram — CNN의 구조적 noise robustness 이식.

    CNN이 noise에 강한 이유 = 2D conv가 주파수×시간 local 패턴의
    상대적 관계를 학습. 이 관계는 noise에 불변(invariant).

    핵심 통찰: DS-CNN-S, BC-ResNet-1 모두 clean만 학습했는데 noise에 강함.
    이는 training data가 아닌 "구조적" 특성. 2D Conv가 주파수 3bin × 시간
    3frame의 상대적 패턴(e.g., formant)을 학습하면, noise가 추가돼도
    상대적 관계가 유지되어 자연스럽게 일반화됨.

    SSM은 시간축만 처리 → 주파수 간 상대적 관계를 볼 수 없음.
    TinyConv2D로 이 gap을 10 params만으로 메운다.

    Single Conv2d(1, 1, 3, 3) + ReLU + residual = 10 params.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size,
                              padding=kernel_size // 2, bias=True)
        # Init near-identity: conv output starts at ~0, residual dominates
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, mel):
        """Apply 2D convolution on mel spectrogram with residual.

        Args:
            mel: (B, n_mels, T) mel spectrogram (before log)
        Returns:
            mel': (B, n_mels, T) enhanced mel spectrogram
        """
        x = mel.unsqueeze(1)                    # (B, 1, n_mels, T)
        out = F.relu(self.conv(x)).squeeze(1)   # (B, n_mels, T)
        return mel + out                         # residual connection


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

        # Initialize A with HiPPO diagonal approximation
        # HiPPO (High-order Polynomial Projection Operators):
        #   Optimal polynomial basis for memorizing continuous history.
        #   Full HiPPO-LegS has A[n,k] = -(2n+1)^0.5 * (2k+1)^0.5 (n>k),
        #   diagonal A[n,n] = -(n+1). Mamba uses diagonal approx: A[n] = n+0.5
        #   → better long-range temporal dependency than simple A=[1,2,...,N]
        A = torch.arange(1, d_state + 1, dtype=torch.float32) + 0.5  # HiPPO shift
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(d_inner))

        # SNR gating strength (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # [NOVEL] Structural noise robustness: NON-LEARNABLE architectural constants.
        # These are register_buffer (not nn.Parameter) so the optimizer CANNOT
        # modify them. This makes noise robustness a true ARCHITECTURAL property,
        # not a learned behavior that can be optimized away.
        #
        # Evidence: When these were nn.Parameter, optimizer destroyed them:
        #   gate_floor: 0.1 → -0.26 (went NEGATIVE!)
        #   delta_floor: 0.127 → 0.030 (reduced 4×)
        #   epsilon: 0.049 → 0.045 (already too small)

        # [NOVEL] SNR-Adaptive Delta Floor: structural guarantee that SSM never freezes.
        # High SNR → delta_floor_max (0.15, fast adaptation, same as original)
        # Low SNR  → delta_floor_min (0.05, longer temporal memory, prevents SSM freezing)
        # At -15dB white noise: fixed 0.15 causes dA decay too fast → SSM forgets
        # Adaptive floor gives SSM longer memory when it needs it most.
        self.register_buffer('delta_floor_min', torch.tensor(0.05))
        self.register_buffer('delta_floor_max', torch.tensor(0.15))

        # [NOVEL] SNR-Adaptive Epsilon: structural residual path scaling.
        # h_t = Ā·h_{t-1} + B̃·x_t + ε·x_t
        # High SNR → epsilon_min (0.08, minimal bypass, trust gating)
        # Low SNR  → epsilon_max (0.20, stronger bypass, rescue information)
        # At extreme noise, gating over-suppresses — adaptive ε ensures info flow.
        self.register_buffer('epsilon_min', torch.tensor(0.08))
        self.register_buffer('epsilon_max', torch.tensor(0.20))

        # B-gate floor: minimum input flow guarantee
        # B_gate = raw * (1 - bgate_floor) + bgate_floor  ∈ [bgate_floor, 1.0]
        self.register_buffer('bgate_floor', torch.tensor(0.3))

    def set_calibration(self, delta_floor_min=None, delta_floor_max=None,
                        epsilon_min=None, epsilon_max=None, bgate_floor=None):
        """Runtime Parameter Calibration: set adaptive constants based on
        estimated noise environment during silence/VAD periods.

        This enables noise-type-aware adaptation at inference time:
          - Clean (20dB+):  revert to training defaults
          - Light (10-20dB): moderate adaptation
          - Heavy (0-10dB):  aggressive adaptation
          - Extreme (<0dB):  maximum adaptation

        Maps directly to hardware registers:
          0x50: FLOOR_MIN, 0x52: FLOOR_MAX,
          0x54: EPS_MIN, 0x56: EPS_MAX, 0x58: BGATE_FLOOR

        Args:
            delta_floor_min: SSM memory floor at low SNR (default 0.05)
            delta_floor_max: SSM memory floor at high SNR (default 0.15)
            epsilon_min: residual bypass at high SNR (default 0.08)
            epsilon_max: residual bypass at low SNR (default 0.20)
            bgate_floor: minimum B-gate value (default 0.3)
        """
        if delta_floor_min is not None:
            self.delta_floor_min.fill_(delta_floor_min)
        if delta_floor_max is not None:
            self.delta_floor_max.fill_(delta_floor_max)
        if epsilon_min is not None:
            self.epsilon_min.fill_(epsilon_min)
        if epsilon_max is not None:
            self.epsilon_max.fill_(epsilon_max)
        if bgate_floor is not None:
            self.bgate_floor.fill_(bgate_floor)

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
            B_gate_raw = torch.sigmoid(snr_mod[..., 1:])  # (B, L, N)
            # [NOVEL] B-Gate Floor: minimum 30% input flow guarantee.
            # At -15dB: raw B_gate ≈ 0.1 → only 55% input passes (with alpha=0.5)
            # With floor: B_gate ∈ [0.3, 1.0] → minimum 65% input guaranteed
            # Prevents compound over-suppression (dt + B both suppressing)
            B_gate = B_gate_raw * (1.0 - self.bgate_floor) + self.bgate_floor
        else:
            B_gate = torch.ones_like(B_param)  # no B gating

        # [NOVEL] SNR-Adaptive Delta Floor:
        # Compute mean SNR across mel bands for floor adaptation
        snr_mean = snr_mel.mean(dim=-1, keepdim=True)  # (B, L, 1)
        # snr_mean ∈ [0, 1] (tanh-normalized by SNREstimator)
        # High SNR → floor=0.15 (fast adaptation, original behavior)
        # Low SNR  → floor=0.05 (longer temporal memory, prevents freezing)
        adaptive_floor = self.delta_floor_min + (
            self.delta_floor_max - self.delta_floor_min
        ) * snr_mean  # (B, L, 1) broadcasts to (B, L, D_inner)

        delta = F.softplus(
            self.dt_proj(dt_raw + dt_snr_shift)
        ) + adaptive_floor  # (B, L, D_inner)

        # [NOVEL] SNR-gated B: B = B_standard * (1 - alpha + alpha * snr_gate)
        if self.mode != 'standard':
            B_param = B_param * (1.0 - self.alpha + self.alpha * B_gate)

        # Get A matrix (negative for stability)
        A = -torch.exp(self.A_log)  # (D_inner, N)

        # Precompute discretized A and B for all timesteps (vectorized)
        dA = torch.exp(
            A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1)
        )  # (B, L, D, N)
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, D, N)
        dBx = dB * x.unsqueeze(-1)  # (B, L, D, N) - gated input

        # ================================================================
        # [NOVEL] Structural Noise Robustness: Adaptive Δ floor + ε residual
        # ================================================================
        # Three non-learnable architectural guarantees (register_buffer):
        #
        # 1. Adaptive Δ floor ∈ [0.05, 0.15] (SNR-dependent):
        #    High SNR → 0.15 (fast adaptation), Low SNR → 0.05 (long memory)
        #    → SSM bandwidth adapts to noise level
        #
        # 2. Adaptive ε ∈ [0.08, 0.20] (SNR-dependent):
        #    High SNR → 0.08 (trust gating), Low SNR → 0.20 (rescue info)
        #    → Ungated residual path scales with noise severity
        #
        # 3. B-gate floor = 0.3 (fixed minimum):
        #    → Prevents compound over-suppression (dt + B)
        #    → Minimum 30% input always flows through
        #
        # All FIXED (not learned) to prevent optimizer from destroying
        # structural guarantees during clean-data training.
        # ================================================================

        # [NOVEL] SNR-Adaptive Epsilon: pre-compute per-timestep
        # Low SNR → higher epsilon (rescue), High SNR → lower epsilon (trust gate)
        adaptive_eps = self.epsilon_max - (
            self.epsilon_max - self.epsilon_min
        ) * snr_mean  # (B, L, 1)

        # Sequential SSM scan
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, N, device=x.device)

        for t in range(L):
            # h_t = Ā·h_{t-1} + B̃·x_t + ε_t·x_t
            #       ─────────   ────────   ────────
            #       state decay  gated in   adaptive residual (SNR-scaled)
            h = (dA[:, t] * h + dBx[:, t] +
                 adaptive_eps[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-1))
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        return y


# ============================================================================
# SA-SSM v2: Enhanced SNR Resolution + PCEN Gate Conditioning
# ============================================================================

class SpectralAwareSSM_v2(nn.Module):
    """Enhanced SA-SSM with three improvements for extreme noise robustness.

    Problem: At -15dB, tanh(snr/10) compresses snr_mel to ~0.006, making all
    adaptive mechanisms (delta_floor, epsilon, bgate) collapse to their extremes
    with no dynamic range. The SSM effectively becomes feedforward.

    Improvements (0 extra learnable parameters):

    1. Internal SNR re-normalization (Michaelis-Menten):
       snr_internal = snr_mel / (snr_mel + 0.05)
       At -15dB: 0.006 → 0.107 (17× more resolution!)
       At clean: 0.95 → 0.95 (barely changed)
       Applied INSIDE SA-SSM only — SNREstimator output unchanged.

    2. Wider adaptive buffer ranges:
       delta_floor: [0.03, 0.15] (was [0.05, 0.15]) → longer memory at extreme
       epsilon: [0.05, 0.30] (was [0.08, 0.20]) → wider rescue range
       bgate: 0.20 (was 0.30) → more modulation freedom

    3. PCEN routing gate conditioning:
       Stationary noise (high gate) → reduce delta_floor 40% → longer memory
       (stationary noise is predictable, SSM can average it out over time)
       Non-stationary (low gate) → keep floor → faster adaptation needed

    Parameters: identical to SpectralAwareSSM (same learnable params).
    """

    def __init__(self, d_inner, d_state, n_mels=40, mode='full'):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_mels = n_mels
        self.mode = mode

        # Standard SSM projections: x -> (dt_raw, B, C)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)

        # SNR modulation projection: snr_mel -> (dt_mod, B_gate)
        self.snr_proj = nn.Linear(n_mels, d_state + 1, bias=True)

        # dt projection to expand dt to d_inner
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # HiPPO-initialized A matrix
        A = torch.arange(1, d_state + 1, dtype=torch.float32) + 0.5
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(d_inner))

        # SNR gating strength (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # [v2] Wider adaptive buffer ranges for extreme noise robustness
        self.register_buffer('delta_floor_min', torch.tensor(0.03))   # was 0.05
        self.register_buffer('delta_floor_max', torch.tensor(0.15))   # unchanged
        self.register_buffer('epsilon_min', torch.tensor(0.05))       # was 0.08
        self.register_buffer('epsilon_max', torch.tensor(0.30))       # was 0.20
        self.register_buffer('bgate_floor', torch.tensor(0.20))       # was 0.30

        # [v2] SNR re-normalization half-saturation constant
        # Chosen to equal delta_floor_min — values below this need more resolution
        self.register_buffer('snr_half_sat', torch.tensor(0.05))

    def set_calibration(self, delta_floor_min=None, delta_floor_max=None,
                        epsilon_min=None, epsilon_max=None, bgate_floor=None):
        """Runtime Parameter Calibration (same interface as v1)."""
        if delta_floor_min is not None:
            self.delta_floor_min.fill_(delta_floor_min)
        if delta_floor_max is not None:
            self.delta_floor_max.fill_(delta_floor_max)
        if epsilon_min is not None:
            self.epsilon_min.fill_(epsilon_min)
        if epsilon_max is not None:
            self.epsilon_max.fill_(epsilon_max)
        if bgate_floor is not None:
            self.bgate_floor.fill_(bgate_floor)

    def forward(self, x, snr_mel, pcen_gate=None):
        """
        Args:
            x: (B, L, d_inner) - feature sequence after conv1d + SiLU
            snr_mel: (B, L, n_mels) - per-mel-band SNR for each frame
            pcen_gate: (B, L) optional - per-frame PCEN routing stationarity score.
                       High = stationary noise detected → longer SSM memory.
                       Per-frame allows different memory behavior within an utterance.
        Returns:
            y: (B, L, d_inner) - SSM output
        """
        B, L, D = x.shape
        N = self.d_state

        # Standard projections from x
        x_proj = self.x_proj(x)  # (B, L, 2N+1)
        dt_raw = x_proj[..., :1]
        B_param = x_proj[..., 1:N + 1]
        C_param = x_proj[..., N + 1:]

        # SNR modulation of selection parameters
        snr_mod = self.snr_proj(snr_mel)  # (B, L, N+1)

        if self.mode in ('full', 'dt_only'):
            dt_snr_shift = snr_mod[..., :1]
        else:
            dt_snr_shift = torch.zeros_like(dt_raw)

        if self.mode in ('full', 'b_only'):
            B_gate_raw = torch.sigmoid(snr_mod[..., 1:])
            B_gate = B_gate_raw * (1.0 - self.bgate_floor) + self.bgate_floor
        else:
            B_gate = torch.ones_like(B_param)

        # ================================================================
        # [v2] Michaelis-Menten SNR re-normalization (internal to SA-SSM)
        # ================================================================
        # snr_mel from SNREstimator: tanh(snr/10) ∈ [0,1]
        # At -15dB: tanh compresses to ~0.006 → all adaptive params collapse
        # Re-normalize: s/(s+K) with K=0.05 spreads low values without
        # affecting high values. Only used for floor/eps adaptation.
        snr_internal = snr_mel / (snr_mel + self.snr_half_sat)
        snr_mean = snr_internal.mean(dim=-1, keepdim=True)  # (B, L, 1)

        # SNR-Adaptive Delta Floor
        adaptive_floor = self.delta_floor_min + (
            self.delta_floor_max - self.delta_floor_min
        ) * snr_mean

        # ================================================================
        # [v2] PCEN gate conditioning: noise-type-aware temporal dynamics
        # ================================================================
        # Stationary noise (pcen_gate→1): reduce floor → longer memory
        #   (stationary noise is predictable, averaging helps)
        # Non-stationary (pcen_gate→0): keep original floor → fast adaptation
        # Per-frame: different regions of an utterance can have different memory
        if pcen_gate is not None:
            # pcen_gate: (B, L) per-frame → (B, L, 1) for broadcasting with adaptive_floor
            pg = pcen_gate.detach().unsqueeze(-1)  # (B, L, 1)
            gate_modulation = 1.0 - 0.4 * pg  # [0.6, 1.0] per frame
            adaptive_floor = adaptive_floor * gate_modulation

        delta = F.softplus(
            self.dt_proj(dt_raw + dt_snr_shift)
        ) + adaptive_floor

        # SNR-gated B
        if self.mode != 'standard':
            B_param = B_param * (1.0 - self.alpha + self.alpha * B_gate)

        # Get A matrix
        A = -torch.exp(self.A_log)

        # Discretized A and B
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1))
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)
        dBx = dB * x.unsqueeze(-1)

        # SNR-Adaptive Epsilon (using re-normalized SNR)
        adaptive_eps = self.epsilon_max - (
            self.epsilon_max - self.epsilon_min
        ) * snr_mean

        # Sequential SSM scan
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, N, device=x.device)

        for t in range(L):
            h = (dA[:, t] * h + dBx[:, t] +
                 adaptive_eps[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-1))
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        return y


# ============================================================================
# Frequency-Interleaved Mamba (FI-Mamba)
# ============================================================================

class FrequencySSM(nn.Module):
    """Standard Selective SSM for frequency-axis scanning.

    Scans across mel bins (low → high frequency) to capture cross-band
    patterns: harmonic structure (speech) vs flat spectrum (noise).
    No SNR modulation — frequency patterns are noise-informative by nature.
    """

    def __init__(self, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Projections: x → (dt_raw, B, C)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # HiPPO-initialized A
        A = torch.arange(1, d_state + 1, dtype=torch.float32) + 0.5
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        """
        Args:
            x: (B, L, d_inner) — L = n_mels (frequency axis)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        x_proj = self.x_proj(x)
        dt_raw = x_proj[..., :1]
        B_param = x_proj[..., 1:N + 1]
        C_param = x_proj[..., N + 1:]

        delta = F.softplus(self.dt_proj(dt_raw)) + 0.1

        A = -torch.exp(self.A_log)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1))
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)
        dBx = dB * x.unsqueeze(-1)

        y = torch.zeros_like(x)
        h = torch.zeros(B, D, N, device=x.device)

        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        return y


class SpectralMambaBlock(nn.Module):
    """Mamba block scanning along the FREQUENCY axis.

    Processes mel spectrogram frame-by-frame: each time frame's n_mels
    mel energies form a length-n_mels sequence scanned by a selective SSM.

    This replaces CNN's 2D convolution for cross-frequency pattern detection
    using the SSM paradigm:
      - Harmonic structure (evenly spaced peaks) → speech
      - Flat spectrum → broadband noise (white, pink)
      - Low-freq concentration → factory hum
      - Speech-like but irregular → babble

    The conv1d (kernel=3) captures local frequency context (3 adjacent mel
    bins), while the SSM captures long-range frequency dependencies
    (harmonics spanning the full spectrum).
    """

    def __init__(self, d_model, d_state=3, d_conv=3, expand=1.5, n_mels=40):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels
        self.d_inner = int(d_model * expand)

        # Embed scalar mel energy → d_model
        self.mel_embed = nn.Linear(1, d_model)

        # Standard Mamba block (no SNR projection)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner)
        self.freq_ssm = FrequencySSM(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Deembed back to 1D
        self.mel_deembed = nn.Linear(d_model, 1)

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T) normalized log-mel spectrogram
        Returns:
            out: (B, n_mels, T) spectrally-enhanced mel (with residual)
        """
        Bs, Fm, Tm = mel.shape

        # Reshape: (B, F, T) → (B*T, F, 1) — process each frame independently
        x = mel.permute(0, 2, 1).reshape(Bs * Tm, Fm, 1)

        # Embed scalar → d_model
        x = self.mel_embed(x)  # (B*T, F, d_model)

        # Mamba block with residual
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Local frequency context (3 adjacent mel bins)
        x_branch = x_branch.transpose(1, 2)  # (B*T, d_inner, F)
        x_branch = self.conv1d(x_branch)[:, :, :Fm]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # Frequency SSM: scan low→high frequency
        y = self.freq_ssm(x_branch)

        # Gate + output projection + residual
        y = y * F.silu(z)
        y = self.out_proj(y) + residual  # (B*T, F, d_model)

        # Deembed to scalar + reshape
        out = self.mel_deembed(y).squeeze(-1)  # (B*T, F)
        out = out.reshape(Bs, Tm, Fm).permute(0, 2, 1)  # (B, F, T)

        return mel + out  # Residual: original mel + spectral correction


class FIMamba(nn.Module):
    """Frequency-Interleaved Mamba for Noise-Robust Keyword Spotting.

    Central thesis: SSMs fail under noise because they collapse the frequency
    axis (via patch projection) before temporal modeling, losing cross-frequency
    pattern information that CNNs capture with 2D convolution.

    FI-Mamba solves this by adding a spectral scanning layer BEFORE projection,
    giving the model native cross-frequency awareness within the SSM paradigm.

    Architecture:
      Audio → STFT → SNR Est → Mel → log → InstanceNorm
            → **SpectralMamba (frequency axis)** ← NEW: cross-band pattern detection
            → PatchProj → Temporal SA-SSM (time axis) × N → Classifier

    The spectral Mamba replaces ALL hand-designed frequency processing:
      - Wiener gain / spectral subtraction → learned frequency-domain filtering
      - PCEN / DualPCEN → learned adaptive normalization across bands
      - TinyConv2D → learned cross-frequency pattern detection
    All with a single SSM mechanism applied to the frequency axis.

    Paper: "Frequency-Interleaved Mamba: Native Cross-Frequency Awareness
           for Noise-Robust Keyword Spotting" (IEEE/ACM TASLP)
    """

    def __init__(self, n_mels=40, n_classes=12,
                 d_model=18, d_state_t=4, d_state_f=3,
                 d_conv=3, expand=1.5,
                 n_temporal_layers=2,
                 sr=16000, n_fft=512, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.d_model = d_model
        n_freq = n_fft // 2 + 1

        # 1. SNR Estimator (for Temporal SA-SSM blocks)
        self.snr_estimator = SNREstimator(n_freq=n_freq, use_running_ema=False)

        # 2. Mel filterbank (fixed, not learnable)
        mel_fb = self._create_mel_fb(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # 3. Instance normalization (before spectral processing)
        self.input_norm = nn.InstanceNorm1d(n_mels)

        # 4. Spectral Mamba: frequency-axis scanning
        #    Learns cross-band patterns: harmonics (speech) vs flat (noise)
        self.spectral_block = SpectralMambaBlock(
            d_model=d_model, d_state=d_state_f,
            d_conv=d_conv, expand=expand, n_mels=n_mels)

        # 5. Patch projection: n_mels → d_model
        self.patch_proj = nn.Linear(n_mels, d_model)

        # 6. Temporal SA-SSM blocks: time-axis scanning with SNR awareness
        self.blocks = nn.ModuleList([
            NanoMambaBlock(
                d_model=d_model,
                d_state=d_state_t,
                d_conv=d_conv,
                expand=expand,
                n_mels=n_mels,
                ssm_mode='full',
                use_ssm_v2=True)
            for _ in range(n_temporal_layers)
        ])

        # 7. Final norm + classifier
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    @staticmethod
    def _create_mel_fb(sr, n_fft, n_mels):
        """Create mel filterbank."""
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

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform at 16 kHz
        Returns:
            logits: (B, n_classes)
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()

        # SNR estimation (for temporal SA-SSM blocks)
        snr_mel = self.snr_estimator(mag, self.mel_fb)

        # Mel features + log compression + normalization
        mel = torch.matmul(self.mel_fb, mag)
        mel = torch.log(mel + 1e-8)
        mel = self.input_norm(mel)  # (B, n_mels, T_frames)

        # ---- SPECTRAL MAMBA: frequency-axis scanning ----
        # Captures cross-band patterns before patch projection destroys them
        mel = self.spectral_block(mel)  # (B, n_mels, T_frames)

        # Patch projection
        x = mel.transpose(1, 2)  # (B, T, n_mels)
        snr = snr_mel.transpose(1, 2)  # (B, T, n_mels)
        x = self.patch_proj(x)  # (B, T, d_model)

        # ---- TEMPORAL SA-SSM: time-axis scanning with SNR ----
        for block in self.blocks:
            x = block(x, snr)

        # Classify
        x = self.final_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

    def set_calibration(self, profile='default', **kwargs):
        """Runtime calibration for SA-SSM blocks."""
        for block in self.blocks:
            if hasattr(block.sa_ssm, 'set_calibration'):
                block.sa_ssm.set_calibration(**kwargs)


# Factory functions for FI-Mamba
def create_fimamba_matched(n_classes=12):
    """FI-Mamba matched to BC-ResNet-1 (~7,439 params).

    Architecture: SpectralMamba(d=18,N=3) → SA-SSM(d=18,N=4) × 2
    """
    return FIMamba(
        n_mels=40, n_classes=n_classes,
        d_model=18, d_state_t=4, d_state_f=3,
        d_conv=3, expand=1.5, n_temporal_layers=2)


def create_fimamba_small(n_classes=12):
    """FI-Mamba small variant (~5,000 params)."""
    return FIMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state_t=4, d_state_f=3,
        d_conv=3, expand=1.5, n_temporal_layers=2)


# ============================================================================
# Integrated Spectral Enhancement (0 learnable parameters)
# ============================================================================

class SpectralEnhancer(nn.Module):
    """Integrated Spectral Enhancement: Wiener Gain + SNR-Adaptive Bypass.

    A 0-parameter signal-processing module that sits BEFORE the STFT/mel
    pipeline.  The module:

      1. Estimates audio-level SNR from the first few frames.
      2. Applies Wiener Gain filtering (running minimum-statistics noise
         estimation, per-frame SNR-adaptive gain, frequency-weighted floor).
      3. Blends original and enhanced audio via a noise-type-aware bypass
         gate driven by spectral flatness.

    **Wiener Gain vs Spectral Subtraction**:
      - SS: enhanced = mag - α*noise → subtractive, can go negative → musical noise
      - Wiener: enhanced = mag * G, G = max(1-(noise/mag)^2, floor) → multiplicative
      - Wiener is smoother, produces fewer artifacts, better for downstream PCEN
      - Both achieve similar ~12dB effective SNR improvement on broadband noise

    At high SNR the bypass gate ≈ 1 → original audio is preserved (no
    quality loss on clean speech).  At low SNR the gate ≈ 0 → the Wiener-
    enhanced audio is used, providing ~20-30 %p accuracy improvement at
    extreme broadband noise (-15 dB white/pink).

    This module adds **0 learnable parameters** to the model.  All
    operations are classical signal processing wrapped in ``torch.no_grad``
    so that no additional GPU memory is consumed by the autograd graph.

    Args:
        n_fft: FFT size (default 512 = 32 ms @ 16 kHz).
        hop_length: STFT hop (default 160 = 10 ms @ 16 kHz).
        bypass_threshold: base bypass threshold in dB (default 8.0).
        bypass_scale: sigmoid steepness for bypass gate (default 1.5).
        alpha_noise: smoothing factor for running noise estimate (default 0.95).
    """

    def __init__(self, n_fft=512, hop_length=160,
                 bypass_threshold=8.0, bypass_scale=1.5,
                 alpha_noise=0.95):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bypass_threshold = bypass_threshold
        self.bypass_scale = bypass_scale
        self.alpha_noise = alpha_noise

        # Pre-compute frequency-weighted gain floor (fixed, not learnable)
        # More protection at low frequencies (speech F0, formants)
        # Less protection at high frequencies (allow more noise removal)
        n_freq = n_fft // 2 + 1
        freq_floor = torch.linspace(0.15, 0.03, n_freq)
        self.register_buffer('freq_floor', freq_floor.view(1, -1, 1))  # (1, F, 1)

    # ------------------------------------------------------------------
    # Audio-level SNR estimation (simple energy-based, 0 params)
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_snr(audio, hop_length, n_noise_frames=5):
        """Estimate utterance-level SNR from first N frames (noise floor).

        Args:
            audio: (B, T) waveform
        Returns:
            snr_db: (B, 1) estimated SNR in dB
        """
        frame_size = hop_length * 2
        noise_samples = min(n_noise_frames * frame_size, audio.size(-1) // 4)
        noise_floor = audio[:, :noise_samples].pow(2).mean(dim=-1, keepdim=True) + 1e-10
        signal_power = audio.pow(2).mean(dim=-1, keepdim=True)
        snr_linear = signal_power / noise_floor
        return 10.0 * torch.log10(snr_linear + 1e-10)  # (B, 1)

    # ------------------------------------------------------------------
    # Spectral flatness for noise-type classification (0 params)
    # ------------------------------------------------------------------
    @staticmethod
    def _spectral_flatness(mag):
        """Spectral flatness from magnitude spectrum.

        High SF (≈0.9) → flat/broadband (white, pink) → enhancement very effective.
        Low  SF (≈0.3) → peaked/modulated (babble)     → enhancement may hurt.

        Args:
            mag: (B, F, T_frames) magnitude spectrogram
        Returns:
            sf: (B,) spectral flatness ∈ [0, 1]
        """
        mag_mean = mag.mean(dim=-1)  # (B, F)
        log_mag = torch.log(mag_mean + 1e-8)
        geo_mean = torch.exp(log_mag.mean(dim=-1))  # (B,)
        arith_mean = mag_mean.mean(dim=-1) + 1e-8   # (B,)
        return (geo_mean / arith_mean).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Wiener Gain Filtering (0 params) — replaces Spectral Subtraction
    # ------------------------------------------------------------------
    def _wiener_gain_filter(self, audio):
        """Wiener Gain filtering: multiplicative noise suppression.

        Unlike spectral subtraction (mag - α*noise), Wiener gain is
        multiplicative (mag * G), which:
          - Never produces negative magnitudes → no musical noise artifacts
          - Smooth gain transition → fewer processing distortions
          - Better preserves speech spectral envelope for downstream PCEN

        Algorithm:
          1. Running minimum statistics noise estimation (same as SS v2)
          2. Per-frame SNR → adaptive oversubtraction factor
          3. Wiener gain: G = max(1 - (α * noise_est / (mag + eps))^2, floor)
          4. Enhanced magnitude = mag * G

        Returns:
            enhanced: (B, T) enhanced waveform
            mag: (B, F, T_frames) original magnitude (for SF computation)
        """
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()    # (B, F, T_frames)
        phase = spec.angle()
        B, F, T_frames = mag.shape

        # ---- Running minimum statistics noise estimation ----
        n_init = min(5, T_frames)
        noise_est = mag[..., :n_init].mean(dim=-1, keepdim=True).expand_as(mag).clone()

        for t in range(1, T_frames):
            frame_mag = mag[..., t:t + 1]
            local_min = torch.minimum(frame_mag, noise_est[..., t - 1:t])
            noise_est[..., t:t + 1] = (
                self.alpha_noise * noise_est[..., t - 1:t]
                + (1.0 - self.alpha_noise) * local_min
            )

        # ---- Per-frame SNR → adaptive oversubtraction ----
        frame_pwr = mag.pow(2).mean(dim=1, keepdim=True)        # (B,1,T)
        noise_pwr = noise_est.pow(2).mean(dim=1, keepdim=True)  # (B,1,T)
        frame_snr = 10.0 * torch.log10(frame_pwr / (noise_pwr + 1e-10) + 1e-10)
        # low SNR → oversubtract ≈ 3.5 ; high SNR → ≈ 1.0
        oversubtract = 1.0 + 2.5 * torch.sigmoid(-0.3 * (frame_snr - 5.0))

        # ---- Wiener Gain: multiplicative suppression ----
        # G = max(1 - (α * noise / (mag + eps))^2, freq_floor)
        # Squared ratio → smoother transition than linear SS
        noise_ratio = (oversubtract * noise_est) / (mag + 1e-8)
        gain = torch.maximum(1.0 - noise_ratio.pow(2), self.freq_floor)
        enhanced_mag = mag * gain

        # ---- Reconstruct waveform ----
        enhanced_spec = enhanced_mag * torch.exp(1j * phase)
        enhanced = torch.istft(enhanced_spec, self.n_fft, self.hop_length,
                               window=window, length=audio.size(-1))
        return enhanced, mag

    # ------------------------------------------------------------------
    # Forward: Wiener gain + noise-aware bypass
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, audio):
        """Apply Wiener gain enhancement with SNR-adaptive bypass.

        The entire computation is wrapped in ``torch.no_grad()`` because all
        operations are fixed signal processing — no learnable parameters.

        Args:
            audio: (B, T) raw waveform at 16 kHz
        Returns:
            out: (B, T) enhanced/original blended waveform
        """
        # 1. Audio-level SNR
        snr_est = self._estimate_snr(audio, self.hop_length)  # (B, 1)

        # 2. Wiener Gain Filtering
        enhanced, mag = self._wiener_gain_filter(audio)

        # 3. Spectral-flatness-aware adaptive bypass
        sf = self._spectral_flatness(mag)  # (B,)
        # High SF (white/pink) → lower threshold → more enhancement
        # Low  SF (babble)     → higher threshold → less enhancement
        adaptive_threshold = (
            self.bypass_threshold + 6.0 * (1.0 - sf.unsqueeze(1))
        )  # (B, 1)
        gate = torch.sigmoid(self.bypass_scale * (snr_est - adaptive_threshold))

        # 4. Blend: gate ≈ 1 → original (clean), gate ≈ 0 → enhanced (noisy)
        return gate * audio + (1.0 - gate) * enhanced


# ============================================================================
# NanoMamba Block
# ============================================================================

class NanoMambaBlock(nn.Module):
    """Single NanoMamba block: LayerNorm -> in_proj -> DWConv -> SA-SSM -> Gate -> out_proj + Residual."""

    def __init__(self, d_model, d_state=4, d_conv=3, expand=1.5, n_mels=40,
                 ssm_mode='full', use_ssm_v2=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.use_ssm_v2 = use_ssm_v2

        self.norm = nn.LayerNorm(d_model)

        # Input projection: (d_model) -> (2 * d_inner) for [x_branch, z_gate]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner)

        # Spectral-Aware SSM (v1 or v2)
        SSMClass = SpectralAwareSSM_v2 if use_ssm_v2 else SpectralAwareSSM
        self.sa_ssm = SSMClass(
            d_inner=self.d_inner,
            d_state=d_state,
            n_mels=n_mels,
            mode=ssm_mode)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, snr_mel, pcen_gate=None):
        """
        Args:
            x: (B, L, d_model) - input sequence
            snr_mel: (B, L, n_mels) - per-mel-band SNR per frame
            pcen_gate: (B, L) optional - per-frame PCEN routing stationarity (v2 only)
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

        # Spectral-Aware SSM (v2 receives pcen_gate for noise-type conditioning)
        if self.use_ssm_v2 and pcen_gate is not None:
            y = self.sa_ssm(x_branch, snr_mel, pcen_gate=pcen_gate)
        else:
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
                 use_moe_freq=False,
                 use_tiny_conv=False, tiny_conv_ks=3,
                 use_pcen=False, use_dual_pcen=False,
                 use_multi_pcen=False, n_pcen_experts=3,
                 use_dual_pcen_v2=False, use_multi_pcen_v2=False,
                 use_ssm_v2=False,
                 use_spectral_enhancer=False,
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
            use_moe_freq: if True, apply SNR-conditioned Mixture-of-Experts
                frequency filter. Uses SNR profile to route between
                narrow/wide/identity experts. Adds ~21 parameters.
            use_tiny_conv: if True, apply 2D convolution on mel spectrogram.
                Transplants CNN's structural noise robustness: 2D conv learns
                relative freq×time local patterns that are noise-invariant.
                Applied AFTER mel projection, BEFORE log. Adds 10 parameters.
            tiny_conv_ks: kernel size for TinyConv2D (default 3).
            use_pcen: if True, replace log(mel) with PCEN (Per-Channel
                Energy Normalization) + FrequencyDependentFloor +
                Running SNR Estimator. Structural noise suppression
                for factory/pink noise. Adds ~162 parameters.
            use_dual_pcen: if True, replace log(mel) with DualPCEN —
                two complementary PCEN experts (δ=2.0 babble + δ=0.01
                factory) with Spectral Flatness routing (0-cost gate).
                Structural robustness to ALL noise types. Adds ~321 params.
                Overrides use_pcen if both True.
            use_multi_pcen: if True, replace log(mel) with MultiPCEN —
                N-expert PCEN with hierarchical routing. Extends DualPCEN
                with additional experts for colored/structured noise.
                Overrides use_dual_pcen and use_pcen if True.
            n_pcen_experts: number of PCEN experts (2=DualPCEN, 3=TriPCEN).
                Only used when use_multi_pcen=True. Default 3.
            use_dual_pcen_v2: if True, use DualPCEN_v2 with enhanced routing:
                TMI + SNR-conditioned temp + temporal smoothing + aux loss.
                0 extra params vs DualPCEN. Overrides use_dual_pcen.
            use_multi_pcen_v2: if True, use MultiPCEN_v2 with enhanced routing.
                0 extra params vs MultiPCEN. Overrides use_multi_pcen.
            use_spectral_enhancer: if True, apply built-in SpectralEnhancer
                (SS v2 + SNR-adaptive bypass) on raw audio BEFORE STFT.
                Provides ~20-30%p improvement at extreme broadband noise
                (-15dB white/pink) with 0 extra parameters.
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
        self.use_moe_freq = use_moe_freq
        self.use_tiny_conv = use_tiny_conv
        self.use_pcen = use_pcen
        self.use_dual_pcen = use_dual_pcen or use_dual_pcen_v2
        self.use_dual_pcen_v2 = use_dual_pcen_v2
        self.use_multi_pcen = use_multi_pcen or use_multi_pcen_v2
        self.use_multi_pcen_v2 = use_multi_pcen_v2
        self.use_ssm_v2 = use_ssm_v2
        self.use_spectral_enhancer = use_spectral_enhancer

        # -1. Integrated Spectral Enhancement (0 params, before STFT)
        if use_spectral_enhancer:
            self.spectral_enhancer = SpectralEnhancer(
                n_fft=n_fft, hop_length=hop_length)

        # 0. Frequency processing plug-in (optional, mutually exclusive)
        if use_freq_filter:
            self.freq_filter = FrequencyFilter(n_freq=n_freq)
        if use_freq_conv:
            self.freq_conv = FreqConv(kernel_size=freq_conv_ks)
        if use_moe_freq:
            self.moe_freq = MoEFreq()

        # 0b. TinyConv2D: CNN structural noise robustness on mel spectrogram
        if use_tiny_conv:
            self.tiny_conv = TinyConv2D(kernel_size=tiny_conv_ks)

        # 0c. Feature normalization front-end
        if use_multi_pcen_v2:
            # MultiPCEN_v2: TMI + SNR-conditioned hierarchical routing
            self.multi_pcen = MultiPCEN_v2(n_mels=n_mels, n_experts=n_pcen_experts)
            self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)
        elif use_multi_pcen:
            # MultiPCEN: N-expert PCEN with hierarchical routing
            self.multi_pcen = MultiPCEN(n_mels=n_mels, n_experts=n_pcen_experts)
            self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)
        elif use_dual_pcen_v2:
            # DualPCEN_v2: TMI + SNR-conditioned routing — enhanced
            self.dual_pcen = DualPCEN_v2(n_mels=n_mels)
            self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)
        elif use_dual_pcen:
            # DualPCEN: noise-adaptive routing — ALL noise types
            self.dual_pcen = DualPCEN(n_mels=n_mels)
            self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)
        elif use_pcen:
            # Single PCEN: factory/pink specialist
            self.pcen = PCEN(n_mels=n_mels)
            self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)

        # 1. SNR Estimator (with running EMA when PCEN/DualPCEN/MultiPCEN is enabled)
        self.snr_estimator = SNREstimator(
            n_freq=n_freq, use_running_ema=(use_pcen or self.use_dual_pcen or self.use_multi_pcen))

        # 2. Mel filterbank (fixed)
        mel_fb = self._create_mel_fb(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # 3. Instance normalization
        self.input_norm = nn.InstanceNorm1d(n_mels)

        # 4. Patch projection: mel bands -> d_model
        self.patch_proj = nn.Linear(n_mels, d_model)

        # 5. SA-SSM Blocks (v1 or v2)
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
                ssm_mode=ssm_mode,
                use_ssm_v2=use_ssm_v2)
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
                    ssm_mode=ssm_mode,
                    use_ssm_v2=use_ssm_v2)
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
        """Extract mel features and SNR from (possibly SS-enhanced) audio.

        Args:
            audio: (B, T) raw or SS-enhanced waveform
        Returns:
            mel: (B, n_mels, T_frames) log-mel or PCEN spectrogram
            snr_mel: (B, n_mels, T_frames) per-mel-band SNR ∈ [0,1]
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

        # [NOVEL] MoE-Freq: SNR-conditioned frequency filtering
        # Applied AFTER SNR estimation so gating can use noise fingerprint
        if self.use_moe_freq:
            mag = self.moe_freq(mag, snr_mel)

        # Mel features
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T)

        # [NOVEL] CNN structural noise robustness: 2D conv on mel spectrogram
        # Learns relative freq×time local patterns (e.g., formant shapes)
        # that are noise-invariant. Applied BEFORE log to operate on
        # linear mel energy where relative patterns are most meaningful.
        if self.use_tiny_conv:
            mel = self.tiny_conv(mel)

        # Feature normalization: MultiPCEN / DualPCEN / PCEN / log
        # v2 variants receive snr_mel for SNR-conditioned routing
        if self.use_multi_pcen:
            mel = self.freq_dep_floor(mel)
            if self.use_multi_pcen_v2:
                mel = self.multi_pcen(mel, snr_mel=snr_mel)
            else:
                mel = self.multi_pcen(mel)
        elif self.use_dual_pcen:
            mel = self.freq_dep_floor(mel)
            if self.use_dual_pcen_v2:
                mel = self.dual_pcen(mel, snr_mel=snr_mel)
            else:
                mel = self.dual_pcen(mel)
        elif self.use_pcen:
            mel = self.freq_dep_floor(mel)   # Low-freq safety net
            mel = self.pcen(mel)             # Single PCEN (factory specialist)
        else:
            mel = torch.log(mel + 1e-8)      # Original log compression

        mel = self.input_norm(mel)

        return mel, snr_mel

    def get_routing_gate(self, per_frame=False):
        """Return last routing gate values.

        Args:
            per_frame: if True, return per-frame (B, T) gate for SA-SSM v2.
                       if False, return per-utterance (B,) mean for aux loss.
        Returns:
            gate: (B,) or (B, T) gate values from last forward pass, or None.
        """
        if per_frame:
            # Per-frame gate for SA-SSM v2 per-timestep conditioning
            if self.use_dual_pcen_v2 and hasattr(self.dual_pcen, '_last_gate_per_frame'):
                return self.dual_pcen._last_gate_per_frame
            if self.use_multi_pcen_v2 and hasattr(self.multi_pcen, '_last_gate_l1_per_frame'):
                return self.multi_pcen._last_gate_l1_per_frame
        else:
            # Per-utterance mean for auxiliary routing loss
            if self.use_dual_pcen_v2 and hasattr(self.dual_pcen, '_last_gate'):
                return self.dual_pcen._last_gate
            if self.use_multi_pcen_v2 and hasattr(self.multi_pcen, '_last_gate_l1'):
                return self.multi_pcen._last_gate_l1
        return None

    def get_routing_gate_l2(self):
        """Return Level 2 routing gate for TriPCEN aux loss.

        Level 2: broadband (white/pink → Expert 1) vs colored (factory/street → Expert 2).
        Only available for MultiPCEN_v2 with n_experts >= 3.

        Returns:
            gate_l2: (B,) mean L2 gate values, or None.
        """
        if self.use_multi_pcen_v2 and hasattr(self.multi_pcen, '_last_gate_l2'):
            return self.multi_pcen._last_gate_l2
        return None

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform at 16kHz
        Returns:
            logits: (B, n_classes)
        """
        # [ISE] Integrated Spectral Enhancement — before STFT
        # SS v2 + SNR-adaptive bypass: clean audio preserved, noisy audio enhanced
        if self.use_spectral_enhancer:
            audio = self.spectral_enhancer(audio)

        # Extract features + SNR
        mel, snr_mel = self.extract_features(audio)
        # mel: (B, n_mels, T), snr_mel: (B, n_mels, T)

        # Transpose to (B, T, n_mels) for sequence processing
        x = mel.transpose(1, 2)  # (B, T, n_mels)
        snr = snr_mel.transpose(1, 2)  # (B, T, n_mels)

        # Patch projection
        x = self.patch_proj(x)  # (B, T, d_model)

        # [v2] Extract PCEN routing gate for SA-SSM conditioning
        # Per-frame gate: stationary frames get longer memory, non-stat get faster adaptation
        pcen_gate = None
        if self.use_ssm_v2:
            pcen_gate = self.get_routing_gate(per_frame=True)  # (B, T) or None

        # SA-SSM blocks (each receives SNR + optional pcen_gate)
        if self.weight_sharing:
            for _ in range(self.n_repeats):
                x = self.blocks[0](x, snr, pcen_gate=pcen_gate)
        else:
            for block in self.blocks:
                x = block(x, snr, pcen_gate=pcen_gate)

        # Final norm + global average pooling
        x = self.final_norm(x)  # (B, T, d_model)
        x = x.mean(dim=1)  # (B, d_model)

        # Classify
        return self.classifier(x)

    def set_calibration(self, profile='default', **kwargs):
        """Runtime Parameter Calibration for all SA-SSM blocks.

        Set adaptive constants based on estimated noise environment.
        Called during silence/VAD periods before keyword detection.

        Args:
            profile: preset name or 'custom'
                'default'  - training defaults (no calibration)
                'clean'    - optimized for clean/quiet environment (20dB+)
                'light'    - light noise (10-20dB)
                'moderate' - moderate noise (0-10dB)
                'extreme'  - extreme noise (<0dB)
                'custom'   - use kwargs directly
            **kwargs: custom values (delta_floor_min, delta_floor_max,
                     epsilon_min, epsilon_max, bgate_floor)
        """
        # Calibration lookup table — domain knowledge driven
        # SA-SSM v2 uses wider default ranges; v1 profiles unchanged for compat
        if self.use_ssm_v2:
            PROFILES = {
                'default':  dict(delta_floor_min=0.03, delta_floor_max=0.15,
                                epsilon_min=0.05, epsilon_max=0.30, bgate_floor=0.2),
                'clean':    dict(delta_floor_min=0.15, delta_floor_max=0.15,
                                epsilon_min=0.05, epsilon_max=0.05, bgate_floor=0.0),
                'light':    dict(delta_floor_min=0.06, delta_floor_max=0.15,
                                epsilon_min=0.05, epsilon_max=0.15, bgate_floor=0.1),
                'moderate': dict(delta_floor_min=0.03, delta_floor_max=0.15,
                                epsilon_min=0.08, epsilon_max=0.25, bgate_floor=0.2),
                'extreme':  dict(delta_floor_min=0.01, delta_floor_max=0.15,
                                epsilon_min=0.10, epsilon_max=0.35, bgate_floor=0.5),
            }
        else:
            PROFILES = {
                'default':  dict(delta_floor_min=0.05, delta_floor_max=0.15,
                                epsilon_min=0.08, epsilon_max=0.20, bgate_floor=0.3),
                'clean':    dict(delta_floor_min=0.15, delta_floor_max=0.15,
                                epsilon_min=0.08, epsilon_max=0.08, bgate_floor=0.0),
                'light':    dict(delta_floor_min=0.08, delta_floor_max=0.15,
                                epsilon_min=0.08, epsilon_max=0.15, bgate_floor=0.2),
                'moderate': dict(delta_floor_min=0.05, delta_floor_max=0.15,
                                epsilon_min=0.10, epsilon_max=0.20, bgate_floor=0.3),
                'extreme':  dict(delta_floor_min=0.02, delta_floor_max=0.15,
                                epsilon_min=0.15, epsilon_max=0.30, bgate_floor=0.5),
            }

        if profile == 'custom':
            params = kwargs
        else:
            params = PROFILES.get(profile, PROFILES['default'])
            params.update(kwargs)  # allow partial override

        # Apply to all SA-SSM blocks
        for block in self.blocks:
            if hasattr(block, 'sa_ssm'):
                block.sa_ssm.set_calibration(**params)


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
# MoE-Freq Variants (SNR-Conditioned Noise-Aware Filtering)
# ============================================================================

def create_nanomamba_tiny_moe(n_classes=12):
    """NanoMamba-Tiny + MoE-Freq: ~4,657 params (+21 from baseline).

    SNR-conditioned mixture-of-experts frequency filter.
    3 experts: narrow(k=3), wide(k=7), identity.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_moe_freq=True)


def create_nanomamba_tiny_ws_moe(n_classes=12):
    """NanoMamba-Tiny-WS-MoE: Weight-Shared + MoE-Freq.

    지피지기 백전불패: ~3,782 params = BC-ResNet-1의 절반.
    Weight sharing (depth=3, params=1 block) + MoE-Freq (21 params).
    Target: Half the params of BC-ResNet-1, superior noise robustness.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=4, d_conv=3, expand=1.5,
        n_layers=1, weight_sharing=True, n_repeats=3,
        use_moe_freq=True)


# ============================================================================
# TinyConv2D Variants (Hybrid CNN-SSM: structural noise robustness)
# ============================================================================

def create_nanomamba_tiny_tc(n_classes=12):
    """NanoMamba-Tiny + TinyConv2D: ~4,646 params (+10 from baseline).

    Hybrid CNN-SSM: 2D conv on mel spectrogram transplants CNN's structural
    noise robustness. Conv2d(1,1,3,3) learns freq×time relative patterns
    that are noise-invariant, even when trained on clean data only.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_tiny_conv=True)


def create_nanomamba_tiny_ws_tc(n_classes=12):
    """NanoMamba-Tiny-WS-TC: Weight-Shared + TinyConv2D.

    Hybrid CNN-SSM with weight sharing: ~3,771 params = BC-ResNet-1의 절반.
    CNN의 구조적 noise robustness + SSM의 temporal modeling.
    Target: Half the params of BC-ResNet-1, superior noise robustness.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=4, d_conv=3, expand=1.5,
        n_layers=1, weight_sharing=True, n_repeats=3,
        use_tiny_conv=True)


# ============================================================================
# PCEN Variants (Structural Factory/Pink Noise Robustness)
# ============================================================================

def create_nanomamba_tiny_pcen(n_classes=12):
    """NanoMamba-Tiny-PCEN: SA-SSM + PCEN + Running SNR + FreqDepFloor.

    3-Layer structural defense against factory/pink noise:
    - PCEN: adaptive AGC replaces log(mel), preserves speech under noise
    - Running SNR: EMA noise tracking handles non-stationary factory impulses
    - FreqDepFloor: low-freq mel band safety net (non-learnable)

    Adds ~162 params over Tiny baseline. Trained on clean data only.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_pcen=True)


def create_nanomamba_small_pcen(n_classes=12):
    """NanoMamba-Small-PCEN: SA-SSM + PCEN + Running SNR + FreqDepFloor."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=24, d_state=4, d_conv=3, expand=1.5,
        n_layers=3, use_pcen=True)


def create_nanomamba_tiny_pcen_tc(n_classes=12):
    """NanoMamba-Tiny-PCEN-TC: PCEN + TinyConv2D (full structural defense).

    Combines PCEN (factory/pink noise) + TinyConv2D (babble noise).
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_pcen=True, use_tiny_conv=True)


# ============================================================================
# DualPCEN Variants — ALL-Noise Structural Robustness
# ============================================================================

def create_nanomamba_tiny_dualpcen(n_classes=12):
    """NanoMamba-Tiny-DualPCEN: SA-SSM + Noise-Adaptive Dual-PCEN Routing.

    The proposed noise-universal model. Two PCEN experts:
      - Expert 1 (δ=2.0): babble/non-stationary champion
      - Expert 2 (δ=0.01): factory/white/stationary champion
    Routed by Spectral Flatness (0-cost signal-based gate).

    Adds ~321 params over Tiny baseline (~4.9K total).
    Trained on clean data only — no noise augmentation needed.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen=True)


def create_nanomamba_small_dualpcen(n_classes=12):
    """NanoMamba-Small-DualPCEN: SA-SSM + Noise-Adaptive Dual-PCEN Routing."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=24, d_state=4, d_conv=3, expand=1.5,
        n_layers=3, use_dual_pcen=True)


def create_nanomamba_matched_dualpcen(n_classes=12):
    """NanoMamba-Matched-DualPCEN: param-matched to BC-ResNet-1 (~7.4K).

    Scales d_model 16→21 and d_state 4→5 to match BC-ResNet-1 parameter count.
    BC-ResNet-1: 7,464 params / NanoMamba-Matched: 7,402 params (0.8% diff).
    Fair comparison: same params, different architecture.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=21, d_state=5, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen=True)


# ============================================================================
# TriPCEN Variants (3-Expert PCEN Routing)
# ============================================================================

def create_nanomamba_tiny_tripcen(n_classes=12):
    """NanoMamba-Tiny-TriPCEN: 3-expert PCEN with hierarchical routing.

    Extends DualPCEN with 3rd expert for colored/structured noise (factory, street).
    Expert 0: Non-stationary (babble) — δ=2.0
    Expert 1: Broadband stationary (white/pink) — δ=0.01
    Expert 2: Colored stationary (factory/street) — δ=0.1 (NEW)
    Adds +161 params over DualPCEN (1 PCEN + 1 gate_temp).
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_multi_pcen=True, n_pcen_experts=3)


def create_nanomamba_matched_tripcen(n_classes=12):
    """NanoMamba-Matched-TriPCEN: 3-expert, param-matched to BC-ResNet-1.

    d_model=20, d_state=6: higher SSM memory (d_state 6 > DualPCEN's 5)
    compensates for slightly smaller model dimension.
    7,414 params vs BC-ResNet-1's 7,464 (-0.7% diff). Fair comparison.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=6, d_conv=3, expand=1.5,
        n_layers=2, use_multi_pcen=True, n_pcen_experts=3)


# ============================================================================
# v2 Enhanced Routing Variants (TMI + SNR-Conditioned, 0 extra params)
# ============================================================================

def create_nanomamba_tiny_dualpcen_v2(n_classes=12):
    """NanoMamba-Tiny-DualPCEN-v2: Enhanced routing, same 4,957 params.

    v2 improvements (0 extra params):
      1. TMI (Temporal Modulation Index) for time-domain stationarity
      2. SNR-conditioned gate temperature (sharper at low SNR)
      3. Temporal smoothing of SF (stable routing at low SNR)
      4. Auxiliary routing loss support (training-time only)
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True)


def create_nanomamba_matched_dualpcen_v2(n_classes=12):
    """NanoMamba-Matched-DualPCEN-v2: Enhanced routing, same 7,402 params.

    Param-matched to BC-ResNet-1 (7,464). v2 routing improvements:
      TMI + SNR-conditioned temp + SF smoothing + aux routing loss.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=21, d_state=5, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True)


def create_nanomamba_tiny_tripcen_v2(n_classes=12):
    """NanoMamba-Tiny-TriPCEN-v2: 3-expert + enhanced routing, same params."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_multi_pcen_v2=True, n_pcen_experts=3)


def create_nanomamba_matched_tripcen_v2(n_classes=12):
    """NanoMamba-Matched-TriPCEN-v2: 3-expert + enhanced routing, same 7,414 params."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=6, d_conv=3, expand=1.5,
        n_layers=2, use_multi_pcen_v2=True, n_pcen_experts=3)


# ============================================================================
# v2 + SSM v2 Factory Functions (PCEN v2 routing + SA-SSM v2 temporal dynamics)
# ============================================================================

def create_nanomamba_tiny_dualpcen_v2_ssmv2(n_classes=12):
    """NanoMamba-Tiny-DualPCEN-v2-SSMv2: Full v2 stack.
    DualPCEN v2 (TMI+SNR routing) + SA-SSM v2 (SNR re-norm + PCEN gate conditioning).
    Same 4,957 params as Tiny-DualPCEN."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True)


def create_nanomamba_matched_dualpcen_v2_ssmv2(n_classes=12):
    """NanoMamba-Matched-DualPCEN-v2-SSMv2: Full v2 stack at matched size.
    DualPCEN v2 + SA-SSM v2. Same 7,402 params as Matched-DualPCEN."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=21, d_state=5, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True)


def create_nanomamba_tiny_tripcen_v2_ssmv2(n_classes=12):
    """NanoMamba-Tiny-TriPCEN-v2-SSMv2: 3-expert v2 + SSM v2.
    Same 5,120 params as Tiny-TriPCEN."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_multi_pcen_v2=True, n_pcen_experts=3,
        use_ssm_v2=True)


def create_nanomamba_matched_tripcen_v2_ssmv2(n_classes=12):
    """NanoMamba-Matched-TriPCEN-v2-SSMv2: 3-expert v2 + SSM v2 at matched size.
    Same 7,414 params as Matched-TriPCEN."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=6, d_conv=3, expand=1.5,
        n_layers=2, use_multi_pcen_v2=True, n_pcen_experts=3,
        use_ssm_v2=True)


# ============================================================================
# Complete Model: v2 + SSMv2 + Integrated Spectral Enhancement (ISE)
# Full noise-robust pipeline: SS → DualPCEN v2 → SA-SSM v2, 0 extra params
# ============================================================================

def create_nanomamba_tiny_dualpcen_v2_ssmv2_se(n_classes=12):
    """NanoMamba-Tiny-SE: Complete noise-robust model (~4,967 params).

    Full pipeline (+10 params over Tiny-DualPCEN for TinyConv2D):
      1. SpectralEnhancer: Wiener gain + SNR-adaptive bypass (broadband defense)
      2. TinyConv2D: 3×3 cross-band feature mixing (+10 params, CNN advantage)
      3. DualPCEN v2: TMI + SNR-conditioned routing + SNR-adaptive AGC speed
      4. SA-SSM v2: Michaelis-Menten SNR re-norm + per-frame gate conditioning
      5. Noise curriculum v2 training + continuous calibration at inference

    Target: Surpass BC-ResNet-1 (7.5K) noise robustness with ~34% fewer params.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True,
        use_spectral_enhancer=True, use_tiny_conv=True)


def create_nanomamba_matched_dualpcen_v2_ssmv2_se(n_classes=12):
    """NanoMamba-Matched-SE: Complete, param-matched (~7,422 params).

    Near-identical to BC-ResNet-1 (7,464 params). Full v2 + ISE pipeline.
    Target: Exceed BC-ResNet-1 in ALL noise types at equal param count.

    Complete noise defense chain:
      - Wiener gain: ~12dB effective SNR boost on broadband noise (0 params)
      - TinyConv2D: cross-band pattern detection (+10 params, CNN advantage)
      - DualPCEN v2: adaptive AGC + routing for all noise types
      - SA-SSM v2: SNR-aware temporal modeling
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=21, d_state=5, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True,
        use_spectral_enhancer=True, use_tiny_conv=True)


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
        'NanoMamba-Tiny-MoE': create_nanomamba_tiny_moe,
        'NanoMamba-Tiny-WS-MoE': create_nanomamba_tiny_ws_moe,
        'NanoMamba-Tiny-TC': create_nanomamba_tiny_tc,
        'NanoMamba-Tiny-WS-TC': create_nanomamba_tiny_ws_tc,
        'NanoMamba-Tiny-WS': create_nanomamba_tiny_ws,
        'NanoMamba-Tiny-WS-FF': create_nanomamba_tiny_ws_ff,
        'NanoMamba-Tiny-PCEN': create_nanomamba_tiny_pcen,
        'NanoMamba-Small-PCEN': create_nanomamba_small_pcen,
        'NanoMamba-Tiny-PCEN-TC': create_nanomamba_tiny_pcen_tc,
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
