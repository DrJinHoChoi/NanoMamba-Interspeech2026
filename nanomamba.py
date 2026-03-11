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
import random
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
        # Prevent noise floor collapse for silence → root cause of NaN cascade
        init_noise = init_noise.clamp(min=1e-5)

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
        # Guarantee clean [0,1] output — catch any residual Inf/NaN
        snr_mel = torch.nan_to_num(snr_mel, nan=0.0, posinf=1.0, neginf=0.0)

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
        # NaN safety: clamp smoothed to prevent extreme gain when smoothed≈0
        smoothed = smoothed.clamp(min=1e-5)
        gain = (self.eps + smoothed) ** (-alpha)
        # NaN safety: clamp gain to prevent overflow in mel * gain
        gain = gain.clamp(max=1e5)
        pcen_out = (mel * gain + delta) ** r - delta ** r
        # Defense-in-depth: catch NaN from fractional power with NaN r (upstream propagation)
        pcen_out = torch.nan_to_num(pcen_out, nan=0.0)

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
        # .detach(): snr_mel is a conditioning signal, not a training target.
        # Without detach, gradient flows back through 101-step IIR loop × 2 experts
        # = 202-step chain → gradient explosion to Inf/NaN on GPU FP32.
        # Same pattern as pcen_gate.detach() in SSM.
        snr_cond = snr_mel.detach() if snr_mel is not None else None
        out_nonstat = self.pcen_nonstat(mel_linear, snr_mel=snr_cond)
        out_stat = self.pcen_stat(mel_linear, snr_mel=snr_cond)

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
        tmi = variance.sqrt() / (ema_E.clamp(min=1e-5) + 1e-8)  # CV coefficient; clamp prevents Inf for silence
        tmi = self._causal_smooth(tmi.clamp(0, 2.0) / 2.0)  # normalize to [0,1]

        # TMI correction: low TMI (temporally stationary) → boost toward stat expert
        tmi_boost = torch.relu(0.5 - tmi) * 0.5
        routing_signal = sf_adjusted + (1.0 - sf_adjusted) * tmi_boost

        # === [v2] SNR-conditioned temperature (0 params) ===
        # Low SNR → noise dominates → noise type is acoustically clear → sharper gate
        # High SNR → speech dominates → routing less critical → softer blending
        if snr_mel is not None:
            # snr_mel: tanh(snr/10) ∈ [0,1], 0=noise-dominated, 1=clean
            snr_global = snr_mel.detach().mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            snr_scale = 1.0 + self.snr_temp_scale * (1.0 - snr_global)
            effective_temp = self.gate_temp * snr_scale
        else:
            effective_temp = self.gate_temp

        # Gate computation
        gate = torch.sigmoid(effective_temp * (routing_signal - 0.5))

        # Weighted blend
        pcen_out = gate * out_stat + (1 - gate) * out_nonstat

        # NaN safety: clean outputs BEFORE storing gate for downstream consumers
        pcen_out = torch.nan_to_num(pcen_out, nan=0.0)
        gate = torch.nan_to_num(gate, nan=0.5)  # 0.5 = neutral blend for NaN frames

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
        snr_safe = snr_mel.clamp(0.0, 1.0)  # Guarantee valid range from SNREstimator
        snr_internal = snr_safe / (snr_safe + self.snr_half_sat)
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
        ).clamp(max=1.0) + adaptive_floor  # clamp softplus to prevent delta explosion

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
            # NaN safety: clamp hidden state to prevent accumulation overflow
            h = h.clamp(-1e4, 1e4)
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        # NaN safety: replace any residual NaN in output
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return y


# ============================================================================
# Selectivity-Modulated SA-SSM (SM-SSM)
# ============================================================================

class SelectivityModulatedSSM(SpectralAwareSSM_v2):
    """Selectivity-Modulated SA-SSM: CNN-like noise immunity meets SSM adaptivity.

    Key insight — Selective SSM vs CNN noise propagation:
      CNN:  y = W·x → noise enters ADDITIVELY through fixed filters W
      SSM:  Δ,B,C = f(x) → noise enters MULTIPLICATIVELY through input-dependent
            selection parameters. At low SNR, x ≈ noise, so Δ·B·x = noise³.

    Solution — SNR-adaptive selectivity:
      σ = sigmoid(scale · SNR_smooth + bias)     ← learnable transition
      Δ = σ_dt · Δ_sel + (1-σ_dt) · Δ_fixed
      B = σ_BC · B_sel + (1-σ_BC) · B_fixed
      C = σ_BC · C_sel + (1-σ_BC) · C_fixed

      High SNR (σ≈1): fully selective = Standard Mamba (input-dependent)
      Low SNR  (σ≈0): fixed dynamics  = LTI-SSM ≈ learned causal convolution

    Enhancements over naive selectivity modulation:
      1. Temporal σ smoothing: causal 3-frame avg reduces SNR estimation noise
         → stabilizes gate behavior, prevents σ flickering
      2. Per-state selectivity: B,C have per-state thresholds via sel_bias_BC
         → some states can stay selective for temporal patterns, others go LTI
         → at extreme low SNR, all states still converge to LTI (scale dominates)
      3. PCEN-conditioned σ: non-stationary noise (high pcen_gate) → lower σ
         → noise-type-aware selectivity modulation

    Extra parameters per block: 3·d_state + 4
      d_state=5: +19 params/block, +38 total (0.51% of 7,402)
      Total model: 7,402 + 38 = 7,440 (still < BC-ResNet-1's 7,464)
    """

    def __init__(self, d_inner, d_state, n_mels=40, mode='full'):
        super().__init__(d_inner, d_state, n_mels, mode)

        # Fixed (LTI) dynamics: learned "CNN-like" fallback
        # Small init (not zero) for gradient stability; near-zero ≈ SA-SSM v2
        self.dt_base = nn.Parameter(torch.zeros(1))
        self.B_base = nn.Parameter(torch.full((d_state,), 1e-3))
        self.C_base = nn.Parameter(torch.full((d_state,), 1e-3))

        # Selectivity gate: controls selective↔fixed interpolation
        # sel_scale=5.0: moderately sharp transition
        self.sel_scale = nn.Parameter(torch.tensor(5.0))

        # [Enhancement 2] Per-state selectivity thresholds
        # σ_dt: scalar gate for Δ (discretization step)
        # σ_BC: per-state gate for B,C (allows state-wise selectivity)
        # At extreme low SNR, all converge to 0 since sel_scale * 0 + bias < 0
        self.sel_bias_dt = nn.Parameter(torch.tensor(-1.0))
        self.sel_bias_BC = nn.Parameter(torch.full((d_state,), -1.0))

        # [Enhancement 3] PCEN-conditioned σ modulation
        # Non-stationary noise (high pcen_gate) → further reduce σ
        self.sigma_pcen_mod = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, snr_mel, pcen_gate=None):
        """
        Args:
            x: (B, L, d_inner) - feature sequence
            snr_mel: (B, L, n_mels) - per-mel-band SNR
            pcen_gate: (B, L) optional - per-frame PCEN routing score
        Returns:
            y: (B, L, d_inner)
        """
        Bs, L, D = x.shape
        N = self.d_state

        # ================================================================
        # 1. Selective parameters (from potentially noisy input x)
        # ================================================================
        x_proj = self.x_proj(x)  # (B, L, 2N+1)
        dt_selective = x_proj[..., :1]
        B_selective = x_proj[..., 1:N + 1]
        C_selective = x_proj[..., N + 1:]

        # ================================================================
        # 2. SNR-based selectivity gate σ (enhanced)
        # ================================================================
        # Michaelis-Menten re-normalization (from SA-SSM v2)
        snr_safe = snr_mel.clamp(0.0, 1.0)  # Guarantee valid range
        snr_internal = snr_safe / (snr_safe + self.snr_half_sat)
        snr_mean = snr_internal.mean(dim=-1, keepdim=True)  # (B, L, 1)

        # [Enhancement 1] Temporal smoothing: causal 3-frame average
        # Reduces noise in SNR estimation → more stable σ gate
        # Causal: pad left only, so no future frame dependency
        snr_smooth = F.pad(snr_mean.transpose(1, 2), (2, 0), mode='replicate')
        snr_smooth = F.avg_pool1d(snr_smooth, kernel_size=3, stride=1)
        snr_smooth = snr_smooth.transpose(1, 2)  # (B, L, 1)

        # [Enhancement 2] Separate σ for dt (scalar) and B,C (per-state)
        sigma_dt = torch.sigmoid(
            self.sel_scale * snr_smooth + self.sel_bias_dt)     # (B, L, 1)
        sigma_BC = torch.sigmoid(
            self.sel_scale * snr_smooth + self.sel_bias_BC)     # (B, L, N)

        # Cache for per-sub-band analysis (following DualPCEN._last_gate pattern)
        self._last_sigma_dt = sigma_dt.detach()     # (B, L, 1)
        self._last_sigma_BC = sigma_BC.detach()     # (B, L, N)

        # [Enhancement 3] PCEN gate modulation on σ
        # Non-stationary noise → pcen_gate high → reduce σ more aggressively
        if pcen_gate is not None:
            pg = pcen_gate.detach().unsqueeze(-1)  # (B, L, 1)
            # Safety: NaN guard on pcen_gate + clamp sigma_pcen_mod to [0, 1]
            pg = torch.nan_to_num(pg, nan=0.0)
            mod_clamped = self.sigma_pcen_mod.clamp(0.0, 1.0)
            pcen_mod = 1.0 - mod_clamped * pg
            sigma_dt = sigma_dt * pcen_mod
            sigma_BC = sigma_BC * pcen_mod

        # ================================================================
        # 3. Selectivity Modulation: blend selective ↔ fixed
        # ================================================================
        dt_raw = sigma_dt * dt_selective + (1.0 - sigma_dt) * self.dt_base
        B_param = sigma_BC * B_selective + (1.0 - sigma_BC) * self.B_base
        C_param = sigma_BC * C_selective + (1.0 - sigma_BC) * self.C_base

        # ================================================================
        # 4. SNR modulation (existing SA-SSM v2 mechanisms, unchanged)
        # ================================================================
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

        # Adaptive delta floor (use smoothed SNR for consistency)
        adaptive_floor = self.delta_floor_min + (
            self.delta_floor_max - self.delta_floor_min
        ) * snr_smooth

        # PCEN gate conditioning on adaptive floor (v2, unchanged)
        if pcen_gate is not None:
            pg = pcen_gate.detach().unsqueeze(-1)
            pg = torch.nan_to_num(pg, nan=0.0)  # NaN safety
            gate_modulation = 1.0 - 0.4 * pg
            adaptive_floor = adaptive_floor * gate_modulation

        delta = F.softplus(
            self.dt_proj(dt_raw + dt_snr_shift)
        ).clamp(max=1.0) + adaptive_floor  # clamp softplus to prevent delta explosion

        # SNR-gated B
        if self.mode != 'standard':
            B_param = B_param * (1.0 - self.alpha + self.alpha * B_gate)

        # ================================================================
        # 5. SSM state update (identical to SA-SSM v2)
        # ================================================================
        A = -torch.exp(self.A_log)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1))
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)
        dBx = dB * x.unsqueeze(-1)

        adaptive_eps = self.epsilon_max - (
            self.epsilon_max - self.epsilon_min
        ) * snr_smooth

        y = torch.zeros_like(x)
        h = torch.zeros(Bs, D, N, device=x.device)

        for t in range(L):
            h = (dA[:, t] * h + dBx[:, t] +
                 adaptive_eps[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-1))
            # NaN safety: clamp hidden state to prevent accumulation overflow
            h = h.clamp(-1e4, 1e4)
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        # NaN safety: replace any residual NaN in output
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return y


class NoiseCondSMSSM(SelectivityModulatedSSM):
    """NC-SSM: Noise-Conditioned Selectivity-Modulated SSM.

    Extends SM-SSM with frequency-aware noise conditioning:

    1. Per-sub-band selectivity (core innovation):
       SM-SSM: mean(40 bands) → scalar σ → all states same SNR
       NC-SSM: pool(40→N sub-bands) → per-state σ → each state uses
               its frequency sub-band's SNR for selectivity control.
       Uses adaptive avg pooling → works with any d_state (5, 6, 8, ...).
       State 0 ↔ Sub-band 0 (low freq): factory noise → σ≈0 (LTI mode)
       State N-1 ↔ Sub-band N-1 (high freq): factory noise → σ≈1 (selective)

    2. Stationarity-conditioned Δ floor:
       Stationary noise → boost Δ floor → longer memory → better averaging
       Non-stationary noise → keep smaller Δ floor → faster adaptation

    3. Spectral-flatness-conditioned B_base:
       Broadband noise (white) → B_base unchanged (all states equal)
       Narrowband noise (factory) → B_base modulated per state

    Extra parameters per block: d_state + 1 + d_state = 2*d_state + 1
      d_state=5: 11/block, 22 total | d_state=6: 13/block, 26 total

    Initialization: all NC-SSM params set to reproduce SM-SSM behavior
    exactly, enabling warm-start from SM-SSM checkpoints.
    """

    def __init__(self, d_inner, d_state=5, n_mels=40, mode='full'):
        super().__init__(d_inner, d_state, n_mels, mode)

        # NC-1: Per-sub-band selectivity scale for B,C gates
        # Each state's σ is driven by its matched frequency sub-band
        # Initialized to 5.0 (same as parent's sel_scale) for SM-SSM equivalence
        self.sel_sub_scale = nn.Parameter(
            torch.full((d_state,), 5.0))               # d_state p

        # NC-2: Stationarity → Δ floor modulation
        # 0.0 = no effect (SM-SSM equivalent)
        self.dt_station_alpha = nn.Parameter(
            torch.tensor(0.0))                          # 1 p

        # NC-3: Spectral flatness → B_base modulation
        # zeros = no effect (SM-SSM equivalent)
        self.B_sf_scale = nn.Parameter(
            torch.zeros(d_state))                       # d_state p

        # Sub-band configuration
        # Supports any d_state via adaptive pooling (40→d_state sub-bands)
        self.n_sub_bands = d_state
        self.n_mels = n_mels

    def forward(self, x, snr_mel, pcen_gate=None):
        """
        Args:
            x: (B, L, d_inner) - feature sequence
            snr_mel: (B, L, n_mels) - per-mel-band SNR in [0,1]
            pcen_gate: (B, L) optional - per-frame PCEN routing score
        Returns:
            y: (B, L, d_inner)
        """
        Bs, L, D = x.shape
        N = self.d_state

        # ================================================================
        # 1. Selective parameters (from potentially noisy input x)
        # ================================================================
        x_proj = self.x_proj(x)  # (B, L, 2N+1)
        dt_selective = x_proj[..., :1]
        B_selective = x_proj[..., 1:N + 1]
        C_selective = x_proj[..., N + 1:]

        # ================================================================
        # 2. NC-SSM: Per-sub-band selectivity gate σ
        # ================================================================
        # Michaelis-Menten re-normalization
        snr_safe = snr_mel.clamp(0.0, 1.0)  # Guarantee valid range
        snr_internal = snr_safe / (snr_safe + self.snr_half_sat)

        # ★ NC-1: Pool n_mels mel bands → d_state sub-bands (one per state)
        # Adaptive pooling handles any n_mels/d_state ratio (e.g., 40/6)
        snr_sub = F.adaptive_avg_pool1d(
            snr_internal.reshape(Bs * L, 1, self.n_mels),
            self.n_sub_bands
        ).reshape(Bs, L, self.n_sub_bands)  # (B, L, d_state)

        # Global mean for dt (scalar) — same as SM-SSM
        snr_mean = snr_sub.mean(dim=-1, keepdim=True)  # (B, L, 1)

        # Temporal smoothing: causal 3-frame average
        # For scalar dt gate:
        snr_smooth_dt = F.pad(
            snr_mean.transpose(1, 2), (2, 0), mode='replicate')
        snr_smooth_dt = F.avg_pool1d(
            snr_smooth_dt, kernel_size=3, stride=1)
        snr_smooth_dt = snr_smooth_dt.transpose(1, 2)  # (B, L, 1)

        # ★ For per-sub-band BC gate: smooth each sub-band independently
        snr_smooth_bc = F.pad(
            snr_sub.transpose(1, 2), (2, 0), mode='replicate')  # (B, 5, L+2)
        snr_smooth_bc = F.avg_pool1d(
            snr_smooth_bc, kernel_size=3, stride=1)
        snr_smooth_bc = snr_smooth_bc.transpose(1, 2)  # (B, L, 5)

        # σ_dt: scalar gate for Δ (unchanged from SM-SSM)
        sigma_dt = torch.sigmoid(
            self.sel_scale * snr_smooth_dt + self.sel_bias_dt)   # (B, L, 1)

        # ★ σ_BC: per-sub-band gate for B,C — CORE NC-SSM CHANGE
        # Each state's σ depends on its frequency sub-band's SNR
        sigma_BC = torch.sigmoid(
            self.sel_sub_scale * snr_smooth_bc + self.sel_bias_BC)  # (B, L, N)

        # Cache for per-sub-band multimodality analysis
        self._last_sigma_dt = sigma_dt.detach()         # (B, L, 1)
        self._last_sigma_BC = sigma_BC.detach()         # (B, L, N)
        self._last_snr_sub = snr_smooth_bc.detach()     # (B, L, N)

        # PCEN gate modulation on σ (unchanged from SM-SSM)
        if pcen_gate is not None:
            pg = pcen_gate.detach().unsqueeze(-1)  # (B, L, 1)
            # Safety: NaN guard on pcen_gate (DualPCEN can produce NaN in edge cases)
            pg = torch.nan_to_num(pg, nan=0.0)
            # Clamp sigma_pcen_mod to [0, 1] to prevent pcen_mod from going negative
            mod_clamped = self.sigma_pcen_mod.clamp(0.0, 1.0)
            pcen_mod = 1.0 - mod_clamped * pg
            sigma_dt = sigma_dt * pcen_mod
            sigma_BC = sigma_BC * pcen_mod

        # ================================================================
        # 3. NC-SSM: Noise-conditioned B_base + selectivity blend
        # ================================================================
        # ★ NC-3: Spectral-flatness-conditioned B_base
        # broadband_score ≈ 1 for white noise (uniform SNR), ≈ 0 for factory
        snr_sub_mean = snr_smooth_bc.mean(dim=-1, keepdim=True)  # (B, L, 1)
        # FIX: torch.std backward divides by std → NaN when std≈0 (clean audio)
        # Use variance + epsilon + sqrt for gradient-safe computation
        snr_sub_var = (snr_smooth_bc - snr_sub_mean).pow(2).mean(dim=-1, keepdim=True)
        snr_sub_std = (snr_sub_var + 1e-6).sqrt()               # (B, L, 1)
        spectral_var = (snr_sub_std / (snr_sub_mean.abs() + 1e-4)).clamp(0, 1)
        broadband_score = (1.0 - spectral_var).detach()  # conditioning signal, no grad needed

        B_base_mod = self.B_base * (
            1.0 + self.B_sf_scale.clamp(-5.0, 5.0) * broadband_score)  # (B, L, N) via broadcast

        # Blend selective ↔ fixed (with modulated B_base)
        dt_raw = sigma_dt * dt_selective + (1.0 - sigma_dt) * self.dt_base
        B_param = sigma_BC * B_selective + (1.0 - sigma_BC) * B_base_mod
        C_param = sigma_BC * C_selective + (1.0 - sigma_BC) * self.C_base

        # ================================================================
        # 4. SNR modulation (SA-SSM v2 mechanisms)
        # ================================================================
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

        # Adaptive delta floor (uses scalar snr_smooth_dt)
        adaptive_floor = self.delta_floor_min + (
            self.delta_floor_max - self.delta_floor_min
        ) * snr_smooth_dt

        # PCEN gate conditioning on adaptive floor
        if pcen_gate is not None:
            pg = pcen_gate.detach().unsqueeze(-1)
            pg = torch.nan_to_num(pg, nan=0.0)  # NaN safety
            gate_modulation = 1.0 - 0.4 * pg

            # ★ NC-2: Stationarity-conditioned Δ floor
            # pcen_gate high → stationary noise → boost floor (longer memory)
            station_boost = 1.0 + self.dt_station_alpha.clamp(-1.0, 1.0) * pg
            adaptive_floor = adaptive_floor * gate_modulation * station_boost

        delta = F.softplus(
            self.dt_proj(dt_raw + dt_snr_shift)
        ).clamp(max=1.0) + adaptive_floor  # clamp softplus to prevent delta explosion

        # SNR-gated B
        if self.mode != 'standard':
            B_param = B_param * (1.0 - self.alpha + self.alpha * B_gate)

        # ================================================================
        # 5. SSM state update (identical to SA-SSM v2)
        # ================================================================
        A = -torch.exp(self.A_log)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1))
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)
        dBx = dB * x.unsqueeze(-1)

        adaptive_eps = self.epsilon_max - (
            self.epsilon_max - self.epsilon_min
        ) * snr_smooth_dt

        y = torch.zeros_like(x)
        h = torch.zeros(Bs, D, N, device=x.device)

        for t in range(L):
            h = (dA[:, t] * h + dBx[:, t] +
                 adaptive_eps[:, t].unsqueeze(-1) * x[:, t].unsqueeze(-1))
            # NaN safety: clamp hidden state to prevent accumulation overflow
            h = h.clamp(-1e4, 1e4)
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        # NaN safety: replace any residual NaN in output
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return y


# ============================================================================
# Frequency-Aware Modules (for NanoApple)
# ============================================================================

class SubSpectralNorm(nn.Module):
    """Sub-Spectral Normalization (from BC-ResNet, Kim et al. 2021).

    Divides frequency axis into sub-bands and applies BatchNorm per sub-band.
    Each sub-band learns its own running statistics → frequency-aware
    normalization that equalizes energy across frequency regions.
    """

    def __init__(self, num_features, num_sub_bands=5):
        super().__init__()
        self.num_sub_bands = num_sub_bands
        self.bn = nn.BatchNorm2d(num_features * num_sub_bands)

    def forward(self, x):
        # x: (B, C, Fr, T)
        B, C, Fr, T = x.shape
        S = self.num_sub_bands
        pad = (S - Fr % S) % S
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            Fr_new = Fr + pad
        else:
            Fr_new = Fr
        x = x.reshape(B, C, S, Fr_new // S, T).reshape(B, C * S, Fr_new // S, T)
        x = self.bn(x)
        x = x.reshape(B, C, S, Fr_new // S, T).reshape(B, C, Fr_new, T)
        if pad > 0:
            x = x[:, :, :Fr_new - pad, :]
        return x


class BCResBlock(nn.Module):
    """BC-ResNet block with broadcasted residual connection.

    From BC-ResNet (Kim et al., Interspeech 2021).
    Used in NanoApple-v3 backbone for deep frequency processing.

    Architecture per block:
      x → Conv2d(1×1) + SSN → DW Conv(1×K) + SSN → Conv2d(1×1) + SSN
        → + Broadcast(freq_pool) → + skip/residual → ReLU
    """

    def __init__(self, in_ch, out_ch, kernel_size=3,
                 stride=(1, 1), dilation=1, num_sub_bands=5):
        super().__init__()
        self.use_residual = (in_ch == out_ch and stride == (1, 1))

        # Pointwise frequency conv: channel mixing
        self.freq_conv1 = nn.Conv2d(in_ch, out_ch, (1, 1))
        self.ssn1 = SubSpectralNorm(out_ch, num_sub_bands)

        # Depthwise temporal conv: local temporal context
        padding = (0, (kernel_size - 1) * dilation // 2)
        self.temp_dw_conv = nn.Conv2d(
            out_ch, out_ch, (1, kernel_size), stride=(1, stride[1]),
            padding=padding, dilation=(1, dilation), groups=out_ch)
        self.ssn2 = SubSpectralNorm(out_ch, num_sub_bands)

        # Pointwise frequency conv: channel refinement
        self.freq_conv2 = nn.Conv2d(out_ch, out_ch, (1, 1))
        self.ssn3 = SubSpectralNorm(out_ch, num_sub_bands)

        # Broadcast: frequency average → all bands
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Skip connection for channel/stride mismatch
        if not self.use_residual and in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (1, 1), stride=stride),
                nn.BatchNorm2d(out_ch))
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = F.relu(self.ssn1(self.freq_conv1(x)))
        out = F.relu(self.ssn2(self.temp_dw_conv(out)))
        out = self.ssn3(self.freq_conv2(out))
        out = out + self.freq_pool(out)  # Broadcast
        if self.use_residual:
            out = out + identity
        elif self.skip is not None:
            out = out + self.skip(identity)
        return F.relu(out)


class FreqConvBlock(nn.Module):
    """Lightweight 2D frequency-time processing block (BC-ResNet-inspired).

    Applies SubSpectralNorm + depthwise temporal conv + Broadcast on the
    mel spectrogram BEFORE patch projection, providing the frequency-axis
    processing that SSM-only architectures lack.

    Key mechanisms from BC-ResNet:
      - SubSpectralNorm: per-sub-band normalization (frequency-aware)
      - Broadcast: temporal average → all frequency bands (time→freq feedback)
      - Residual: skip connection for gradient flow

    All convolutions use causal padding for streaming compatibility.
    """

    def __init__(self, n_mels=40, c_mid=8, num_sub_bands=5, temp_ks=3):
        super().__init__()
        # Pointwise expand: 1 → c_mid channels
        self.freq_conv1 = nn.Conv2d(1, c_mid, (1, 1))
        self.ssn1 = SubSpectralNorm(c_mid, num_sub_bands)

        # Causal depthwise temporal conv
        self.temp_dw_conv = nn.Conv2d(
            c_mid, c_mid, (1, temp_ks),
            padding=(0, temp_ks - 1),  # left-pad only for causal
            groups=c_mid)
        self.ssn2 = SubSpectralNorm(c_mid, num_sub_bands)

        # Pointwise compress: c_mid → 1 channel
        self.freq_conv2 = nn.Conv2d(c_mid, 1, (1, 1))
        self.ssn3 = SubSpectralNorm(1, num_sub_bands)

        # Broadcast: frequency average → all bands
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.temp_ks = temp_ks

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T) - mel spectrogram after DualPCEN
        Returns:
            (B, n_mels, T) - frequency-enhanced mel spectrogram
        """
        x = mel.unsqueeze(1)  # (B, 1, 40, T)
        identity = x

        out = F.relu(self.ssn1(self.freq_conv1(x)))
        # Causal conv: trim future frames
        out = self.temp_dw_conv(out)[:, :, :, :mel.size(-1)]
        out = F.relu(self.ssn2(out))
        out = self.ssn3(self.freq_conv2(out))
        out = out + self.freq_pool(out)  # Broadcast
        out = F.relu(out + identity)     # Residual

        return out.squeeze(1)  # (B, 40, T)


class GroupedProj(nn.Module):
    """Sub-band-preserving projection (replaces Linear patch_proj).

    Instead of Linear(40, d_model) which freely mixes all frequency bands
    and destroys frequency structure, GroupedProj maps each sub-band
    independently:  8 mel bins → d_sub dims, preserving sub-band identity.

    Output dims are organized: [sub0_d0..d3, sub1_d0..d3, ..., sub4_d0..d3]
    This allows downstream SubBandNormBroadcast to re-establish frequency
    structure between SSM blocks.

    Param savings: Linear(40, 20) = 820 vs GroupedProj(5×Linear(8,4)) = 180
    → 640 params saved, reinvested in frequency-aware processing.
    """

    def __init__(self, n_mels=40, n_sub_bands=5, d_sub=4):
        super().__init__()
        self.n_sub_bands = n_sub_bands
        self.d_sub = d_sub
        self.group_size = n_mels // n_sub_bands  # 8
        self.projs = nn.ModuleList([
            nn.Linear(self.group_size, d_sub)
            for _ in range(n_sub_bands)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, T, n_mels) - mel features (transposed)
        Returns:
            (B, T, d_model) where d_model = n_sub_bands × d_sub
        """
        chunks = x.split(self.group_size, dim=-1)  # 5 × (B, T, 8)
        projected = [self.projs[i](chunks[i]) for i in range(self.n_sub_bands)]
        return torch.cat(projected, dim=-1)  # (B, T, 20)


class SubBandNormBroadcast(nn.Module):
    """Re-establish frequency sub-band structure after SSM processing.

    After SSM mixes all d_model dims together, this module:
    1. Reshapes to sub-band view: (B, T, d_model) → (B, T, n_sub, d_sub)
    2. Applies per-sub-band LayerNorm (independent normalization)
    3. Learnable per-sub-band scale/bias
    4. Broadcasts: mean across sub-bands → project to full d_model

    This is analogous to BC-ResNet's SubSpectralNorm + Broadcast,
    but applied in the SSM's latent space rather than on mel spectrograms.
    """

    def __init__(self, d_model=20, n_sub_bands=5):
        super().__init__()
        self.n_sub_bands = n_sub_bands
        self.d_sub = d_model // n_sub_bands

        # Per-sub-band normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.d_sub) for _ in range(n_sub_bands)
        ])

        # Broadcast: sub-band average → full d_model
        self.broadcast_proj = nn.Linear(self.d_sub, d_model, bias=False)

        # Learnable per-sub-band affine
        self.sub_scale = nn.Parameter(torch.ones(n_sub_bands, self.d_sub))
        self.sub_bias = nn.Parameter(torch.zeros(n_sub_bands, self.d_sub))

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with re-established sub-band structure
        """
        B, T, D = x.shape
        residual = x

        # Reshape to sub-band view
        x_sub = x.view(B, T, self.n_sub_bands, self.d_sub)

        # Per-sub-band LayerNorm
        normed = []
        for i in range(self.n_sub_bands):
            normed.append(self.norms[i](x_sub[:, :, i, :]))
        x_sub = torch.stack(normed, dim=2)  # (B, T, 5, 4)

        # Learnable affine
        x_sub = x_sub * self.sub_scale + self.sub_bias

        # Broadcast: average across sub-bands → project to full dim
        x_avg = x_sub.mean(dim=2)  # (B, T, d_sub)
        broadcast = self.broadcast_proj(x_avg)  # (B, T, d_model)

        return x_sub.reshape(B, T, D) + broadcast + residual


class SubBandSSMBlock(nn.Module):
    """Sub-band Parallel SSM: processes each sub-band independently.

    Key insight: BC-ResNet-1 wins because it maintains per-sub-band processing
    throughout all 7 blocks. Standard SSM mixes all d_model dims into a flat
    vector, destroying sub-band identity.

    SubBandSSMBlock processes each sub-band's temporal sequence independently,
    then applies cross-band broadcast for inter-sub-band communication.

    Architecture per sub-band (d_sub dims each):
      1. LayerNorm(d_sub)
      2. Linear(d_sub → d_inner_sub)  [in_proj: gate + x]
      3. Causal Conv1d(d_inner_sub//2)
      4. Simplified SSM scan (A, dt, B, C per sub-band)
      5. Linear(d_inner_sub//2 → d_sub)  [out_proj]
    Then: cross-band broadcast + residual

    This maintains frequency structure INSIDE the SSM blocks,
    analogous to BC-ResNet's SSN at every layer.
    """

    def __init__(self, d_model=20, n_sub_bands=5, d_state=4,
                 d_conv=3, expand=1.5, n_mels=40):
        super().__init__()
        self.n_sub_bands = n_sub_bands
        self.d_sub = d_model // n_sub_bands  # 4
        d_inner = int(self.d_sub * expand)  # 6

        # Per-sub-band: shared-weight SSM (weight sharing across sub-bands)
        # This is parameter-efficient: one set of SSM weights serves all 5 sub-bands
        self.norm = nn.LayerNorm(self.d_sub)
        self.in_proj = nn.Linear(self.d_sub, d_inner * 2, bias=False)  # gate + x
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                padding=d_conv - 1, groups=d_inner)  # causal

        # SSM parameters (shared across sub-bands)
        self.A_log = nn.Parameter(torch.log(
            torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # dt, B, C
        self.out_proj = nn.Linear(d_inner, self.d_sub, bias=False)

        # Cross-band broadcast
        self.broadcast_proj = nn.Linear(self.d_sub, d_model, bias=False)

        # SNR projection (shared, from full n_mels)
        self.snr_proj = nn.Linear(n_mels, d_state + 1, bias=True)

        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv

    def _ssm_scan(self, x, A, dt, B, C, D):
        """Simplified parallel SSM scan."""
        B_batch, T, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize
        dtA = torch.einsum('btd,dn->btdn', dt, A)
        dA = torch.exp(dtA)
        dB = torch.einsum('btd,btn->btdn', dt, B)

        # Sequential scan
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        return y + x * D

    def forward(self, x, snr, pcen_gate=None):
        """
        Args:
            x: (B, T, d_model)   - SSM latent
            snr: (B, T, n_mels)  - per-mel-band SNR
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        residual = x

        # SNR → shared modulation
        snr_mean = snr.mean(dim=1, keepdim=True)  # (B, 1, n_mels)
        snr_feat = self.snr_proj(snr_mean).expand(B, T, -1)  # (B, T, d_state+1)

        # Split into sub-bands: (B, T, 5, 4)
        x_sub = x.view(B, T, self.n_sub_bands, self.d_sub)

        # Process each sub-band with shared-weight SSM
        out_subs = []
        A = -torch.exp(self.A_log)

        for s in range(self.n_sub_bands):
            xs = self.norm(x_sub[:, :, s, :])  # (B, T, d_sub)

            # In-projection: gate + features
            xz = self.in_proj(xs)  # (B, T, 2*d_inner)
            x_feat, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

            # Causal conv1d
            x_conv = x_feat.transpose(1, 2)  # (B, d_inner, T)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_feat = F.silu(x_conv.transpose(1, 2))

            # SSM projections
            x_dbc = self.x_proj(x_feat)  # (B, T, 2*d_state+1)
            dt_raw = x_dbc[:, :, :1]  # (B, T, 1)
            bc_raw = x_dbc[:, :, 1:]  # (B, T, 2*d_state)

            # SNR modulation on dt
            dt_snr = snr_feat[:, :, :1]  # (B, T, 1)
            dt = F.softplus(self.dt_proj(dt_raw + dt_snr * 0.1))

            # B, C from combined projections + SNR
            B_param = bc_raw[:, :, :self.d_state] + snr_feat[:, :, 1:] * 0.1
            C_param = bc_raw[:, :, self.d_state:]

            # SSM scan
            y = self._ssm_scan(x_feat, A, dt, B_param, C_param, self.D)

            # Gated output
            y = y * F.silu(z)
            y = self.out_proj(y)  # (B, T, d_sub)
            out_subs.append(y)

        # Concatenate sub-bands
        out = torch.stack(out_subs, dim=2)  # (B, T, 5, d_sub)

        # Cross-band broadcast
        out_avg = out.mean(dim=2)  # (B, T, d_sub)
        broadcast = self.broadcast_proj(out_avg)  # (B, T, d_model)

        out = out.reshape(B, T, D) + broadcast + residual
        return out


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
# Learnable Spectral Enhancer (differentiable, 516 params)
# ============================================================================

class LearnableSpectralEnhancer(nn.Module):
    """Differentiable Wiener-style spectral enhancement on STFT magnitude.

    Replaces the non-learnable SpectralEnhancer with a trainable version
    that operates directly on STFT magnitude (inside the STFT pipeline)
    rather than on raw waveforms (before STFT).

    Key differences from SpectralEnhancer:
      1. Operates on mag (B, F, T) — no redundant STFT/iSTFT
      2. Per-frequency learnable oversubtraction and spectral floor
      3. Gain computation is differentiable (end-to-end learning)
      4. Noise estimation remains detached (stable training)
      5. Simple SNR-adaptive bypass (2 learnable params)

    The learnable gain enables the network to optimize noise suppression
    specifically for KWS accuracy — not generic SNR improvement — by
    learning which frequency bands to suppress and how aggressively,
    through end-to-end backpropagation from the classification loss.

    Parameters: 257 (oversub) + 257 (floor) + 1 (bypass_scale) + 1 (bypass_thresh) = 516
    """

    def __init__(self, n_freq=257, alpha_noise=0.95):
        super().__init__()
        self.n_freq = n_freq
        self.alpha_noise = alpha_noise

        # Per-frequency oversubtraction factor (learned)
        # softplus(0.5) ≈ 0.97 → effective oversubtraction starts near 1.0
        self.oversub_raw = nn.Parameter(torch.full((n_freq,), 0.5))

        # Per-frequency spectral floor (learned)
        # sigmoid(-1.5) ≈ 0.18 → reasonable starting floor
        self.floor_raw = nn.Parameter(torch.full((n_freq,), -1.5))

        # Bypass parameters (learned)
        self.bypass_scale = nn.Parameter(torch.tensor(1.5))
        self.bypass_threshold = nn.Parameter(torch.tensor(8.0))

    def _running_min_noise(self, mag):
        """Running minimum statistics noise estimation (detached).

        Same algorithm as SpectralEnhancer._wiener_gain_filter noise
        estimation.  Tracks a slowly-varying noise floor using exponential
        smoothing of local minimum statistics.

        Args:
            mag: (B, F, T) magnitude spectrogram
        Returns:
            noise_est: (B, F, T) estimated noise floor (detached)
        """
        B, F, T = mag.shape
        n_init = min(5, T)
        noise_est = mag[..., :n_init].mean(
            dim=-1, keepdim=True).expand_as(mag).clone()

        for t in range(1, T):
            frame_mag = mag[..., t:t + 1]
            local_min = torch.minimum(frame_mag, noise_est[..., t - 1:t])
            noise_est[..., t:t + 1] = (
                self.alpha_noise * noise_est[..., t - 1:t]
                + (1.0 - self.alpha_noise) * local_min
            )
        return noise_est

    def forward(self, mag):
        """Apply learnable Wiener gain with SNR-adaptive bypass.

        Noise estimation is wrapped in ``torch.no_grad()`` for training
        stability.  Only the gain computation and bypass gate are
        differentiable, allowing end-to-end optimization.

        Args:
            mag: (B, n_freq, T) magnitude spectrogram from STFT
        Returns:
            enhanced_mag: (B, n_freq, T) enhanced magnitude
        """
        # 1. Noise estimation (detached — stable target, no vanishing grads)
        with torch.no_grad():
            noise_est = self._running_min_noise(mag)

        # 2. Learnable Wiener gain (differentiable)
        oversub = F.softplus(self.oversub_raw).view(1, -1, 1)  # (1, F, 1)
        floor = torch.sigmoid(self.floor_raw).view(1, -1, 1)   # (1, F, 1)

        noise_ratio = (oversub * noise_est) / (mag + 1e-8)
        noise_ratio = torch.clamp(noise_ratio, max=10.0)  # prevent extreme values when mag ≈ 0
        gain = torch.clamp(1.0 - noise_ratio.pow(2), min=0.0)
        gain = torch.maximum(gain, floor)
        enhanced = mag * gain

        # 3. Per-frame SNR-adaptive bypass
        # frame_snr is detached: log10 of small ratios causes gradient explosion
        # bypass_scale/threshold still get gradients through sigmoid(scale*(snr-thresh))
        with torch.no_grad():
            frame_snr = 10.0 * torch.log10(
                mag.pow(2).mean(dim=1, keepdim=True) /
                (noise_est.pow(2).mean(dim=1, keepdim=True) + 1e-10) + 1e-10
            )  # (B, 1, T)
        gate = torch.sigmoid(
            self.bypass_scale * (frame_snr - self.bypass_threshold))

        # gate ≈ 1 → original mag (clean), gate ≈ 0 → enhanced (noisy)
        return gate * mag + (1.0 - gate) * enhanced


# ============================================================================
# NanoMamba Block
# ============================================================================

class NanoMambaBlock(nn.Module):
    """Single NanoMamba block: LayerNorm -> in_proj -> DWConv -> SA-SSM -> Gate -> out_proj + Residual."""

    def __init__(self, d_model, d_state=4, d_conv=3, expand=1.5, n_mels=40,
                 ssm_mode='full', use_ssm_v2=False, use_sm_ssm=False,
                 use_nc_ssm=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.use_ssm_v2 = use_ssm_v2
        self.use_sm_ssm = use_sm_ssm
        self.use_nc_ssm = use_nc_ssm

        self.norm = nn.LayerNorm(d_model)

        # Input projection: (d_model) -> (2 * d_inner) for [x_branch, z_gate]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner)

        # SSM variant selection:
        # NC-SSM: Noise-Conditioned SM-SSM (per-sub-band selectivity)
        # SM-SSM: Selectivity-Modulated (CNN↔Mamba blend based on SNR)
        # SA-SSM v2: SNR-aware with Michaelis-Menten + PCEN gate
        # SA-SSM v1: Basic SNR modulation
        if use_nc_ssm:
            SSMClass = NoiseCondSMSSM
        elif use_sm_ssm:
            SSMClass = SelectivityModulatedSSM
        elif use_ssm_v2:
            SSMClass = SpectralAwareSSM_v2
        else:
            SSMClass = SpectralAwareSSM
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

        # Spectral-Aware SSM (v2/SM-SSM/NC-SSM receive pcen_gate for noise-type conditioning)
        if (self.use_ssm_v2 or self.use_sm_ssm or self.use_nc_ssm) and pcen_gate is not None:
            y = self.sa_ssm(x_branch, snr_mel, pcen_gate=pcen_gate)
        else:
            y = self.sa_ssm(x_branch, snr_mel)

        # Gate with z branch
        y = y * F.silu(z)

        # Output projection + residual
        out = self.out_proj(y) + residual

        # NaN safety: unconditional guard (avoids gradient graph discontinuity)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out


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
                 use_sm_ssm=False,
                 use_nc_ssm=False,
                 use_lsg=False,
                 use_spectral_enhancer=False,
                 use_learnable_enhancer=False,
                 use_spectral_block=False, d_state_f=3,
                 use_spec_augment=False,
                 use_freq_aware=False, n_sub_bands=5, d_sub=4,
                 use_subband_ssm=False,
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
            use_freq_aware: if True, enable Frequency-Aware architecture:
                (1) FreqConvBlock on mel before patch_proj (BC-ResNet-style)
                (2) GroupedProj replaces Linear patch_proj (sub-band preserving)
                (3) SubBandNormBroadcast between SSM blocks
                Bridges CNN frequency processing + SSM streaming.
            n_sub_bands: number of frequency sub-bands for GroupedProj and
                SubBandNormBroadcast. Default 5 (8 mel bins per sub-band).
            d_sub: dimension per sub-band in GroupedProj. d_model = n_sub_bands × d_sub.
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
        self.use_sm_ssm = use_sm_ssm
        self.use_nc_ssm = use_nc_ssm
        self.use_lsg = use_lsg
        self.use_spectral_enhancer = use_spectral_enhancer
        self.use_learnable_enhancer = use_learnable_enhancer
        self.use_spectral_block = use_spectral_block
        self.use_spec_augment = use_spec_augment
        self.use_freq_aware = use_freq_aware
        self.n_sub_bands = n_sub_bands
        self.d_sub = d_sub
        self.use_subband_ssm = use_subband_ssm

        # Mutual exclusion: waveform-domain vs magnitude-domain enhancer
        assert not (use_spectral_enhancer and use_learnable_enhancer), \
            "Cannot use both SpectralEnhancer (waveform) and " \
            "LearnableSpectralEnhancer (magnitude) simultaneously"

        # -1. Integrated Spectral Enhancement (0 params, before STFT)
        if use_spectral_enhancer:
            self.spectral_enhancer = SpectralEnhancer(
                n_fft=n_fft, hop_length=hop_length)

        # -1b. Learnable Spectral Enhancement (516 params, on STFT magnitude)
        if use_learnable_enhancer:
            self.learnable_enhancer = LearnableSpectralEnhancer(
                n_freq=n_freq, alpha_noise=0.95)

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

        # 1b. Learned Spectral Gate (NC-SSM: explicit noise suppression before PCEN)
        if use_lsg:
            self.spectral_gate = LearnedSpectralGate(n_mels=n_mels)

        # 2. Mel filterbank (fixed)
        mel_fb = self._create_mel_fb(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # 3. Instance normalization
        self.input_norm = nn.InstanceNorm1d(n_mels)

        # 3b. [FI] Spectral Mamba Block: frequency-axis SSM after normalization
        # Captures cross-frequency patterns (harmonics, spectral tilt) on
        # PCEN-normalized features before PatchProj collapses frequency axis.
        if use_spectral_block:
            self.spectral_block = SpectralMambaBlock(
                d_model=d_model, d_state=d_state_f,
                d_conv=d_conv, expand=expand, n_mels=n_mels)

        # 3c. [FA] Frequency-Aware modules (NanoApple)
        if use_freq_aware:
            # BC-ResNet-style 2D conv + SSN + Broadcast on mel spectrogram
            self.freq_aware_block = FreqConvBlock(
                n_mels=n_mels, c_mid=8,
                num_sub_bands=n_sub_bands, temp_ks=3)

        # 4. Patch projection: mel bands -> d_model
        if use_freq_aware:
            # Sub-band-preserving grouped projection
            self.patch_proj = GroupedProj(
                n_mels=n_mels, n_sub_bands=n_sub_bands, d_sub=d_sub)
        else:
            self.patch_proj = nn.Linear(n_mels, d_model)

        # 5. SA-SSM Blocks (v1, v2, SM-SSM, or NC-SSM)
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
                use_ssm_v2=use_ssm_v2,
                use_sm_ssm=use_sm_ssm,
                use_nc_ssm=use_nc_ssm)
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
                    use_ssm_v2=use_ssm_v2,
                    use_sm_ssm=use_sm_ssm,
                    use_nc_ssm=use_nc_ssm)
                for _ in range(n_layers)
            ])
            self.n_repeats = n_layers

        # 5b. [FA] SubBandNormBroadcast between SSM blocks
        if use_freq_aware:
            self.sub_band_norms = nn.ModuleList([
                SubBandNormBroadcast(d_model=d_model, n_sub_bands=n_sub_bands)
                for _ in range(n_layers)
            ])

        # 5c. [SB-SSM] Sub-band Parallel SSM: per-sub-band independent processing
        if use_subband_ssm:
            self.subband_block = SubBandSSMBlock(
                d_model=d_model, n_sub_bands=n_sub_bands,
                d_state=d_state, d_conv=d_conv, expand=expand,
                n_mels=n_mels)

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

        # SNR estimation from ORIGINAL magnitude (accurate noise level for SA-SSM)
        snr_mel = self.snr_estimator(mag, self.mel_fb)  # (B, n_mels, T)

        # [NOVEL] Learnable Spectral Enhancement: differentiable Wiener gain
        # Applied AFTER SNR estimation (SNR sees original noise level) but
        # BEFORE mel projection (mel features benefit from cleaner magnitude).
        if self.use_learnable_enhancer:
            mag = self.learnable_enhancer(mag)

        # [NOVEL] MoE-Freq: SNR-conditioned frequency filtering
        # Applied AFTER SNR estimation so gating can use noise fingerprint
        if self.use_moe_freq:
            mag = self.moe_freq(mag, snr_mel)

        # Mel features (from possibly enhanced magnitude)
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T)

        # [NOVEL] CNN structural noise robustness: 2D conv on mel spectrogram
        # Learns relative freq×time local patterns (e.g., formant shapes)
        # that are noise-invariant. Applied BEFORE log to operate on
        # linear mel energy where relative patterns are most meaningful.
        if self.use_tiny_conv:
            mel = self.tiny_conv(mel)

        # [NC-SSM] Learned Spectral Gate: explicit per-freq, per-frame noise suppression
        # Applied BEFORE PCEN normalization — suppresses noisy mel bins using SNR
        if self.use_lsg:
            mel = self.spectral_gate(mel, snr_mel)

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

        # NaN safety: catch NaN from PCEN/InstanceNorm pipeline
        if torch.isnan(mel).any():
            mel = torch.nan_to_num(mel, nan=0.0)

        # [FA] Frequency-Aware: BC-ResNet-style 2D conv + SSN + Broadcast
        # Applied AFTER DualPCEN + InstanceNorm, BEFORE patch projection.
        # Provides frequency-axis processing that SSM-only architectures lack.
        if self.use_freq_aware:
            mel = self.freq_aware_block(mel)  # (B, n_mels, T)

        # [FI] Frequency-axis SSM: captures cross-band patterns
        # (harmonics, spectral tilt) on normalized features.
        # Applied AFTER DualPCEN + InstanceNorm, BEFORE PatchProj.
        if self.use_spectral_block:
            mel = self.spectral_block(mel)  # (B, n_mels, T) → (B, n_mels, T)

        # [SpecAugment] Frequency + time masking (training only, 0 params)
        # Masks random freq bands and time frames to improve generalization.
        # Conservative params for tiny models: freq_mask=5, time_mask=10.
        if self.training and self.use_spec_augment:
            mel = self._spec_augment(mel)

        return mel, snr_mel

    def _spec_augment(self, mel, n_freq_masks=2, freq_mask_param=5,
                      n_time_masks=2, time_mask_param=10):
        """SpecAugment: frequency & time masking on mel spectrogram.

        Applied during training only. Masks random contiguous bands in
        frequency and time, forcing the model to be robust to partial
        information loss. Universal technique for all audio models.

        Args:
            mel: (B, n_mels, T) normalized mel features
        Returns:
            mel: (B, n_mels, T) with random masks applied
        """
        B, F, T = mel.shape
        mel = mel.clone()

        for _ in range(n_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(0, F - f))
            mel[:, f0:f0 + f, :] = 0.0

        for _ in range(n_time_masks):
            t = random.randint(0, min(time_mask_param, T))
            t0 = random.randint(0, max(0, T - t))
            mel[:, :, t0:t0 + t] = 0.0

        return mel

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

        # [v2/SM/NC] Extract PCEN routing gate for SA-SSM conditioning
        # Per-frame gate: stationary frames get longer memory, non-stat get faster adaptation
        pcen_gate = None
        if self.use_ssm_v2 or self.use_sm_ssm or self.use_nc_ssm:
            pcen_gate = self.get_routing_gate(per_frame=True)  # (B, T) or None

        # SA-SSM blocks (each receives SNR + optional pcen_gate)
        if self.weight_sharing:
            for i in range(self.n_repeats):
                x = self.blocks[0](x, snr, pcen_gate=pcen_gate)
                if self.use_freq_aware:
                    x = self.sub_band_norms[min(i, len(self.sub_band_norms) - 1)](x)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, snr, pcen_gate=pcen_gate)
                if self.use_freq_aware:
                    x = self.sub_band_norms[i](x)

        # [SB-SSM] Sub-band Parallel SSM: per-sub-band independent processing
        # Applied after full-mixing SSM blocks to re-establish frequency structure
        if self.use_subband_ssm:
            x = self.subband_block(x, snr, pcen_gate=pcen_gate)

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
# Learnable Spectral Enhancement (LSE) variants
# ============================================================================

def create_nanomamba_tiny_dualpcen_v2_ssmv2_lse(n_classes=12):
    """NanoMamba-Tiny-LSE: Complete model with Learnable Spectral Enhancement.

    Full pipeline (~5,483 params):
      1. LearnableSpectralEnhancer: differentiable Wiener gain (+516 params)
         - Per-frequency learnable oversubtraction (257)
         - Per-frequency learnable spectral floor (257)
         - Learnable SNR-adaptive bypass (2)
      2. TinyConv2D: 3x3 cross-band feature mixing (+10 params)
      3. DualPCEN v2: TMI + SNR-conditioned routing + SNR-adaptive AGC speed
      4. SA-SSM v2: Michaelis-Menten SNR re-norm + per-frame gate conditioning

    Key advantage over SpectralEnhancer (SE):
      - SE has 0 learnable params (fixed Wiener gain, @torch.no_grad)
      - LSE has 516 learnable params (end-to-end optimized for KWS accuracy)
      - LSE operates on STFT magnitude (no redundant STFT/iSTFT)
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True,
        use_learnable_enhancer=True, use_tiny_conv=True)


def create_nanomamba_matched_dualpcen_v2_ssmv2_lse(n_classes=12):
    """NanoMamba-Matched-LSE: param-matched with Learnable Spectral Enhancement.

    ~7,938 params — comparable to BC-ResNet-1 (7,464 params).
    Full v2 pipeline with learnable spectral enhancement.

    Complete noise defense chain:
      - LearnableSpectralEnhancer: end-to-end trained noise suppression (516 params)
      - TinyConv2D: cross-band pattern detection (+10 params)
      - DualPCEN v2: adaptive AGC + routing for all noise types
      - SA-SSM v2: SNR-aware temporal modeling
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=21, d_state=5, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True,
        use_learnable_enhancer=True, use_tiny_conv=True)


# ============================================================================
# v2 + SSM v2 + FI (Frequency-Interleaved spectral scanning add-on)
# ============================================================================

def create_nanomamba_matched_dualpcen_v2_ssmv2_fi(n_classes=12):
    """NanoMamba-Matched-FI: DualPCEN v2 + SA-SSM v2 + SpectralMambaBlock.

    Full v2 stack + frequency-axis SSM scanning (~9,988 params).
    SpectralMambaBlock captures cross-frequency patterns (harmonics,
    spectral tilt) on PCEN-normalized features before PatchProj.

    Pipeline:
      STFT → SNR → Mel → DualPCEN_v2 → InstanceNorm
           → SpectralMamba (freq axis) → PatchProj → SA-SSM_v2 ×2 → Classifier
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=21, d_state=5, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True,
        use_spectral_block=True, d_state_f=3)


def create_nanomamba_tiny_dualpcen_v2_ssmv2_fi(n_classes=12):
    """NanoMamba-Tiny-FI: DualPCEN v2 + SA-SSM v2 + SpectralMambaBlock.

    Tiny variant with frequency-axis SSM scanning (~6,598 params).
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True, use_ssm_v2=True,
        use_spectral_block=True, d_state_f=3)


# ============================================================================
# SM-SSM: Selectivity-Modulated SA-SSM (CNN noise immunity + Mamba adaptivity)
# ============================================================================

def create_nanomamba_matched_dualpcen_v2_smssm(n_classes=12):
    """NanoMamba-Matched-SM: DualPCEN v2 + Selectivity-Modulated SA-SSM.

    Novel architecture: SM-SSM smoothly interpolates between selective
    (Mamba, input-dependent) and fixed (LTI ≈ learned convolution) dynamics
    based on estimated SNR.

    At high SNR: fully selective → Standard Mamba adaptivity
    At low SNR:  fixed dynamics  → CNN-like noise immunity

    Key insight: Selective SSM creates multiplicative noise through
    input-dependent Δ,B,C. Fixed LTI-SSM creates only additive noise
    (like CNN's fixed filters). SM-SSM bridges both paradigms.

    ~7,440 params (+38 from SM, 0.51% overhead).
    Enhanced: temporal σ smoothing, per-state selectivity, PCEN-conditioned σ.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=21, d_state=5, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True,
        use_ssm_v2=True, use_sm_ssm=True)


def create_nanomamba_tiny_dualpcen_v2_smssm(n_classes=12):
    """NanoMamba-Tiny-SM: DualPCEN v2 + SM-SSM. ~4,987 params.
    Enhanced: temporal σ smoothing, per-state selectivity, PCEN-conditioned σ."""
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=16, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True,
        use_ssm_v2=True, use_sm_ssm=True)


def create_nanomamba_nc_matched(n_classes=12):
    """NanoMamba-NC-Matched: DualPCEN v2 + NC-SSM + Learned Spectral Gate.

    Novel architecture: NC-SSM (Noise-Conditioned SM-SSM) extends SM-SSM with
    per-sub-band selectivity — each SSM state is controlled by its own
    frequency sub-band's SNR rather than a scalar average.

    Key innovations over SM-SSM:
    1. Per-sub-band σ_BC: 40 mel → 6 sub-bands → per-state selectivity
       Factory noise (low-freq) → only low-freq states go LTI, rest stay selective
       6 sub-bands (vs SM-SSM's 5): finer frequency resolution
    2. Stationarity-conditioned Δ floor: pcen_gate modulates minimum step size
    3. Spectral-flatness-conditioned B_base: broadband noise → stronger fixed input
    4. Learned Spectral Gate (LSG): explicit per-freq noise suppression before PCEN

    d_model=20, d_state=6: capacity-matched to BC-ResNet-1.
    7,443 params (21p margin under BC-ResNet-1's 7,464).
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=6, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True,
        use_ssm_v2=True, use_nc_ssm=True, use_lsg=True)


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
# NanoMamba v3: Pure Representation Efficiency (Beat BC-ResNet-1)
# ============================================================================
#
# Design Principle: At 7.4K params, EVERY parameter must contribute to
# representation learning. Noise robustness is achieved through:
#   1. PCEN (structural AGC normalization, superior to log-mel + BatchNorm)
#   2. SpecAugment (0-param data augmentation)
#   3. Standard noise-augmented training
# NOT through explicit SNR estimation, adaptive floors, or selectivity gates.
#
# Why SSM should beat CNN at equal params:
#   - PCEN provides stronger noise normalization than BatchNorm
#   - SSM captures full temporal trajectory of keywords (1s = 100 frames)
#   - CNN only captures local freq×time patches (bounded receptive field)
#   - 100% params for representation = parameter parity with BC-ResNet-1
# ============================================================================

class PureSSM(nn.Module):
    """Standard Selective SSM — maximum parameter efficiency.

    HiPPO-initialized diagonal A-matrix. No SNR modulation, no adaptive
    floors, no epsilon bypass. Pure Mamba dynamics where selection parameters
    (dt, B, C) are derived from the input alone.

    Parameters: d_inner * (3*d_state + 4)
    """

    def __init__(self, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Projections: x → (dt_raw[1], B[N], C[N])
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # HiPPO diagonal initialization: A[n] = -(n + 0.5)
        # Superior long-range temporal memory vs simple A[n] = -n
        A = torch.arange(1, d_state + 1, dtype=torch.float32) + 0.5
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone())
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        """
        Args:
            x: (B, L, d_inner) feature sequence
        Returns:
            y: (B, L, d_inner) SSM output
        """
        B, L, D = x.shape
        N = self.d_state

        proj = self.x_proj(x)  # (B, L, 2N+1)
        dt_raw = proj[..., :1]
        B_param = proj[..., 1:N + 1]
        C_param = proj[..., N + 1:]

        delta = F.softplus(self.dt_proj(dt_raw))  # (B, L, D)

        A = -torch.exp(self.A_log)  # (D, N), all negative → BIBO stable
        dA = torch.exp(
            A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1))  # (B, L, D, N)
        dBx = (delta.unsqueeze(-1) * B_param.unsqueeze(2)
               * x.unsqueeze(-1))  # (B, L, D, N)

        # Sequential scan
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, N, device=x.device)

        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            y[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]

        return y


class PureNanoMambaBlock(nn.Module):
    """Minimal Mamba block: LN → in_proj → DWConv → PureSSM → SiLU gate → out_proj + Residual.

    expand=1.0 (no expansion): d_inner = d_model.
    At 7.4K total budget, expansion wastes params on projection matrices.
    Better to increase d_model directly.
    """

    def __init__(self, d_model, d_state=4, d_conv=3):
        super().__init__()
        self.d_model = d_model

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv,
            padding=d_conv - 1, groups=d_model)
        self.ssm = PureSSM(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)  # (B, L, 2*d_model)
        x_branch, z = xz.chunk(2, dim=-1)

        # Local context via depthwise conv
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :residual.size(1)]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # Temporal modeling via SSM
        y = self.ssm(x_branch)

        # SiLU gating + output projection + residual
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class NanoMambaV3(nn.Module):
    """NanoMamba v3: Pure representation efficiency for KWS.

    Architecture:
      Raw Audio → STFT → Mel → PCEN → InstanceNorm → [SpecAugment]
      → PatchProj → N × PureSSM → LayerNorm → GAP → Classifier

    Key insight: at 7.4K params, 100% must go to representation.
    PCEN provides structural noise normalization (AGC). SpecAugment
    provides free regularization. No SNR estimation, no routing,
    no adaptive parameters.

    Configs:
      v3-Matched:  d=27, N=5, L=2       → 7,461 params (< BC-ResNet-1)
      v3-Deep:     d=37, N=6, L=1×3(WS) → 7,430 params (deeper, wider)
    """

    def __init__(self, n_mels=40, n_classes=12,
                 d_model=27, d_state=5, d_conv=3,
                 n_layers=2, sr=16000, n_fft=512, hop_length=160,
                 weight_sharing=False, n_repeats=3,
                 use_spec_augment=True):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.use_spec_augment = use_spec_augment

        # Mel filterbank (fixed, non-learnable)
        mel_fb = NanoMamba._create_mel_fb(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # PCEN: structural AGC — superior to log-mel + BatchNorm
        # Tracks energy envelope per channel, divides out stationary noise
        # 4 params per mel band = 160 total
        self.pcen = PCEN(n_mels=n_mels)

        # Instance normalization (per-sample, no batch dependence)
        self.input_norm = nn.InstanceNorm1d(n_mels)

        # Patch projection: mel → d_model
        self.patch_proj = nn.Linear(n_mels, d_model)

        # Pure Mamba blocks
        self.weight_sharing = weight_sharing
        if weight_sharing:
            block = PureNanoMambaBlock(d_model, d_state, d_conv)
            self.blocks = nn.ModuleList([block])
            self.n_repeats = n_repeats
        else:
            self.blocks = nn.ModuleList([
                PureNanoMambaBlock(d_model, d_state, d_conv)
                for _ in range(n_layers)
            ])
            self.n_repeats = n_layers

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def get_routing_gate(self, per_frame=False):
        """Stub for compatibility with train_colab.py."""
        return None

    def get_routing_gate_l2(self):
        """Stub for compatibility with train_colab.py."""
        return None

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform at 16kHz
        Returns:
            logits: (B, n_classes)
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # (B, F, T)

        # Mel projection
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T)

        # PCEN: structural noise suppression via AGC
        mel = self.pcen(mel)

        # Instance normalization
        mel = self.input_norm(mel)

        # SpecAugment (training only, 0 params)
        if self.training and self.use_spec_augment:
            mel = self._spec_augment(mel)

        # Sequence processing
        x = mel.transpose(1, 2)   # (B, T, n_mels)
        x = self.patch_proj(x)    # (B, T, d_model)

        # Mamba blocks
        if self.weight_sharing:
            for _ in range(self.n_repeats):
                x = self.blocks[0](x)
        else:
            for block in self.blocks:
                x = block(x)

        # Classification
        x = self.final_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

    def _spec_augment(self, mel, n_freq_masks=2, freq_mask_param=5,
                      n_time_masks=2, time_mask_param=10):
        """SpecAugment: freq & time masking, 0 params."""
        B, F, T = mel.shape
        mel = mel.clone()
        for _ in range(n_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(0, F - f))
            mel[:, f0:f0 + f, :] = 0.0
        for _ in range(n_time_masks):
            t = random.randint(0, min(time_mask_param, T))
            t0 = random.randint(0, max(0, T - t))
            mel[:, :, t0:t0 + t] = 0.0
        return mel


class NanoAppleV3(nn.Module):
    """NanoApple-v3: DualPCEN v2 + BC-ResNet backbone for noise-robust KWS.

    Strategy: BC-ResNet-1's deep frequency processing (7 BCResBlocks)
    with DualPCEN v2 noise-adaptive front-end + SS Training.

    Architecture:
      Audio → STFT → SNR Est → Mel → FreqDepFloor → DualPCEN v2
      → InstanceNorm → Conv2d(1→8) + BN + ReLU
      → Stage 1: BCResBlock(8→8) × 2
      → Stage 2: BCResBlock(8→12, s=(1,2)) + BCResBlock(12→12, d=2)
      → Stage 3: BCResBlock(12→16, s=(1,2)) + BCResBlock(16→16, d=4)
      → Stage 4: BCResBlock(16→20)
      → GAP → Classifier(20→12)

    ~7,409 params (55p margin under BC-ResNet-1's 7,464).

    Key advantages over BC-ResNet-1:
      - DualPCEN v2: noise-type adaptive routing (321p, 0 inference overhead)
      - SS Training: +20%p on broadband noise (training-time only, 0 cost)
      - Same clean accuracy (identical backbone depth/structure)
    """

    def __init__(self, n_mels=40, n_classes=12, num_sub_bands=5,
                 sr=16000, n_fft=512, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        n_freq = n_fft // 2 + 1

        # Mel filterbank (fixed, non-learnable)
        mel_fb = NanoMamba._create_mel_fb(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # SNR Estimator (with running EMA for DualPCEN conditioning)
        self.snr_estimator = SNREstimator(
            n_freq=n_freq, use_running_ema=True)

        # Feature normalization front-end: DualPCEN v2
        # DualPCEN v2 routes between nonstationary (babble) and stationary
        # (white/pink/factory) PCEN experts based on spectral flatness,
        # spectral tilt, and TMI — all zero-cost features.
        self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)
        self.dual_pcen = DualPCEN_v2(n_mels=n_mels)
        self.input_norm = nn.InstanceNorm1d(n_mels)

        # BC-ResNet backbone: 7 BCResBlocks across 4 stages
        # Channel progression: 8 → 8 → 12 → 16 → 20
        c = 8
        self.conv1 = nn.Conv2d(1, c, (5, 5), stride=(2, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(c)

        # Stage 1: same-channel, freq structure establishment
        self.stage1 = nn.Sequential(
            BCResBlock(c, c, num_sub_bands=num_sub_bands),
            BCResBlock(c, c, num_sub_bands=num_sub_bands))

        # Stage 2: channel expansion 8→12, temporal downsampling ×2
        c2 = int(c * 1.5)  # 12
        self.stage2 = nn.Sequential(
            BCResBlock(c, c2, stride=(1, 2), num_sub_bands=num_sub_bands),
            BCResBlock(c2, c2, dilation=2, num_sub_bands=num_sub_bands))

        # Stage 3: channel expansion 12→16, temporal downsampling ×2
        c3 = c * 2  # 16
        self.stage3 = nn.Sequential(
            BCResBlock(c2, c3, stride=(1, 2), num_sub_bands=num_sub_bands),
            BCResBlock(c3, c3, dilation=4, num_sub_bands=num_sub_bands))

        # Stage 4: channel expansion 16→20 (1 block)
        c4 = int(c * 2.5)  # 20
        self.stage4 = BCResBlock(c3, c4, num_sub_bands=num_sub_bands)

        # Classification head (no head conv — budget used for DualPCEN)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c4, n_classes)

    def get_routing_gate(self, per_frame=False):
        """Return DualPCEN v2 routing gate for aux loss or conditioning."""
        if per_frame:
            if hasattr(self.dual_pcen, '_last_gate_per_frame'):
                return self.dual_pcen._last_gate_per_frame
        else:
            if hasattr(self.dual_pcen, '_last_gate'):
                return self.dual_pcen._last_gate
        return None

    def get_routing_gate_l2(self):
        """Stub for compatibility with train_colab.py."""
        return None

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform at 16kHz
        Returns:
            logits: (B, n_classes)
        """
        # 1. STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # (B, F, T)

        # 2. SNR estimation (for DualPCEN conditioning)
        snr_mel = self.snr_estimator(mag, self.mel_fb)  # (B, n_mels, T)

        # 3. Mel projection
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T)

        # 4. DualPCEN v2 (noise-adaptive front-end)
        mel = self.freq_dep_floor(mel)
        mel = self.dual_pcen(mel, snr_mel=snr_mel)
        mel = self.input_norm(mel)

        # 5. BC-ResNet backbone (deep frequency processing)
        x = mel.unsqueeze(1)                  # (B, 1, 40, T)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 8, 20, T)
        x = self.stage1(x)                    # (B, 8, 20, T)
        x = self.stage2(x)                    # (B, 12, 20, T//2)
        x = self.stage3(x)                    # (B, 16, 20, T//4)
        x = self.stage4(x)                    # (B, 20, 20, T//4)

        # 6. Classification
        x = self.pool(x).flatten(1)           # (B, 20)
        return self.classifier(x)             # (B, n_classes)


# ============================================================================
# SAGN: SNR-Adaptive Gated Network
# ============================================================================

class LearnedSpectralGate(nn.Module):
    """Learned per-frequency, per-frame spectral gating for noise suppression.

    Replaces external spectral subtraction with an end-to-end trainable gate
    that suppresses noisy frequency bands while preserving clean ones.

    Unlike CNN's fixed kernels, this module creates a different gain mask
    for every input frame based on its per-band SNR estimate:
      - High SNR band → gain ≈ 1 (pass through)
      - Low SNR band  → gain ≈ 0 (suppress to floor)
      - KWS-important bands learn conservative suppression (large w)

    Parameters: 3 × n_mels = 120 (for n_mels=40)
    """

    def __init__(self, n_mels=40):
        super().__init__()
        # Per-frequency SNR sensitivity (how sharply to gate)
        self.w = nn.Parameter(torch.ones(n_mels))       # 40p
        # Per-frequency threshold bias
        self.b = nn.Parameter(torch.zeros(n_mels))       # 40p
        # Per-frequency minimum floor (prevents complete zeroing)
        self.floor_raw = nn.Parameter(torch.full((n_mels,), -3.0))  # 40p

    def forward(self, mel, snr):
        """Apply learned spectral gating.

        Args:
            mel: (B, n_mels, T) mel spectrogram (linear energy)
            snr: (B, n_mels, T) per-band SNR estimate in [0,1]
        Returns:
            gated_mel: (B, n_mels, T) noise-suppressed mel
        """
        # SNR-dependent gain: sigmoid(w * snr + b)
        gain = torch.sigmoid(self.w[:, None] * snr + self.b[:, None])
        # Floor prevents complete signal loss at extreme low SNR
        floor = torch.sigmoid(self.floor_raw[:, None])
        # Output: gain × mel + (1 - gain) × floor × mel
        #       = mel × (gain + (1 - gain) × floor)
        #       = mel × (gain × (1 - floor) + floor)
        return mel * (gain * (1.0 - floor) + floor)


class SNRCondScale(nn.Module):
    """Per-channel scaling conditioned on global SNR.

    Makes each layer of the backbone noise-aware by scaling channel
    activations based on the estimated SNR. At low SNR, noise-sensitive
    channels are automatically suppressed; at high SNR, all channels
    contribute fully.

    This addresses CNN's weakness of applying identical processing
    regardless of input noise level.

    Parameters: 2 × channels
    """

    def __init__(self, channels):
        super().__init__()
        # alpha: SNR sensitivity per channel (initialized to 0 = no effect)
        self.alpha = nn.Parameter(torch.zeros(channels))   # C p
        # beta: base scale per channel (initialized to 1 = identity)
        self.beta = nn.Parameter(torch.ones(channels))     # C p

    def forward(self, x, snr_global):
        """Apply SNR-conditioned channel scaling.

        Args:
            x: (B, C, Fr, T) feature map from BCResBlock
            snr_global: (B, 1) mean SNR scalar in [0,1]
        Returns:
            scaled_x: (B, C, Fr, T) SNR-adaptively scaled features
        """
        # scale = beta + alpha × snr_global
        # At snr_global ≈ 0 (noise): scale = beta (some channels suppressed)
        # At snr_global ≈ 1 (clean): scale = beta + alpha (all channels active)
        scale = self.beta + self.alpha * snr_global  # (B, C)
        return x * scale[:, :, None, None]            # (B, C, 1, 1)


class SAGN(nn.Module):
    """SAGN: SNR-Adaptive Gated Network for noise-robust keyword spotting.

    Combines BC-ResNet-1's proven deep frequency processing with two
    structural innovations that exploit CNN's weaknesses:

    1. Learned Spectral Gate (LSG): Per-frequency, per-frame noise
       suppression conditioned on estimated SNR — replaces external
       spectral subtraction with end-to-end learned denoising.
       CNN cannot do this because fixed kernels apply identical
       processing regardless of per-band noise level.

    2. SNR-Conditioned Channel Scale (SCCS): Per-layer channel scaling
       based on global SNR — makes backbone noise-aware without
       changing the core convolution structure.
       CNN's BatchNorm uses fixed statistics averaged over all SNR levels.

    Architecture:
      Audio → STFT → SNR Est → Mel → FreqDepFloor → LSG (★)
      → InstanceNorm → Conv2d(1→8) + BN + ReLU
      → Stage 1: BCResBlock(8→8) × 2 + SCCS (★)
      → Stage 2: BCResBlock(8→12, s=(1,2)) + BCResBlock(12→12, d=2) + SCCS
      → Stage 3: BCResBlock(12→16, s=(1,2)) + BCResBlock(16→16, d=4) + SCCS
      → Stage 4: BCResBlock(16→20) + SCCS
      → GAP → Classifier(20→12)

    ~7,080 params (384p margin under BC-ResNet-1's 7,464).
    """

    def __init__(self, n_mels=40, n_classes=12, num_sub_bands=5,
                 sr=16000, n_fft=512, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        n_freq = n_fft // 2 + 1

        # Mel filterbank (fixed, non-learnable)
        mel_fb = NanoMamba._create_mel_fb(sr, n_fft, n_mels)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb))

        # SNR Estimator (with running EMA, 4p)
        self.snr_estimator = SNREstimator(
            n_freq=n_freq, use_running_ema=True)

        # ★ Learned Spectral Gate — replaces DualPCEN with explicit denoising
        self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)
        self.spectral_gate = LearnedSpectralGate(n_mels=n_mels)  # 120p
        self.input_norm = nn.InstanceNorm1d(n_mels)  # 0p (affine=False)

        # BC-ResNet backbone: 7 BCResBlocks across 4 stages
        c = 8
        self.conv1 = nn.Conv2d(1, c, (5, 5), stride=(2, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(c)

        # Stage 1: freq structure establishment (8→8)
        self.stage1 = nn.Sequential(
            BCResBlock(c, c, num_sub_bands=num_sub_bands),
            BCResBlock(c, c, num_sub_bands=num_sub_bands))

        # Stage 2: channel expansion 8→12, temporal downsampling ×2
        c2 = int(c * 1.5)  # 12
        self.stage2 = nn.Sequential(
            BCResBlock(c, c2, stride=(1, 2), num_sub_bands=num_sub_bands),
            BCResBlock(c2, c2, dilation=2, num_sub_bands=num_sub_bands))

        # Stage 3: channel expansion 12→16, temporal downsampling ×2
        c3 = c * 2  # 16
        self.stage3 = nn.Sequential(
            BCResBlock(c2, c3, stride=(1, 2), num_sub_bands=num_sub_bands),
            BCResBlock(c3, c3, dilation=4, num_sub_bands=num_sub_bands))

        # Stage 4: channel expansion 16→20 (1 block)
        c4 = int(c * 2.5)  # 20
        self.stage4 = BCResBlock(c3, c4, num_sub_bands=num_sub_bands)

        # ★ SNR-Conditioned Channel Scale per stage
        self.snr_scale1 = SNRCondScale(c)    # 2×8  = 16p
        self.snr_scale2 = SNRCondScale(c2)   # 2×12 = 24p
        self.snr_scale3 = SNRCondScale(c3)   # 2×16 = 32p
        self.snr_scale4 = SNRCondScale(c4)   # 2×20 = 40p

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c4, n_classes)

    def get_routing_gate(self, per_frame=False):
        """Stub for compatibility with train_colab.py."""
        return None

    def get_routing_gate_l2(self):
        """Stub for compatibility with train_colab.py."""
        return None

    def forward(self, audio):
        """
        Args:
            audio: (B, T) raw waveform at 16kHz
        Returns:
            logits: (B, n_classes)
        """
        # 1. STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # (B, F, T)

        # 2. SNR estimation (per-band, per-frame)
        snr_mel = self.snr_estimator(mag, self.mel_fb)  # (B, n_mels, T)
        snr_global = snr_mel.mean(dim=(1, 2), keepdim=False).unsqueeze(1)  # (B, 1)

        # 3. Mel projection + Learned Spectral Gate (★ noise suppression)
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T)
        mel = self.freq_dep_floor(mel)
        mel = self.spectral_gate(mel, snr_mel)  # ★ per-band denoising
        mel = self.input_norm(mel)

        # 4. BC-ResNet backbone + SNR-Conditioned Scale (★ per-layer adaptation)
        x = mel.unsqueeze(1)                  # (B, 1, 40, T)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 8, 20, T)
        x = self.snr_scale1(self.stage1(x), snr_global)   # (B, 8, 20, T)
        x = self.snr_scale2(self.stage2(x), snr_global)   # (B, 12, 20, T//2)
        x = self.snr_scale3(self.stage3(x), snr_global)   # (B, 16, 20, T//4)
        x = self.snr_scale4(self.stage4(x), snr_global)   # (B, 20, 20, T//4)

        # 5. Classification
        x = self.pool(x).flatten(1)           # (B, 20)
        return self.classifier(x)             # (B, n_classes)


def create_sagn(n_classes=12):
    """SAGN: SNR-Adaptive Gated Network for noise-robust KWS.

    Exploits CNN's structural weaknesses:
    1. Learned Spectral Gate: per-frequency, per-frame noise suppression (120p)
    2. SNR-Conditioned Scale: per-layer noise adaptation (112p)
    3. BC-ResNet backbone: proven deep frequency processing (6,848p)

    ~7,080 params (384p margin under BC-ResNet-1's 7,464).

    Training recipe:
        python train_colab.py --models SAGN --noise_aug \\
            --noise_curriculum_v2 --calibrate
    """
    return SAGN(n_mels=40, n_classes=n_classes)


def create_nanoapple_v3(n_classes=12):
    """NanoApple-v3: DualPCEN v2 + BC-ResNet backbone for noise-robust KWS.

    Uses BC-ResNet-1's deep frequency processing (7 BCResBlocks) with
    DualPCEN v2 noise-adaptive front-end. Combined with SS Training,
    achieves asymmetric noise robustness advantage over vanilla BC-ResNet-1.

    ~7,409 params (55p margin under BC-ResNet-1's 7,464).

    Key advantages over BC-ResNet-1:
      - DualPCEN v2: noise-type adaptive routing (321p, 0 inference overhead)
      - SS Training: +20%p broadband noise (training-time only, 0 cost)
      - Same clean accuracy (identical backbone architecture)

    Training recipe:
        python train_colab.py --models NanoApple-v3 --noise_aug \\
            --noise_curriculum_v2 --ss_train --calibrate
    """
    return NanoAppleV3(n_mels=40, n_classes=n_classes)


def create_nanoapple(n_classes=12):
    """NanoApple: Frequency-Aware SSM for noise-robust KWS.

    Bridges CNN's frequency processing (BC-ResNet) with SSM's streaming:
      1. FreqConvBlock: BC-ResNet-style 2D conv + SSN + Broadcast on mel
      2. GroupedProj: Sub-band-preserving projection (5×Linear(8,4))
      3. SubBandNormBroadcast: Re-establishes freq structure between SSM blocks
      4. SM-SSM: Selectivity-Modulated SSM for multiplicative noise suppression
      5. DualPCEN v2: Adaptive per-channel energy normalization

    ~7,204 params. Addresses the TWO structural weaknesses vs BC-ResNet-1:
      - Frequency-axis blindness → FreqConvBlock + GroupedProj + SubBandNorm
      - Multiplicative noise    → SM-SSM σ-gate
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=6, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True,
        use_ssm_v2=True, use_sm_ssm=True,
        use_freq_aware=True, n_sub_bands=5, d_sub=4)


def create_nanoapple_v2(n_classes=12):
    """NanoApple-v2: Sub-band Parallel SSM for noise-robust KWS.

    Extends NanoApple with SubBandSSMBlock: processes each frequency
    sub-band independently through a shared-weight SSM, then applies
    cross-band broadcast. This maintains frequency structure INSIDE
    the SSM processing, analogous to BC-ResNet's per-sub-band SSN.

    Architecture:
      DualPCEN → FreqConvBlock → GroupedProj(5×8→4) → SM-SSM Block
      → SubBandNormBroadcast → SubBandSSMBlock → GAP → Classifier

    Uses 2 SM-SSM blocks (d_state=4) + 1 SubBandSSMBlock,
    fitting within BC-ResNet-1's 7,464 param budget.
    """
    return NanoMamba(
        n_mels=40, n_classes=n_classes,
        d_model=20, d_state=4, d_conv=3, expand=1.5,
        n_layers=2, use_dual_pcen_v2=True,
        use_ssm_v2=True, use_sm_ssm=True,
        use_freq_aware=True, n_sub_bands=5, d_sub=4,
        use_subband_ssm=True)


def create_nanomamba_v3_matched(n_classes=12):
    """NanoMamba-v3-Matched: 2 unique layers, d=27, N=5.

    7,461 params — 3 FEWER than BC-ResNet-1 (7,464).
    100% params for representation. PCEN for noise normalization.
    """
    return NanoMambaV3(
        n_mels=40, n_classes=n_classes,
        d_model=27, d_state=5, d_conv=3, n_layers=2)


def create_nanomamba_v3_deep(n_classes=12):
    """NanoMamba-v3-Deep: 1 block weight-shared 3×, d=37, N=6.

    7,430 params — 34 FEWER than BC-ResNet-1 (7,464).
    37% wider d_model + 20% larger state space + 50% deeper.
    Weight sharing trades layer specialization for depth + width.
    """
    return NanoMambaV3(
        n_mels=40, n_classes=n_classes,
        d_model=37, d_state=6, d_conv=3,
        n_layers=1, weight_sharing=True, n_repeats=3)


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
        # SM-SSM: Selectivity-Modulated SA-SSM (CNN↔Mamba blend based on SNR)
        'NanoMamba-Matched-SM': create_nanomamba_matched_dualpcen_v2_smssm,
        'NanoMamba-Tiny-SM': create_nanomamba_tiny_dualpcen_v2_smssm,
        # NC-SSM: Noise-Conditioned SM-SSM (per-sub-band selectivity + LSG)
        'NanoMamba-NC-Matched': create_nanomamba_nc_matched,
        # NanoApple: Frequency-Aware SSM (CNN freq processing + SSM streaming)
        'NanoApple': create_nanoapple,
        'NanoApple-v2': create_nanoapple_v2,
        'NanoApple-v3': create_nanoapple_v3,
        # SAGN: SNR-Adaptive Gated Network (CNN backbone + learned spectral gate)
        'SAGN': create_sagn,
        # v3: Pure representation efficiency — beat BC-ResNet-1
        'NanoMamba-v3-Matched': create_nanomamba_v3_matched,
        'NanoMamba-v3-Deep': create_nanomamba_v3_deep,
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
