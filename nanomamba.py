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

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T) LINEAR mel energy (before log!)
        Returns:
            pcen_out: (B, n_mels, T) PCEN-normalized features
        """
        # Constrained parameters (noise-biased clamping prevents clean drift)
        s = torch.sigmoid(self.log_s).clamp(0.05, 0.3).unsqueeze(0).unsqueeze(-1)       # (1, M, 1)
        alpha = torch.sigmoid(self.log_alpha).clamp(0.9, 0.999).unsqueeze(0).unsqueeze(-1)
        delta = torch.exp(self.log_delta).clamp(*self.delta_clamp).unsqueeze(0).unsqueeze(-1)
        r = torch.sigmoid(self.log_r).clamp(0.05, 0.25).unsqueeze(0).unsqueeze(-1)

        # IIR smoothing of energy envelope (AGC)
        B, M, T = mel.shape
        smoother = mel[:, :, :1]  # Initialize with first frame

        smoothed_frames = []
        for t in range(T):
            smoother = (1 - s) * smoother + s * mel[:, :, t:t+1]
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
                 use_moe_freq=False,
                 use_tiny_conv=False, tiny_conv_ks=3,
                 use_pcen=False, use_dual_pcen=False,
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
        self.use_dual_pcen = use_dual_pcen

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
        if use_dual_pcen:
            # DualPCEN: noise-adaptive routing — ALL noise types
            self.dual_pcen = DualPCEN(n_mels=n_mels)
            self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)
        elif use_pcen:
            # Single PCEN: factory/pink specialist
            self.pcen = PCEN(n_mels=n_mels)
            self.freq_dep_floor = FrequencyDependentFloor(n_mels=n_mels)

        # 1. SNR Estimator (with running EMA when PCEN/DualPCEN is enabled)
        self.snr_estimator = SNREstimator(
            n_freq=n_freq, use_running_ema=(use_pcen or use_dual_pcen))

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

        # Feature normalization: DualPCEN / PCEN / log
        if self.use_dual_pcen:
            mel = self.freq_dep_floor(mel)   # Low-freq safety net
            mel = self.dual_pcen(mel)        # Noise-adaptive dual expert routing
        elif self.use_pcen:
            mel = self.freq_dep_floor(mel)   # Low-freq safety net
            mel = self.pcen(mel)             # Single PCEN (factory specialist)
        else:
            mel = torch.log(mel + 1e-8)      # Original log compression

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
