#!/usr/bin/env python3
# coding=utf-8
# NC-Vibration: Noise-Conditioned Models for Vibration-Based Fault Detection
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.
# Dual License: Free for academic/research use. Commercial use requires license.
"""
NC-Vibration - Noise-Conditioned SSM/TCN for Vibration Fault Detection
=======================================================================

Extends the NC framework (NC-SSM, NC-TCN) from speech to vibration signals.
Same core idea: estimate environment quality -> condition the model -> robust inference.

Domain Mapping (Speech -> Vibration):
  - Audio waveform (16kHz)     -> Accelerometer signal (12.8-25.6kHz)
  - Mel spectrogram            -> Envelope spectrum (bearing fault frequencies)
  - SNR (speech vs noise)      -> VNR (Vibration-to-Noise Ratio: fault vs background)
  - Spectral Gate (per-band)   -> Spectral Gate (per-frequency-band)
  - DualPCEN (stat/nonstat)    -> DualPCEN (RPM-varying vs constant-speed)
  - KWS 12-class               -> Fault type classification (Normal/Inner/Outer/Ball/Combo)

Key Differences from Speech:
  1. VNR Estimator: noise floor from running minimum (not first-N-frames)
     - Machine vibration has no "silence" → use adaptive minimum tracking
  2. Order Tracking: normalize by RPM to handle speed variation
     - Fault frequencies are RPM-proportional (BPFO, BPFI, BSF, FTF)
  3. Envelope Analysis: amplitude demodulation extracts fault impulses
     - Bearing faults create periodic impulses modulated on carrier
  4. Kurtosis Gate: high kurtosis = impulsive fault, low = normal vibration
     - Replaces spectral flatness for vibration stationarity estimation

Target Datasets:
  - CWRU Bearing Dataset (Case Western Reserve University)
  - MFPT Bearing Dataset
  - Paderborn University Bearing Dataset

Target Deployment:
  - Cortex-M4/M7 MCU on motor controller board
  - NC-Vib-Tiny: ~3.5K params -> $0.5 MCU for per-motor monitoring
  - NC-Vib-20K: ~20K params -> $7 MCU for multi-sensor fusion
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# VNR Estimator (Vibration-to-Noise Ratio)
# ============================================================================

class VNREstimator(nn.Module):
    """Vibration-to-Noise Ratio estimator for fault signal detection.

    Unlike speech SNR (noise from first-N silent frames), vibration has no
    silence period. Instead, uses adaptive minimum tracking to estimate
    the background vibration floor, then computes VNR per frequency band.

    The key insight: fault-induced vibration creates spectral peaks at
    characteristic frequencies (BPFO, BPFI, BSF), while background
    mechanical noise is relatively flat. VNR highlights these peaks.

    Parameters: ~4 (noise_scale, floor, beta, gamma)
    """

    def __init__(self, n_freq=257, min_track_len=20):
        super().__init__()
        self.min_track_len = min_track_len

        # Learnable noise floor parameters
        self.noise_scale = nn.Parameter(torch.tensor(1.5))
        self.floor = nn.Parameter(torch.tensor(0.02))

        # Adaptive minimum tracking rates
        # sigmoid(-1.5) ~ 0.18: rise rate (when frame > min)
        self.raw_rise = nn.Parameter(torch.tensor(-1.5))
        # sigmoid(-3.0) ~ 0.05: fall rate (when frame < min, track faster)
        self.raw_fall = nn.Parameter(torch.tensor(-3.0))

    def forward(self, mag, freq_fb):
        """
        Args:
            mag: (B, F, T) magnitude spectrogram of vibration signal
            freq_fb: (n_bands, F) frequency band filterbank matrix
        Returns:
            vnr_bands: (B, n_bands, T) per-band VNR estimate in [0, 1]
        """
        rise = torch.sigmoid(self.raw_rise)
        fall = torch.sigmoid(self.raw_fall)

        B, F, T = mag.shape

        # Adaptive minimum tracking (no silence assumption)
        # Initialize from first few frames' minimum
        init_min = mag[:, :, :min(self.min_track_len, T)].min(dim=2, keepdim=True).values
        init_min = init_min.clamp(min=1e-5)

        noise_floor = init_min.clone()
        noise_estimates = []

        for t in range(T):
            frame = mag[:, :, t:t + 1]
            # Asymmetric: fast fall (track true minimum), slow rise (ignore transients)
            is_above = (frame > noise_floor).float()
            alpha_t = rise * is_above + fall * (1 - is_above)
            noise_floor = (1 - alpha_t) * noise_floor + alpha_t * frame
            # Safety: never go below half of initial minimum
            noise_floor = torch.maximum(noise_floor, init_min * 0.5)
            noise_estimates.append(noise_floor)

        running_noise = torch.cat(noise_estimates, dim=-1)  # (B, F, T)

        # VNR = signal / noise_floor
        vnr = mag / (self.noise_scale.abs() * running_noise + 1e-8)

        # Project to analysis bands
        vnr_bands = torch.matmul(freq_fb, vnr)  # (B, n_bands, T)

        # Normalize to [0, 1]
        vnr_bands = torch.tanh(vnr_bands / 10.0)
        vnr_bands = torch.nan_to_num(vnr_bands, nan=0.0, posinf=1.0, neginf=0.0)

        return vnr_bands


# ============================================================================
# Kurtosis Gate (replaces Spectral Flatness for vibration)
# ============================================================================

class KurtosisGate(nn.Module):
    """Per-band kurtosis-based impulsiveness gate.

    In vibration analysis, kurtosis is the primary indicator of fault:
    - Normal bearing: Gaussian vibration, kurtosis ~ 3.0
    - Faulty bearing: Periodic impulses, kurtosis >> 3.0 (can reach 20+)

    This module computes per-band kurtosis and gates the signal to
    enhance fault-related frequency bands while suppressing normal
    vibration bands.

    Parameters: 3 * n_bands (weight, bias, floor per band)
    """

    def __init__(self, n_bands=40):
        super().__init__()
        self.n_bands = n_bands

        # Per-band gate parameters
        self.gate_weight = nn.Parameter(torch.ones(n_bands) * 0.5)
        self.gate_bias = nn.Parameter(torch.zeros(n_bands))
        self.gate_floor = nn.Parameter(torch.ones(n_bands) * 0.1)

    def forward(self, x, vnr_bands=None):
        """
        Args:
            x: (B, n_bands, T) frequency band features
            vnr_bands: (B, n_bands, T) optional VNR for conditioning
        Returns:
            gated: (B, n_bands, T) kurtosis-gated features
        """
        # Compute per-band kurtosis over time dimension
        # Kurtosis = E[(x-mu)^4] / (E[(x-mu)^2])^2
        mu = x.mean(dim=-1, keepdim=True)
        diff = x - mu
        var = (diff ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
        kurt = (diff ** 4).mean(dim=-1, keepdim=True) / (var ** 2 + 1e-8)
        # kurt: (B, n_bands, 1), normal ~ 3.0, fault >> 3.0

        # Normalize kurtosis: (kurt - 3) / 3 so normal -> 0, fault -> positive
        kurt_norm = (kurt - 3.0) / 3.0

        # Gate: sigmoid(weight * kurt_norm + bias)
        w = self.gate_weight.unsqueeze(0).unsqueeze(-1)  # (1, n_bands, 1)
        b = self.gate_bias.unsqueeze(0).unsqueeze(-1)
        floor = torch.sigmoid(self.gate_floor).unsqueeze(0).unsqueeze(-1)

        gate = torch.sigmoid(w * kurt_norm + b)
        gate = gate * (1.0 - floor) + floor  # minimum floor per band

        # Optionally boost gate with VNR
        if vnr_bands is not None:
            gate = gate * (0.5 + 0.5 * vnr_bands)

        gated = x * gate
        gated = torch.nan_to_num(gated, nan=0.0, posinf=1e4, neginf=-1e4)
        return gated


# ============================================================================
# Envelope Extractor (bearing fault impulse demodulation)
# ============================================================================

class EnvelopeExtractor(nn.Module):
    """Learnable envelope extraction for bearing fault impulse detection.

    Bearing faults create periodic impulses at characteristic frequencies
    (BPFO, BPFI, BSF). These impulses are amplitude-modulated on a
    high-frequency carrier. Envelope analysis demodulates the impulses.

    Traditional: bandpass filter -> Hilbert transform -> lowpass
    Learned: Conv1d bandpass -> abs() -> Conv1d lowpass (end-to-end)

    Parameters: 2 * n_filters * kernel_size (bandpass + lowpass convs)
    """

    def __init__(self, n_filters=8, kernel_size=31):
        super().__init__()
        # Learnable bandpass filters
        self.bandpass = nn.Conv1d(1, n_filters, kernel_size,
                                  padding=kernel_size // 2, bias=False)
        # Learnable lowpass for envelope smoothing
        self.lowpass = nn.Conv1d(n_filters, n_filters, kernel_size,
                                 padding=kernel_size // 2, groups=n_filters,
                                 bias=False)

        # Initialize bandpass as bandpass-like filters
        with torch.no_grad():
            for i in range(n_filters):
                freq = (i + 1) / (n_filters + 1)
                t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
                # Gabor-like initialization
                self.bandpass.weight[i, 0] = torch.cos(2 * math.pi * freq * t) * \
                    torch.exp(-t ** 2 / (2 * (kernel_size / 4) ** 2))
            # Initialize lowpass as moving average
            self.lowpass.weight.fill_(1.0 / kernel_size)

    def forward(self, x):
        """
        Args:
            x: (B, T) raw vibration signal
        Returns:
            envelope: (B, n_filters, T) multi-band envelope features
        """
        x = x.unsqueeze(1)  # (B, 1, T)
        filtered = self.bandpass(x)  # (B, n_filters, T)
        rectified = filtered.abs()  # amplitude demodulation
        envelope = self.lowpass(rectified)  # smooth envelope
        return envelope


# ============================================================================
# NC-Vibration Frontend
# ============================================================================

class NCVibrationFrontend(nn.Module):
    """Noise-Conditioned frontend for vibration signals.

    Pipeline:
      Raw vibration (sr Hz, 1 sec)
        -> STFT (n_fft=512, hop=64) -> mag spectrogram
        -> Frequency band projection (like mel, but linear/log bands)
        -> VNR Estimation (adaptive minimum tracking)
        -> Kurtosis Gate (fault impulse enhancement)
        -> DualPCEN (RPM-adaptive normalization)
        -> InstanceNorm

    Parameters: ~600 (VNR: 4, KurtosisGate: 120, DualPCEN: ~480)
    """

    def __init__(self, n_bands=40, sr=25600, n_fft=512, hop_length=64,
                 use_envelope=False, use_dual_pcen=True):
        super().__init__()
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.use_envelope = use_envelope
        self.n_freq = n_fft // 2 + 1

        # Frequency band filterbank (log-spaced, like mel but for vibration)
        freq_fb = self._create_vibration_fb(sr, n_fft, n_bands)
        self.register_buffer('freq_fb', torch.from_numpy(freq_fb))

        # VNR Estimator (replaces SNR Estimator)
        self.vnr_estimator = VNREstimator(n_freq=self.n_freq)

        # Kurtosis Gate (replaces Learned Spectral Gate)
        self.kurtosis_gate = KurtosisGate(n_bands=n_bands)

        # Optional envelope extractor
        if use_envelope:
            self.envelope = EnvelopeExtractor(n_filters=8, kernel_size=31)

        # DualPCEN for vibration (reuse from speech — same math)
        if use_dual_pcen:
            from nanomamba import DualPCEN_v2
            self.dual_pcen = DualPCEN_v2(n_mels=n_bands)
        else:
            self.dual_pcen = None

        # Instance normalization
        self.input_norm = nn.InstanceNorm1d(n_bands)

        # Hann window buffer
        self.register_buffer('window', torch.hann_window(n_fft))

    @staticmethod
    def _create_vibration_fb(sr, n_fft, n_bands):
        """Create log-spaced frequency band filterbank for vibration.

        Unlike mel (perceptual), vibration uses log-spaced bands to
        capture harmonics of fault frequencies evenly across octaves.
        """
        n_freq = n_fft // 2 + 1
        freq_hz = np.linspace(0, sr / 2, n_freq)

        # Log-spaced center frequencies from 10 Hz to sr/2
        low_freq = 10.0
        high_freq = sr / 2.0
        centers = np.logspace(np.log10(low_freq), np.log10(high_freq), n_bands + 2)

        fb = np.zeros((n_bands, n_freq), dtype=np.float32)
        for i in range(n_bands):
            low = centers[i]
            center = centers[i + 1]
            high = centers[i + 2]

            # Rising slope
            mask_up = (freq_hz >= low) & (freq_hz <= center)
            if np.any(mask_up):
                fb[i, mask_up] = (freq_hz[mask_up] - low) / (center - low + 1e-8)

            # Falling slope
            mask_down = (freq_hz > center) & (freq_hz <= high)
            if np.any(mask_down):
                fb[i, mask_down] = (high - freq_hz[mask_down]) / (high - center + 1e-8)

        # Normalize each band
        fb_sum = fb.sum(axis=1, keepdims=True)
        fb = fb / (fb_sum + 1e-8)

        return fb

    def forward(self, vibration):
        """
        Args:
            vibration: (B, T) raw vibration signal
        Returns:
            features: (B, n_bands, T_frames) normalized features
            vnr_bands: (B, n_bands, T_frames) per-band VNR
        """
        # 1. STFT
        spec = torch.stft(vibration, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True)
        mag = spec.abs()  # (B, F, T_frames)

        # 2. VNR estimation
        vnr_bands = self.vnr_estimator(mag, self.freq_fb)

        # 3. Band projection
        band_features = torch.matmul(self.freq_fb, mag)  # (B, n_bands, T_frames)

        # 4. Kurtosis gate (enhance fault impulses)
        band_features = self.kurtosis_gate(band_features, vnr_bands)

        # 5. DualPCEN (RPM-adaptive normalization)
        if self.dual_pcen is not None:
            band_features = self.dual_pcen(band_features, snr_mel=vnr_bands)

        # 6. Instance normalization
        features = self.input_norm(band_features)

        return features, vnr_bands


# ============================================================================
# NC-Vibration Models (SSM and TCN backends)
# ============================================================================

class NCVibrationSSM(nn.Module):
    """NC-Vibration with SSM backend for fault detection.

    Same NC-SSM architecture as speech, with vibration-specific frontend.
    """

    def __init__(self, n_bands=40, n_classes=5, d_model=37, d_state=10,
                 d_conv=3, expand=1.5, n_layers=2, sr=25600,
                 n_fft=512, hop_length=64):
        super().__init__()
        from nanomamba import NanoMambaBlock

        self.frontend = NCVibrationFrontend(
            n_bands=n_bands, sr=sr, n_fft=n_fft, hop_length=hop_length)

        self.n_bands = n_bands

        # Patch projection
        self.patch_proj = nn.Linear(n_bands, d_model)

        # SSM blocks (reuse from nanomamba)
        self.blocks = nn.ModuleList([
            NanoMambaBlock(d_model=d_model, d_state=d_state,
                           d_conv=d_conv, expand=expand, n_mels=n_bands)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def get_routing_gate(self, per_frame=False):
        if self.frontend.dual_pcen is not None:
            g = self.frontend.dual_pcen._last_gate_per_frame if per_frame \
                else self.frontend.dual_pcen._last_gate
            return g
        return None

    def get_routing_gate_l2(self):
        return None

    def forward(self, vibration):
        """
        Args:
            vibration: (B, T) raw vibration waveform
        Returns:
            logits: (B, n_classes) fault classification logits
        """
        features, vnr_bands = self.frontend(vibration)

        # (B, n_bands, T_frames) -> (B, T_frames, d_model)
        x = self.patch_proj(features.transpose(1, 2))

        # vnr_bands: (B, n_bands, T_frames) -> (B, T_frames, n_bands)
        snr_mel = vnr_bands.transpose(1, 2)

        for block in self.blocks:
            x = block(x, snr_mel=snr_mel)

        x = self.final_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


class NCVibrationTCN(nn.Module):
    """NC-Vibration with TCN backend for fault detection.

    Fully parallel, INT8-safe, SIMD-optimized — ideal for MCU deployment.
    """

    def __init__(self, n_bands=40, n_classes=5, d_model=37,
                 d_conv=3, expand=1.5, n_layers=3, dilations=None,
                 sr=25600, n_fft=512, hop_length=64):
        super().__init__()
        from nanomamba import DilatedTCNBlock

        self.frontend = NCVibrationFrontend(
            n_bands=n_bands, sr=sr, n_fft=n_fft, hop_length=hop_length)

        # Patch projection
        self.patch_proj = nn.Linear(n_bands, d_model)

        # TCN blocks
        if dilations is None:
            dilations = [2 ** i for i in range(n_layers)]

        self.blocks = nn.ModuleList([
            DilatedTCNBlock(d_model=d_model, d_conv=d_conv,
                            expand=expand, dilation=d)
            for d in dilations
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def get_routing_gate(self, per_frame=False):
        if self.frontend.dual_pcen is not None:
            g = self.frontend.dual_pcen._last_gate_per_frame if per_frame \
                else self.frontend.dual_pcen._last_gate
            return g
        return None

    def get_routing_gate_l2(self):
        return None

    def forward(self, vibration):
        features, vnr_bands = self.frontend(vibration)

        x = self.patch_proj(features.transpose(1, 2))

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================================================
# Factory Functions
# ============================================================================

# CWRU Bearing: 4 classes (Normal, Inner, Outer, Ball)
# Extended: 5 classes (+ Combination fault)

def create_nc_vib_ssm_20k(n_classes=4):
    """NC-Vibration-SSM-20K: Full SSM model for bearing fault detection."""
    return NCVibrationSSM(
        n_bands=40, n_classes=n_classes,
        d_model=37, d_state=10, d_conv=3, expand=1.5,
        n_layers=2, sr=25600, n_fft=512, hop_length=64)


def create_nc_vib_tcn_20k(n_classes=4):
    """NC-Vibration-TCN-20K: Full TCN model for bearing fault detection."""
    return NCVibrationTCN(
        n_bands=40, n_classes=n_classes,
        d_model=37, d_conv=3, expand=1.5,
        n_layers=3, dilations=[1, 2, 4],
        sr=25600, n_fft=512, hop_length=64)


def create_nc_vib_ssm_matched(n_classes=4):
    """NC-Vibration-SSM-Matched: Mid-tier SSM for bearing fault detection."""
    return NCVibrationSSM(
        n_bands=40, n_classes=n_classes,
        d_model=20, d_state=6, d_conv=3, expand=1.5,
        n_layers=2, sr=25600, n_fft=512, hop_length=64)


def create_nc_vib_tcn_matched(n_classes=4):
    """NC-Vibration-TCN-Matched: Mid-tier TCN for bearing fault detection."""
    return NCVibrationTCN(
        n_bands=40, n_classes=n_classes,
        d_model=20, d_conv=3, expand=1.5,
        n_layers=2, dilations=[1, 2],
        sr=25600, n_fft=512, hop_length=64)


def create_nc_vib_tcn_tiny(n_classes=4):
    """NC-Vibration-TCN-Tiny: Ultra-small TCN for per-motor MCU monitoring.

    Target: $0.5 Cortex-M0+ MCU attached to each motor.
    """
    return NCVibrationTCN(
        n_bands=40, n_classes=n_classes,
        d_model=16, d_conv=3, expand=1.0,
        n_layers=2, dilations=[1, 2],
        sr=25600, n_fft=512, hop_length=64)


# ============================================================================
# Verification
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  NC-Vibration: Noise-Conditioned Models for Fault Detection")
    print("=" * 70)

    # Simulated vibration signal: 25.6kHz, 1 second
    vib = torch.randn(2, 25600)

    configs = {
        'NC-Vib-TCN-Tiny': create_nc_vib_tcn_tiny,
        'NC-Vib-TCN-Matched': create_nc_vib_tcn_matched,
        'NC-Vib-SSM-Matched': create_nc_vib_ssm_matched,
        'NC-Vib-TCN-20K': create_nc_vib_tcn_20k,
        'NC-Vib-SSM-20K': create_nc_vib_ssm_20k,
    }

    print(f"\n  {'Model':<24} | {'Params':>8} | {'FP32 KB':>8} | {'INT8 KB':>8} | Output")
    print("  " + "-" * 75)

    for name, create_fn in configs.items():
        model = create_fn()
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        fp32_kb = params * 4 / 1024
        int8_kb = params / 1024

        with torch.no_grad():
            out = model(vib)

        print(f"  {name:<24} | {params:>8,} | {fp32_kb:>7.1f} | {int8_kb:>7.1f} | "
              f"{list(out.shape)}")

    print("\n  NC-Vibration Novelty:")
    print("  - VNR: adaptive minimum tracking (no silence assumption)")
    print("  - Kurtosis Gate: fault impulse enhancement per band")
    print("  - DualPCEN: RPM-adaptive normalization")
    print("  - Same NC framework, vibration-specific frontend")
