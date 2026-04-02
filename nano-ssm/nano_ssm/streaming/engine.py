# nano_ssm/streaming/engine.py
# Real-time streaming inference engine for NC-SSM
# Copyright (c) 2026 Jin Ho Choi. All rights reserved.

"""
StreamingEngine v8 — State Machine

States:
  IDLE      → Waiting for speech. Energy monitoring only.
              Sends prediction every 500ms for UI probability bars.
  ONSET     → Speech energy detected. Accumulate audio + classify every 200ms.
              If confidence > threshold → DETECTED.
              If timeout (2.5s) or energy drops → IDLE.
  DETECTED  → Keyword confirmed. Send detection event. Transition to COOLDOWN.
  COOLDOWN  → Block all classification for cooldown_ms.
              Buffer cleared. Energy reset. Then → IDLE.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
from collections import deque

from ..models.core import NCSSM, GSC_LABELS_12


class StreamingEngine:
    """State-machine streaming engine for NC-SSM keyword spotting."""

    SHORT_WORDS = {'go', 'no', 'on', 'off', 'up'}

    # States
    IDLE = 'idle'
    ONSET = 'onset'
    DETECTED = 'detected'
    COOLDOWN = 'cooldown'

    def __init__(self, model: NCSSM,
                 chunk_ms: int = 100,
                 sr: int = 16000,
                 confidence_threshold: float = 0.5,
                 classify_interval_ms: int = 200,
                 cooldown_chunks: int = 15,
                 smoothing_window: int = 2,
                 energy_onset: float = -42.0,
                 energy_offset: float = -55.0):
        self.model = model
        self.sr = sr
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(sr * chunk_ms / 1000)
        self.confidence_threshold = confidence_threshold
        self.cooldown_chunks = cooldown_chunks
        self.labels = model.labels

        # Energy thresholds
        self._energy_onset_thresh = energy_onset    # dB to trigger onset
        self._energy_offset_thresh = energy_offset  # dB to end speech

        # How often to classify (in chunks)
        self.classify_every = max(1, classify_interval_ms // chunk_ms)

        # Idle classify interval (in chunks) — for UI prob bars
        self._idle_classify_every = max(1, 500 // chunk_ms)

        # Full rolling buffer
        self._buf = torch.zeros(0)
        self._max_buf = sr * 3

        # State machine
        self._state = self.IDLE
        self._energy_smooth = -60.0
        self._onset_idx = 0
        self._chunks_since_onset = 0
        self._cooldown_remaining = 0
        self._chunk_count = 0

        # Smoothing
        self._history = deque(maxlen=smoothing_window)

    def reset(self):
        self._buf = torch.zeros(0)
        self._state = self.IDLE
        self._energy_smooth = -60.0
        self._onset_idx = 0
        self._chunks_since_onset = 0
        self._cooldown_remaining = 0
        self._chunk_count = 0
        self._history.clear()

    def _enter_idle(self):
        """Transition → IDLE."""
        self._state = self.IDLE
        self._onset_idx = 0
        self._chunks_since_onset = 0
        self._history.clear()

    def _enter_onset(self):
        """Transition → ONSET."""
        self._state = self.ONSET
        pre_onset = int(self.sr * 0.1)  # 100ms before
        self._onset_idx = max(0, len(self._buf) - self.chunk_samples - pre_onset)
        self._chunks_since_onset = 0
        self._history.clear()

    def _enter_cooldown(self):
        """Transition → COOLDOWN. Full reset."""
        self._state = self.COOLDOWN
        self._cooldown_remaining = self.cooldown_chunks
        self._buf = torch.zeros(0)
        self._onset_idx = 0
        self._chunks_since_onset = 0
        self._energy_smooth = -60.0
        self._history.clear()

    @torch.no_grad()
    def _classify(self, audio_1d: torch.Tensor):
        """Classify audio. Pad to 1s if shorter."""
        if len(audio_1d) < self.sr:
            audio_1d = F.pad(audio_1d, (0, self.sr - len(audio_1d)))
        if len(audio_1d) > self.sr:
            audio_1d = audio_1d[-self.sr:]
        x = audio_1d.unsqueeze(0)
        logits = self.model.model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        probs_np = probs.cpu().numpy()
        idx = int(np.argmax(probs_np))
        return probs_np, self.labels[idx], float(probs_np[idx])

    @torch.no_grad()
    def feed(self, chunk: torch.Tensor) -> Optional[Dict]:
        if chunk.dim() == 2:
            chunk = chunk.squeeze(0)

        # Buffer management
        self._buf = torch.cat([self._buf, chunk])
        if len(self._buf) > self._max_buf:
            trim = len(self._buf) - self._max_buf
            self._buf = self._buf[trim:]
            self._onset_idx = max(0, self._onset_idx - trim)

        self._chunk_count += 1

        # Energy
        audio_np = chunk.numpy()
        rms = np.sqrt(np.mean(audio_np ** 2))
        energy_db = 20 * np.log10(max(rms, 1e-10))
        self._energy_smooth = 0.6 * self._energy_smooth + 0.4 * energy_db

        # Default result
        result = {
            'label': 'silence', 'confidence': 0.0,
            'smoothed_label': 'silence', 'smoothed_confidence': 0.0,
            'raw_probs': np.zeros(len(self.labels)),
            'energy_db': energy_db, 'detected': False,
        }

        # ════════════════════════════════════════
        # STATE: COOLDOWN — block everything
        # ════════════════════════════════════════
        if self._state == self.COOLDOWN:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self._enter_idle()
            return result

        # Not enough audio yet
        if len(self._buf) < self.sr * 0.2:
            return None

        # ════════════════════════════════════════
        # STATE: IDLE — wait for speech energy
        # ════════════════════════════════════════
        if self._state == self.IDLE:
            if self._energy_smooth > self._energy_onset_thresh:
                self._enter_onset()
                # Fall through to ONSET processing below
            else:
                # Occasional classify for UI probability bars
                if (self._chunk_count % self._idle_classify_every == 0 and
                        len(self._buf) >= self.sr):
                    probs_np, label, conf = self._classify(self._buf[-self.sr:])
                    result['label'] = label
                    result['confidence'] = conf
                    result['raw_probs'] = probs_np
                    result['smoothed_label'] = label
                    result['smoothed_confidence'] = conf
                return result

        # ════════════════════════════════════════
        # STATE: ONSET — accumulate & classify
        # ════════════════════════════════════════
        if self._state == self.ONSET:
            self._chunks_since_onset += 1

            should_classify = (self._chunks_since_onset % self.classify_every == 0)

            if should_classify:
                speech_audio = self._buf[self._onset_idx:]
                probs_np, label, conf = self._classify(speech_audio)

                result['label'] = label
                result['confidence'] = conf
                result['raw_probs'] = probs_np

                # Smoothing
                self._history.append(torch.from_numpy(probs_np))
                if len(self._history) >= 2:
                    avg = torch.stack(list(self._history)).mean(dim=0)
                    idx = avg.argmax().item()
                    result['smoothed_label'] = self.labels[idx]
                    result['smoothed_confidence'] = float(avg[idx])
                else:
                    result['smoothed_label'] = label
                    result['smoothed_confidence'] = conf

                s_label = result['smoothed_label']
                s_conf = result['smoothed_confidence']

                # Short words: lower threshold
                is_short = s_label in self.SHORT_WORDS
                thresh = self.confidence_threshold * 0.8 if is_short else self.confidence_threshold

                # ── Detection check ──
                if (s_conf >= thresh and s_label not in ('silence', 'unknown')):
                    result['detected'] = True
                    result['smoothed_label'] = s_label
                    result['smoothed_confidence'] = s_conf
                    self._enter_cooldown()
                    return result

            else:
                # Between classifications: return last known
                if len(self._history) > 0:
                    avg = torch.stack(list(self._history)).mean(dim=0)
                    idx = avg.argmax().item()
                    result['label'] = self.labels[idx]
                    result['confidence'] = float(avg[idx])
                    result['smoothed_label'] = result['label']
                    result['smoothed_confidence'] = result['confidence']
                    result['raw_probs'] = avg.numpy()

            # ── Timeout / Speech end ──
            time_since = self._chunks_since_onset * self.chunk_ms / 1000.0

            # Speech ended (energy dropped)
            if time_since > 0.8 and self._energy_smooth < self._energy_offset_thresh:
                self._enter_idle()

            # Hard timeout
            if time_since > 2.5:
                self._enter_idle()

        return result

    @property
    def buffer_duration_ms(self) -> float:
        return len(self._buf) / self.sr * 1000

    def __repr__(self):
        return (f"StreamingEngine(state={self._state}, "
                f"interval={self.classify_every * self.chunk_ms}ms, "
                f"threshold={self.confidence_threshold})")
