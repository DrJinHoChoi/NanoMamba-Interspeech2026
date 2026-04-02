#!/usr/bin/env python3
"""
MCU Inference Demo - NC-SSM vs DS-CNN-S
========================================
Shows WHY NC-SSM wins on edge hardware (MCU/C)
even though CNN wins in Python.

Three environments compared side-by-side:
  1. Python (PyTorch CPU) - CNN wins due to interpreter overhead
  2. C SDK (x86 -O3)     - NC-SSM catches up
  3. MCU (Cortex-M7 est) - NC-SSM wins (pure MAC advantage)

Live mic input -> both models run -> results displayed with
per-environment latency and a dramatic "race" visualization.

    python mcu_demo_server.py

Opens http://localhost:8091
"""

import asyncio
import json
import os
import sys
import time
import threading
import queue
import subprocess
import struct
import tempfile

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

PARENT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(PARENT, 'nano-ssm'))
sys.path.insert(0, PARENT)

import nano_ssm
from nano_ssm.streaming import StreamingEngine

# ---- Config ----
SR = 16000
CHUNK_MS = 100
CHUNK_SAMPLES = int(SR * CHUNK_MS / 1000)
PORT = 8091

CSDK_DIR = os.path.join(PARENT, 'nano-ssm', 'csdk')
C_EXE = os.path.join(CSDK_DIR, 'ncssm_20k.exe')

GSC_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right',
              'on', 'off', 'stop', 'go', 'silence', 'unknown']

# MAC counts (pre-calculated)
NCSSM_MACS = 5_056_244
DSCNN_MACS = 10_029_256
CORTEX_M7_MHZ = 480
MAC_PER_CYCLE = 0.8  # effective with CMSIS-DSP


def estimate_mcu_ms(macs):
    cycles = macs / MAC_PER_CYCLE
    return cycles / (CORTEX_M7_MHZ * 1e6) * 1000


NCSSM_MCU_MS = estimate_mcu_ms(NCSSM_MACS)  # ~13.2ms
DSCNN_MCU_MS = estimate_mcu_ms(DSCNN_MACS)   # ~26.1ms


# ---- CNN SimpleEngine ----
class SimpleEngine:
    def __init__(self, wrapper, sr=16000, threshold=0.35, cooldown_chunks=3):
        self.wrapper = wrapper
        self.sr = sr
        self.threshold = threshold
        self.labels = GSC_LABELS
        self.buffer = torch.zeros(0)
        self.chunks_since = cooldown_chunks
        self.cooldown_chunks = cooldown_chunks
        n_fft = 512
        n_mels = 40
        self.n_fft = n_fft
        self.hop_length = 160
        self.window = torch.hann_window(n_fft)
        n_freq = n_fft // 2 + 1
        low_mel = 2595 * np.log10(1 + 20 / 700)
        high_mel = 2595 * np.log10(1 + sr / 2 / 700)
        mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        mel_fb = np.zeros((n_mels, n_freq))
        for i in range(n_mels):
            for j in range(bins[i], bins[i+1]):
                mel_fb[i, j] = (j - bins[i]) / max(bins[i+1] - bins[i], 1)
            for j in range(bins[i+1], bins[i+2]):
                mel_fb[i, j] = (bins[i+2] - j) / max(bins[i+2] - bins[i+1], 1)
        self.mel_fb = torch.from_numpy(mel_fb).float()

    @torch.no_grad()
    def classify_1s(self, audio_1s):
        """Classify exactly 1s of audio. Returns (label, conf, latency_ms)."""
        x = audio_1s.unsqueeze(0) if audio_1s.dim() == 1 else audio_1s
        t0 = time.perf_counter()
        spec = torch.stft(x, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True)
        mag = spec.abs()
        mel = torch.matmul(self.mel_fb, mag)
        log_mel = torch.log(mel + 1e-8)
        logits = self.wrapper.model(log_mel)
        probs = torch.softmax(logits, -1)[0].cpu().numpy()
        lat = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx]), lat, probs

    @torch.no_grad()
    def feed(self, chunk):
        if chunk.dim() == 2:
            chunk = chunk.squeeze(0)
        self.buffer = torch.cat([self.buffer, chunk])
        if len(self.buffer) > self.sr * 3:
            self.buffer = self.buffer[-self.sr * 3:]
        self.chunks_since += 1
        if len(self.buffer) < self.sr:
            return None
        audio = self.buffer[-self.sr:]
        label, conf, lat, probs = self.classify_1s(audio)
        detected = (label not in ('silence', 'unknown') and
                    conf >= self.threshold and
                    self.chunks_since >= self.cooldown_chunks)
        if detected:
            self.chunks_since = 0
        return {
            'label': label, 'confidence': conf,
            'latency_ms': lat, 'detected': detected,
            'raw_probs': probs,
        }

    def reset(self):
        self.buffer = torch.zeros(0)
        self.chunks_since = self.cooldown_chunks


class CNNWrapper:
    def __init__(self, model, name, n_params):
        self.model = model
        self.name = name
        self.n_params = n_params
        self.labels = GSC_LABELS


# ---- C SDK runner ----
def run_c_inference(audio_1s_np):
    """Run C executable on 1s audio, return (label, conf, latency_ms)."""
    if not os.path.exists(C_EXE):
        return None, 0.0, 0.0

    # Write audio to temp file
    tmp = os.path.join(CSDK_DIR, '_tmp_audio.raw')
    audio_1s_np.astype(np.float32).tofile(tmp)

    try:
        # CREATE_NO_WINDOW prevents cmd flash on Windows
        startupinfo = None
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
        result = subprocess.run(
            [C_EXE, tmp],
            capture_output=True, text=True, timeout=5,
            encoding='utf-8', errors='replace',
            startupinfo=startupinfo,
        )
        label = 'unknown'
        conf = 0.0
        lat = 0.0
        for line in result.stdout.split('\n'):
            if 'Prediction:' in line:
                parts = line.split(':')[1].strip()
                label = parts.split('(')[0].strip()
                conf = float(parts.split('(')[1].replace('%)', '').strip()) / 100
            if 'Latency:' in line:
                lat = float(line.split(':')[1].strip().replace('ms', '').strip())
        return label, conf, lat
    except Exception as e:
        print(f'  C exec error: {e}', flush=True)
        return None, 0.0, 0.0


# ---- Global state ----
models = {}
clients = set()
audio_queue = queue.Queue()
result_queue = queue.Queue()

state = {
    'is_listening': False,
    'stream': None,
    'thread': None,
    'ncssm_engine': None,
    'cnn_engine': None,
    'audio_buf': np.zeros(0, dtype=np.float32),  # rolling 1s buffer for C
    'threshold': 0.35,
    'round_number': 0,
    'scores': {'ncssm': 0, 'cnn': 0},
    'history': [],
    # Accumulated latency stats
    'latency_accum': {
        'ncssm_py': [], 'cnn_py': [],
        'ncssm_c': [], 'ncssm_mcu': [], 'cnn_mcu': [],
    },
}


# ---- Load models ----
def load_models():
    print("\n  Loading models for MCU Demo...")

    ckpt_dir = os.path.join(PARENT, 'checkpoints_full')

    # NC-SSM 20K
    ckpt_path = os.path.join(ckpt_dir, 'NanoMamba-NC-20K', 'best.pt')
    if os.path.exists(ckpt_path):
        try:
            m = nano_ssm.create('ncssm-20k', pretrained=ckpt_path)
            models['ncssm'] = ('ncssm', m)
            print(f"    NC-SSM-20K: {m.n_params:,} params OK")
        except Exception as e:
            print(f"    NC-SSM-20K: FAIL ({e})")

    if 'ncssm' not in models:
        ckpt_path = os.path.join(ckpt_dir, 'NanoMamba-NC-Matched', 'best.pt')
        if os.path.exists(ckpt_path):
            try:
                m = nano_ssm.create('ncssm', pretrained=ckpt_path)
                models['ncssm'] = ('ncssm', m)
                print(f"    NC-SSM: {m.n_params:,} params OK")
            except Exception as e:
                print(f"    NC-SSM: FAIL ({e})")

    # DS-CNN-S
    try:
        from train_colab import DSCNN_S
        dscnn_ckpt = os.path.join(ckpt_dir, 'DS-CNN-S', 'best.pt')
        if os.path.exists(dscnn_ckpt):
            m = DSCNN_S(n_classes=12)
            ckpt = torch.load(dscnn_ckpt, map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            n_params = sum(p.numel() for p in m.parameters())
            models['cnn'] = ('cnn', CNNWrapper(m, 'DS-CNN-S', n_params))
            print(f"    DS-CNN-S: {n_params:,} params OK")
    except Exception as e:
        print(f"    DS-CNN-S: FAIL ({e})")

    # Check C executable
    if os.path.exists(C_EXE):
        print(f"    C SDK: {C_EXE} OK")
    else:
        print(f"    C SDK: NOT FOUND ({C_EXE})")


# ---- Audio ----
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())


def broadcast_sync(msg):
    result_queue.put(msg)


# ---- Processing thread ----
def process_loop():
    ncssm_engine = state['ncssm_engine']
    cnn_engine = state['cnn_engine']
    classify_counter = 0

    while state['is_listening']:
        try:
            audio_np = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            chunk = torch.from_numpy(audio_np).float()

            # Accumulate 1s buffer for C inference
            state['audio_buf'] = np.concatenate([state['audio_buf'], audio_np])
            if len(state['audio_buf']) > SR * 2:
                state['audio_buf'] = state['audio_buf'][-SR * 2:]

            # Energy
            rms = np.sqrt(np.mean(audio_np ** 2))
            energy_db = 20 * np.log10(max(rms, 1e-10))

            # Python NC-SSM
            t0 = time.perf_counter()
            result_ncssm = ncssm_engine.feed(chunk) if ncssm_engine else None
            lat_ncssm_py = (time.perf_counter() - t0) * 1000

            # Python CNN
            t0 = time.perf_counter()
            result_cnn = cnn_engine.feed(chunk) if cnn_engine else None
            lat_cnn_py = (time.perf_counter() - t0) * 1000

            classify_counter += 1

            # Every 10 chunks (~1s), also run C inference for comparison
            c_label, c_conf, c_lat = None, 0.0, 0.0
            run_c = (classify_counter % 10 == 0 and len(state['audio_buf']) >= SR)
            if run_c:
                audio_1s = state['audio_buf'][-SR:]
                c_label, c_conf, c_lat = run_c_inference(audio_1s)

            # Build message
            msg = {
                'type': 'tick',
                'energy_db': round(float(energy_db), 1),
            }

            # NC-SSM Python results
            if result_ncssm:
                ncssm_label = result_ncssm.get('smoothed_label',
                                                result_ncssm.get('label', 'silence'))
                ncssm_conf = float(result_ncssm.get('smoothed_confidence',
                                                     result_ncssm.get('confidence', 0.0)))
                probs = result_ncssm.get('raw_probs', None)
                probs_list = [float(p) for p in probs] if probs is not None else None
                prob_sum = sum(probs_list) if probs_list else 0
                msg['ncssm_py'] = {
                    'label': str(ncssm_label),
                    'confidence': round(ncssm_conf, 4),
                    'latency_ms': round(lat_ncssm_py, 2),
                    'probs': probs_list if prob_sum > 0.01 else None,
                    'detected': result_ncssm.get('detected', False),
                }
            else:
                msg['ncssm_py'] = {'label': 'buffering', 'confidence': 0.0,
                                   'latency_ms': 0, 'detected': False}

            # CNN Python results
            if result_cnn:
                msg['cnn_py'] = {
                    'label': str(result_cnn['label']),
                    'confidence': round(float(result_cnn['confidence']), 4),
                    'latency_ms': round(float(result_cnn.get('latency_ms', lat_cnn_py)), 2),
                    'detected': result_cnn.get('detected', False),
                }
            else:
                msg['cnn_py'] = {'label': 'buffering', 'confidence': 0.0,
                                 'latency_ms': 0, 'detected': False}

            # C SDK results (when available)
            if run_c and c_label:
                msg['ncssm_c'] = {
                    'label': c_label,
                    'confidence': round(c_conf, 4),
                    'latency_ms': round(c_lat, 2),
                }

            # --- Detection event: when keyword detected, run full 1s comparison ---
            ncssm_detected = (result_ncssm and result_ncssm.get('detected', False) and
                              msg['ncssm_py']['label'] not in ('silence', 'unknown', 'buffering'))
            cnn_detected = (result_cnn and result_cnn.get('detected', False) and
                            msg['cnn_py']['label'] not in ('silence', 'unknown', 'buffering'))

            if (ncssm_detected or cnn_detected) and len(state['audio_buf']) >= SR:
                try:
                    audio_1s = state['audio_buf'][-SR:]
                    audio_1s_t = torch.from_numpy(audio_1s.copy()).float()

                    det_word = msg['ncssm_py']['label'] if ncssm_detected else msg['cnn_py']['label']

                    # Full forward: Python NC-SSM (1s audio)
                    ncssm_wrapper = models['ncssm'][1]
                    with torch.no_grad():
                        t0 = time.perf_counter()
                        logits = ncssm_wrapper.model(audio_1s_t.unsqueeze(0))
                        full_ncssm_ms = (time.perf_counter() - t0) * 1000
                        probs_n = torch.softmax(logits, -1)[0]
                        ncssm_full_label = GSC_LABELS[probs_n.argmax().item()]
                        ncssm_full_conf = probs_n.max().item()

                    # Full forward: Python CNN (1s audio)
                    cnn_eng = state['cnn_engine']
                    with torch.no_grad():
                        cnn_full_label, cnn_full_conf, full_cnn_ms, _ = cnn_eng.classify_1s(audio_1s_t)

                    # 1-core Python tests
                    orig_threads = torch.get_num_threads()
                    torch.set_num_threads(1)
                    with torch.no_grad():
                        t0 = time.perf_counter()
                        logits_1c = ncssm_wrapper.model(audio_1s_t.unsqueeze(0))
                        ncssm_1core_ms = (time.perf_counter() - t0) * 1000
                    with torch.no_grad():
                        _, _, cnn_1core_ms, _ = cnn_eng.classify_1s(audio_1s_t)
                    torch.set_num_threads(orig_threads)

                    # C SDK
                    c_label2, c_conf2, c_lat2 = run_c_inference(audio_1s)

                    detection_msg = {
                        'type': 'detection',
                        'word': det_word.upper(),
                        'ncssm_py': {
                            'label': ncssm_full_label, 'conf': round(ncssm_full_conf, 3),
                            'ms': round(full_ncssm_ms, 1),
                        },
                        'cnn_py': {
                            'label': cnn_full_label, 'conf': round(cnn_full_conf, 3),
                            'ms': round(full_cnn_ms, 1),
                        },
                        'ncssm_py_1core': {
                            'ms': round(ncssm_1core_ms, 1),
                        },
                        'cnn_py_1core': {
                            'ms': round(cnn_1core_ms, 1),
                        },
                        'ncssm_c': {
                            'label': c_label2 or '?', 'conf': round(c_conf2, 3),
                            'ms': round(c_lat2, 1),
                        } if c_label2 else None,
                        'mcu': {
                            'ncssm_ms': round(NCSSM_MCU_MS, 1),
                            'cnn_ms': round(DSCNN_MCU_MS, 1),
                        },
                    }
                    broadcast_sync(detection_msg)
                    print(f'  [DET] {det_word} | ncssm_py={full_ncssm_ms:.0f}ms cnn_py={full_cnn_ms:.0f}ms 1core_n={ncssm_1core_ms:.0f}ms 1core_c={cnn_1core_ms:.0f}ms c={c_lat2:.0f}ms', flush=True)
                except Exception as e:
                    print(f'  [DET ERR] {e}', flush=True)

            broadcast_sync(msg)
        except Exception as e:
            print(f'  [LOOP ERR] {e}', flush=True)


# ---- Create engines ----
def create_engines():
    if 'ncssm' in models:
        _, wrapper = models['ncssm']
        state['ncssm_engine'] = StreamingEngine(
            wrapper, chunk_ms=CHUNK_MS, sr=SR,
            confidence_threshold=state['threshold'],
            cooldown_chunks=6,
        )
        state['ncssm_engine']._energy_speech = -50.0
        state['ncssm_engine']._energy_floor = -60.0
    if 'cnn' in models:
        _, wrapper = models['cnn']
        state['cnn_engine'] = SimpleEngine(
            wrapper, sr=SR, threshold=state['threshold'],
            cooldown_chunks=3,
        )


# ---- Start/Stop ----
def start_listening():
  try:
    if state['is_listening']:
        return
    if 'ncssm' not in models or 'cnn' not in models:
        broadcast_sync({'type': 'error', 'message': 'Models not loaded'})
        return

    create_engines()
    state['audio_buf'] = np.zeros(0, dtype=np.float32)

    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break

    import scipy.signal as sig

    def try_open_mic(device=None):
        dev_info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device[0])
        native_sr = int(dev_info['default_samplerate'])
        dev_name = dev_info['name']
        if native_sr == SR:
            stream = sd.InputStream(
                samplerate=SR, channels=1, dtype='float32',
                blocksize=CHUNK_SAMPLES, callback=audio_callback, device=device)
            stream.start()
            print(f'  Mic: [{device}] {dev_name} @ {SR}Hz', flush=True)
            return stream
        native_block = int(native_sr * CHUNK_MS / 1000)
        def resample_cb(indata, frames, ti, status):
            audio = indata[:, 0].astype(np.float32)
            target_len = int(len(audio) * SR / native_sr)
            resampled = sig.resample(audio, target_len).astype(np.float32)
            audio_queue.put(resampled)
        stream = sd.InputStream(
            samplerate=native_sr, channels=1, dtype='float32',
            blocksize=native_block, callback=resample_cb, device=device)
        stream.start()
        print(f'  Mic: [{device}] {dev_name} @ {native_sr}Hz -> {SR}Hz', flush=True)
        return stream

    device_priority = [34, None, 31, 20, 21]
    opened = False
    for dev_id in device_priority:
        try:
            state['stream'] = try_open_mic(dev_id)
            opened = True
            break
        except Exception as e:
            print(f'  Mic [{dev_id}] failed: {str(e)[:60]}', flush=True)

    if not opened:
        broadcast_sync({'type': 'error', 'message': 'No audio device'})
        return

    state['is_listening'] = True
    state['thread'] = threading.Thread(target=process_loop, daemon=True)
    state['thread'].start()

    ncssm_w = models['ncssm'][1]
    cnn_w = models['cnn'][1]
    broadcast_sync({
        'type': 'started',
        'ncssm_params': ncssm_w.n_params,
        'cnn_params': cnn_w.n_params,
        'ncssm_macs': NCSSM_MACS,
        'cnn_macs': DSCNN_MACS,
        'ncssm_mcu_ms': round(NCSSM_MCU_MS, 1),
        'cnn_mcu_ms': round(DSCNN_MCU_MS, 1),
        'c_available': os.path.exists(C_EXE),
    })
    print(f'  Audio started', flush=True)
  except Exception as e:
    print(f'  [START ERR] {e}', flush=True)
    import traceback
    traceback.print_exc()
    broadcast_sync({'type': 'error', 'message': str(e)[:100]})


def stop_listening():
    state['is_listening'] = False
    if state['stream']:
        try:
            state['stream'].stop()
            state['stream'].close()
        except: pass
        state['stream'] = None
    if state['ncssm_engine']:
        state['ncssm_engine'].reset()
    if state['cnn_engine']:
        state['cnn_engine'].reset()

    # Compute summary stats
    stats = {}
    for key, vals in state['latency_accum'].items():
        if vals:
            stats[key] = {
                'avg': round(sum(vals) / len(vals), 1),
                'median': round(sorted(vals)[len(vals)//2], 1),
                'n': len(vals),
            }
    broadcast_sync({'type': 'stopped', 'stats': stats})


def run_benchmark():
    """Run quick benchmark with test audio files."""
    test_dir = os.path.join(CSDK_DIR, 'test')
    results = []
    test_files = sorted([f for f in os.listdir(test_dir) if f.startswith('audio_') and f.endswith('.raw')])

    if not test_files:
        return results

    ncssm_model = models.get('ncssm', (None, None))[1]
    cnn_wrapper = models.get('cnn', (None, None))[1]

    for fname in test_files:
        word = fname.replace('audio_', '').replace('.raw', '')
        fpath = os.path.join(test_dir, fname)
        audio_np = np.fromfile(fpath, dtype=np.float32)
        audio_t = torch.from_numpy(audio_np.copy())

        row = {'word': word}

        # Python NC-SSM
        if ncssm_model:
            with torch.no_grad():
                t0 = time.perf_counter()
                logits = ncssm_model.model(audio_t.unsqueeze(0))
                lat = (time.perf_counter() - t0) * 1000
                probs = torch.softmax(logits, -1)[0]
                idx = probs.argmax().item()
                row['ncssm_py'] = {
                    'label': GSC_LABELS[idx],
                    'conf': round(probs[idx].item(), 3),
                    'ms': round(lat, 1),
                    'correct': GSC_LABELS[idx] == word,
                }

        # Python CNN
        if cnn_wrapper:
            with torch.no_grad():
                n_fft = 512
                hop = 160
                window = torch.hann_window(n_fft)
                spec = torch.stft(audio_t.unsqueeze(0), n_fft, hop,
                                  window=window, return_complex=True)
                mag = spec.abs()
                low_mel = 2595 * np.log10(1 + 20 / 700)
                high_mel = 2595 * np.log10(1 + SR / 2 / 700)
                mel_pts = np.linspace(low_mel, high_mel, 42)
                hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
                bins = np.floor((n_fft + 1) * hz_pts / SR).astype(int)
                mel_fb = np.zeros((40, 257))
                for i in range(40):
                    for j in range(bins[i], bins[i+1]):
                        mel_fb[i, j] = (j - bins[i]) / max(bins[i+1] - bins[i], 1)
                    for j in range(bins[i+1], bins[i+2]):
                        mel_fb[i, j] = (bins[i+2] - j) / max(bins[i+2] - bins[i+1], 1)
                mel_fb_t = torch.from_numpy(mel_fb).float()
                mel = torch.matmul(mel_fb_t, mag)
                log_mel = torch.log(mel + 1e-8)

                t0 = time.perf_counter()
                logits = cnn_wrapper.model(log_mel)
                lat = (time.perf_counter() - t0) * 1000
                probs = torch.softmax(logits, -1)[0]
                idx = probs.argmax().item()
                row['cnn_py'] = {
                    'label': GSC_LABELS[idx],
                    'conf': round(probs[idx].item(), 3),
                    'ms': round(lat, 1),
                    'correct': GSC_LABELS[idx] == word,
                }

        # Python 1-core (torch.set_num_threads(1))
        orig_threads = torch.get_num_threads()
        torch.set_num_threads(1)

        if ncssm_model:
            with torch.no_grad():
                t0 = time.perf_counter()
                logits = ncssm_model.model(audio_t.unsqueeze(0))
                lat = (time.perf_counter() - t0) * 1000
                probs = torch.softmax(logits, -1)[0]
                idx = probs.argmax().item()
                row['ncssm_py_1core'] = {
                    'label': GSC_LABELS[idx],
                    'conf': round(probs[idx].item(), 3),
                    'ms': round(lat, 1),
                    'correct': GSC_LABELS[idx] == word,
                }

        if cnn_wrapper:
            with torch.no_grad():
                n_fft = 512
                hop = 160
                window = torch.hann_window(n_fft)
                spec = torch.stft(audio_t.unsqueeze(0), n_fft, hop,
                                  window=window, return_complex=True)
                mag = spec.abs()
                low_mel = 2595 * np.log10(1 + 20 / 700)
                high_mel = 2595 * np.log10(1 + SR / 2 / 700)
                mel_pts = np.linspace(low_mel, high_mel, 42)
                hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
                bins = np.floor((n_fft + 1) * hz_pts / SR).astype(int)
                mel_fb = np.zeros((40, 257))
                for i in range(40):
                    for j in range(bins[i], bins[i+1]):
                        mel_fb[i, j] = (j - bins[i]) / max(bins[i+1] - bins[i], 1)
                    for j in range(bins[i+1], bins[i+2]):
                        mel_fb[i, j] = (bins[i+2] - j) / max(bins[i+2] - bins[i+1], 1)
                mel_fb_t = torch.from_numpy(mel_fb).float()
                mel = torch.matmul(mel_fb_t, mag)
                log_mel = torch.log(mel + 1e-8)

                t0 = time.perf_counter()
                logits = cnn_wrapper.model(log_mel)
                lat = (time.perf_counter() - t0) * 1000
                probs = torch.softmax(logits, -1)[0]
                idx = probs.argmax().item()
                row['cnn_py_1core'] = {
                    'label': GSC_LABELS[idx],
                    'conf': round(probs[idx].item(), 3),
                    'ms': round(lat, 1),
                    'correct': GSC_LABELS[idx] == word,
                }

        torch.set_num_threads(orig_threads)

        # C SDK
        if os.path.exists(C_EXE):
            c_label, c_conf, c_lat = run_c_inference(audio_np)
            if c_label:
                row['ncssm_c'] = {
                    'label': c_label,
                    'conf': round(c_conf, 3),
                    'ms': round(c_lat, 1),
                    'correct': c_label == word,
                }

        # MCU estimates
        row['mcu'] = {
            'ncssm_ms': round(NCSSM_MCU_MS, 1),
            'cnn_ms': round(DSCNN_MCU_MS, 1),
        }

        results.append(row)

    return results


# ---- FastAPI ----
app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'mcu_demo.html'))


@app.get("/api/benchmark")
async def api_benchmark():
    results = run_benchmark()
    return {
        'results': results,
        'ncssm_params': models['ncssm'][1].n_params if 'ncssm' in models else 0,
        'cnn_params': models['cnn'][1].n_params if 'cnn' in models else 0,
        'ncssm_macs': NCSSM_MACS,
        'cnn_macs': DSCNN_MACS,
        'ncssm_mcu_ms': round(NCSSM_MCU_MS, 1),
        'cnn_mcu_ms': round(DSCNN_MCU_MS, 1),
        'c_available': os.path.exists(C_EXE),
    }


async def result_pusher():
    while True:
        try:
            msg = result_queue.get_nowait()
            try:
                data = json.dumps(msg)
            except TypeError:
                data = json.dumps(msg, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
            dead = set()
            for ws in clients:
                try:
                    await ws.send_text(data)
                except:
                    dead.add(ws)
            clients.difference_update(dead)
        except queue.Empty:
            pass
        except Exception:
            pass
        await asyncio.sleep(0.02)


_pusher_started = False

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _pusher_started
    await ws.accept()
    clients.add(ws)
    # Send MCU estimates immediately on connect
    try:
        await ws.send_text(json.dumps({
            'type': 'init',
            'ncssm_mcu_ms': round(NCSSM_MCU_MS, 1),
            'cnn_mcu_ms': round(DSCNN_MCU_MS, 1),
            'ncssm_macs': NCSSM_MACS,
            'cnn_macs': DSCNN_MACS,
            'ncssm_params': models['ncssm'][1].n_params if 'ncssm' in models else 0,
            'cnn_params': models['cnn'][1].n_params if 'cnn' in models else 0,
        }))
    except Exception:
        pass
    if not _pusher_started:
        _pusher_started = True
        asyncio.create_task(result_pusher())
    try:
        while True:
            data = await ws.receive_text()
            cmd = json.loads(data)
            action = cmd.get('action', '')
            if action == 'start':
                start_listening()
            elif action == 'stop':
                stop_listening()
            elif action == 'benchmark':
                results = run_benchmark()
                ncssm_w = models.get('ncssm', (None, None))[1]
                cnn_w = models.get('cnn', (None, None))[1]
                await ws.send_text(json.dumps({
                    'type': 'benchmark_result',
                    'results': results,
                    'ncssm_params': ncssm_w.n_params if ncssm_w else 0,
                    'cnn_params': cnn_w.n_params if cnn_w else 0,
                    'ncssm_macs': NCSSM_MACS,
                    'cnn_macs': DSCNN_MACS,
                    'c_available': os.path.exists(C_EXE),
                }))
    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)


@app.on_event("startup")
async def startup():
    load_models()
    print(f"\n  MCU Demo ready: http://localhost:{PORT}", flush=True)
    print(f"  Models: {list(models.keys())}", flush=True)
    print(f"  C SDK:  {'OK' if os.path.exists(C_EXE) else 'NOT FOUND'}\n", flush=True)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')
