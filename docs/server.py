#!/usr/bin/env python3
"""
NC-SSM CES 2027 Live Demo Server (v3)
======================================
All 4 models. Simple 1s buffer classify. Instant WebSocket push.

    python server.py
"""

import asyncio
import json
import os
import sys
import time
import threading
import queue

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# Add paths
PARENT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(PARENT, 'nano-ssm'))
sys.path.insert(0, PARENT)

import nano_ssm
from nano_ssm.streaming import StreamingEngine
from nanomamba import create_nc_tcn_20k, NanoTCN
from train_colab import spectral_subtraction_v2

# ── Config ──
SR = 16000
CHUNK_MS = 100
CHUNK_SAMPLES = int(SR * CHUNK_MS / 1000)
PORT = 8088
CKPT_DIR = os.path.join(PARENT, 'checkpoints_full')

GSC_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right',
              'on', 'off', 'stop', 'go', 'silence', 'unknown']

# ── Global state ──
models = {}       # key -> (model, model_type)  model_type: 'ncssm' or 'cnn'
clients = set()
audio_queue = queue.Queue()
state = {
    'is_listening': False,
    'current_model': 'ncssm-20k',
    'threshold': 0.45,
    'engine': None,
    'stream': None,
    'thread': None,
}


# ── CNN wrapper (DS-CNN-S, BC-ResNet-1) ──
class CNNWrapper:
    """Wrap CNN models to have same interface as NCSSM."""
    def __init__(self, model, name, n_params):
        self.model = model
        self.name = name
        self.n_params = n_params
        self.labels = GSC_LABELS
        self.mel = torch.nn.Identity()  # placeholder

    @torch.no_grad()
    def predict(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        logits = self.model(audio)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = probs.argmax().item()
        return {'label': self.labels[idx], 'confidence': float(probs[idx]),
                'probs': probs.cpu().numpy()}


# ── Simple CNN streaming engine ──
class SimpleEngine:
    """1-second buffer → mel → classify. For CNN models."""
    def __init__(self, wrapper, sr=16000, threshold=0.45, cooldown_chunks=8):
        self.wrapper = wrapper
        self.sr = sr
        self.threshold = threshold
        self.confidence_threshold = threshold
        self.cooldown_chunks = cooldown_chunks
        self.labels = wrapper.labels
        self.buffer = torch.zeros(0)
        self.chunks_since = cooldown_chunks

        # Build mel filterbank exactly like training code
        n_fft = 512
        n_mels = 40
        self.n_fft = n_fft
        self.hop_length = 160
        self.window = torch.hann_window(n_fft)

        # Create mel filterbank (same as train_colab.py)
        n_freq = n_fft // 2 + 1
        low_hz, high_hz = 20, sr // 2
        low_mel = 2595 * np.log10(1 + low_hz / 700)
        high_mel = 2595 * np.log10(1 + high_hz / 700)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        mel_fb = np.zeros((n_mels, n_freq))
        for i in range(n_mels):
            for j in range(bin_points[i], bin_points[i+1]):
                mel_fb[i, j] = (j - bin_points[i]) / max(bin_points[i+1] - bin_points[i], 1)
            for j in range(bin_points[i+1], bin_points[i+2]):
                mel_fb[i, j] = (bin_points[i+2] - j) / max(bin_points[i+2] - bin_points[i+1], 1)
        self.mel_fb = torch.from_numpy(mel_fb).float()

    @torch.no_grad()
    def feed(self, chunk):
        if chunk.dim() == 2:
            chunk = chunk.squeeze(0)
        self.buffer = torch.cat([self.buffer, chunk])
        if len(self.buffer) > self.sr * 3:
            self.buffer = self.buffer[-self.sr * 3:]
        self.chunks_since += 1

        audio_np = chunk.numpy()
        rms = np.sqrt(np.mean(audio_np ** 2))
        energy_db = 20 * np.log10(max(rms, 1e-10))

        result = {
            'label': 'silence', 'confidence': 0.0,
            'smoothed_label': 'silence', 'smoothed_confidence': 0.0,
            'raw_probs': np.zeros(len(self.labels)),
            'energy_db': energy_db, 'detected': False,
        }

        if len(self.buffer) < self.sr:
            return None

        # Classify latest 1s — exact same mel as training
        audio = self.buffer[-self.sr:].unsqueeze(0)  # (1, 16000)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True)
        mag = spec.abs()  # (1, F, T)
        mel = torch.matmul(self.mel_fb, mag)  # (1, n_mels, T)
        log_mel = torch.log(mel + 1e-8)

        model = self.wrapper.model
        logits = model(log_mel)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        conf = float(probs[idx])

        result['label'] = label
        result['confidence'] = conf
        result['smoothed_label'] = label
        result['smoothed_confidence'] = conf
        result['raw_probs'] = probs

        if (label not in ('silence', 'unknown') and
                conf >= self.confidence_threshold and
                self.chunks_since >= self.cooldown_chunks):
            result['detected'] = True
            self.chunks_since = 0

        return result

    @property
    def buffer_duration_ms(self):
        return len(self.buffer) / self.sr * 1000

    def reset(self):
        self.buffer = torch.zeros(0)
        self.chunks_since = self.cooldown_chunks


# ── NC-TCN+SS Streaming Engine ──
class NanoTCNSSEngine:
    """NC-TCN-20K + External Spectral Subtraction streaming engine.
    1-second buffer → SS → NanoTCN → classify.
    """
    def __init__(self, model, sr=16000, threshold=0.45, cooldown_chunks=6,
                 ss_bypass_threshold_db=-5.0):
        self.model = model
        self.sr = sr
        self.confidence_threshold = threshold
        self.cooldown_chunks = cooldown_chunks
        self.ss_bypass_threshold_db = ss_bypass_threshold_db
        self.labels = GSC_LABELS
        self.buffer = torch.zeros(0)
        self.chunks_since = cooldown_chunks

    @torch.no_grad()
    def feed(self, chunk):
        if chunk.dim() == 2:
            chunk = chunk.squeeze(0)
        self.buffer = torch.cat([self.buffer, chunk])
        if len(self.buffer) > self.sr * 3:
            self.buffer = self.buffer[-self.sr * 3:]
        self.chunks_since += 1

        audio_np = chunk.numpy()
        rms = np.sqrt(np.mean(audio_np ** 2))
        energy_db = 20 * np.log10(max(rms, 1e-10))

        result = {
            'label': 'silence', 'confidence': 0.0,
            'smoothed_label': 'silence', 'smoothed_confidence': 0.0,
            'raw_probs': np.zeros(len(self.labels)),
            'energy_db': energy_db, 'detected': False,
        }

        if len(self.buffer) < self.sr:
            return None

        # Latest 1s audio
        audio = self.buffer[-self.sr:].unsqueeze(0)  # (1, 16000)

        # External SS with hard bypass (threshold = -5dB)
        rms_full = audio.pow(2).mean().sqrt()
        snr_est_db = 20 * np.log10(max(float(rms_full), 1e-10)) + 30  # rough SNR
        if snr_est_db < self.ss_bypass_threshold_db:
            # Apply spectral subtraction for noisy audio
            try:
                audio_ss = spectral_subtraction_v2(audio)
                audio = audio_ss
            except Exception:
                pass

        # Forward through NanoTCN (raw audio → internal STFT/mel/etc.)
        logits = self.model(audio)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        conf = float(probs[idx])

        result['label'] = label
        result['confidence'] = conf
        result['smoothed_label'] = label
        result['smoothed_confidence'] = conf
        result['raw_probs'] = probs

        if (label not in ('silence', 'unknown') and
                conf >= self.confidence_threshold and
                self.chunks_since >= self.cooldown_chunks):
            result['detected'] = True
            self.chunks_since = 0

        return result

    @property
    def buffer_duration_ms(self):
        return len(self.buffer) / self.sr * 1000

    def reset(self):
        self.buffer = torch.zeros(0)
        self.chunks_since = self.cooldown_chunks


# ── Load models ──
def load_models():
    print("\n  Loading models...")

    # NC-SSM models via nano_ssm
    ncssm_map = {
        'ncssm': ('NanoMamba-NC-Matched', 'ncssm'),
        'ncssm-20k': ('NanoMamba-NC-20K', 'ncssm-20k'),
    }
    for key, (ckpt_name, factory_name) in ncssm_map.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name, 'best.pt')
        if not os.path.exists(ckpt_path):
            continue
        try:
            m = nano_ssm.create(factory_name, pretrained=ckpt_path)
            models[key] = ('ncssm', m)
            print(f"    {key}: {m.n_params:,} params OK")
        except Exception as e:
            print(f"    {key}: FAIL ({e})")

    # NC-TCN-20K + external SS
    try:
        tcn_ckpt_path = os.path.join(CKPT_DIR, 'NC-TCN-20K', 'best.pt')
        if os.path.exists(tcn_ckpt_path):
            m = create_nc_tcn_20k(n_classes=12)
            ckpt = torch.load(tcn_ckpt_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in ckpt:
                m.load_state_dict(ckpt['model_state_dict'])
            else:
                m.load_state_dict(ckpt)
            m.eval()
            n_params = sum(p.numel() for p in m.parameters())
            models['nc-tcn-20k-ss'] = ('nctcn-ss', m)
            print(f"    nc-tcn-20k-ss: {n_params:,} params (+ext SS) OK")
        else:
            print(f"    nc-tcn-20k-ss: SKIP (no checkpoint at {tcn_ckpt_path})")
    except Exception as e:
        print(f"    nc-tcn-20k-ss: FAIL ({e})")

    # CNN models via train_colab.py
    try:
        from train_colab import DSCNN_S, BCResNet
        print("  [OK] train_colab.py loaded")

        # DS-CNN-S
        dscnn_ckpt = os.path.join(CKPT_DIR, 'DS-CNN-S', 'best.pt')
        if os.path.exists(dscnn_ckpt):
            m = DSCNN_S(n_classes=12)
            ckpt = torch.load(dscnn_ckpt, map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            n_params = sum(p.numel() for p in m.parameters())
            models['ds-cnn-s'] = ('cnn', CNNWrapper(m, 'DS-CNN-S', n_params))
            print(f"    ds-cnn-s: {n_params:,} params OK")

        # BC-ResNet-1
        bcres_ckpt = os.path.join(CKPT_DIR, 'BC-ResNet-1', 'best.pt')
        if os.path.exists(bcres_ckpt):
            m = BCResNet(n_classes=12, scale=1)
            ckpt = torch.load(bcres_ckpt, map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            n_params = sum(p.numel() for p in m.parameters())
            models['bc-resnet-1'] = ('cnn', CNNWrapper(m, 'BC-ResNet-1', n_params))
            print(f"    bc-resnet-1: {n_params:,} params OK")

    except Exception as e:
        print(f"  [WARN] CNN models: {e}")


# ── Audio callback ──
def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    audio_queue.put(indata[:, 0].copy())


# ── Broadcast ──
_main_loop = None

async def broadcast(msg: dict):
    data = json.dumps(msg)
    dead = set()
    for ws in clients:
        try:
            await ws.send_text(data)
        except:
            dead.add(ws)
    clients.difference_update(dead)


result_queue = queue.Queue()

def broadcast_sync(msg: dict):
    """Put message in queue. Async pusher will send to WS clients."""
    result_queue.put(msg)


# ── Processing thread ──
def process_loop():
    engine = state['engine']
    if engine is None:
        return

    while state['is_listening']:
        try:
            audio_np = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        chunk = torch.from_numpy(audio_np).float()
        t0 = time.perf_counter()
        result = engine.feed(chunk)
        latency = (time.perf_counter() - t0) * 1000

        if result is None:
            broadcast_sync({'type': 'buffering',
                            'progress': engine.buffer_duration_ms / 1000.0})
            continue

        label = result.get('smoothed_label', result.get('label', 'silence'))
        conf = float(result.get('smoothed_confidence', result.get('confidence', 0.0)))
        detected = result.get('detected', False)
        probs = result.get('raw_probs', None)
        probs_list = [float(p) for p in probs] if probs is not None else [0.0] * 12
        energy_db = float(result.get('energy_db', -60))

        msg = {
            'type': 'detection' if detected else 'prediction',
            'label': str(label),
            'confidence': round(conf, 4),
            'probs': probs_list,
            'latency_ms': round(float(latency), 1),
            'energy_db': round(energy_db, 1),
            'timestamp': time.time(),
        }
        broadcast_sync(msg)


# ── Create engine ──
def create_engine(model_key):
    model_type, wrapper = models[model_key]
    if model_type == 'ncssm':
        return StreamingEngine(
            wrapper, chunk_ms=CHUNK_MS, sr=SR,
            confidence_threshold=state['threshold'],
            cooldown_chunks=6,
        )
    elif model_type == 'nctcn-ss':
        return NanoTCNSSEngine(
            wrapper, sr=SR,
            threshold=state['threshold'],
            cooldown_chunks=6,
            ss_bypass_threshold_db=-5.0,
        )
    else:
        return SimpleEngine(
            wrapper, sr=SR,
            threshold=state['threshold'],
            cooldown_chunks=8,
        )


# ── Start / Stop ──
def start_listening():
    if state['is_listening']:
        return
    model_key = state['current_model']
    if model_key not in models:
        return

    state['engine'] = create_engine(model_key)

    # Clear queue
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break

    # Try to open audio — fallback through devices if needed
    import scipy.signal as sig

    def try_mic(device=None):
        dev_info = sd.query_devices(device if device is not None else sd.default.device[0])
        native_sr = int(dev_info['default_samplerate'])
        if native_sr == SR:
            s = sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                               blocksize=CHUNK_SAMPLES, callback=audio_callback, device=device)
            s.start()
            print(f"  Mic [{device}] {dev_info['name']} @ {SR}Hz", flush=True)
            return s
        native_block = int(native_sr * CHUNK_MS / 1000)
        def resample_cb(indata, frames, ti, status):
            audio = indata[:, 0].astype(np.float32)
            resampled = sig.resample(audio, int(len(audio) * SR / native_sr)).astype(np.float32)
            audio_queue.put(resampled)
        s = sd.InputStream(samplerate=native_sr, channels=1, dtype='float32',
                           blocksize=native_block, callback=resample_cb, device=device)
        s.start()
        print(f"  Mic [{device}] {dev_info['name']} @ {native_sr}Hz -> {SR}Hz", flush=True)
        return s

    stream = None
    for dev_id in [None, 1, 9, 21, 22, 23]:
        try:
            stream = try_mic(dev_id)
            break
        except Exception:
            continue
    if stream is None:
        print("  [ERR] No working mic found", flush=True)
        return
    state['stream'] = stream
    state['is_listening'] = True

    state['thread'] = threading.Thread(target=process_loop, daemon=True)
    state['thread'].start()

    broadcast_sync({'type': 'started', 'model': model_key})


def stop_listening():
    state['is_listening'] = False
    if state['stream']:
        try:
            state['stream'].stop()
            state['stream'].close()
        except: pass
        state['stream'] = None
    if state['engine']:
        state['engine'].reset()
    broadcast_sync({'type': 'stopped'})


# ── FastAPI ──
app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'live.html'))


@app.get("/api/models")
async def get_models():
    info = {}
    for k, (mtype, wrapper) in models.items():
        if mtype == 'ncssm':
            info[k] = {'params': wrapper.n_params, 'type': 'ssm'}
        else:
            info[k] = {'params': wrapper.n_params, 'type': 'cnn'}
    return {'models': info, 'current': state['current_model']}


async def result_pusher():
    """Async loop: drain result_queue and send to all WS clients."""
    while True:
        try:
            msg = result_queue.get_nowait()
            try:
                data = json.dumps(msg)
            except TypeError:
                # Force convert all values to native Python types
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
        await asyncio.sleep(0.02)  # 50Hz polling


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    # Start pusher if not running
    if not hasattr(state, '_pusher_started'):
        state['_pusher_started'] = True
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

            elif action == 'switch_model':
                new_model = cmd.get('model', 'ncssm-20k')
                if new_model in models:
                    was_listening = state['is_listening']
                    if was_listening:
                        stop_listening()
                    state['current_model'] = new_model
                    if was_listening:
                        await asyncio.sleep(0.3)
                        start_listening()
                    _, w = models[new_model]
                    await ws.send_text(json.dumps({
                        'type': 'model_changed',
                        'model': new_model,
                        'params': w.n_params,
                    }))

            elif action == 'set_threshold':
                state['threshold'] = float(cmd.get('value', 0.45))
                if state['engine']:
                    state['engine'].confidence_threshold = state['threshold']

    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)


@app.on_event("startup")
async def startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    load_models()
    print(f"\n  Server ready: http://localhost:{PORT}")
    print(f"  Models loaded: {list(models.keys())}\n")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')
