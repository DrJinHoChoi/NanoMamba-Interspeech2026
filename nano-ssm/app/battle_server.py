#!/usr/bin/env python3
"""
Voice Reflex Battle — CES 2027 Showcase
========================================
NC-SSM vs DS-CNN-S: Real-time keyword detection speed battle.

Both models process the SAME audio stream simultaneously.
NC-SSM detects ~350ms, CNN detects ~1050ms → visible speed gap.

    python battle_server.py

Opens http://localhost:8090
"""

import asyncio
import json
import os
import sys
import time
import threading
import queue
import random

# Fix Windows console encoding
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
from fastapi.staticfiles import StaticFiles

# Add paths
PARENT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(PARENT, 'nano-ssm'))
sys.path.insert(0, PARENT)

import nano_ssm
from nano_ssm.streaming import StreamingEngine

# ── Config ──
SR = 16000
CHUNK_MS = 100
CHUNK_SAMPLES = int(SR * CHUNK_MS / 1000)
PORT = 8090
CKPT_DIR = os.path.join(PARENT, 'checkpoints_full')

GSC_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right',
              'on', 'off', 'stop', 'go', 'silence', 'unknown']

# Keywords for the battle (exclude silence/unknown)
BATTLE_KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right',
                   'on', 'off', 'stop', 'go']


# ── CNN SimpleEngine (reused from server.py) ──
class SimpleEngine:
    """1-second buffer → mel → classify. For CNN models."""
    def __init__(self, wrapper, sr=16000, threshold=0.45, cooldown_chunks=8):
        self.wrapper = wrapper
        self.sr = sr
        self.threshold = threshold
        self.confidence_threshold = threshold
        self.cooldown_chunks = cooldown_chunks
        self.labels = GSC_LABELS
        self.buffer = torch.zeros(0)
        self.chunks_since = cooldown_chunks

        n_fft = 512
        n_mels = 40
        self.n_fft = n_fft
        self.hop_length = 160
        self.window = torch.hann_window(n_fft)

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
            'raw_probs': np.zeros(len(self.labels)),
            'energy_db': energy_db, 'detected': False,
        }

        if len(self.buffer) < self.sr:
            return None

        audio = self.buffer[-self.sr:].unsqueeze(0)
        spec = torch.stft(audio, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True)
        mag = spec.abs()
        mel = torch.matmul(self.mel_fb, mag)
        log_mel = torch.log(mel + 1e-8)

        model = self.wrapper.model
        logits = model(log_mel)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        conf = float(probs[idx])

        result['label'] = label
        result['confidence'] = conf
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


class CNNWrapper:
    def __init__(self, model, name, n_params):
        self.model = model
        self.name = name
        self.n_params = n_params
        self.labels = GSC_LABELS


# ── Global state ──
models = {}  # key -> (type, wrapper)
clients = set()
audio_queue = queue.Queue()
result_queue = queue.Queue()

state = {
    'is_listening': False,
    'stream': None,
    'thread': None,
    'ncssm_engine': None,
    'cnn_engine': None,
    # Battle state
    'target_keyword': None,
    'round_active': False,
    'round_start_time': None,
    'ncssm_detected_time': None,
    'cnn_detected_time': None,
    'scores': {'ncssm': 0, 'cnn': 0, 'ties': 0},
    'round_number': 0,
    'threshold': 0.35,
    'noise_mode': False,
    'history': [],  # list of round results
}


# ── Load models ──
def load_models():
    print("\n  Loading models for Battle...")

    # NC-SSM 20K
    ckpt_path = os.path.join(CKPT_DIR, 'NanoMamba-NC-20K', 'best.pt')
    if os.path.exists(ckpt_path):
        try:
            m = nano_ssm.create('ncssm-20k', pretrained=ckpt_path)
            models['ncssm'] = ('ncssm', m)
            print(f"    NC-SSM-20K: {m.n_params:,} params OK")
        except Exception as e:
            print(f"    NC-SSM-20K: FAIL ({e})")

    # Fallback: NC-SSM matched
    if 'ncssm' not in models:
        ckpt_path = os.path.join(CKPT_DIR, 'NanoMamba-NC-Matched', 'best.pt')
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
        dscnn_ckpt = os.path.join(CKPT_DIR, 'DS-CNN-S', 'best.pt')
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

    # Fallback: BC-ResNet-1
    if 'cnn' not in models:
        try:
            from train_colab import BCResNet
            bcres_ckpt = os.path.join(CKPT_DIR, 'BC-ResNet-1', 'best.pt')
            if os.path.exists(bcres_ckpt):
                m = BCResNet(n_classes=12, scale=1)
                ckpt = torch.load(bcres_ckpt, map_location='cpu', weights_only=False)
                m.load_state_dict(ckpt['model_state_dict'])
                m.eval()
                n_params = sum(p.numel() for p in m.parameters())
                models['cnn'] = ('cnn', CNNWrapper(m, 'BC-ResNet-1', n_params))
                print(f"    BC-ResNet-1: {n_params:,} params OK")
        except Exception as e:
            print(f"    BC-ResNet-1: FAIL ({e})")


# ── Audio callback ──
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())


# ── Broadcast ──
def broadcast_sync(msg: dict):
    result_queue.put(msg)


# ── Dual processing thread ──
def battle_process_loop():
    """Process audio through BOTH models simultaneously."""
    ncssm_engine = state['ncssm_engine']
    cnn_engine = state['cnn_engine']

    if ncssm_engine is None or cnn_engine is None:
        print("  [ERR] Engines not initialized")
        return

    while state['is_listening']:
        try:
            audio_np = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        chunk = torch.from_numpy(audio_np).float()

        # Feed BOTH models with the SAME audio
        t0_ncssm = time.perf_counter()
        result_ncssm = ncssm_engine.feed(chunk)
        lat_ncssm = (time.perf_counter() - t0_ncssm) * 1000

        t0_cnn = time.perf_counter()
        result_cnn = cnn_engine.feed(chunk)
        lat_cnn = (time.perf_counter() - t0_cnn) * 1000

        now = time.perf_counter()

        # Energy from chunk
        rms = np.sqrt(np.mean(audio_np ** 2))
        energy_db = 20 * np.log10(max(rms, 1e-10))

        # Build message
        msg = {
            'type': 'battle_tick',
            'energy_db': round(float(energy_db), 1),
            'timestamp': time.time(),
        }

        # NC-SSM result
        if result_ncssm is not None:
            ncssm_label = result_ncssm.get('smoothed_label', result_ncssm.get('label', 'silence'))
            ncssm_conf = float(result_ncssm.get('smoothed_confidence', result_ncssm.get('confidence', 0.0)))
            ncssm_probs = result_ncssm.get('raw_probs', None)
            ncssm_probs_list = [float(p) for p in ncssm_probs] if ncssm_probs is not None else [0.0] * 12
            prob_sum = sum(ncssm_probs_list)

            msg['ncssm'] = {
                'label': str(ncssm_label),
                'confidence': round(ncssm_conf, 4),
                'probs': ncssm_probs_list if prob_sum > 0.01 else None,
                'latency_ms': round(float(lat_ncssm), 1),
                'detected': result_ncssm.get('detected', False),
            }
        else:
            msg['ncssm'] = {'label': 'buffering', 'confidence': 0.0,
                            'probs': None, 'latency_ms': 0, 'detected': False}

        # CNN result
        if result_cnn is not None:
            cnn_label = result_cnn.get('label', 'silence')
            cnn_conf = float(result_cnn.get('confidence', 0.0))
            cnn_probs = result_cnn.get('raw_probs', None)
            cnn_probs_list = [float(p) for p in cnn_probs] if cnn_probs is not None else [0.0] * 12

            msg['cnn'] = {
                'label': str(cnn_label),
                'confidence': round(cnn_conf, 4),
                'probs': cnn_probs_list,
                'latency_ms': round(float(lat_cnn), 1),
                'detected': result_cnn.get('detected', False),
            }
        else:
            msg['cnn'] = {'label': 'buffering', 'confidence': 0.0,
                          'probs': None, 'latency_ms': 0, 'detected': False}

        # Debug: log top predictions during active round
        if state['round_active'] and state['target_keyword']:
            target = state['target_keyword']
            n_label = msg['ncssm']['label']
            n_conf = msg['ncssm']['confidence']
            c_label = msg['cnn']['label']
            c_conf = msg['cnn']['confidence']
            n_det = msg['ncssm']['detected']
            c_det = msg['cnn']['detected']
            if n_label != 'silence' or c_label not in ('silence', 'unknown'):
                print(f"  [R{state['round_number']}] target={target} | ncssm={n_label}({n_conf:.2f},det={n_det}) | cnn={c_label}({c_conf:.2f},det={c_det}) | e={msg['energy_db']}dB", flush=True)

            # Check NC-SSM detection (any keyword, not just target)
            if (state['ncssm_detected_time'] is None and
                    msg['ncssm']['detected'] and
                    msg['ncssm']['label'] not in ('silence', 'unknown')):
                state['ncssm_detected_time'] = now
                reaction_ms = (now - state['round_start_time']) * 1000
                broadcast_sync({
                    'type': 'model_detected',
                    'model': 'ncssm',
                    'label': msg['ncssm']['label'],
                    'reaction_ms': round(reaction_ms, 0),
                    'confidence': msg['ncssm']['confidence'],
                })

            # Check CNN detection (any keyword, not just target)
            if (state['cnn_detected_time'] is None and
                    msg['cnn']['detected'] and
                    msg['cnn']['label'] not in ('silence', 'unknown')):
                state['cnn_detected_time'] = now
                reaction_ms = (now - state['round_start_time']) * 1000
                broadcast_sync({
                    'type': 'model_detected',
                    'model': 'cnn',
                    'label': target,
                    'reaction_ms': round(reaction_ms, 0),
                    'confidence': msg['cnn']['confidence'],
                })

            # Both detected or timeout
            both_done = (state['ncssm_detected_time'] is not None and
                         state['cnn_detected_time'] is not None)
            elapsed = (now - state['round_start_time'])
            timed_out = elapsed > 5.0  # 5 second timeout

            if both_done or timed_out:
                # Determine winner
                ncssm_t = state['ncssm_detected_time']
                cnn_t = state['cnn_detected_time']

                if ncssm_t and cnn_t:
                    ncssm_ms = (ncssm_t - state['round_start_time']) * 1000
                    cnn_ms = (cnn_t - state['round_start_time']) * 1000
                    if ncssm_ms < cnn_ms - 50:  # 50ms margin
                        winner = 'ncssm'
                        state['scores']['ncssm'] += 1
                    elif cnn_ms < ncssm_ms - 50:
                        winner = 'cnn'
                        state['scores']['cnn'] += 1
                    else:
                        winner = 'tie'
                        state['scores']['ties'] += 1
                elif ncssm_t:
                    winner = 'ncssm'
                    ncssm_ms = (ncssm_t - state['round_start_time']) * 1000
                    cnn_ms = None
                    state['scores']['ncssm'] += 1
                elif cnn_t:
                    winner = 'cnn'
                    ncssm_ms = None
                    cnn_ms = (cnn_t - state['round_start_time']) * 1000
                    state['scores']['cnn'] += 1
                else:
                    winner = 'timeout'
                    ncssm_ms = None
                    cnn_ms = None

                round_result = {
                    'round': state['round_number'],
                    'keyword': target,
                    'winner': winner,
                    'ncssm_ms': round(ncssm_ms, 0) if ncssm_ms else None,
                    'cnn_ms': round(cnn_ms, 0) if cnn_ms else None,
                }
                state['history'].append(round_result)

                broadcast_sync({
                    'type': 'round_result',
                    **round_result,
                    'scores': dict(state['scores']),
                })

                state['round_active'] = False

        broadcast_sync(msg)


# ── Create engines ──
def create_engines():
    if 'ncssm' in models:
        _, wrapper = models['ncssm']
        state['ncssm_engine'] = StreamingEngine(
            wrapper, chunk_ms=CHUNK_MS, sr=SR,
            confidence_threshold=state['threshold'],
            cooldown_chunks=6,
        )
        # Lower onset threshold for USB mic (speaks at ~-15 to -30dB)
        state['ncssm_engine']._energy_speech = -50.0
        state['ncssm_engine']._energy_floor = -60.0
    if 'cnn' in models:
        _, wrapper = models['cnn']
        state['cnn_engine'] = SimpleEngine(
            wrapper, sr=SR,
            threshold=state['threshold'],
            cooldown_chunks=3,
        )


# ── Start / Stop ──
def start_listening():
    if state['is_listening']:
        return
    if 'ncssm' not in models or 'cnn' not in models:
        broadcast_sync({'type': 'error', 'message': 'Models not loaded'})
        return

    create_engines()

    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break

    # Find a working audio input device
    import scipy.signal as sig

    def try_open_mic(device=None):
        """Try to open mic, with resampling if needed."""
        dev_info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device[0])
        native_sr = int(dev_info['default_samplerate'])
        dev_name = dev_info['name']

        if native_sr == SR:
            # Direct 16kHz
            stream = sd.InputStream(
                samplerate=SR, channels=1, dtype='float32',
                blocksize=CHUNK_SAMPLES, callback=audio_callback,
                device=device,
            )
            stream.start()
            print(f'  Mic: [{device}] {dev_name} @ {SR}Hz direct', flush=True)
            return stream

        # Need resampling
        native_block = int(native_sr * CHUNK_MS / 1000)

        def resample_callback(indata, frames, time_info, status):
            audio = indata[:, 0].astype(np.float32)
            target_len = int(len(audio) * SR / native_sr)
            resampled = sig.resample(audio, target_len).astype(np.float32)
            audio_queue.put(resampled)

        stream = sd.InputStream(
            samplerate=native_sr, channels=1, dtype='float32',
            blocksize=native_block, callback=resample_callback,
            device=device,
        )
        stream.start()
        print(f'  Mic: [{device}] {dev_name} @ {native_sr}Hz -> resample {SR}Hz', flush=True)
        return stream

    # Try devices in priority order
    device_priority = [34, None, 31, 20, 21]  # USB mic first, then fallbacks
    opened = False
    for dev_id in device_priority:
        try:
            state['stream'] = try_open_mic(dev_id)
            opened = True
            break
        except Exception as e:
            print(f'  Mic [{dev_id}] failed: {str(e)[:60]}', flush=True)
            continue

    if not opened:
        err_msg = 'No working audio input device found'
        print(f'  [ERR] {err_msg}', flush=True)
        broadcast_sync({'type': 'error', 'message': err_msg})
        return

    state['is_listening'] = True

    state['thread'] = threading.Thread(target=battle_process_loop, daemon=True)
    state['thread'].start()

    ncssm_type, ncssm_w = models['ncssm']
    cnn_type, cnn_w = models['cnn']
    broadcast_sync({
        'type': 'started',
        'ncssm_name': getattr(ncssm_w, 'name', 'NC-SSM-20K'),
        'ncssm_params': ncssm_w.n_params,
        'cnn_name': getattr(cnn_w, 'name', 'DS-CNN-S'),
        'cnn_params': cnn_w.n_params,
    })
    print(f'  Audio started (device={sd.default.device})', flush=True)


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
    state['round_active'] = False
    broadcast_sync({'type': 'stopped'})


def new_round():
    """Start a new battle round with a random keyword."""
    if not state['is_listening']:
        return

    # Reset engines
    if state['ncssm_engine']:
        state['ncssm_engine'].reset()
        state['ncssm_engine']._energy_speech = -50.0
        state['ncssm_engine']._energy_floor = -60.0
    if state['cnn_engine']:
        state['cnn_engine'].reset()

    # Clear audio queue
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break

    state['round_number'] += 1
    state['target_keyword'] = random.choice(BATTLE_KEYWORDS)
    state['round_active'] = True
    state['round_start_time'] = time.perf_counter()
    state['ncssm_detected_time'] = None
    state['cnn_detected_time'] = None

    broadcast_sync({
        'type': 'new_round',
        'round': state['round_number'],
        'keyword': state['target_keyword'],
    })


# ── FastAPI ──
app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'battle.html'))


@app.get("/api/status")
async def get_status():
    return {
        'models_loaded': list(models.keys()),
        'is_listening': state['is_listening'],
        'scores': state['scores'],
        'round': state['round_number'],
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

            elif action == 'new_round':
                new_round()

            elif action == 'reset_scores':
                state['scores'] = {'ncssm': 0, 'cnn': 0, 'ties': 0}
                state['round_number'] = 0
                state['history'] = []
                await ws.send_text(json.dumps({
                    'type': 'scores_reset',
                    'scores': state['scores'],
                }))

            elif action == 'set_threshold':
                state['threshold'] = float(cmd.get('value', 0.45))
                if state['ncssm_engine']:
                    state['ncssm_engine'].confidence_threshold = state['threshold']
                if state['cnn_engine']:
                    state['cnn_engine'].confidence_threshold = state['threshold']

    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)


@app.on_event("startup")
async def startup():
    load_models()
    print(f"\n  Voice Reflex Battle ready: http://localhost:{PORT}", flush=True)
    print(f"  Models: {list(models.keys())}\n", flush=True)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')
