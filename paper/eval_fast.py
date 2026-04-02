#!/usr/bin/env python3
"""Fast evaluation - bypasses slow torchaudio dataset loading."""
import sys, os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io.wavfile as wavfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanomamba import create_nanomamba_nc_20k, create_nanomamba_nc_matched, create_nc_tcn_20k
from train_colab import (DSCNN_S, BCResNet, generate_noise_signal,
                          mix_audio_at_snr, spectral_subtraction_v2)

DEVICE = 'cpu'
SR = 16000
DATA_ROOT = Path(__file__).parent.parent / 'data' / 'SpeechCommands' / 'speech_commands_v0.02'
CKPT_DIR = Path(__file__).parent.parent / 'checkpoints_full'

CORE_WORDS = {'yes','no','up','down','left','right','on','off','stop','go'}
LABELS = ['yes','no','up','down','left','right','on','off','stop','go','silence','unknown']
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

def load_test_set():
    """Load test set from testing_list.txt + silence samples."""
    test_list = DATA_ROOT / 'testing_list.txt'
    samples = []
    with open(test_list) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            keyword = line.split('/')[0]
            label = keyword if keyword in CORE_WORDS else 'unknown'
            wav_path = DATA_ROOT / line
            samples.append((str(wav_path), LABEL2IDX[label]))

    # Add silence samples from background noise
    bg_dir = DATA_ROOT / '_background_noise_'
    if bg_dir.is_dir():
        noise_files = sorted(bg_dir.glob('*.wav'))
        n_silence = 500  # standard for test set
        silence_idx = LABEL2IDX['silence']
        for i in range(n_silence):
            nf = noise_files[i % len(noise_files)]
            samples.append((str(nf) + f'#silence_{i}', silence_idx))
    return samples

def load_audio(path, target_len=16000):
    """Load and pad/trim audio to target length using scipy."""
    actual_path = path.split('#')[0]
    is_silence = '#silence' in path
    try:
        sr, data = wavfile.read(actual_path)
        audio = torch.from_numpy(data.astype(np.float32) / 32768.0)
        if sr != SR:
            ratio = SR / sr
            new_len = int(len(audio) * ratio)
            audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0),
                                  size=new_len, mode='linear').squeeze()
    except Exception as e:
        print(f"  WARNING: Failed to load {actual_path}: {e}")
        audio = torch.zeros(target_len)

    # For silence: random segment, scaled down
    if is_silence:
        if len(audio) > target_len:
            start = np.random.randint(0, len(audio) - target_len)
            audio = audio[start:start + target_len]
        audio = audio * 0.1

    if len(audio) < target_len:
        audio = F.pad(audio, (0, target_len - len(audio)))
    elif len(audio) > target_len:
        audio = audio[:target_len]
    return audio

def compute_mel(audio, mel_fb):
    """Compute log-mel spectrogram."""
    window = torch.hann_window(400)
    spec = torch.stft(audio, 512, 160, 400, window=window, return_complex=True)
    mag = spec.abs()
    mel = torch.matmul(mel_fb, mag)
    return torch.log(mel + 1e-8)

def create_mel_fb():
    n_mels, n_fft, sr = 40, 512, 16000
    n_freq = n_fft // 2 + 1
    mel_high = 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(0, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i+1]):
            if j < n_freq:
                fb[i,j] = (j - bin_points[i]) / max(bin_points[i+1] - bin_points[i], 1)
        for j in range(bin_points[i+1], bin_points[i+2]):
            if j < n_freq:
                fb[i,j] = (bin_points[i+2] - j) / max(bin_points[i+2] - bin_points[i+1], 1)
    return torch.from_numpy(fb)

def batch_evaluate(model, samples, batch_size=128, is_cnn=False, mel_fb=None,
                   noise_type=None, snr_db=None, use_ss=False):
    """Evaluate model on samples with optional noise."""
    model.eval()
    correct = total = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        audios = torch.stack([load_audio(p) for p, _ in batch])
        labels = torch.tensor([l for _, l in batch])

        if noise_type is not None and snr_db is not None:
            noise = generate_noise_signal(noise_type, audios.size(-1), sr=SR)
            if noise.dim() == 1:
                noise = noise.unsqueeze(0).expand_as(audios)
            elif noise.size(0) != audios.size(0):
                noise = noise[:1].expand_as(audios)
            audios = mix_audio_at_snr(audios, noise, snr_db)

        if use_ss:
            audios = spectral_subtraction_v2(audios)

        with torch.no_grad():
            if is_cnn and mel_fb is not None:
                mels = torch.stack([compute_mel(a, mel_fb) for a in audios])
                logits = model(mels)
            else:
                logits = model(audios)

        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total

def main():
    print("=" * 80)
    print("  NC-TCN Paper - Full Evaluation")
    print(f"  Device: {DEVICE}")
    print("=" * 80)

    # Load test set
    print("\n[1] Loading test set...", flush=True)
    samples = load_test_set()
    print(f"  {len(samples)} test samples loaded", flush=True)

    # Load models
    print("\n[2] Loading models...", flush=True)
    configs = [
        ('DS-CNN-S', DSCNN_S(12), 'DS-CNN-S/best.pt', True),
        ('BC-ResNet-1', BCResNet(12, scale=1), 'BC-ResNet-1/best.pt', True),
        ('NC-SSM', create_nanomamba_nc_matched(12), 'NC-SSM/best.pt', False),
        ('NC-SSM-20K', create_nanomamba_nc_20k(12), 'NanoMamba-NC-20K/best.pt', False),
        ('NC-TCN-20K', create_nc_tcn_20k(12), 'NC-TCN-20K/best.pt', False),
    ]

    models = {}
    for name, model, ckpt_name, is_cnn in configs:
        ckpt = torch.load(str(CKPT_DIR / ckpt_name), map_location='cpu', weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
        except:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"  WARNING: {name} loaded with strict=False")
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        models[name] = (model, is_cnn)
        print(f"  {name:>15s}: {params:>6,} params", flush=True)

    mel_fb = create_mel_fb()
    results = {}

    # Clean accuracy
    print("\n[3] Clean accuracy...", flush=True)
    for name, (model, is_cnn) in models.items():
        t0 = time.time()
        acc = batch_evaluate(model, samples, is_cnn=is_cnn, mel_fb=mel_fb)
        results[name] = {'clean': round(acc, 2), 'noise': {}}
        print(f"  {name:>15s}: {acc:.2f}% ({time.time()-t0:.1f}s)", flush=True)

    # NC-TCN-20K-SS = NC-TCN-20K clean (SS bypass at high SNR)
    results['NC-TCN-20K-SS'] = {'clean': results['NC-TCN-20K']['clean'], 'noise': {}}
    print(f"  {'NC-TCN-20K-SS':>15s}: {results['NC-TCN-20K-SS']['clean']:.2f}% (=NC-TCN-20K)", flush=True)

    # Noise evaluation
    noise_types = ['babble', 'factory', 'street', 'white', 'pink']
    snr_levels = [-15, -10, -5, 0, 5, 10, 15]

    print(f"\n[4] Noise robustness ({len(noise_types)} types x {len(snr_levels)} SNRs)...", flush=True)

    all_model_names = list(models.keys()) + ['NC-TCN-20K-SS']

    for name in all_model_names:
        if name == 'NC-TCN-20K-SS':
            model, is_cnn = models['NC-TCN-20K']
            use_ss = True
        else:
            model, is_cnn = models[name]
            use_ss = False

        print(f"\n  --- {name} ---", flush=True)
        for nt in noise_types:
            if nt not in results[name]['noise']:
                results[name]['noise'][nt] = {}
            for snr in snr_levels:
                t0 = time.time()
                acc = batch_evaluate(model, samples, is_cnn=is_cnn, mel_fb=mel_fb,
                                    noise_type=nt, snr_db=snr, use_ss=use_ss)
                results[name]['noise'][nt][str(snr)] = round(acc, 1)
                dt = time.time() - t0
                print(f"    {nt:>8s} {snr:>4d}dB: {acc:.1f}% ({dt:.0f}s)", flush=True)

        # Save intermediate results
        with open('paper/eval_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Final summary tables
    print("\n" + "=" * 90)
    print("  TABLE I: Accuracy (%) averaged over 5 noise types")
    hdr = f"  {'Model':>15s} | {'Clean':>6s}"
    for s in snr_levels:
        hdr += f" | {s:>5d}dB"
    print(hdr)
    print("  " + "-" * 85)

    for name in all_model_names:
        r = results[name]
        line = f"  {name:>15s} | {r['clean']:>5.1f}%"
        for snr in snr_levels:
            vals = [r['noise'].get(nt, {}).get(str(snr), 0) for nt in noise_types]
            avg = np.mean(vals) if vals else 0
            line += f" | {avg:>5.1f}%"
        print(line)

    print(f"\n  TABLE III: Per-noise at 0dB")
    hdr = f"  {'Model':>15s}"
    for nt in noise_types:
        hdr += f" | {nt:>8s}"
    print(hdr)
    print("  " + "-" * 65)
    for name in all_model_names:
        line = f"  {name:>15s}"
        for nt in noise_types:
            v = results[name]['noise'].get(nt, {}).get('0', 0)
            line += f" | {v:>7.1f}%"
        print(line)

    print(f"\nResults saved to paper/eval_results.json")

if __name__ == '__main__':
    main()
