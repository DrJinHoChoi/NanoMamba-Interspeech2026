#!/usr/bin/env python3
"""Evaluate NC-TCN-20K and NC-TCN-20K-SS noise robustness on CPU."""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io.wavfile as wavfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanomamba import create_nc_tcn_20k
from train_colab import generate_noise_signal, mix_audio_at_snr, spectral_subtraction_v2

SR = 16000
DATA_ROOT = Path(__file__).parent.parent / 'data' / 'SpeechCommands' / 'speech_commands_v0.02'
CKPT_DIR = Path(__file__).parent.parent / 'checkpoints_full'

CORE_WORDS = {'yes','no','up','down','left','right','on','off','stop','go'}
LABELS = ['yes','no','up','down','left','right','on','off','stop','go','silence','unknown']
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

def load_test_set():
    test_list = DATA_ROOT / 'testing_list.txt'
    samples = []
    with open(test_list) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            keyword = line.split('/')[0]
            label = keyword if keyword in CORE_WORDS else 'unknown'
            samples.append((str(DATA_ROOT / line), LABEL2IDX[label]))
    bg_dir = DATA_ROOT / '_background_noise_'
    if bg_dir.is_dir():
        noise_files = sorted(bg_dir.glob('*.wav'))
        for i in range(500):
            nf = noise_files[i % len(noise_files)]
            samples.append((str(nf) + f'#silence_{i}', LABEL2IDX['silence']))
    return samples

def load_audio(path, target_len=16000):
    actual_path = path.split('#')[0]
    is_silence = '#silence' in path
    try:
        sr, data = wavfile.read(actual_path)
        audio = torch.from_numpy(data.astype(np.float32) / 32768.0)
        if sr != SR:
            ratio = SR / sr
            new_len = int(len(audio) * ratio)
            audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear').squeeze()
    except:
        audio = torch.zeros(target_len)
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

def evaluate(model, samples, batch_size=128, noise_type=None, snr_db=None, use_ss=False):
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
            logits = model(audios)
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total

def main():
    print("Loading test set...", flush=True)
    samples = load_test_set()
    print(f"  {len(samples)} samples", flush=True)

    print("Loading NC-TCN-20K...", flush=True)
    model = create_nc_tcn_20k(12)
    ckpt = torch.load(str(CKPT_DIR / 'NC-TCN-20K' / 'best.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    results = {'NC-TCN-20K': {'noise': {}}, 'NC-TCN-20K-SS': {'noise': {}}}

    # Clean
    t0 = time.time()
    acc = evaluate(model, samples)
    results['NC-TCN-20K']['clean'] = round(acc, 2)
    results['NC-TCN-20K-SS']['clean'] = round(acc, 2)
    print(f"  Clean: {acc:.2f}% ({time.time()-t0:.0f}s)", flush=True)

    noise_types = ['babble', 'factory', 'street', 'white', 'pink']
    snr_levels = [-10, -5, 0, 5, 10, 15]

    for name, use_ss in [('NC-TCN-20K', False), ('NC-TCN-20K-SS', True)]:
        print(f"\n=== {name} ===", flush=True)
        for nt in noise_types:
            results[name]['noise'][nt] = {}
            for snr in snr_levels:
                t0 = time.time()
                acc = evaluate(model, samples, noise_type=nt, snr_db=snr, use_ss=use_ss)
                results[name]['noise'][nt][str(snr)] = round(acc, 1)
                print(f"  {nt:>8s} {snr:>4d}dB: {acc:.1f}% ({time.time()-t0:.0f}s)", flush=True)

        with open('paper/eval_nctcn_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n=== SUMMARY (5-noise avg) ===")
    for name in ['NC-TCN-20K', 'NC-TCN-20K-SS']:
        r = results[name]
        line = f"  {name:>15s} | Clean={r['clean']:.1f}"
        for snr in snr_levels:
            vals = [r['noise'][nt][str(snr)] for nt in noise_types]
            line += f" | {snr}dB={np.mean(vals):.1f}"
        print(line)

    print(f"\nSaved to paper/eval_nctcn_results.json")

if __name__ == '__main__':
    main()
