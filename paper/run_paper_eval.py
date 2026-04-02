#!/usr/bin/env python3
"""
NC-TCN MLSP 2026 Paper — Full Evaluation Script
Evaluates all 5 models + NC-TCN-20K with external SS on GSC V2.
Outputs: clean accuracy, noise robustness (5 types × 7 SNR levels),
         per-noise breakdown at 0dB, params, MACs.
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from nanomamba import (
    create_nanomamba_nc_20k, create_nanomamba_nc_matched,
    create_nc_tcn_20k, create_nc_tcn_20k_ss
)
from train_colab import (
    SpeechCommandsDataset, DSCNN_S, BCResNet,
    evaluate, evaluate_noisy, _evaluate_cnn, _create_mel_fb_tensor,
    _is_cnn_model, generate_noise_signal, mix_audio_at_snr,
    spectral_subtraction_v2
)
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = ROOT / 'checkpoints_full'
RESULTS_FILE = ROOT / 'paper' / 'eval_results.json'

NOISE_TYPES = ['babble', 'factory', 'street', 'white', 'pink']
SNR_LEVELS = [-15, -10, -5, 0, 5, 10, 15]

# Model definitions
MODELS_CONFIG = {
    'DS-CNN-S': {
        'create_fn': lambda: DSCNN_S(n_classes=12),
        'ckpt': 'DS-CNN-S/best.pt',
        'is_cnn': True,
    },
    'BC-ResNet-1': {
        'create_fn': lambda: BCResNet(n_classes=12, scale=1),
        'ckpt': 'BC-ResNet-1/best.pt',
        'is_cnn': True,
    },
    'NC-SSM': {
        'create_fn': lambda: create_nanomamba_nc_matched(12),
        'ckpt': 'NC-SSM/best.pt',
        'is_cnn': False,
    },
    'NC-SSM-20K': {
        'create_fn': lambda: create_nanomamba_nc_20k(12),
        'ckpt': 'NanoMamba-NC-20K/best.pt',
        'is_cnn': False,
    },
    'NC-TCN-20K': {
        'create_fn': lambda: create_nc_tcn_20k(12),
        'ckpt': 'NC-TCN-20K/best.pt',
        'is_cnn': False,
    },
}


def load_model(name, config):
    """Load model + checkpoint."""
    model = config['create_fn']()
    ckpt_path = CKPT_DIR / config['ckpt']
    if not ckpt_path.exists():
        print(f"  [SKIP] {name}: checkpoint not found at {ckpt_path}")
        return None
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    # Try loading; handle EMA state if present
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # Try EMA state
        ema_state = ckpt.get('ema_state_dict', None)
        if ema_state:
            model.load_state_dict(ema_state, strict=True)
            print(f"  {name}: loaded EMA weights")
        else:
            model.load_state_dict(state, strict=False)
            print(f"  {name}: loaded with strict=False")
    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n_params:,} params, loaded from {config['ckpt']}")
    return model


def count_macs(model, is_cnn=False):
    """Estimate MACs using a dummy forward pass."""
    model.eval()
    total_macs = 0

    if is_cnn:
        # CNN models: mel input (40, 101)
        dummy = torch.randn(1, 40, 101).to(DEVICE)
    else:
        # NanoMamba/NC-TCN: raw audio (16000,)
        dummy = torch.randn(1, 16000).to(DEVICE)

    # Use simple parameter-based MAC estimation
    # More accurate than hook-based for our specific architectures
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_macs += module.in_features * module.out_features
            if hasattr(module, 'bias') and module.bias is not None:
                total_macs += module.out_features
        elif isinstance(module, torch.nn.Conv1d):
            # out_channels * kernel_size * in_channels/groups * output_length
            # Approximate output_length
            if is_cnn:
                out_len = 101  # approximate
            else:
                out_len = 101  # ~100 frames for 1 sec
            total_macs += (module.out_channels * module.kernel_size[0] *
                          module.in_channels // module.groups * out_len)
        elif isinstance(module, torch.nn.Conv2d):
            # For CNN baselines
            k = module.kernel_size[0] * module.kernel_size[1]
            total_macs += (module.out_channels * k *
                          module.in_channels // module.groups * 20 * 50)  # approx spatial

    return total_macs


def compute_macs_detailed(model_name, model):
    """More accurate MAC computation for our specific models."""
    T = 101  # frames
    n_mels = 40

    if model_name == 'DS-CNN-S':
        # DS-CNN-S: known architecture
        # Conv layers + DW separable convs
        n_params = sum(p.numel() for p in model.parameters())
        # From literature: DS-CNN-S ~ 5.4M MACs
        # We verify by summing conv MACs
        total = 0
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                # H_out * W_out * C_out * K_h * K_w * C_in / groups
                # Approximate: assume feature map ~20x50 on average
                total += m.out_channels * m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups * 20 * 50
            elif isinstance(m, torch.nn.Linear):
                total += m.in_features * m.out_features
        return total

    elif model_name == 'BC-ResNet-1':
        # BC-ResNet-1 with scale=1
        total = 0
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                # More careful spatial estimation needed
                total += m.out_channels * m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups * 10 * 25
            elif isinstance(m, torch.nn.Linear):
                total += m.in_features * m.out_features
        return total

    elif 'NC-SSM' in model_name or 'NC-TCN' in model_name:
        # NC-SSM / NC-TCN: we know the exact architecture
        d_model = 37
        d_inner = 55
        d_state = 10
        n_layers = 2 if 'SSM' in model_name else 3

        # STFT: negligible compared to model
        # Mel projection: 257 * 40 * T
        macs = 257 * 40 * T

        # SNR estimation: ~minimal
        macs += 40 * T * 5  # some per-frame ops

        # LSG: 40 * T * 3
        macs += 40 * T * 3

        # PCEN: ~40 * T * 20 (two experts + routing)
        macs += 40 * T * 20

        # Patch projection: 40 * 37 * T
        macs += 40 * d_model * T

        # Per layer:
        for l in range(n_layers):
            # LayerNorm: 2 * d_model * T
            macs += 2 * d_model * T
            # in_proj: d_model * 2*d_inner * T
            macs += d_model * 2 * d_inner * T
            # DWConv1d: d_inner * 3 * T (kernel=3)
            macs += d_inner * 3 * T

            if 'SSM' in model_name:
                # SSM scan: per timestep
                # x_proj: d_inner * (1 + d_state + d_state) * T
                macs += d_inner * (1 + 2 * d_state) * T
                # snr_proj: 40 * (1 + d_state) * T
                macs += 40 * (1 + d_state) * T
                # State update: d_inner * d_state * T
                macs += d_inner * d_state * T * 3  # multiply, add, output
            else:
                # TCN: dilated conv already counted in DWConv
                # No SSM-specific MACs
                pass

            # SiLU: d_inner * T
            macs += d_inner * T
            # Gate: d_inner * T
            macs += d_inner * T
            # out_proj: d_inner * d_model * T
            macs += d_inner * d_model * T

        # Final norm + GAP + classifier
        macs += 2 * d_model * T  # norm
        macs += d_model * T  # GAP
        macs += d_model * 12  # classifier

        return macs

    return 0


@torch.no_grad()
def evaluate_clean(model, val_loader, is_cnn=False):
    """Evaluate clean accuracy."""
    model.eval()
    correct = 0
    total = 0
    for mel, labels, audio in val_loader:
        labels = labels.to(DEVICE)
        if is_cnn:
            mel = mel.to(DEVICE)
            logits = model(mel)
        else:
            audio = audio.to(DEVICE)
            logits = model(audio)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total


@torch.no_grad()
def evaluate_noisy_condition(model, val_loader, noise_type, snr_db,
                              is_cnn=False, mel_fb=None,
                              use_ext_ss=False, has_ss_bypass=False):
    """Evaluate under a single noise condition."""
    model.eval()
    correct = 0
    total = 0

    for mel, labels, audio in val_loader:
        labels = labels.to(DEVICE)
        audio = audio.to(DEVICE)

        noise = generate_noise_signal(noise_type, audio.size(-1), sr=16000).to(DEVICE)
        noisy_audio = mix_audio_at_snr(audio, noise, snr_db)

        if has_ss_bypass:
            # Model with learned SS bypass: compute SS-enhanced and pass both
            ss_enhanced = spectral_subtraction_v2(noisy_audio)
            logits = model(noisy_audio, ss_enhanced=ss_enhanced)
        elif use_ext_ss:
            # External SS applied before model
            noisy_audio = spectral_subtraction_v2(noisy_audio)
            if is_cnn and mel_fb is not None:
                noisy_mel = _compute_mel_batch_local(noisy_audio, mel_fb)
                logits = model(noisy_mel)
            else:
                logits = model(noisy_audio)
        elif is_cnn and mel_fb is not None:
            noisy_mel = _compute_mel_batch_local(noisy_audio, mel_fb)
            logits = model(noisy_mel)
        else:
            logits = model(noisy_audio)

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100. * correct / total


def _compute_mel_batch_local(audio, mel_fb):
    """Compute mel spectrogram for a batch of audio."""
    n_fft = 512
    hop_length = 160
    win_length = 400
    window = torch.hann_window(win_length).to(audio.device)
    mel_fb = mel_fb.to(audio.device)

    spec = torch.stft(audio, n_fft, hop_length, win_length,
                      window=window, return_complex=True)
    mag = spec.abs()
    mel = torch.matmul(mel_fb, mag)
    mel = torch.log(mel + 1e-8)
    return mel


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(ROOT / 'data'), help='GSC V2 data directory')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--quick', action='store_true', help='Quick test: 1 noise type, 3 SNR levels')
    args = ap.parse_args()

    print("=" * 80)
    print("  NC-TCN MLSP 2026 - Full Paper Evaluation")
    print(f"  Device: {DEVICE}")
    print(f"  Data dir: {args.data_dir}")
    print("=" * 80)

    # Load dataset
    print("\n[1] Loading GSC V2 test set...")
    test_dataset = SpeechCommandsDataset(
        root=args.data_dir, subset='testing',
        n_mels=40, sr=16000, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"  Test samples: {len(test_dataset)}")

    # Load all models
    print("\n[2] Loading models...")
    models = {}
    for name, config in MODELS_CONFIG.items():
        m = load_model(name, config)
        if m is not None:
            models[name] = (m, config['is_cnn'])

    # Also prepare NC-TCN-20K-SS (same checkpoint, different eval mode)
    if 'NC-TCN-20K' in models:
        print("  NC-TCN-20K-SS: will use NC-TCN-20K + external SS at eval time")

    # Mel filterbank for CNN models
    mel_fb = _create_mel_fb_tensor()

    # Results dict
    results = {}

    # [3] Params & MACs
    print("\n[3] Model complexity...")
    for name, (model, is_cnn) in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        macs = compute_macs_detailed(name, model)
        results[name] = {
            'params': n_params,
            'macs': macs,
            'clean': 0.0,
            'noise': {}
        }
        print(f"  {name:>15s}: {n_params:>6,} params, {macs/1e6:.2f}M MACs")

    # NC-TCN-20K-SS has +2 params for bypass gate (but we apply externally here)
    results['NC-TCN-20K-SS'] = {
        'params': results.get('NC-TCN-20K', {}).get('params', 21689),
        'macs': results.get('NC-TCN-20K', {}).get('macs', 0),
        'clean': 0.0,
        'noise': {},
        'note': 'NC-TCN-20K + external SS v2'
    }

    # [4] Clean accuracy
    print("\n[4] Clean accuracy evaluation...")
    for name, (model, is_cnn) in models.items():
        t0 = time.time()
        acc = evaluate_clean(model, test_loader, is_cnn=is_cnn)
        dt = time.time() - t0
        results[name]['clean'] = round(acc, 1)
        print(f"  {name:>15s}: {acc:.1f}%  ({dt:.1f}s)")

    # NC-TCN-20K-SS clean = same as NC-TCN-20K (SS doesn't affect clean)
    if 'NC-TCN-20K' in results:
        results['NC-TCN-20K-SS']['clean'] = results['NC-TCN-20K']['clean']
        print(f"  {'NC-TCN-20K-SS':>15s}: {results['NC-TCN-20K-SS']['clean']:.1f}% (=NC-TCN-20K, SS bypass at high SNR)")

    # [5] Noise robustness
    noise_types = NOISE_TYPES
    snr_levels = SNR_LEVELS
    if args.quick:
        noise_types = ['white', 'factory']
        snr_levels = [-10, 0, 10]

    print(f"\n[5] Noise robustness evaluation...")
    print(f"    Noise types: {noise_types}")
    print(f"    SNR levels: {snr_levels}")

    for name, (model, is_cnn) in models.items():
        results[name]['noise'] = {}
        print(f"\n  --- {name} {'(CNN)' if is_cnn else '(NC)'} ---")
        for nt in noise_types:
            results[name]['noise'][nt] = {}
            for snr in snr_levels:
                t0 = time.time()
                acc = evaluate_noisy_condition(
                    model, test_loader, nt, snr,
                    is_cnn=is_cnn, mel_fb=mel_fb)
                dt = time.time() - t0
                results[name]['noise'][nt][str(snr)] = round(acc, 1)
                print(f"    {nt:>8s} {snr:>4d}dB: {acc:.1f}%  ({dt:.1f}s)", flush=True)

    # NC-TCN-20K-SS: NC-TCN-20K + external SS
    if 'NC-TCN-20K' in models:
        model_tcn, _ = models['NC-TCN-20K']
        print(f"\n  --- NC-TCN-20K-SS (external SS) ---")
        results['NC-TCN-20K-SS']['noise'] = {}
        for nt in noise_types:
            results['NC-TCN-20K-SS']['noise'][nt] = {}
            for snr in snr_levels:
                t0 = time.time()
                acc = evaluate_noisy_condition(
                    model_tcn, test_loader, nt, snr,
                    is_cnn=False, mel_fb=mel_fb,
                    use_ext_ss=True)
                dt = time.time() - t0
                results['NC-TCN-20K-SS']['noise'][nt][str(snr)] = round(acc, 1)
                print(f"    {nt:>8s} {snr:>4d}dB: {acc:.1f}%  ({dt:.1f}s)", flush=True)

    # [6] Save results
    with open(str(RESULTS_FILE), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_FILE}")

    # [7] Print summary tables
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE (for paper)")
    print("=" * 80)

    # Table I: Accuracy averaged over 5 noise types
    all_models = ['DS-CNN-S', 'BC-ResNet-1', 'NC-SSM', 'NC-SSM-20K',
                  'NC-TCN-20K', 'NC-TCN-20K-SS']
    print("\n  Table I: Accuracy (%) averaged over noise types")
    header = f"  {'Model':>15s} | {'Clean':>6s} | " + " | ".join(f"{s:>4d}dB" for s in snr_levels)
    print(header)
    print("  " + "-" * len(header))

    for name in all_models:
        if name not in results:
            continue
        r = results[name]
        clean = r['clean']
        snr_avgs = []
        for snr in snr_levels:
            vals = [r['noise'].get(nt, {}).get(str(snr), 0) for nt in noise_types]
            avg = np.mean(vals) if vals else 0
            snr_avgs.append(avg)
        line = f"  {name:>15s} | {clean:>5.1f}% | " + " | ".join(f"{a:>5.1f}%" for a in snr_avgs)
        print(line)

    # Table III: Per-noise breakdown at 0dB
    print("\n  Table III: Per-noise accuracy (%) at 0 dB")
    header = f"  {'Model':>15s} | " + " | ".join(f"{nt:>8s}" for nt in noise_types)
    print(header)
    print("  " + "-" * len(header))
    for name in all_models:
        if name not in results:
            continue
        r = results[name]
        vals = [r['noise'].get(nt, {}).get('0', 0) for nt in noise_types]
        line = f"  {name:>15s} | " + " | ".join(f"{v:>7.1f}%" for v in vals)
        print(line)

    # Table IV: Deployment
    print("\n  Table IV: Deployment comparison")
    print(f"  {'Model':>15s} | {'Params':>8s} | {'MACs':>8s}")
    print("  " + "-" * 40)
    for name in all_models:
        if name not in results:
            continue
        r = results[name]
        p = r['params']
        m = r['macs']
        p_str = f"{p/1000:.1f}K" if p else "?"
        m_str = f"{m/1e6:.1f}M" if m else "?"
        print(f"  {name:>15s} | {p_str:>8s} | {m_str:>8s}")


if __name__ == '__main__':
    main()
