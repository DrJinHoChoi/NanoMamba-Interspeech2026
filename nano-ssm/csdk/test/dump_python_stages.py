#!/usr/bin/env python3
"""
Dump Python model intermediate values for C SDK comparison.
Saves each stage output as raw float32 files.
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2].parent))
from nanomamba import create_nanomamba_nc_20k

CKPT = str(Path(__file__).parents[2].parent / 'checkpoints_full' / 'NanoMamba-NC-20K' / 'best.pt')
OUT_DIR = Path(__file__).parent / 'debug_data'
OUT_DIR.mkdir(exist_ok=True)

def save(name, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = arr.astype(np.float32)
    arr.flatten().tofile(str(OUT_DIR / f'{name}.raw'))
    print(f"  {name}: shape={arr.shape}, range=[{arr.min():.6f}, {arr.max():.6f}], mean={arr.mean():.6f}")

def main():
    model = create_nanomamba_nc_20k()
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Generate test audio (same as C test: seed=42 noise)
    np.random.seed(42)
    audio_np = np.random.randn(16000).astype(np.float32) * 0.01
    audio = torch.from_numpy(audio_np).unsqueeze(0)

    save('audio', audio_np)

    print("\n=== Stage-by-stage Python outputs ===\n")

    with torch.no_grad():
        # 1. STFT
        spec = torch.stft(audio, n_fft=512, hop_length=160, win_length=512,
                         window=torch.hann_window(512), return_complex=True, center=True)
        mag = spec.abs()[0]  # (257, 101)
        save('stft_mag', mag)

        # 2. Full feature extraction
        mel, snr_mel = model.extract_features(audio)
        save('mel_after_frontend', mel[0])
        save('snr_mel', snr_mel[0])

        # Get intermediate values manually
        # Raw mel (before any gating/PCEN)
        mel_fb = model.mel_fb
        raw_mel = torch.matmul(mel_fb, mag)  # (40, 101)
        save('raw_mel', raw_mel)

        # SNR from estimator
        snr_from_est = model.snr_estimator(mag.unsqueeze(0), mel_fb)
        save('snr_estimator_output', snr_from_est[0])

        # After LSG
        mel_for_gate = raw_mel.clone().unsqueeze(0)
        snr_for_gate = snr_from_est.clone()
        gated = model.spectral_gate(mel_for_gate, snr_for_gate)
        save('after_lsg', gated[0])

        # After freq floor + PCEN + InstNorm (full frontend)
        # We need these intermediate states. Let's trace them manually:
        mel_trace = raw_mel.clone().unsqueeze(0)  # (1, 40, 101)
        snr_trace = snr_from_est.clone()
        # LSG
        mel_trace = model.spectral_gate(mel_trace, snr_trace)
        # Freq floor
        if hasattr(model, 'freq_dep_floor'):
            mel_trace = model.freq_dep_floor(mel_trace)
        save('after_freqfloor', mel_trace[0])
        # DualPCEN
        mel_before_pcen = mel_trace.clone()
        mel_trace = model.dual_pcen(mel_trace, snr_mel=snr_trace)
        save('after_pcen', mel_trace[0])
        # InstanceNorm
        mel_trace = model.input_norm(mel_trace)
        save('after_instnorm', mel_trace[0])

        # 3. Transpose + Patch projection
        x_seq = mel[0].T  # (T, 40)
        snr_seq = snr_mel[0].T  # (T, 40)
        save('x_seq', x_seq)
        save('snr_seq', snr_seq)

        feat = model.patch_proj(x_seq.unsqueeze(0))  # (1, T, 37)
        save('patch_proj', feat[0])

        # 4. Block 0
        block = model.blocks[0]
        residual = feat.clone()

        # LayerNorm
        x_norm = block.norm(feat)
        save('block0_layernorm', x_norm[0])

        # In projection
        xz = block.in_proj(x_norm)
        x_branch = xz[..., :55]
        z = xz[..., 55:]
        save('block0_x_branch', x_branch[0])
        save('block0_z', z[0])

        # DWConv
        x_t = x_branch.transpose(1, 2)  # (1, 55, T)
        x_conv = block.conv1d(x_t)  # (1, 55, T)
        save('block0_conv_out', x_conv[0])

        # SiLU (trim to match input length)
        x_silu = torch.nn.functional.silu(x_conv)
        T_conv = x_silu.shape[2]
        T_in = x_branch.shape[1]
        if T_conv > T_in:
            x_silu = x_silu[:, :, :T_in]
        x_silu_t = x_silu.transpose(1, 2)  # (1, T, 55)
        save('block0_silu', x_silu_t[0])

        pcen_gate = model.get_routing_gate(per_frame=True)
        save('pcen_gate', pcen_gate)

        # ── Block 0 internals (manual step-by-step) ──
        # Use hooks to get SSM internal values
        import types

        # Capture SSM scan output
        ssm_capture = {}
        orig_forward = block.sa_ssm.forward

        def hooked_forward(self_ssm, *args, **kwargs):
            result = orig_forward(*args, **kwargs)
            ssm_capture['ssm_out'] = result.clone()
            return result

        block.sa_ssm.forward = types.MethodType(lambda self, *a, **kw: hooked_forward(self, *a, **kw), block.sa_ssm)

        # Run block forward
        block0_out = block(feat, snr_seq.unsqueeze(0), pcen_gate=pcen_gate)
        save('block0_output', block0_out[0])

        # Now run step-by-step manually for detailed comparison
        x_in = feat.clone()
        residual2 = x_in.clone()
        x_norm = block.norm(x_in)
        save('block0_norm', x_norm[0])

        xz = block.in_proj(x_norm)  # (1, T, 110)
        x_b = xz[..., :55]
        z_b = xz[..., 55:]

        # DWConv (needs channel-first)
        x_bt = x_b.transpose(1, 2)  # (1, 55, T)
        x_conv = block.conv1d(x_bt)  # (1, 55, T+pad)
        # Trim to match input length
        T_in = x_b.shape[1]
        if x_conv.shape[2] > T_in:
            x_conv = x_conv[:, :, :T_in]
        x_act = torch.nn.functional.silu(x_conv)
        x_act_t = x_act.transpose(1, 2)  # (1, T, 55)
        save('block0_ssm_input', x_act_t[0])

        # SSM output (from the hooked forward)
        if 'ssm_out' in ssm_capture:
            save('block0_ssm_output', ssm_capture['ssm_out'][0])

        # Gate + out_proj
        gated2 = ssm_capture.get('ssm_out', x_act_t) * torch.nn.functional.silu(z_b)
        save('block0_gated', gated2[0])

        out2 = block.out_proj(gated2) + residual2
        save('block0_full_output', out2[0])

        # Full model output
        logits = model(audio)
        save('logits', logits[0])

        probs = torch.softmax(logits, dim=-1)
        labels = ['yes','no','up','down','left','right','on','off','stop','go','silence','unknown']
        pred = probs[0].argmax().item()
        print(f"\nPython prediction: {labels[pred]} ({probs[0][pred]:.4f})")
        print(f"Python logits: {logits[0].numpy()}")

if __name__ == '__main__':
    main()
