#!/usr/bin/env python3
"""
Export NC-SSM PyTorch weights to C header file.
Generates ncssm_weights.h with all model parameters as float arrays.

Usage:
    python export_weights.py --checkpoint ../../checkpoints_full/NanoMamba-NC-20K/best.pt
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2].parent))
from nanomamba import (NanoMamba, create_nanomamba_nc_matched,
                       create_nanomamba_nc_large, create_nanomamba_nc_20k,
                       create_nanomamba_nc_15k)

FACTORY_MAP = {
    'NanoMamba-NC-Matched': create_nanomamba_nc_matched,
    'NC-SSM': create_nanomamba_nc_matched,
    'NanoMamba-NC-Large': create_nanomamba_nc_large,
    'NanoMamba-NC-Large-NASG': lambda: create_nanomamba_nc_large(use_nasg=True),
    'NanoMamba-NC-15K': create_nanomamba_nc_15k,
    'NanoMamba-NC-20K': create_nanomamba_nc_20k,
}


def export_weights(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_name = ckpt.get('model_name', 'NanoMamba-NC-20K')
    print(f"Model: {model_name}")

    factory = FACTORY_MAP.get(model_name, create_nanomamba_nc_20k)
    model = factory()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Collect all weights in order
    weights = []
    offsets = {}
    offset = 0

    def add_weight(name, tensor):
        nonlocal offset
        data = tensor.detach().cpu().numpy().flatten()
        offsets[name] = offset
        weights.append(data)
        offset += len(data)
        return data

    # ── SNR Estimator ──
    add_weight('snr_noise_scale', model.snr_estimator.noise_scale)
    add_weight('snr_floor', model.snr_estimator.floor)
    # Running EMA parameters (beta = sigmoid(raw_beta), gamma = sigmoid(raw_gamma))
    if model.snr_estimator.use_running_ema:
        add_weight('snr_raw_beta', model.snr_estimator.raw_beta)
        add_weight('snr_raw_gamma', model.snr_estimator.raw_gamma)
    else:
        # Placeholder values (won't be used)
        add_weight('snr_raw_beta', torch.tensor(-2.2))
        add_weight('snr_raw_gamma', torch.tensor(-3.0))

    # ── DualPCEN v2 ──
    pcen = model.dual_pcen
    # Expert 0 (nonstat): log_s, log_alpha, log_delta, log_r
    add_weight('pcen_expert0_log_s', pcen.pcen_nonstat.log_s)
    add_weight('pcen_expert0_log_alpha', pcen.pcen_nonstat.log_alpha)
    add_weight('pcen_expert0_log_delta', pcen.pcen_nonstat.log_delta)
    add_weight('pcen_expert0_log_r', pcen.pcen_nonstat.log_r)
    # Expert 1 (stat)
    add_weight('pcen_expert1_log_s', pcen.pcen_stat.log_s)
    add_weight('pcen_expert1_log_alpha', pcen.pcen_stat.log_alpha)
    add_weight('pcen_expert1_log_delta', pcen.pcen_stat.log_delta)
    add_weight('pcen_expert1_log_r', pcen.pcen_stat.log_r)
    # Gate temp
    add_weight('pcen_gate_temp', pcen.gate_temp.unsqueeze(0))
    # FreqDependentFloor (register_buffer, but include for completeness)
    add_weight('freq_floor', model.freq_dep_floor.freq_floor.flatten())

    # ── Learned Spectral Gate ──
    add_weight('lsg_w', model.spectral_gate.w)
    add_weight('lsg_b', model.spectral_gate.b)
    add_weight('lsg_floor', model.spectral_gate.floor_raw)

    # ── Patch Projection ──
    add_weight('patch_proj_w', model.patch_proj.weight)
    add_weight('patch_proj_b', model.patch_proj.bias)

    # ── Blocks ──
    for blk_idx in range(2):
        block = model.blocks[blk_idx]
        prefix = f'block{blk_idx}'

        # LayerNorm
        add_weight(f'{prefix}_norm_g', block.norm.weight)
        add_weight(f'{prefix}_norm_b', block.norm.bias)

        # In projection (no bias)
        add_weight(f'{prefix}_in_proj', block.in_proj.weight)

        # DWConv1d
        add_weight(f'{prefix}_conv_w', block.conv1d.weight)
        add_weight(f'{prefix}_conv_b', block.conv1d.bias)

        # NC-SSM weights
        ssm = block.sa_ssm
        ssm_start = offset

        add_weight(f'{prefix}_ssm_x_proj_w', ssm.x_proj.weight)
        add_weight(f'{prefix}_ssm_snr_proj_w', ssm.snr_proj.weight)
        add_weight(f'{prefix}_ssm_snr_proj_b', ssm.snr_proj.bias)
        add_weight(f'{prefix}_ssm_dt_proj_w', ssm.dt_proj.weight)
        add_weight(f'{prefix}_ssm_dt_proj_b', ssm.dt_proj.bias)
        add_weight(f'{prefix}_ssm_A_log', ssm.A_log)
        add_weight(f'{prefix}_ssm_D', ssm.D)
        add_weight(f'{prefix}_ssm_alpha', ssm.alpha.unsqueeze(0))
        add_weight(f'{prefix}_ssm_sel_scale', ssm.sel_scale.unsqueeze(0))
        add_weight(f'{prefix}_ssm_sel_bias_dt', ssm.sel_bias_dt.unsqueeze(0))
        add_weight(f'{prefix}_ssm_sel_bias_BC', ssm.sel_bias_BC)
        add_weight(f'{prefix}_ssm_sel_sub_scale', ssm.sel_sub_scale)
        add_weight(f'{prefix}_ssm_dt_base', ssm.dt_base.unsqueeze(0) if ssm.dt_base.dim() == 0 else ssm.dt_base)
        add_weight(f'{prefix}_ssm_B_base', ssm.B_base)
        add_weight(f'{prefix}_ssm_C_base', ssm.C_base)
        add_weight(f'{prefix}_ssm_dt_station_alpha',
                   ssm.dt_station_alpha.unsqueeze(0) if ssm.dt_station_alpha.dim() == 0 else ssm.dt_station_alpha)
        add_weight(f'{prefix}_ssm_B_sf_scale', ssm.B_sf_scale)
        add_weight(f'{prefix}_ssm_sigma_pcen_mod',
                   ssm.sigma_pcen_mod.unsqueeze(0) if ssm.sigma_pcen_mod.dim() == 0 else ssm.sigma_pcen_mod)

        offsets[f'{prefix}_ssm'] = ssm_start

        # Out projection (no bias)
        add_weight(f'{prefix}_out_proj', block.out_proj.weight)

    # ── Final Norm ──
    add_weight('final_norm_g', model.final_norm.weight)
    add_weight('final_norm_b', model.final_norm.bias)

    # ── Classifier ──
    add_weight('classifier_w', model.classifier.weight)
    add_weight('classifier_b', model.classifier.bias)

    # ── Mel filterbank (constant) ──
    mel_fb = model.mel_fb.numpy()

    # Concatenate all weights
    all_weights = np.concatenate(weights)
    print(f"Total weights: {len(all_weights):,} floats ({len(all_weights)*4/1024:.1f} KB)")
    print(f"Mel filterbank: {mel_fb.size:,} floats ({mel_fb.size*4/1024:.1f} KB)")

    # Write C header
    with open(output_path, 'w') as f:
        f.write("/**\n")
        f.write(" * NC-SSM-20K Weights - Auto-generated\n")
        f.write(f" * Source: {checkpoint_path}\n")
        f.write(f" * Total: {len(all_weights):,} floats ({len(all_weights)*4/1024:.1f} KB)\n")
        f.write(" * DO NOT EDIT\n")
        f.write(" */\n\n")
        f.write("#ifndef NCSSM_WEIGHTS_H\n")
        f.write("#define NCSSM_WEIGHTS_H\n\n")

        # Weight offset defines
        f.write("/* Weight offsets */\n")
        offset_map = {
            'NCSSM_W_SNR_NOISE_SCALE': 'snr_noise_scale',
            'NCSSM_W_SNR_RAW_BETA': 'snr_raw_beta',
            'NCSSM_W_SNR_RAW_GAMMA': 'snr_raw_gamma',
            'NCSSM_W_SNR_FLOOR': 'snr_floor',
            'NCSSM_W_PCEN': 'pcen_expert0_log_s',
            'NCSSM_W_FREQ_FLOOR': 'freq_floor',
            'NCSSM_W_LSG_W': 'lsg_w',
            'NCSSM_W_LSG_B': 'lsg_b',
            'NCSSM_W_LSG_FLOOR': 'lsg_floor',
            'NCSSM_W_PATCH_PROJ_W': 'patch_proj_w',
            'NCSSM_W_PATCH_PROJ_B': 'patch_proj_b',
            'NCSSM_W_BLOCK0_NORM_G': 'block0_norm_g',
            'NCSSM_W_BLOCK0_NORM_B': 'block0_norm_b',
            'NCSSM_W_BLOCK0_IN_PROJ': 'block0_in_proj',
            'NCSSM_W_BLOCK0_CONV_W': 'block0_conv_w',
            'NCSSM_W_BLOCK0_CONV_B': 'block0_conv_b',
            'NCSSM_W_BLOCK0_SSM': 'block0_ssm',
            'NCSSM_W_BLOCK0_OUT_PROJ': 'block0_out_proj',
            'NCSSM_W_BLOCK1_NORM_G': 'block1_norm_g',
            'NCSSM_W_BLOCK1_NORM_B': 'block1_norm_b',
            'NCSSM_W_BLOCK1_IN_PROJ': 'block1_in_proj',
            'NCSSM_W_BLOCK1_CONV_W': 'block1_conv_w',
            'NCSSM_W_BLOCK1_CONV_B': 'block1_conv_b',
            'NCSSM_W_BLOCK1_SSM': 'block1_ssm',
            'NCSSM_W_BLOCK1_OUT_PROJ': 'block1_out_proj',
            'NCSSM_W_FINAL_NORM_G': 'final_norm_g',
            'NCSSM_W_FINAL_NORM_B': 'final_norm_b',
            'NCSSM_W_CLASSIFIER_W': 'classifier_w',
            'NCSSM_W_CLASSIFIER_B': 'classifier_b',
        }
        for cname, pyname in offset_map.items():
            f.write(f"const int {cname} = {offsets[pyname]};\n")

        # Mel filterbank
        f.write(f"\n/* Mel filterbank ({mel_fb.shape[0]}x{mel_fb.shape[1]}) */\n")
        f.write(f"const float ncssm_mel_fb[{mel_fb.size}] = {{\n")
        for i, v in enumerate(mel_fb.flatten()):
            if i % 10 == 0: f.write("  ")
            f.write(f"{v:.8f}f,")
            if i % 10 == 9: f.write("\n")
        f.write("\n};\n\n")

        # All weights
        f.write(f"/* Model weights ({len(all_weights)} floats) */\n")
        f.write(f"const float ncssm_weights_data[{len(all_weights)}] = {{\n")
        for i, v in enumerate(all_weights):
            if i % 8 == 0: f.write("  ")
            f.write(f"{v:.8f}f,")
            if i % 8 == 7: f.write("\n")
        f.write("\n};\n\n")

        f.write("#endif /* NCSSM_WEIGHTS_H */\n")

    # Also generate config header with actual model dimensions
    d_model = model.d_model
    d_inner = int(d_model * 1.5)
    d_state = model.blocks[0].sa_ssm.d_state
    config_path = output_path.replace('weights', 'config')
    with open(config_path, 'w') as f:
        f.write("/* Auto-generated config for %s */\n" % model_name)
        f.write("#ifndef NCSSM_CONFIG_H\n#define NCSSM_CONFIG_H\n\n")
        f.write(f"#define NCSSM_SR            16000\n")
        f.write(f"#define NCSSM_AUDIO_LEN     16000\n")
        f.write(f"#define NCSSM_N_FFT         512\n")
        f.write(f"#define NCSSM_HOP_LENGTH    160\n")
        f.write(f"#define NCSSM_N_FREQ        257\n")
        f.write(f"#define NCSSM_N_FRAMES      100\n")
        f.write(f"#define NCSSM_N_MELS        40\n")
        f.write(f"#define NCSSM_N_CLASSES     12\n")
        f.write(f"#define NCSSM_D_MODEL       {d_model}\n")
        f.write(f"#define NCSSM_D_INNER       {d_inner}\n")
        f.write(f"#define NCSSM_D_STATE       {d_state}\n")
        f.write(f"#define NCSSM_D_CONV        3\n")
        f.write(f"#define NCSSM_N_LAYERS      2\n")
        f.write(f"#define NCSSM_EXPAND        1.5f\n")
        f.write(f"#define NCSSM_DELTA_FLOOR_MIN 0.03f\n")
        f.write(f"#define NCSSM_DELTA_FLOOR_MAX 0.15f\n")
        f.write(f"#define NCSSM_EPSILON_MIN   0.05f\n")
        f.write(f"#define NCSSM_EPSILON_MAX   0.30f\n")
        f.write(f"#define NCSSM_BGATE_FLOOR   0.20f\n")
        f.write(f"#define NCSSM_SNR_HALF_SAT  0.05f\n")
        f.write(f"#define NCSSM_NOISE_FRAMES  5\n")
        f.write(f"#define NCSSM_PCEN_EXPERTS  2\n")
        f.write(f"#define NCSSM_LUT_SIZE      256\n")
        f.write(f"#define NCSSM_LUT_RANGE     8.0f\n")
        f.write(f"#define NCSSM_SSM_STATE_SIZE (NCSSM_D_INNER * NCSSM_D_STATE)\n")
        f.write(f"#define NCSSM_MEL_BUF_SIZE  (NCSSM_N_MELS * NCSSM_N_FRAMES)\n")
        f.write(f"#define NCSSM_FEAT_BUF_SIZE (NCSSM_D_MODEL * NCSSM_N_FRAMES)\n")
        f.write(f"\n#endif\n")
    print(f"Config: {config_path} (d_model={d_model}, d_state={d_state})")

    print(f"\nExported to: {output_path}")
    print(f"Weight offsets: {len(offset_map)} entries")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='../../checkpoints_full/NanoMamba-NC-20K/best.pt')
    parser.add_argument('--output', type=str,
                        default='../include/ncssm_weights.h')
    args = parser.parse_args()
    export_weights(args.checkpoint, args.output)
