/**
 * DualPCEN v2: Noise-adaptive dual-expert PCEN
 * Full Python-matching implementation:
 *   - Two PCEN experts with SNR-adaptive r and s
 *   - Spectral flatness + tilt routing
 *   - Temporal smoothing (causal K=7)
 *   - TMI (Temporal Modulation Index)
 *   - SNR-conditioned gate temperature
 */

#include "ncssm.h"
#include <math.h>
#include <string.h>

/* PCEN parameters per expert: s, alpha, delta, r (per mel band = 4 * N_MELS) */
#define PCEN_PARAMS_PER_EXPERT (4 * NCSSM_N_MELS)
#define SMOOTH_K 7

/* ================================================================
 * Causal moving average (K=7, matches Python _causal_smooth)
 * in: (T,), out: (T,)
 * ================================================================ */
static void causal_smooth(const float *in, float *out, int T) {
    /* Equivalent to F.conv1d with left-padding K-1 */
    for (int t = 0; t < T; t++) {
        float sum = 0.0f;
        int count = 0;
        for (int k = 0; k < SMOOTH_K; k++) {
            int idx = t - k;
            if (idx >= 0) {
                sum += in[idx];
                count++;
            }
        }
        out[t] = sum / SMOOTH_K;  /* Always divide by K (zero-padded) */
    }
}

/* ================================================================
 * Single PCEN expert forward
 * mel: (N_MELS, T) linear mel input
 * out: (N_MELS, T) PCEN output
 * params: log_s, log_alpha, log_delta, log_r (each N_MELS)
 * snr_mel: (N_MELS, T) for SNR-adaptive compression (can be NULL)
 * ================================================================ */
static void pcen_expert(const float *mel, float *out,
                        const float *params, const float *snr_mel,
                        int n_mels, int T,
                        float delta_lo, float delta_hi) {
    /* Parse params */
    const float *log_s     = &params[0];
    const float *log_alpha = &params[n_mels];
    const float *log_delta = &params[2 * n_mels];
    const float *log_r     = &params[3 * n_mels];

    for (int m = 0; m < n_mels; m++) {
        float s = ncssm_sigmoid(log_s[m]);
        if (s < 0.05f) s = 0.05f; if (s > 0.3f) s = 0.3f;
        float alpha = ncssm_sigmoid(log_alpha[m]);
        if (alpha < 0.9f) alpha = 0.9f; if (alpha > 0.999f) alpha = 0.999f;
        float delta = expf(log_delta[m]);
        if (delta < delta_lo) delta = delta_lo; if (delta > delta_hi) delta = delta_hi;
        float r = ncssm_sigmoid(log_r[m]);
        if (r < 0.05f) r = 0.05f; if (r > 0.25f) r = 0.25f;

        /* IIR smoothing */
        float smoother = mel[m * T];  /* init with first frame */
        for (int t = 0; t < T; t++) {
            float x = mel[m * T + t];

            /* SNR-adaptive r and s */
            float r_eff = r, s_eff = s;
            if (snr_mel != NULL) {
                float snr = snr_mel[m * T + t];
                r_eff = r * (1.0f + 0.5f * (1.0f - snr));
                if (r_eff > 0.40f) r_eff = 0.40f;
                s_eff = s * (1.0f + 0.3f * (1.0f - snr));
                if (s_eff > 0.40f) s_eff = 0.40f;
            }

            smoother = (1.0f - s_eff) * smoother + s_eff * x;
            if (smoother < 1e-5f) smoother = 1e-5f;

            float gain = powf(1e-6f + smoother, -alpha);
            if (gain > 1e5f) gain = 1e5f;

            float val = powf(x * gain + delta, r_eff) - powf(delta, r_eff);
            if (isnan(val)) val = 0.0f;
            out[m * T + t] = val;
        }
    }
}

/* ================================================================
 * DualPCEN v2 forward (full Python-matching implementation)
 *
 * mel: (N_MELS, T) linear mel input, MODIFIED in place
 * snr_mel: (N_MELS, T) per-mel-band SNR [0,1]
 * pcen_weights: expert0 params (4*40) + expert1 params (4*40) + gate_temp (1)
 * pcen_gate: (T,) output per-frame routing gate for SSM conditioning
 * ================================================================ */
void ncssm_dual_pcen_v2(float *mel, const float *snr_mel,
                        const float *pcen_weights, float *pcen_gate,
                        int n_mels, int n_frames) {
    int T = n_frames;

    /* Expert weights */
    const float *expert0_params = &pcen_weights[0];                    /* nonstat */
    const float *expert1_params = &pcen_weights[PCEN_PARAMS_PER_EXPERT]; /* stat */
    float gate_temp = pcen_weights[2 * PCEN_PARAMS_PER_EXPERT];

    /* Run both experts (with SNR-adaptive compression) */
    float out0[NCSSM_N_MELS * NCSSM_N_FRAMES];  /* nonstat */
    float out1[NCSSM_N_MELS * NCSSM_N_FRAMES];  /* stat */
    pcen_expert(mel, out0, expert0_params, snr_mel, n_mels, T, 0.5f, 5.0f);
    pcen_expert(mel, out1, expert1_params, snr_mel, n_mels, T, 0.001f, 0.1f);

    /* ── Step 1: Compute per-frame raw spectral flatness SF ── */
    float sf_raw[NCSSM_N_FRAMES];
    float frame_energy[NCSSM_N_FRAMES];
    for (int t = 0; t < T; t++) {
        float log_sum = 0.0f, arith_sum = 0.0f;
        float e_sum = 0.0f;
        for (int m = 0; m < n_mels; m++) {
            float v = mel[m * T + t];
            if (v < 1e-4f) v = 1e-4f;
            log_sum += logf(v);
            arith_sum += v;
            e_sum += v;
        }
        float geo = expf(log_sum / n_mels);
        float arith = arith_sum / n_mels;
        float sf = geo / (arith + 1e-8f);
        if (sf < 0.0f) sf = 0.0f;
        if (sf > 1.0f) sf = 1.0f;
        sf_raw[t] = sf;
        frame_energy[t] = e_sum / n_mels;  /* mean energy per frame */
    }

    /* ── Step 2: Temporal smoothing of SF (K=7 causal MA) ── */
    float sf_smooth[NCSSM_N_FRAMES];
    causal_smooth(sf_raw, sf_smooth, T);

    /* ── Step 3: Spectral tilt (per-frame) ── */
    float tilt[NCSSM_N_FRAMES];
    int n_low = n_mels / 3;
    int n_high = n_mels - 2 * n_mels / 3;
    for (int t = 0; t < T; t++) {
        float low_e = 0.0f, high_e = 0.0f;
        for (int m = 0; m < n_low; m++) {
            float v = mel[m * T + t]; if (v < 1e-4f) v = 1e-4f;
            low_e += v;
        }
        for (int m = 2 * n_mels / 3; m < n_mels; m++) {
            float v = mel[m * T + t]; if (v < 1e-4f) v = 1e-4f;
            high_e += v;
        }
        low_e /= n_low; high_e /= n_high;
        float t_val = low_e / (low_e + high_e + 1e-8f);
        if (t_val > 1.0f) t_val = 1.0f;
        tilt[t] = t_val;
    }

    /* ── Step 4: SF + Tilt correction → sf_adjusted ── */
    float sf_adj[NCSSM_N_FRAMES];
    for (int t = 0; t < T; t++) {
        float relu_tilt = (tilt[t] - 0.6f) > 0.0f ? (tilt[t] - 0.6f) : 0.0f;
        sf_adj[t] = sf_smooth[t] + (1.0f - sf_smooth[t]) * relu_tilt;
    }

    /* ── Step 5: TMI (Temporal Modulation Index) ── */
    float ema_E[NCSSM_N_FRAMES], ema_E2[NCSSM_N_FRAMES];
    float fe_sq[NCSSM_N_FRAMES];
    for (int t = 0; t < T; t++) fe_sq[t] = frame_energy[t] * frame_energy[t];
    causal_smooth(frame_energy, ema_E, T);
    causal_smooth(fe_sq, ema_E2, T);

    float tmi_raw[NCSSM_N_FRAMES], tmi[NCSSM_N_FRAMES];
    for (int t = 0; t < T; t++) {
        float variance = ema_E2[t] - ema_E[t] * ema_E[t];
        if (variance < 0.0f) variance = 0.0f;
        float ema_e_safe = ema_E[t]; if (ema_e_safe < 1e-5f) ema_e_safe = 1e-5f;
        float tmi_val = sqrtf(variance + 1e-6f) / (ema_e_safe + 1e-8f);
        if (tmi_val > 2.0f) tmi_val = 2.0f;
        tmi_raw[t] = tmi_val / 2.0f;  /* normalize to [0,1] */
    }
    causal_smooth(tmi_raw, tmi, T);

    /* TMI correction: low TMI → temporally stationary → boost toward stat expert */
    float routing[NCSSM_N_FRAMES];
    for (int t = 0; t < T; t++) {
        float tmi_boost = (0.5f - tmi[t]) > 0.0f ? (0.5f - tmi[t]) * 0.5f : 0.0f;
        routing[t] = sf_adj[t] + (1.0f - sf_adj[t]) * tmi_boost;
    }

    /* ── Step 6: SNR-conditioned gate temperature ── */
    /* snr_global = mean(snr_mel) over all bands and frames */
    float snr_global = 0.0f;
    if (snr_mel != NULL) {
        for (int i = 0; i < n_mels * T; i++) snr_global += snr_mel[i];
        snr_global /= (n_mels * T);
    } else {
        snr_global = 0.5f;
    }
    /* Low SNR → sharper routing, High SNR → softer */
    float snr_scale = 1.0f + 2.0f * (1.0f - snr_global);
    float effective_temp = gate_temp * snr_scale;

    /* ── Step 7: Gate + blend ── */
    for (int t = 0; t < T; t++) {
        float gate = ncssm_sigmoid(effective_temp * (routing[t] - 0.5f));
        /* NaN safety */
        if (isnan(gate)) gate = 0.5f;
        if (pcen_gate) pcen_gate[t] = gate;

        /* Blend: gate * stat_expert + (1-gate) * nonstat_expert */
        for (int m = 0; m < n_mels; m++) {
            int idx = m * T + t;
            mel[idx] = gate * out1[idx] + (1.0f - gate) * out0[idx];
            if (isnan(mel[idx])) mel[idx] = 0.0f;
        }
    }
}
