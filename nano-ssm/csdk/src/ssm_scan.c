/**
 * NC-SSM Core Scan Loop
 * The heart of the model: sequential state update
 *
 * h[d][n] = dA[d][n] * h[d][n] + dBx[d][n] + eps * x[d]
 * y[d] = sum_n(h[d][n] * C[n]) + D[d] * x[d]
 *
 * This is THE function to optimize for edge deployment.
 * On Cortex-M7: ~1650 MACs per timestep × 100 steps = 165K MACs total
 */

#include "ncssm.h"
#include <math.h>
#include <string.h>

/* ================================================================
 * NC-SSM Scan: full block computation
 *
 * Inputs:
 *   x:       (L, D_INNER) feature sequence after conv+silu
 *   snr_mel: (L, N_MELS) per-mel-band SNR [0,1]
 *   weights: pointer to weight struct for this block's SSM
 *
 * Outputs:
 *   y:       (L, D_INNER) SSM output
 *   h:       (D_INNER, D_STATE) persistent state (streaming)
 *
 * Weight layout (contiguous):
 *   x_proj_w:   (2*D_STATE+1, D_INNER) = (21, 55)
 *   snr_proj_w:  (D_STATE+1, N_MELS)   = (11, 40)
 *   snr_proj_b:  (D_STATE+1)            = (11,)
 *   dt_proj_w:   (D_INNER, 1)           = (55,)
 *   dt_proj_b:   (D_INNER,)             = (55,)
 *   A_log:       (D_INNER, D_STATE)     = (55, 10)
 *   D_param:     (D_INNER,)             = (55,)
 *   alpha:       (1,)
 *   sel_scale:   (1,)
 *   sel_bias_dt: (1,)
 *   sel_bias_BC: (D_STATE,)             = (10,)
 *   sel_sub_scale: (D_STATE,)           = (10,)
 *   dt_base:     (1,)
 *   B_base:      (D_STATE,)             = (10,)
 *   C_base:      (D_STATE,)             = (10,)
 *   dt_station_alpha: (1,)
 *   B_sf_scale:  (D_STATE,)             = (10,)
 *   sigma_pcen_mod: (1,)
 *   snr_half_sat: (1,) [buffer]
 * ================================================================ */

/* Helper: clamp */
static inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

void ncssm_ssm_scan(const float *x, const float *snr_mel,
                    float *y, float *h,
                    const float *w, int L,
                    const float *pcen_gate) {
    const int D = NCSSM_D_INNER;   /* 55 */
    const int N = NCSSM_D_STATE;   /* 10 */
    const int M = NCSSM_N_MELS;    /* 40 */

    /* Parse weight pointers */
    const float *x_proj_w   = w;              w += (2*N+1) * D;
    const float *snr_proj_w = w;              w += (N+1) * M;
    const float *snr_proj_b = w;              w += (N+1);
    const float *dt_proj_w  = w;              w += D;
    const float *dt_proj_b  = w;              w += D;
    const float *A_log      = w;              w += D * N;
    const float *D_param    = w;              w += D;
    const float *alpha_p    = w;              w += 1;
    const float *sel_scale_p = w;             w += 1;
    const float *sel_bias_dt_p = w;           w += 1;
    const float *sel_bias_BC = w;             w += N;
    const float *sel_sub_scale = w;           w += N;
    const float *dt_base_p  = w;              w += 1;
    const float *B_base     = w;              w += N;
    const float *C_base     = w;              w += N;
    const float *dt_station_alpha_p = w;      w += 1;
    const float *B_sf_scale = w;              w += N;

    float alpha = *alpha_p;
    float sel_scale = *sel_scale_p;
    float sel_bias_dt = *sel_bias_dt_p;
    float dt_base = *dt_base_p;
    float dt_station_alpha = clampf(*dt_station_alpha_p, -1.0f, 1.0f);

    /* Precompute A = -exp(A_log) */
    float A[NCSSM_D_INNER * NCSSM_D_STATE];
    for (int i = 0; i < D * N; i++) {
        A[i] = -expf(A_log[i]);
    }

    /* Temporary buffers (stack-allocated for Cortex-M) */
    float x_proj[2 * NCSSM_D_STATE + 1];     /* 21 */
    float snr_mod[NCSSM_D_STATE + 1];        /* 11 */
    float snr_sub[NCSSM_D_STATE];            /* 10 */
    float snr_smooth_bc[NCSSM_D_STATE];      /* 10 */

    /* Running causal smoothing buffers (3-frame) */
    float snr_dt_hist[3] = {0};
    float snr_bc_hist[3][NCSSM_D_STATE] = {{0}};
    int hist_idx = 0;

    /* Main scan loop */
    for (int t = 0; t < L; t++) {
        const float *xt = &x[t * D];
        const float *snr_t = &snr_mel[t * M];

        /* ── 1. Selective parameter projection ── */
        /* x_proj = x_proj_w @ x  → (21,) */
        for (int o = 0; o < 2*N+1; o++) {
            float sum = 0.0f;
            for (int i = 0; i < D; i++) {
                sum += x_proj_w[o * D + i] * xt[i];
            }
            x_proj[o] = sum;
        }
        float dt_selective = x_proj[0];
        float *B_selective = &x_proj[1];
        float *C_selective = &x_proj[N + 1];

        /* ── 2. SNR modulation ── */
        /* snr_mod = snr_proj_w @ snr_mel + snr_proj_b → (11,) */
        for (int o = 0; o < N+1; o++) {
            float sum = snr_proj_b[o];
            for (int i = 0; i < M; i++) {
                sum += snr_proj_w[o * M + i] * snr_t[i];
            }
            snr_mod[o] = sum;
        }
        float dt_snr_shift = snr_mod[0];

        /* B_gate = sigmoid(snr_mod[1:]) * (1-bgate_floor) + bgate_floor */
        float B_gate[NCSSM_D_STATE];
        for (int n = 0; n < N; n++) {
            float raw = ncssm_sigmoid(snr_mod[1 + n]);
            B_gate[n] = raw * (1.0f - NCSSM_BGATE_FLOOR) + NCSSM_BGATE_FLOOR;
        }

        /* ── 3. NC-SSM per-sub-band selectivity ── */
        /* Michaelis-Menten re-normalization */
        float snr_internal[NCSSM_N_MELS];
        for (int m = 0; m < M; m++) {
            float s = clampf(snr_t[m], 0.0f, 1.0f);
            snr_internal[m] = s / (s + NCSSM_SNR_HALF_SAT);
        }

        /* Adaptive avg pool: 40 mels → 10 sub-bands */
        int bins_per_sub = M / N;  /* 4 */
        float snr_dt_val = 0.0f;
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int b = 0; b < bins_per_sub; b++) {
                sum += snr_internal[n * bins_per_sub + b];
            }
            snr_sub[n] = sum / bins_per_sub;
            snr_dt_val += snr_sub[n];
        }
        snr_dt_val /= N;

        /* Causal 3-frame smoothing */
        snr_dt_hist[hist_idx % 3] = snr_dt_val;
        for (int n = 0; n < N; n++)
            snr_bc_hist[hist_idx % 3][n] = snr_sub[n];

        float snr_smooth_dt = 0.0f;
        int count = (t < 2) ? (t + 1) : 3;
        for (int i = 0; i < count; i++) snr_smooth_dt += snr_dt_hist[i];
        snr_smooth_dt /= count;

        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int i = 0; i < count; i++) sum += snr_bc_hist[i][n];
            snr_smooth_bc[n] = sum / count;
        }
        hist_idx++;

        /* Selectivity gates */
        float sigma_dt = ncssm_sigmoid(sel_scale * snr_smooth_dt + sel_bias_dt);
        float sigma_BC[NCSSM_D_STATE];
        for (int n = 0; n < N; n++) {
            sigma_BC[n] = ncssm_sigmoid(sel_sub_scale[n] * snr_smooth_bc[n] + sel_bias_BC[n]);
        }

        /* ── 4. Blend selective ↔ fixed ── */
        /* Spectral flatness for B_base modulation */
        float snr_sub_mean = 0.0f;
        for (int n = 0; n < N; n++) snr_sub_mean += snr_smooth_bc[n];
        snr_sub_mean /= N;
        float snr_sub_var = 0.0f;
        for (int n = 0; n < N; n++) {
            float diff = snr_smooth_bc[n] - snr_sub_mean;
            snr_sub_var += diff * diff;
        }
        snr_sub_var /= N;
        float spectral_var = sqrtf(snr_sub_var + 1e-6f) / (fabsf(snr_sub_mean) + 1e-4f);
        if (spectral_var > 1.0f) spectral_var = 1.0f;
        float broadband = 1.0f - spectral_var;

        float dt_raw = sigma_dt * dt_selective + (1.0f - sigma_dt) * dt_base;

        float B_param[NCSSM_D_STATE], C_param[NCSSM_D_STATE];
        for (int n = 0; n < N; n++) {
            float b_mod = B_base[n] * (1.0f + clampf(B_sf_scale[n], -5.0f, 5.0f) * broadband);
            B_param[n] = sigma_BC[n] * B_selective[n] + (1.0f - sigma_BC[n]) * b_mod;
            C_param[n] = sigma_BC[n] * C_selective[n] + (1.0f - sigma_BC[n]) * C_base[n];
        }

        /* ── 5. Delta computation ── */
        float adaptive_floor = NCSSM_DELTA_FLOOR_MIN +
            (NCSSM_DELTA_FLOOR_MAX - NCSSM_DELTA_FLOOR_MIN) * snr_smooth_dt;

        /* PCEN gate conditioning on adaptive floor (NC-2) */
        if (pcen_gate != NULL) {
            float pg = pcen_gate[t];
            if (pg != pg) pg = 0.0f;  /* NaN safety */
            float gate_modulation = 1.0f - 0.4f * pg;
            float station_boost = 1.0f + dt_station_alpha * pg;
            adaptive_floor = adaptive_floor * gate_modulation * station_boost;
        }

        /* dt_proj: scalar → d_inner expansion */
        float dt_input = dt_raw + dt_snr_shift;
        float delta[NCSSM_D_INNER];
        for (int d = 0; d < D; d++) {
            float val = ncssm_softplus(dt_proj_w[d] * dt_input + dt_proj_b[d]);
            if (val > 1.0f) val = 1.0f;
            delta[d] = val + adaptive_floor;
        }

        /* SNR-gated B */
        for (int n = 0; n < N; n++) {
            B_param[n] = B_param[n] * (1.0f - alpha + alpha * B_gate[n]);
        }

        /* Adaptive epsilon */
        float adaptive_eps = NCSSM_EPSILON_MAX -
            (NCSSM_EPSILON_MAX - NCSSM_EPSILON_MIN) * snr_smooth_dt;

        /* ── 6. SSM state update (★ CORE) ── */
        float *yt = &y[t * D];
        for (int d = 0; d < D; d++) {
            /* Precompute dA, dBx for this (d, n) */
            for (int n = 0; n < N; n++) {
                int idx = d * N + n;
                float dA = expf(A[idx] * delta[d]);
                float dB = delta[d] * B_param[n];
                float dBx = dB * xt[d];

                /* h_t = dA * h_{t-1} + dBx + eps * x */
                h[idx] = dA * h[idx] + dBx + adaptive_eps * xt[d];

                /* Clamp for stability */
                if (h[idx] > 1e4f) h[idx] = 1e4f;
                if (h[idx] < -1e4f) h[idx] = -1e4f;
            }

            /* y[d] = sum(h[d,:] * C) + D * x[d] */
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += h[d * N + n] * C_param[n];
            }
            yt[d] = sum + D_param[d] * xt[d];
        }
    }
}
