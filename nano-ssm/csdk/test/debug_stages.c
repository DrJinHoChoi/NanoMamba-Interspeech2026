/**
 * Debug C SDK: dump intermediate values at each stage
 * Compare with dump_python_stages.py output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ncssm.h"
#include "ncssm_weights.h"

/* Print stats for a buffer */
static void print_stats(const char *name, const float *buf, int size) {
    float mn = buf[0], mx = buf[0], sum = 0;
    for (int i = 0; i < size; i++) {
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
        sum += buf[i];
    }
    printf("  %-24s: size=%d, range=[%.6f, %.6f], mean=%.6f\n",
           name, size, mn, mx, sum / size);
}

/* Save raw float32 file */
static void save_raw(const char *name, const float *buf, int size) {
    char path[256];
    snprintf(path, sizeof(path), "test/debug_data/%s_c.raw", name);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(buf, sizeof(float), size, f);
        fclose(f);
    }
}

extern const float ncssm_mel_fb[];
extern const float ncssm_weights_data[];
extern const int NCSSM_W_SNR_NOISE_SCALE;
extern const int NCSSM_W_SNR_RAW_BETA;
extern const int NCSSM_W_SNR_RAW_GAMMA;
extern const int NCSSM_W_SNR_FLOOR;
extern const int NCSSM_W_PCEN;
extern const int NCSSM_W_FREQ_FLOOR;
extern const int NCSSM_W_LSG_W;
extern const int NCSSM_W_LSG_B;
extern const int NCSSM_W_LSG_FLOOR;
extern const int NCSSM_W_PATCH_PROJ_W;
extern const int NCSSM_W_PATCH_PROJ_B;
extern const int NCSSM_W_BLOCK0_NORM_G;
extern const int NCSSM_W_BLOCK0_NORM_B;
extern const int NCSSM_W_BLOCK0_IN_PROJ;
extern const int NCSSM_W_BLOCK0_CONV_W;
extern const int NCSSM_W_BLOCK0_CONV_B;
extern const int NCSSM_W_BLOCK0_SSM;
extern const int NCSSM_W_BLOCK0_OUT_PROJ;
extern const int NCSSM_W_FINAL_NORM_G;
extern const int NCSSM_W_FINAL_NORM_B;
extern const int NCSSM_W_CLASSIFIER_W;
extern const int NCSSM_W_CLASSIFIER_B;

int main(void) {
    const float *W = ncssm_weights_data;
    const int T = NCSSM_AUDIO_LEN / NCSSM_HOP_LENGTH + 1;  /* 101 with center=True */
    const int D = NCSSM_D_MODEL;
    const int DI = NCSSM_D_INNER;
    const int M = NCSSM_N_MELS;

    printf("NC-SSM C SDK Stage Debug\n");
    printf("========================\n");
    printf("T=%d frames (center=True)\n\n", T);

    /* Generate same test audio as Python (seed=42, *0.01) */
    float audio[NCSSM_AUDIO_LEN];
    srand(42);
    for (int i = 0; i < NCSSM_AUDIO_LEN; i++) {
        /* Box-Muller for normal distribution matching numpy */
        float u1 = ((float)rand() / RAND_MAX);
        float u2 = ((float)rand() / RAND_MAX);
        if (u1 < 1e-10f) u1 = 1e-10f;
        audio[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2) * 0.01f;
    }

    /* Actually, let's load from the Python dump */
    FILE *af = fopen("test/debug_data/audio.raw", "rb");
    if (af) {
        fread(audio, sizeof(float), NCSSM_AUDIO_LEN, af);
        fclose(af);
        printf("Loaded audio from Python dump\n");
    } else {
        printf("WARNING: Using C-generated audio (may differ from Python)\n");
    }
    print_stats("audio", audio, NCSSM_AUDIO_LEN);

    /* Stage 1: STFT */
    float mag[NCSSM_N_FREQ * NCSSM_N_FRAMES];
    memset(mag, 0, sizeof(mag));
    ncssm_stft_mag(audio, mag, NCSSM_AUDIO_LEN);
    print_stats("stft_mag", mag, NCSSM_N_FREQ * T);
    save_raw("stft_mag", mag, NCSSM_N_FREQ * T);

    /* Stage 2: Mel Projection */
    float mel[NCSSM_N_MELS * NCSSM_N_FRAMES];
    ncssm_mel_projection(mag, ncssm_mel_fb, mel, T);
    print_stats("raw_mel", mel, M * T);
    save_raw("raw_mel", mel, M * T);

    /* Stage 3: SNR estimation */
    float snr_mel[NCSSM_N_MELS * NCSSM_N_FRAMES];
    ncssm_snr_estimate_ema(mag, ncssm_mel_fb, snr_mel, T,
                           W[NCSSM_W_SNR_NOISE_SCALE],
                           W[NCSSM_W_SNR_RAW_BETA],
                           W[NCSSM_W_SNR_RAW_GAMMA]);
    print_stats("snr_mel", snr_mel, M * T);
    save_raw("snr_mel", snr_mel, M * T);

    /* Stage 4: LSG */
    ncssm_spectral_gate(mel, snr_mel,
                        &W[NCSSM_W_LSG_W], &W[NCSSM_W_LSG_B],
                        &W[NCSSM_W_LSG_FLOOR], T);
    print_stats("after_lsg", mel, M * T);
    save_raw("after_lsg", mel, M * T);

    /* Stage 4b: Freq floor */
    for (int m = 0; m < M; m++)
        for (int t = 0; t < T; t++) {
            float fl = W[NCSSM_W_FREQ_FLOOR + m];
            if (mel[m * T + t] < fl) mel[m * T + t] = fl;
        }

    /* Stage 5: DualPCEN */
    float pcen_gate_buf[NCSSM_N_FRAMES];
    ncssm_dual_pcen_v2(mel, snr_mel, &W[NCSSM_W_PCEN], pcen_gate_buf, M, T);
    print_stats("after_pcen", mel, M * T);
    save_raw("after_pcen", mel, M * T);

    /* Stage 6: Instance Norm */
    ncssm_instance_norm(mel, M, T);
    print_stats("after_instnorm", mel, M * T);
    save_raw("after_instnorm", mel, M * T);

    /* Stage 7: Transpose */
    float x_seq[NCSSM_N_FRAMES * NCSSM_N_MELS];
    float snr_seq[NCSSM_N_FRAMES * NCSSM_N_MELS];
    for (int t = 0; t < T; t++)
        for (int m = 0; m < M; m++) {
            x_seq[t * M + m] = mel[m * T + t];
            snr_seq[t * M + m] = snr_mel[m * T + t];
        }

    /* Stage 8: Patch projection */
    float feat[NCSSM_N_FRAMES * NCSSM_D_MODEL];
    for (int t = 0; t < T; t++) {
        ncssm_linear(&x_seq[t * M],
                     &W[NCSSM_W_PATCH_PROJ_W], &W[NCSSM_W_PATCH_PROJ_B],
                     &feat[t * D], M, D);
    }
    print_stats("patch_proj", feat, T * D);
    save_raw("patch_proj", feat, T * D);

    /* Compare with Python reference */
    float py_patch[NCSSM_N_FRAMES * NCSSM_D_MODEL];
    FILE *pf = fopen("test/debug_data/patch_proj.raw", "rb");
    if (pf) {
        fread(py_patch, sizeof(float), T * D, pf);
        fclose(pf);
        float max_diff = 0;
        for (int i = 0; i < T * D; i++) {
            float d = fabsf(feat[i] - py_patch[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("  >> patch_proj vs Python: max_diff = %.6f\n", max_diff);
    }

    /* ── Block 0 internals ── */
    /* LayerNorm */
    float feat_copy[NCSSM_N_FRAMES * NCSSM_D_MODEL];
    memcpy(feat_copy, feat, T * D * sizeof(float));
    float residual[NCSSM_N_FRAMES * NCSSM_D_MODEL];
    memcpy(residual, feat, T * D * sizeof(float));
    ncssm_layer_norm(feat_copy, &W[NCSSM_W_BLOCK0_NORM_G], &W[NCSSM_W_BLOCK0_NORM_B], D, T);
    print_stats("block0_norm", feat_copy, T * D);
    save_raw("block0_norm", feat_copy, T * D);

    /* In projection */
    float xz_buf[NCSSM_N_FRAMES * (NCSSM_D_INNER * 2)];
    for (int t = 0; t < T; t++) {
        ncssm_linear(&feat_copy[t * D], &W[NCSSM_W_BLOCK0_IN_PROJ], NULL,
                     &xz_buf[t * DI * 2], D, DI * 2);
    }

    /* Split + transpose x_branch to (DI, T) for conv */
    float x_br[NCSSM_D_INNER * NCSSM_N_FRAMES];
    float z_br[NCSSM_N_FRAMES * NCSSM_D_INNER];
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < DI; d++) {
            x_br[d * T + t] = xz_buf[t * DI * 2 + d];
            z_br[t * DI + d] = xz_buf[t * DI * 2 + DI + d];
        }
    }

    /* DWConv1d */
    float conv_out2[NCSSM_D_INNER * NCSSM_N_FRAMES];
    ncssm_dwconv1d_causal(x_br, &W[NCSSM_W_BLOCK0_CONV_W], &W[NCSSM_W_BLOCK0_CONV_B],
                          conv_out2, DI, T, NCSSM_D_CONV);

    /* Transpose + SiLU → SSM input */
    float ssm_input[NCSSM_N_FRAMES * NCSSM_D_INNER];
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < DI; d++) {
            ssm_input[t * DI + d] = ncssm_silu(conv_out2[d * T + t]);
        }
    }
    print_stats("block0_ssm_input", ssm_input, T * DI);
    save_raw("block0_ssm_input", ssm_input, T * DI);

    /* SSM scan */
    float ssm_out2[NCSSM_N_FRAMES * NCSSM_D_INNER];
    float h_tmp[NCSSM_D_INNER * NCSSM_D_STATE];
    memset(h_tmp, 0, sizeof(h_tmp));
    ncssm_ssm_scan(ssm_input, snr_seq, ssm_out2, h_tmp, &W[NCSSM_W_BLOCK0_SSM], T, pcen_gate_buf);
    print_stats("block0_ssm_output", ssm_out2, T * DI);
    save_raw("block0_ssm_output", ssm_out2, T * DI);

    /* Gate: y * silu(z) */
    float gated2[NCSSM_N_FRAMES * NCSSM_D_INNER];
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < DI; d++) {
            gated2[t * DI + d] = ssm_out2[t * DI + d] * ncssm_silu(z_br[t * DI + d]);
        }
    }
    print_stats("block0_gated", gated2, T * DI);
    save_raw("block0_gated", gated2, T * DI);

    /* Out proj + residual */
    float block0_out[NCSSM_N_FRAMES * NCSSM_D_MODEL];
    for (int t = 0; t < T; t++) {
        ncssm_linear(&gated2[t * DI], &W[NCSSM_W_BLOCK0_OUT_PROJ], NULL,
                     &block0_out[t * D], DI, D);
        for (int d = 0; d < D; d++)
            block0_out[t * D + d] += residual[t * D + d];
    }
    print_stats("block0_full_output", block0_out, T * D);
    save_raw("block0_full_output", block0_out, T * D);

    /* Compare block0 ssm_output with Python */
    float py_ssm[NCSSM_N_FRAMES * NCSSM_D_INNER];
    FILE *sf2 = fopen("test/debug_data/block0_ssm_output.raw", "rb");
    if (sf2) {
        fread(py_ssm, sizeof(float), T * DI, sf2);
        fclose(sf2);
        float max_d = 0;
        for (int i = 0; i < T * DI; i++) {
            float d = fabsf(ssm_out2[i] - py_ssm[i]);
            if (d > max_d) max_d = d;
        }
        printf("  >> block0_ssm_output vs Python: max_diff = %.6f\n", max_d);
    }

    /* Run full forward for final logits comparison */
    ncssm_ctx_t ctx;
    ncssm_init(&ctx);
    float logits[NCSSM_N_CLASSES];
    ncssm_forward(&ctx, audio, logits);
    print_stats("logits", logits, NCSSM_N_CLASSES);
    save_raw("logits", logits, NCSSM_N_CLASSES);

    /* Compare logits with Python */
    float py_logits[NCSSM_N_CLASSES];
    FILE *lf = fopen("test/debug_data/logits.raw", "rb");
    if (lf) {
        fread(py_logits, sizeof(float), NCSSM_N_CLASSES, lf);
        fclose(lf);
        printf("\n  Logit comparison (C vs Python):\n");
        for (int i = 0; i < NCSSM_N_CLASSES; i++) {
            printf("    %-10s: C=%.4f  Py=%.4f  diff=%.4f\n",
                   NCSSM_LABELS[i], logits[i], py_logits[i],
                   logits[i] - py_logits[i]);
        }
    }

    return 0;
}
