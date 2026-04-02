/**
 * NC-SSM: Noise-Conditioned State Space Model - C SDK
 * Ultra-lightweight keyword spotting for edge devices
 * Copyright (c) 2026 Jin Ho Choi. All rights reserved.
 */

#ifndef NCSSM_H
#define NCSSM_H

#include "ncssm_config.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Context: holds model weights + runtime state
 * ================================================================ */
typedef struct {
    /* SSM hidden state (persistent across streaming chunks) */
    float h[NCSSM_N_LAYERS][NCSSM_D_INNER * NCSSM_D_STATE];

    /* Weights pointer (loaded from header or file) */
    const float *weights;

    /* Streaming accumulation buffer */
    float audio_buf[NCSSM_AUDIO_LEN];
    int audio_pos;

    /* Last prediction */
    int last_label;
    float last_confidence;
} ncssm_ctx_t;

/* ================================================================
 * Public API
 * ================================================================ */

/**
 * Initialize context. Call once before inference.
 */
void ncssm_init(ncssm_ctx_t *ctx);

/**
 * Reset SSM hidden state. Call between utterances.
 */
void ncssm_reset(ncssm_ctx_t *ctx);

/**
 * Run full inference on 1-second audio.
 * @param ctx    Model context
 * @param audio  Input audio samples (16000 float32 @ 16kHz)
 * @param logits Output logits (12 classes)
 */
void ncssm_forward(ncssm_ctx_t *ctx, const float *audio, float *logits);

/**
 * Get predicted class label (argmax of logits).
 */
int ncssm_predict(ncssm_ctx_t *ctx, const float *audio, float *confidence);

/**
 * Feed audio chunk for streaming inference.
 * @param ctx     Model context
 * @param chunk   Audio chunk samples
 * @param n_samples Number of samples in chunk
 * @return label index if detection, -1 if accumulating
 */
int ncssm_feed(ncssm_ctx_t *ctx, const float *chunk, int n_samples);

/* ================================================================
 * Internal functions (exposed for unit testing)
 * ================================================================ */

/* Activation functions */
float ncssm_sigmoid(float x);
float ncssm_silu(float x);
float ncssm_softplus(float x);
float ncssm_tanh_approx(float x);

/* Linear algebra */
void ncssm_linear(const float *input, const float *weight, const float *bias,
                  float *output, int in_dim, int out_dim);
void ncssm_matmul(const float *A, const float *B, float *C,
                  int M, int K, int N);

/* Feature extraction */
void ncssm_stft_mag(const float *audio, float *mag, int audio_len);
void ncssm_mel_projection(const float *mag, const float *mel_fb,
                          float *mel, int n_frames);
void ncssm_snr_estimate(const float *mag, const float *mel_fb,
                        float *snr_mel, int n_frames,
                        float noise_scale, float floor_val);
void ncssm_snr_estimate_ema(const float *mag, const float *mel_fb,
                            float *snr_mel, int n_frames,
                            float noise_scale, float raw_beta, float raw_gamma);
void ncssm_log_mel(float *mel, int n_mels, int n_frames);
void ncssm_spectral_gate(float *mel, const float *snr,
                         const float *w, const float *b,
                         const float *floor_raw, int n_frames);
void ncssm_instance_norm(float *data, int channels, int length);
void ncssm_dual_pcen_v2(float *mel, const float *snr_mel,
                        const float *pcen_weights, float *pcen_gate,
                        int n_mels, int n_frames);
void ncssm_dwconv1d_causal(const float *input, const float *weight,
                           const float *bias, float *output,
                           int channels, int L, int kernel_size);

/* SSM core */
void ncssm_ssm_scan(const float *x, const float *snr_mel,
                    float *y, float *h,
                    const float *weights, int L,
                    const float *pcen_gate);

/* Layer norm */
void ncssm_layer_norm(float *x, const float *gamma, const float *beta,
                      int dim, int len);

/* Softmax */
void ncssm_softmax(float *x, int n);

/* Labels */
extern const char *NCSSM_LABELS[NCSSM_N_CLASSES];

#ifdef __cplusplus
}
#endif

#endif /* NCSSM_H */
