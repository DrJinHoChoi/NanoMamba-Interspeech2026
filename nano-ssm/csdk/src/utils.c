/**
 * Utility functions: LayerNorm, InstanceNorm, pooling, softmax
 */

#include "ncssm.h"
#include <math.h>
#include <string.h>

const char *NCSSM_LABELS[NCSSM_N_CLASSES] = {
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "silence", "unknown"
};

/* ================================================================
 * Layer Normalization
 * x: (len, dim), gamma/beta: (dim,)
 * Normalizes along last dim for each timestep
 * ================================================================ */
void ncssm_layer_norm(float *x, const float *gamma, const float *beta,
                      int dim, int len) {
    for (int t = 0; t < len; t++) {
        float *xt = &x[t * dim];

        /* Mean */
        float mean = 0.0f;
        for (int d = 0; d < dim; d++) mean += xt[d];
        mean /= dim;

        /* Variance */
        float var = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = xt[d] - mean;
            var += diff * diff;
        }
        var /= dim;

        /* Normalize */
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < dim; d++) {
            xt[d] = (xt[d] - mean) * inv_std * gamma[d] + beta[d];
        }
    }
}

/* ================================================================
 * Instance Normalization
 * data: (channels, length), normalizes per channel
 * ================================================================ */
void ncssm_instance_norm(float *data, int channels, int length) {
    for (int c = 0; c < channels; c++) {
        float *ch = &data[c * length];

        float mean = 0.0f;
        for (int t = 0; t < length; t++) mean += ch[t];
        mean /= length;

        float var = 0.0f;
        for (int t = 0; t < length; t++) {
            float diff = ch[t] - mean;
            var += diff * diff;
        }
        var /= length;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int t = 0; t < length; t++) {
            ch[t] = (ch[t] - mean) * inv_std;
        }
    }
}

/* ================================================================
 * Softmax
 * ================================================================ */
void ncssm_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

/* ================================================================
 * Depthwise Conv1d (causal, kernel=3)
 *
 * Matches PyTorch: Conv1d(padding=kernel_size-1) + trim to L.
 * PyTorch pads kernel_size-1 on EACH side with ZEROS, then output[:,:,:L].
 * Effective: left zero-pad of kernel_size-1, no right pad = causal.
 *
 * input:  (channels, L)
 * output: (channels, L)
 * weight: (channels, 1, kernel_size) → (channels, kernel_size)
 * bias:   (channels,)
 * ================================================================ */
void ncssm_dwconv1d_causal(const float *input, const float *weight,
                           const float *bias, float *output,
                           int channels, int L, int kernel_size) {
    int pad = kernel_size - 1;  /* causal: left zero-pad only */

    for (int c = 0; c < channels; c++) {
        const float *in_ch = &input[c * L];
        const float *w_ch = &weight[c * kernel_size];
        float *out_ch = &output[c * L];
        float b = (bias != NULL) ? bias[c] : 0.0f;

        for (int t = 0; t < L; t++) {
            float sum = b;
            for (int k = 0; k < kernel_size; k++) {
                int idx = t - pad + k;
                /* Zero padding (matches PyTorch default) */
                float val = (idx >= 0 && idx < L) ? in_ch[idx] : 0.0f;
                sum += w_ch[k] * val;
            }
            out_ch[t] = sum;
        }
    }
}
