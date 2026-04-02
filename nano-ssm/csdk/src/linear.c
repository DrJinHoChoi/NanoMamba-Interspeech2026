/**
 * Linear algebra operations for NC-SSM
 * GEMM, Linear layer, vector operations
 */

#include "ncssm.h"
#include <string.h>

/* ================================================================
 * Linear layer: output = input @ weight^T + bias
 * weight shape: (out_dim, in_dim) stored row-major
 * input shape:  (in_dim,)
 * output shape: (out_dim,)
 * ================================================================ */
void ncssm_linear(const float *input, const float *weight, const float *bias,
                  float *output, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = (bias != NULL) ? bias[o] : 0.0f;
        const float *w_row = &weight[o * in_dim];
        for (int i = 0; i < in_dim; i++) {
            sum += w_row[i] * input[i];
        }
        output[o] = sum;
    }
}

/* ================================================================
 * Batched linear: apply linear to each timestep
 * input:  (L, in_dim)
 * output: (L, out_dim)
 * ================================================================ */
void ncssm_linear_batched(const float *input, const float *weight,
                          const float *bias, float *output,
                          int L, int in_dim, int out_dim) {
    for (int t = 0; t < L; t++) {
        ncssm_linear(&input[t * in_dim], weight, bias,
                     &output[t * out_dim], in_dim, out_dim);
    }
}

/* ================================================================
 * Matrix multiply: C = A @ B
 * A: (M, K), B: (K, N), C: (M, N)
 * ================================================================ */
void ncssm_matmul(const float *A, const float *B, float *C,
                  int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}
