/**
 * Activation functions for NC-SSM
 * Fast approximations suitable for Cortex-M7
 */

#include "ncssm.h"
#include <math.h>

/* ================================================================
 * Sigmoid: σ(x) = 1 / (1 + exp(-x))
 * Fast piecewise linear approximation for edge
 * ================================================================ */
float ncssm_sigmoid(float x) {
    if (x > 6.0f) return 1.0f;
    if (x < -6.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* ================================================================
 * SiLU (Swish): x * σ(x)
 * ================================================================ */
float ncssm_silu(float x) {
    return x * ncssm_sigmoid(x);
}

/* ================================================================
 * Softplus: log(1 + exp(x))
 * ================================================================ */
float ncssm_softplus(float x) {
    if (x > 20.0f) return x;  /* avoid overflow */
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

/* ================================================================
 * Tanh approximation
 * ================================================================ */
float ncssm_tanh_approx(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    return tanhf(x);
}
