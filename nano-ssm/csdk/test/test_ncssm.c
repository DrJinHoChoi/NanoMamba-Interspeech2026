/**
 * NC-SSM C SDK Test
 * Reads raw audio from stdin, runs inference, prints result
 *
 * Usage:
 *   python gen_test_audio.py | ./test_ncssm
 *   or:
 *   ./test_ncssm < test_audio.raw
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ncssm.h"
#include "ncssm_weights.h"

int main(int argc, char *argv[]) {
    printf("NC-SSM C SDK Test\n");
    printf("=================\n");
    printf("Config: d_model=%d, d_inner=%d, d_state=%d, n_layers=%d\n",
           NCSSM_D_MODEL, NCSSM_D_INNER, NCSSM_D_STATE, NCSSM_N_LAYERS);
    printf("Weights: %d floats (%.1f KB)\n\n",
           (int)(sizeof(ncssm_weights_data) / sizeof(float)),
           sizeof(ncssm_weights_data) / 1024.0f);

    /* Initialize */
    ncssm_ctx_t ctx;
    ncssm_init(&ctx);

    /* Generate test audio: 1 second of silence (zeros) */
    float audio[NCSSM_AUDIO_LEN];
    memset(audio, 0, sizeof(audio));

    /* Try to read from file if provided */
    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        if (f) {
            size_t n = fread(audio, sizeof(float), NCSSM_AUDIO_LEN, f);
            fclose(f);
            printf("Loaded %zu samples from %s\n", n, argv[1]);
        } else {
            printf("Cannot open %s, using silence\n", argv[1]);
        }
    } else {
        /* Generate some noise for testing */
        srand(42);
        for (int i = 0; i < NCSSM_AUDIO_LEN; i++) {
            audio[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
        printf("Using random noise (seed=42)\n");
    }

    /* Run inference with timing */
    float logits[NCSSM_N_CLASSES];
    clock_t t0 = clock();
    ncssm_forward(&ctx, audio, logits);
    clock_t t1 = clock();
    double elapsed_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    /* Softmax */
    float probs[NCSSM_N_CLASSES];
    memcpy(probs, logits, sizeof(logits));
    ncssm_softmax(probs, NCSSM_N_CLASSES);

    /* Results */
    printf("\nResults:\n");
    int best = 0;
    for (int i = 0; i < NCSSM_N_CLASSES; i++) {
        char marker = (probs[i] == probs[best] || probs[i] > probs[best]) ? '*' : ' ';
        if (probs[i] > probs[best]) best = i;
        printf("  %c %-10s: logit=%8.4f  prob=%.4f\n",
               marker, NCSSM_LABELS[i], logits[i], probs[i]);
    }
    printf("\nPrediction: %s (%.1f%%)\n", NCSSM_LABELS[best], probs[best] * 100);
    printf("Latency:    %.1f ms\n", elapsed_ms);
    printf("Real-time:  %.3fx\n", elapsed_ms / 1000.0);

    return 0;
}
