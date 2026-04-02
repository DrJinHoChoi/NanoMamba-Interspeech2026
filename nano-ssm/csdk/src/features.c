/**
 * Feature extraction: STFT, Mel projection, SNR estimation
 */

#include "ncssm.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Radix-2 Cooley-Tukey FFT (in-place, N=512)
 * ================================================================ */
static void fft_radix2(float *re, float *im, int N) {
    /* Bit-reversal permutation */
    int j = 0;
    for (int i = 0; i < N; i++) {
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        int m = N >> 1;
        while (m >= 1 && j >= m) { j -= m; m >>= 1; }
        j += m;
    }

    /* Butterfly stages */
    for (int step = 2; step <= N; step <<= 1) {
        int half = step >> 1;
        float angle = -2.0f * (float)M_PI / step;
        float wr = cosf(angle), wi = sinf(angle);
        for (int k = 0; k < N; k += step) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (int m = 0; m < half; m++) {
                float tr = cur_r * re[k+half+m] - cur_i * im[k+half+m];
                float ti = cur_r * im[k+half+m] + cur_i * re[k+half+m];
                re[k+half+m] = re[k+m] - tr;
                im[k+half+m] = im[k+m] - ti;
                re[k+m] += tr;
                im[k+m] += ti;
                float nr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nr;
            }
        }
    }
}

/* ================================================================
 * STFT magnitude spectrogram using FFT
 * audio: (audio_len,) → mag: (N_FREQ, n_frames)
 *
 * Matches PyTorch torch.stft(center=True):
 *   - Pads signal by n_fft//2 on each side (reflect padding)
 *   - Produces n_frames = audio_len / hop_length + 1 = 101 for 16000 samples
 * ================================================================ */
void ncssm_stft_mag(const float *audio, float *mag, int audio_len) {
    float window[NCSSM_N_FFT];
    for (int i = 0; i < NCSSM_N_FFT; i++)
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / NCSSM_N_FFT));

    /* center=True: pad by n_fft//2 on each side */
    int pad = NCSSM_N_FFT / 2;
    int padded_len = audio_len + 2 * pad;
    int n_frames = (padded_len - NCSSM_N_FFT) / NCSSM_HOP_LENGTH + 1;
    if (n_frames > NCSSM_N_FRAMES) n_frames = NCSSM_N_FRAMES;

    float re[NCSSM_N_FFT], im[NCSSM_N_FFT];

    for (int f = 0; f < n_frames; f++) {
        int offset = f * NCSSM_HOP_LENGTH;  /* offset in padded signal */

        /* Window + load (with reflect padding) */
        for (int n = 0; n < NCSSM_N_FFT; n++) {
            int src_idx = offset + n - pad;  /* index into original audio */
            float s;
            if (src_idx < 0) {
                /* Reflect padding at start */
                src_idx = -src_idx;
                if (src_idx >= audio_len) src_idx = audio_len - 1;
            } else if (src_idx >= audio_len) {
                /* Reflect padding at end */
                src_idx = 2 * audio_len - 2 - src_idx;
                if (src_idx < 0) src_idx = 0;
            }
            s = audio[src_idx];
            re[n] = s * window[n];
            im[n] = 0.0f;
        }

        /* FFT */
        fft_radix2(re, im, NCSSM_N_FFT);

        /* Magnitude for first N_FREQ bins */
        for (int k = 0; k < NCSSM_N_FREQ; k++) {
            mag[k * n_frames + f] = sqrtf(re[k] * re[k] + im[k] * im[k]);
        }
    }
}

/* ================================================================
 * Mel projection: mag (N_FREQ, T) → mel (N_MELS, T)
 * mel_fb: (N_MELS, N_FREQ) filterbank
 * ================================================================ */
void ncssm_mel_projection(const float *mag, const float *mel_fb,
                          float *mel, int n_frames) {
    /* mel[m][t] = sum_f(mel_fb[m][f] * mag[f][t]) */
    for (int m = 0; m < NCSSM_N_MELS; m++) {
        for (int t = 0; t < n_frames; t++) {
            float sum = 0.0f;
            for (int f = 0; f < NCSSM_N_FREQ; f++) {
                sum += mel_fb[m * NCSSM_N_FREQ + f] * mag[f * n_frames + t];
            }
            mel[m * n_frames + t] = sum;
        }
    }
}

/* ================================================================
 * SNR estimation (static noise floor)
 * mag: (N_FREQ, T), mel_fb: (N_MELS, N_FREQ)
 * snr_mel: (N_MELS, T) output in [0, 1]
 * ================================================================ */
void ncssm_snr_estimate(const float *mag, const float *mel_fb,
                        float *snr_mel, int n_frames,
                        float noise_scale, float floor_val) {
    /* 1. Noise floor from first NOISE_FRAMES frames */
    float noise_floor[NCSSM_N_FREQ];
    memset(noise_floor, 0, sizeof(noise_floor));

    int nf = (n_frames < NCSSM_NOISE_FRAMES) ? n_frames : NCSSM_NOISE_FRAMES;
    for (int f = 0; f < NCSSM_N_FREQ; f++) {
        float sum = 0.0f;
        for (int t = 0; t < nf; t++) {
            sum += mag[f * n_frames + t];
        }
        noise_floor[f] = sum / nf;
        if (noise_floor[f] < 1e-5f) noise_floor[f] = 1e-5f;
    }

    /* 2. Compute per-freq SNR = mag / (noise_scale * noise_floor) */
    float snr_freq[NCSSM_N_FREQ];
    float ns = fabsf(noise_scale);

    for (int t = 0; t < n_frames; t++) {
        /* Per-freq SNR */
        for (int f = 0; f < NCSSM_N_FREQ; f++) {
            snr_freq[f] = mag[f * n_frames + t] / (ns * noise_floor[f] + 1e-8f);
        }

        /* Project to mel bands */
        for (int m = 0; m < NCSSM_N_MELS; m++) {
            float sum = 0.0f;
            for (int f = 0; f < NCSSM_N_FREQ; f++) {
                sum += mel_fb[m * NCSSM_N_FREQ + f] * snr_freq[f];
            }
            /* tanh(x/10) normalization to [0, 1] */
            float val = ncssm_tanh_approx(sum / 10.0f);
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            snr_mel[m * n_frames + t] = val;
        }
    }
}

/* ================================================================
 * SNR estimation with Running EMA noise floor tracking
 * Matches Python SNREstimator(use_running_ema=True)
 *
 * Asymmetric EMA: slow rise (speech), faster fall (true noise)
 *   beta = sigmoid(raw_beta)  ~0.014 (when frame < noise: update fast)
 *   gamma = sigmoid(raw_gamma) ~0.022 (when frame > noise: update slow)
 * ================================================================ */
void ncssm_snr_estimate_ema(const float *mag, const float *mel_fb,
                            float *snr_mel, int n_frames,
                            float noise_scale, float raw_beta, float raw_gamma) {
    float ns = fabsf(noise_scale);
    float beta = ncssm_sigmoid(raw_beta);    /* ~0.014 */
    float gamma = ncssm_sigmoid(raw_gamma);  /* ~0.022 */

    /* Phase 1: Initial noise estimate from first NOISE_FRAMES */
    float init_noise[NCSSM_N_FREQ];
    int nf = (n_frames < NCSSM_NOISE_FRAMES) ? n_frames : NCSSM_NOISE_FRAMES;
    for (int f = 0; f < NCSSM_N_FREQ; f++) {
        float sum = 0.0f;
        for (int t = 0; t < nf; t++) {
            sum += mag[f * n_frames + t];
        }
        init_noise[f] = sum / nf;
        if (init_noise[f] < 1e-5f) init_noise[f] = 1e-5f;
    }

    /* Phase 2: Running EMA noise floor */
    float noise_floor[NCSSM_N_FREQ];
    memcpy(noise_floor, init_noise, sizeof(noise_floor));

    float snr_freq[NCSSM_N_FREQ];
    float half_init[NCSSM_N_FREQ];
    for (int f = 0; f < NCSSM_N_FREQ; f++) {
        half_init[f] = init_noise[f] * 0.5f;
    }

    for (int t = 0; t < n_frames; t++) {
        /* Update noise floor with asymmetric EMA */
        for (int f = 0; f < NCSSM_N_FREQ; f++) {
            float frame = mag[f * n_frames + t];
            float alpha_t = (frame > noise_floor[f]) ? gamma : beta;
            noise_floor[f] = (1.0f - alpha_t) * noise_floor[f] + alpha_t * frame;

            /* Safety: never below half initial estimate */
            if (noise_floor[f] < half_init[f])
                noise_floor[f] = half_init[f];
        }

        /* Per-freq SNR */
        for (int f = 0; f < NCSSM_N_FREQ; f++) {
            snr_freq[f] = mag[f * n_frames + t] / (ns * noise_floor[f] + 1e-8f);
        }

        /* Project to mel bands + normalize */
        for (int m = 0; m < NCSSM_N_MELS; m++) {
            float sum = 0.0f;
            for (int f = 0; f < NCSSM_N_FREQ; f++) {
                sum += mel_fb[m * NCSSM_N_FREQ + f] * snr_freq[f];
            }
            float val = ncssm_tanh_approx(sum / 10.0f);
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            snr_mel[m * n_frames + t] = val;
        }
    }
}

/* ================================================================
 * Log mel: mel = log(mel + eps)
 * ================================================================ */
void ncssm_log_mel(float *mel, int n_mels, int n_frames) {
    for (int i = 0; i < n_mels * n_frames; i++) {
        mel[i] = logf(mel[i] + 1e-8f);
    }
}

/* ================================================================
 * Learned Spectral Gate (LSG)
 * mel: (N_MELS, T), snr: (N_MELS, T)
 * w, b, floor_raw: (N_MELS,) learned params
 * ================================================================ */
void ncssm_spectral_gate(float *mel, const float *snr,
                         const float *w, const float *b,
                         const float *floor_raw,
                         int n_frames) {
    for (int m = 0; m < NCSSM_N_MELS; m++) {
        float fl = ncssm_sigmoid(floor_raw[m]);
        for (int t = 0; t < n_frames; t++) {
            int idx = m * n_frames + t;
            float gain = ncssm_sigmoid(w[m] * snr[idx] + b[m]);
            mel[idx] = mel[idx] * (gain * (1.0f - fl) + fl);
        }
    }
}
