/* Auto-generated config for NanoMamba-NC-20K */
#ifndef NCSSM_CONFIG_H
#define NCSSM_CONFIG_H

#define NCSSM_SR            16000
#define NCSSM_AUDIO_LEN     16000
#define NCSSM_N_FFT         512
#define NCSSM_HOP_LENGTH    160
#define NCSSM_N_FREQ        257
#define NCSSM_N_FRAMES      101  /* center=True */
#define NCSSM_N_MELS        40
#define NCSSM_N_CLASSES     12
#define NCSSM_D_MODEL       37
#define NCSSM_D_INNER       55
#define NCSSM_D_STATE       10
#define NCSSM_D_CONV        3
#define NCSSM_N_LAYERS      2
#define NCSSM_EXPAND        1.5f
#define NCSSM_DELTA_FLOOR_MIN 0.03f
#define NCSSM_DELTA_FLOOR_MAX 0.15f
#define NCSSM_EPSILON_MIN   0.05f
#define NCSSM_EPSILON_MAX   0.30f
#define NCSSM_BGATE_FLOOR   0.20f
#define NCSSM_SNR_HALF_SAT  0.05f
#define NCSSM_NOISE_FRAMES  5
#define NCSSM_PCEN_EXPERTS  2
#define NCSSM_LUT_SIZE      256
#define NCSSM_LUT_RANGE     8.0f
#define NCSSM_SSM_STATE_SIZE (NCSSM_D_INNER * NCSSM_D_STATE)
#define NCSSM_MEL_BUF_SIZE  (NCSSM_N_MELS * NCSSM_N_FRAMES)
#define NCSSM_FEAT_BUF_SIZE (NCSSM_D_MODEL * NCSSM_N_FRAMES)

#endif
