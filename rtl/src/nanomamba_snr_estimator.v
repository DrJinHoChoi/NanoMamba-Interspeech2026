// ============================================================================
// NanoMamba SNR Estimator — Per-Mel-Band SNR from Magnitude Spectrum
// ============================================================================
// Computes: SNR_mel[i] = 10 * log10(mag_mel[i] / noise_floor_mel[i])
//
// Noise floor estimation:
//   Phase 1: Average of first 5 frames (initialization)
//   Phase 2: Asymmetric EMA tracking (optional)
//     frame > floor → slow rise (gamma ≈ 0.05)
//     frame < floor → fast fall (beta ≈ 0.10)
//
// Pipeline: Mag input → Mel projection → Noise tracking → SNR → Output
//
// Resources: ~1 BRAM (mel filterbank 40×257) + 1 divider + 1 log2 LUT
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_snr_estimator #(
    parameter N_FREQ       = 257,
    parameter N_MELS       = 40,
    parameter NOISE_FRAMES = 5,
    parameter DATA_WIDTH   = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Magnitude spectrum input (from STFT)
    input  wire [15:0]             mag_in,
    input  wire [8:0]              mag_index,
    input  wire                     mag_valid,
    input  wire                     frame_done,

    // Per-mel-band SNR output (INT8, dB scale)
    output reg  [DATA_WIDTH-1:0]   snr_mel_out,
    output reg  [5:0]              snr_mel_index,
    output reg                      snr_valid,
    output reg                      snr_frame_done
);

    // ---- Mel Filterbank ROM ----
    // 40 × 257 sparse matrix (triangular filters)
    // Stored as: {start_bin, end_bin, peak_bin} per filter + weights
    // In practice, each mel filter spans ~10-20 bins → sparse storage
    reg [7:0] mel_fb_start [0:N_MELS-1];  // Start bin index
    reg [7:0] mel_fb_end   [0:N_MELS-1];  // End bin index
    reg [7:0] mel_fb_peak  [0:N_MELS-1];  // Peak bin index

    // Mel-band accumulators
    reg [31:0] mel_energy [0:N_MELS-1];

    // Noise floor (per mel band, Q8.8 fixed point)
    reg [15:0] noise_floor [0:N_MELS-1];
    reg [3:0]  frame_count;
    reg        noise_initialized;

    // ---- Frame accumulation state ----
    reg [31:0] mel_accum [0:N_MELS-1];

    // ---- log2 LUT (256 entries, 8-bit output) ----
    // log2_lut[x] ≈ 8 * log2(x+1), x ∈ [0, 255]
    reg [7:0] log2_lut [0:255];
    integer lut_i;
    initial begin
        log2_lut[0] = 8'd0;
        for (lut_i = 1; lut_i < 256; lut_i = lut_i + 1) begin
            // Approximate log2 scaled to INT8 range
            log2_lut[lut_i] = 8'd0;  // Filled from init file
        end
    end

    // ---- Processing state machine ----
    localparam S_IDLE      = 3'd0,
               S_ACCUMULATE = 3'd1,
               S_NOISE_UPD = 3'd2,
               S_SNR_CALC  = 3'd3,
               S_OUTPUT    = 3'd4;

    reg [2:0] snr_state;
    reg [5:0] mel_idx;

    // ---- SNR computation ----
    // For each mel band: snr = mel_energy / noise_floor
    // Then: snr_db = 10 * log10(snr) ≈ 3.32 * log2(snr)
    // Quantized to INT8: [-40dB, +40dB] → [0, 255]

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            snr_state        <= S_IDLE;
            frame_count      <= 4'd0;
            noise_initialized <= 1'b0;
            snr_valid        <= 1'b0;
            snr_frame_done   <= 1'b0;
            mel_idx          <= 6'd0;
        end else begin
            snr_valid      <= 1'b0;
            snr_frame_done <= 1'b0;

            case (snr_state)
                S_IDLE: begin
                    if (mag_valid) begin
                        snr_state <= S_ACCUMULATE;
                    end
                end

                S_ACCUMULATE: begin
                    // Accumulate magnitude into mel bands
                    // mel_energy[i] += mag_in * mel_weight[i][bin]
                    // (Sparse: only process non-zero filter coefficients)
                    if (frame_done) begin
                        snr_state <= S_NOISE_UPD;
                        mel_idx   <= 6'd0;
                    end
                end

                S_NOISE_UPD: begin
                    // Update noise floor estimate
                    if (mel_idx < N_MELS) begin
                        if (!noise_initialized) begin
                            // Phase 1: Accumulate first NOISE_FRAMES frames
                            if (frame_count < NOISE_FRAMES) begin
                                // noise_floor[i] += mel_energy[i]
                                noise_floor[mel_idx] <= noise_floor[mel_idx]
                                    + mel_energy[mel_idx][15:0];
                            end else begin
                                // Average: noise_floor[i] /= NOISE_FRAMES
                                noise_floor[mel_idx] <= noise_floor[mel_idx] / NOISE_FRAMES;
                            end
                        end else begin
                            // Phase 2: Asymmetric EMA (optional)
                            // if (mel_energy > noise_floor)
                            //   floor = (1-gamma)*floor + gamma*energy  (slow rise)
                            // else
                            //   floor = (1-beta)*floor + beta*energy    (fast fall)
                            //
                            // INT fixed-point: gamma=0.05→13/256, beta=0.10→26/256
                            if (mel_energy[mel_idx] > {16'b0, noise_floor[mel_idx]}) begin
                                // Slow rise: alpha = 13/256
                                noise_floor[mel_idx] <= noise_floor[mel_idx]
                                    - (noise_floor[mel_idx] >> 4)   // × (1 - 1/16 ≈ 0.9375)
                                    + (mel_energy[mel_idx][15:0] >> 4);
                            end else begin
                                // Fast fall: alpha = 26/256
                                noise_floor[mel_idx] <= noise_floor[mel_idx]
                                    - (noise_floor[mel_idx] >> 3)   // × (1 - 1/8 = 0.875)
                                    + (mel_energy[mel_idx][15:0] >> 3);
                            end
                        end
                        mel_idx <= mel_idx + 1;
                    end else begin
                        if (frame_count < NOISE_FRAMES) begin
                            frame_count <= frame_count + 1;
                            if (frame_count == NOISE_FRAMES - 1)
                                noise_initialized <= 1'b1;
                            snr_state <= S_IDLE;  // Wait for more frames
                        end else begin
                            mel_idx   <= 6'd0;
                            snr_state <= S_SNR_CALC;
                        end
                    end
                end

                S_SNR_CALC: begin
                    // SNR[i] = mel_energy[i] / (noise_scale * noise_floor[i] + floor_param)
                    // snr_db = clamp(3.32 * log2(SNR) * scale + offset, 0, 255)
                    if (mel_idx < N_MELS) begin
                        // Simplified: use log2 LUT on ratio
                        // ratio = mel_energy[i] / noise_floor[i]
                        // snr_int8 = log2_lut[clamp(ratio, 0, 255)]
                        snr_mel_out   <= 8'd128;  // Placeholder (actual: LUT lookup)
                        snr_mel_index <= mel_idx;
                        snr_valid     <= 1'b1;
                        mel_idx       <= mel_idx + 1;
                    end else begin
                        snr_frame_done <= 1'b1;
                        snr_state      <= S_IDLE;
                    end
                end
            endcase
        end
    end

    // ---- Mel Filterbank Initialization ----
    // In real implementation: loaded from ROM or computed at synthesis time
    // Triangular filters in mel scale, 40 bands covering 0-8kHz
    integer mi;
    initial begin
        for (mi = 0; mi < N_MELS; mi = mi + 1) begin
            mel_fb_start[mi] = 8'd0;
            mel_fb_end[mi]   = 8'd0;
            mel_fb_peak[mi]  = 8'd0;
        end
    end

endmodule
