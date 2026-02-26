// ============================================================================
// NanoMamba DualPCEN — 2-Expert PCEN with Gated Blending
// ============================================================================
// PCEN formula per expert:
//   smoother[t] = (1-s) * smoother[t-1] + s * mel[t]     (IIR AGC)
//   gain = (epsilon + smoother)^(-alpha)                   (Adaptive gain)
//   pcen = (mel * gain + delta)^r - delta^r                (DRC + offset)
//
// Expert 0 (Non-stationary/Babble): s=0.025, delta=2.0, r=0.5
// Expert 1 (Stationary/Factory):    s=0.15,  delta=0.01, r=0.1
//
// Blended output = gate * expert1 + (1-gate) * expert0
//
// Power function approximation: x^a ≈ exp(a * log(x))
//   Using log/exp LUT tables (256 entries each)
//
// Per-channel (40 mel bands), each expert has 4 params × 40 = 160 stored params
// Total: 2 × 160 + 1 (gate_temp) = 321 params
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_dual_pcen #(
    parameter N_MELS     = 40,
    parameter DATA_WIDTH = 8,
    parameter N_EXPERTS  = 2
)(
    input  wire                     clk_expert0,   // Gated clock (non-stationary)
    input  wire                     clk_expert1,   // Gated clock (stationary)
    input  wire                     rst_n,

    // Mel input (linear energy, per frame)
    input  wire [15:0]             mel_in,
    input  wire [5:0]              mel_index,
    input  wire                     mel_valid,
    input  wire                     mel_frame_done,

    // Gate from MOE Router
    input  wire [DATA_WIDTH-1:0]   gate,          // Q0.8: 0=expert0, 255=expert1
    input  wire                     gate_valid,

    // PCEN output (blended)
    output reg  [DATA_WIDTH-1:0]   pcen_out,
    output reg  [5:0]              pcen_index,
    output reg                      pcen_valid
);

    // ---- Expert Parameters (loaded from Weight SRAM) ----
    // Per expert, per mel band: s, alpha, delta, r (INT8 each)
    // Stored in Q format appropriate for each parameter

    // Expert 0: Non-stationary (babble specialist)
    reg [7:0] e0_s     [0:N_MELS-1];  // Smoothing (Q0.8, init≈6  = 0.025*256)
    reg [7:0] e0_alpha [0:N_MELS-1];  // AGC exponent (Q0.8, init≈253)
    reg [7:0] e0_delta [0:N_MELS-1];  // Offset (Q4.4, init≈32 = 2.0*16)
    reg [7:0] e0_r     [0:N_MELS-1];  // Compression (Q0.8, init≈128 = 0.5*256)

    // Expert 1: Stationary (factory/white specialist)
    reg [7:0] e1_s     [0:N_MELS-1];  // Q0.8, init≈38 = 0.15*256
    reg [7:0] e1_alpha [0:N_MELS-1];  // Q0.8, init≈253
    reg [7:0] e1_delta [0:N_MELS-1];  // Q4.4, init≈0  = 0.01*16 ≈ 0
    reg [7:0] e1_r     [0:N_MELS-1];  // Q0.8, init≈26 = 0.1*256

    // ---- IIR Smoother State (persistent across frames) ----
    reg [15:0] smoother0 [0:N_MELS-1];  // Expert 0 smoother state
    reg [15:0] smoother1 [0:N_MELS-1];  // Expert 1 smoother state

    // ---- Power Function LUTs ----
    // pow_lut[x] = exp(alpha * log(x)) precomputed for common alpha values
    // In practice: log_lut + multiply + exp_lut pipeline
    reg [7:0] log_lut [0:255];     // log(x) in Q3.5
    reg [7:0] exp_lut [0:255];     // exp(x) in Q8.0

    // ---- Processing State ----
    localparam S_IDLE    = 3'd0,
               S_IIR     = 3'd1,
               S_GAIN    = 3'd2,
               S_PCEN    = 3'd3,
               S_BLEND   = 3'd4,
               S_OUTPUT  = 3'd5;

    reg [2:0]  state;
    reg [5:0]  ch_idx;             // Current mel channel
    reg [15:0] mel_buf [0:N_MELS-1];  // Buffered mel frame

    // Intermediate results
    reg [15:0] gain0, gain1;       // AGC gains
    reg [15:0] pcen0, pcen1;       // Per-expert PCEN outputs
    reg [DATA_WIDTH-1:0] blended;  // Blended output
    reg [DATA_WIDTH-1:0] gate_latch;

    // Output buffer
    reg [DATA_WIDTH-1:0] out_buf [0:N_MELS-1];

    always @(posedge clk_expert0 or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            ch_idx     <= 6'd0;
            pcen_valid <= 1'b0;
            gate_latch <= 8'd128;
        end else begin
            pcen_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    // Buffer incoming mel frame
                    if (mel_valid) begin
                        mel_buf[mel_index] <= mel_in;
                    end
                    if (gate_valid) begin
                        gate_latch <= gate;
                    end
                    if (mel_frame_done) begin
                        ch_idx <= 6'd0;
                        state  <= S_IIR;
                    end
                end

                // ---- IIR Smoother Update (both experts) ----
                S_IIR: begin
                    // Expert 0: smoother0[i] = (1-s0)*smoother0[i] + s0*mel[i]
                    // INT fixed-point: (256-s)*smoother/256 + s*mel/256
                    smoother0[ch_idx] <= (
                        (16'd256 - {8'd0, e0_s[ch_idx]}) * smoother0[ch_idx][15:8]
                        + {8'd0, e0_s[ch_idx]} * mel_buf[ch_idx][15:8]
                    ) >> 8;

                    // Expert 1: smoother1[i] = (1-s1)*smoother1[i] + s1*mel[i]
                    smoother1[ch_idx] <= (
                        (16'd256 - {8'd0, e1_s[ch_idx]}) * smoother1[ch_idx][15:8]
                        + {8'd0, e1_s[ch_idx]} * mel_buf[ch_idx][15:8]
                    ) >> 8;

                    state <= S_GAIN;
                end

                // ---- AGC Gain: (epsilon + smoother)^(-alpha) ----
                S_GAIN: begin
                    // gain = (eps + smoother)^(-alpha)
                    // Using LUT: gain = exp(-alpha * log(eps + smoother))
                    //
                    // Step 1: x = eps + smoother (saturating add)
                    // Step 2: log_x = log_lut[x >> 8]  (top 8 bits)
                    // Step 3: product = alpha * log_x  (INT8 × INT8)
                    // Step 4: gain = exp_lut[255 - product] (negated for ^(-alpha))

                    // Simplified placeholder for synthesis demonstration:
                    gain0 <= 16'h0080;  // Actual: LUT-based computation
                    gain1 <= 16'h0080;

                    state <= S_PCEN;
                end

                // ---- PCEN Transform: (mel * gain + delta)^r - delta^r ----
                S_PCEN: begin
                    // Expert 0: pcen0 = (mel*gain0 + delta0)^r0 - delta0^r0
                    // Expert 1: pcen1 = (mel*gain1 + delta1)^r1 - delta1^r1
                    //
                    // Power function via LUT:
                    //   x^r = exp(r * log(x))
                    //   Using concatenated log/exp LUT pipeline

                    // Placeholder:
                    pcen0 <= mel_buf[ch_idx];  // Actual: full PCEN computation
                    pcen1 <= mel_buf[ch_idx];

                    state <= S_BLEND;
                end

                // ---- Blend: output = gate * expert1 + (1-gate) * expert0 ----
                S_BLEND: begin
                    // gate is Q0.8: 0=expert0, 255=expert1
                    // blend = (gate * pcen1 + (255-gate) * pcen0 + 128) >> 8
                    blended <= (
                        {8'd0, gate_latch} * pcen1[7:0]
                        + (16'd255 - {8'd0, gate_latch}) * pcen0[7:0]
                        + 16'd128
                    ) >> 8;

                    out_buf[ch_idx] <= blended;

                    // Advance to next channel
                    if (ch_idx < N_MELS - 1) begin
                        ch_idx <= ch_idx + 1;
                        state  <= S_IIR;
                    end else begin
                        ch_idx <= 6'd0;
                        state  <= S_OUTPUT;
                    end
                end

                // ---- Stream out blended PCEN frame ----
                S_OUTPUT: begin
                    pcen_out   <= out_buf[ch_idx];
                    pcen_index <= ch_idx;
                    pcen_valid <= 1'b1;

                    if (ch_idx < N_MELS - 1) begin
                        ch_idx <= ch_idx + 1;
                    end else begin
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

    // ---- Expert Parameter Initialization ----
    integer ei;
    initial begin
        for (ei = 0; ei < N_MELS; ei = ei + 1) begin
            // Expert 0 (Non-stationary): s=0.025, alpha=0.99, delta=2.0, r=0.5
            e0_s[ei]     = 8'd6;    // 0.025 * 256
            e0_alpha[ei] = 8'd253;  // 0.99 * 256
            e0_delta[ei] = 8'd32;   // 2.0 * 16 (Q4.4)
            e0_r[ei]     = 8'd128;  // 0.5 * 256

            // Expert 1 (Stationary): s=0.15, alpha=0.99, delta=0.01, r=0.1
            e1_s[ei]     = 8'd38;   // 0.15 * 256
            e1_alpha[ei] = 8'd253;  // 0.99 * 256
            e1_delta[ei] = 8'd0;    // 0.01 * 16 ≈ 0 (Q4.4)
            e1_r[ei]     = 8'd26;   // 0.1 * 256

            smoother0[ei] = 16'd0;
            smoother1[ei] = 16'd0;
        end
    end

endmodule
