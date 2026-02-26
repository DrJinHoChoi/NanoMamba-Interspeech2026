// ============================================================================
// NanoMamba MOE Router — Spectral Flatness Based Expert Selection
// ============================================================================
// Computes Spectral Flatness per frame (0 learnable params for routing signal):
//
//   SF = exp(mean(log(mel))) / mean(mel) = geometric_mean / arithmetic_mean
//
// Gate = sigmoid(gate_temp * (SF - 0.5))
//   SF → 1.0 → stationary noise → gate=1 → Expert 1 (AGC, delta=0.01)
//   SF → 0.0 → non-stationary   → gate=0 → Expert 0 (offset, delta=2.0)
//
// Fixed-point implementation:
//   - log/exp: 256-entry LUT (INT8 in, INT16 out)
//   - sigmoid: 256-entry LUT (INT8 in, INT8 out)
//   - Division: iterative or reciprocal LUT
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_moe_router #(
    parameter N_MELS     = 40,
    parameter DATA_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Mel energy input (linear scale, per frame)
    input  wire [15:0]             mel_in,
    input  wire [5:0]              mel_index,
    input  wire                     mel_valid,
    input  wire                     mel_frame_done,

    // Configuration
    input  wire [15:0]             gate_temp,    // FP16 gate temperature

    // Gate output
    output reg  [DATA_WIDTH-1:0]   gate_out,     // 0-255 (Q0.8: 0.0 to 1.0)
    output reg                      gate_valid,
    output reg                      expert_sel    // 0=nonstat, 1=stat (for clock gating)
);

    // ---- Log LUT (256 entries) ----
    // log_lut[x] = round(log(x+1) * 32), x ∈ [0, 255], output Q3.5
    reg signed [7:0] log_lut [0:255];

    // ---- Exp LUT (256 entries) ----
    // exp_lut[x] = round(exp(x/32)), x ∈ [-128, 127], output Q8.0
    reg [7:0] exp_lut [0:255];

    // ---- Sigmoid LUT (256 entries) ----
    // sigmoid_lut[x] = round(sigmoid(x/16 - 8) * 255)
    // Maps INT8 [-128,127] → [0,255] through sigmoid
    reg [7:0] sigmoid_lut [0:255];

    // Initialize LUTs (in FPGA: use .mem file or BRAM init)
    integer li;
    initial begin
        for (li = 0; li < 256; li = li + 1) begin
            log_lut[li]     = 8'd0;    // Filled from init file
            exp_lut[li]     = 8'd0;
            sigmoid_lut[li] = 8'd0;
        end
    end

    // ---- Accumulation registers ----
    reg [31:0] log_sum;           // Sum of log(mel[i]) for geometric mean
    reg [31:0] arith_sum;         // Sum of mel[i] for arithmetic mean
    reg [5:0]  mel_count;

    // ---- State machine ----
    localparam S_IDLE      = 3'd0,
               S_ACCUMULATE = 3'd1,
               S_COMPUTE_SF = 3'd2,
               S_GATE       = 3'd3,
               S_OUTPUT     = 3'd4;

    reg [2:0] state;

    // Intermediate results
    reg [15:0] geo_mean;         // Geometric mean (from exp of mean of logs)
    reg [15:0] arith_mean;       // Arithmetic mean
    reg [7:0]  sf_q8;            // Spectral flatness in Q0.8

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            log_sum    <= 32'd0;
            arith_sum  <= 32'd0;
            mel_count  <= 6'd0;
            gate_valid <= 1'b0;
            expert_sel <= 1'b0;
            gate_out   <= 8'd128;  // Default: 0.5 (equal blend)
        end else begin
            gate_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (mel_valid) begin
                        log_sum   <= 32'd0;
                        arith_sum <= 32'd0;
                        mel_count <= 6'd0;
                        state     <= S_ACCUMULATE;
                    end
                end

                S_ACCUMULATE: begin
                    if (mel_valid) begin
                        // Accumulate log(mel) and mel
                        // log_sum += log_lut[clamp(mel_in >> 8, 0, 255)]
                        log_sum   <= log_sum + {{24{log_lut[mel_in[15:8]][7]}},
                                                 log_lut[mel_in[15:8]]};
                        arith_sum <= arith_sum + {16'b0, mel_in};
                        mel_count <= mel_count + 1;
                    end

                    if (mel_frame_done) begin
                        state <= S_COMPUTE_SF;
                    end
                end

                S_COMPUTE_SF: begin
                    // Geometric mean = exp(log_sum / N_MELS)
                    // Arithmetic mean = arith_sum / N_MELS
                    //
                    // Division by 40: multiply by (256/40) ≈ 6.4 then shift right 8
                    // Or use shift approximation: /40 ≈ /32 - /128 - /512
                    //
                    // SF = geo_mean / arith_mean
                    // Quantized: sf_q8 = (geo_mean * 256) / arith_mean

                    // Simplified: compute mean of logs, then exp
                    // mean_log = log_sum / 40 (approx with shifts)
                    // geo_mean = exp_lut[mean_log]
                    // arith_mean = arith_sum / 40

                    // For now: placeholder computation
                    // In real HW: pipelined divider (8-16 cycles)
                    sf_q8 <= 8'd128;  // Placeholder
                    state <= S_GATE;
                end

                S_GATE: begin
                    // gate = sigmoid(gate_temp * (SF - 0.5))
                    //
                    // In INT8: sf_centered = sf_q8 - 128 (signed, -128 to 127)
                    // gate_temp is FP16, but for INT8 computation:
                    //   scale = gate_temp_int8 (e.g., 5.0 → 40 in Q3.5)
                    //   product = sf_centered * scale (INT16)
                    //   gate = sigmoid_lut[clamp(product >> 5, 0, 255)]

                    begin
                        reg signed [7:0] sf_centered;
                        reg signed [15:0] scaled;
                        reg [7:0] lut_index;

                        sf_centered = sf_q8 - 8'd128;
                        scaled = sf_centered * 8'sd40;  // gate_temp=5.0 → 40 in Q3.5
                        lut_index = (scaled[15]) ? 8'd0 :
                                    (scaled > 16'sd255*32) ? 8'd255 :
                                    scaled[12:5];

                        gate_out <= sigmoid_lut[lut_index];
                    end

                    state <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    gate_valid <= 1'b1;
                    // Expert selection for clock gating
                    // gate > 0.75 (192) → mostly stationary → can gate expert 0
                    // gate < 0.25 (64)  → mostly non-stat  → can gate expert 1
                    expert_sel <= (gate_out > 8'd128) ? 1'b1 : 1'b0;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
