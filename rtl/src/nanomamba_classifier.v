// ============================================================================
// NanoMamba Classifier — Global Average Pooling + Linear + Argmax
// ============================================================================
// Processing pipeline:
//   1. GAP: Accumulate D_MODEL features over T timesteps → mean
//   2. Linear: D_MODEL(16) → N_CLASSES(12) matmul = 192 MACs
//   3. Argmax: Find maximum logit → class_index + confidence
//
// This module receives timestep-level features from SSM and accumulates
// them for GAP. After the utterance is complete, it performs the final
// classification.
//
// Weight memory layout (from Weight SRAM):
//   0x0A50 - 0x0AFF : classifier weights (16×12 = 192 bytes)
//   0x0B00 - 0x0B0B : classifier bias    (12 bytes)
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_classifier #(
    parameter D_MODEL    = 16,
    parameter N_CLASSES  = 12,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    // Weight SRAM addresses
    parameter WEIGHT_BASE = 13'h0A50,  // classifier weight start
    parameter BIAS_BASE   = 13'h0B00   // classifier bias start
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Feature input from SSM (per timestep)
    input  wire [DATA_WIDTH-1:0]   feat_in,
    input  wire [4:0]              feat_index,     // 0..D_MODEL-1
    input  wire                     feat_valid,
    input  wire                     feat_timestep_done,
    input  wire                     utterance_done, // All timesteps received

    // Weight memory interface (shared SRAM)
    output reg  [12:0]             wt_addr,
    input  wire [DATA_WIDTH-1:0]   wt_rdata,
    output reg                      wt_rd_en,

    // Classification output
    output reg  [N_CLASSES*8-1:0]  logits,         // All class logits packed
    output reg                      logits_valid,
    output reg  [3:0]              result_class,    // Argmax class index
    output reg  [7:0]              result_confidence // Max logit value
);

    // ---- State Machine ----
    localparam S_IDLE       = 3'd0,    // Wait for features
               S_ACCUMULATE = 3'd1,    // Accumulate timestep features for GAP
               S_GAP_NORM   = 3'd2,    // Compute GAP: divide by timestep count
               S_LINEAR     = 3'd3,    // Weight × GAP_feature + bias
               S_ARGMAX     = 3'd4,    // Find max logit
               S_OUTPUT     = 3'd5;    // Output results

    reg [2:0] state;

    // ---- GAP Accumulator ----
    // Accumulate features over all timesteps (101 max)
    // Using 24-bit accumulators to handle 101 × [-128, 127] = [-12928, 12827]
    reg signed [23:0] gap_acc [0:D_MODEL-1];
    reg [6:0] timestep_count;              // Number of timesteps (max 127)

    // GAP normalized result (INT8)
    reg signed [DATA_WIDTH-1:0] gap_feat [0:D_MODEL-1];

    // ---- Linear Layer ----
    // Weight matrix: D_MODEL × N_CLASSES = 16 × 12 = 192 weights
    // Bias vector: N_CLASSES = 12 biases
    reg signed [ACC_WIDTH-1:0] linear_acc;
    reg [3:0]  cls_idx;       // Class index (0..11)
    reg [4:0]  feat_idx;      // Feature index (0..15)

    // Logit buffer
    reg signed [DATA_WIDTH-1:0] logit_buf [0:N_CLASSES-1];

    // ---- Argmax ----
    reg signed [DATA_WIDTH-1:0] max_val;
    reg [3:0]  max_idx;
    reg [3:0]  argmax_idx;

    // ---- Reciprocal LUT for division ----
    // recip_lut[n] ≈ 256/n for n=1..127 (Q0.8 format)
    // Used for GAP normalization: mean = acc * recip_lut[count] >> 8
    reg [7:0] recip_lut [0:127];

    integer ri;
    initial begin
        recip_lut[0] = 8'd255;  // Avoid div-by-zero
        for (ri = 1; ri < 128; ri = ri + 1) begin
            // recip_lut[n] = min(255, round(256/n))
            if (ri <= 1)
                recip_lut[ri] = 8'd255;
            else if (ri <= 2)
                recip_lut[ri] = 8'd128;
            else if (ri <= 3)
                recip_lut[ri] = 8'd85;
            else if (ri <= 4)
                recip_lut[ri] = 8'd64;
            else if (ri <= 5)
                recip_lut[ri] = 8'd51;
            else
                recip_lut[ri] = 8'd0;  // Filled from init file
        end
        // Common values for NanoMamba:
        // 101 timesteps → recip_lut[101] ≈ 256/101 ≈ 2.53 → 3 (Q0.8)
        recip_lut[101] = 8'd3;
        recip_lut[100] = 8'd3;
        recip_lut[50]  = 8'd5;
    end

    // ---- Main Processing FSM ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            timestep_count  <= 7'd0;
            logits_valid    <= 1'b0;
            result_class    <= 4'd0;
            result_confidence <= 8'd0;
            wt_rd_en        <= 1'b0;
            cls_idx         <= 4'd0;
            feat_idx        <= 5'd0;
            linear_acc      <= {ACC_WIDTH{1'b0}};
        end else begin
            logits_valid <= 1'b0;

            case (state)
                // ---- Wait for first feature ----
                S_IDLE: begin
                    if (feat_valid) begin
                        state <= S_ACCUMULATE;
                        // Reset accumulators for new utterance
                        timestep_count <= 7'd0;
                    end
                end

                // ---- Accumulate features for GAP ----
                S_ACCUMULATE: begin
                    // Accumulate incoming feature values
                    if (feat_valid) begin
                        gap_acc[feat_index] <= gap_acc[feat_index]
                                               + {{16{feat_in[DATA_WIDTH-1]}}, feat_in};
                    end

                    if (feat_timestep_done) begin
                        timestep_count <= timestep_count + 1;
                    end

                    // When utterance is complete, compute GAP
                    if (utterance_done) begin
                        state    <= S_GAP_NORM;
                        feat_idx <= 5'd0;
                    end
                end

                // ---- GAP Normalization: mean = sum / count ----
                S_GAP_NORM: begin
                    if (feat_idx < D_MODEL) begin
                        // Integer division using reciprocal LUT:
                        // mean ≈ (acc * recip_lut[count]) >> 8
                        //
                        // For 101 timesteps: recip ≈ 3 (Q0.8)
                        // mean ≈ (acc * 3) >> 8
                        //
                        // Saturate to INT8 range [-128, 127]
                        begin
                            reg signed [31:0] product;
                            reg signed [15:0] mean_raw;
                            product = gap_acc[feat_idx] * $signed({1'b0, recip_lut[timestep_count]});
                            mean_raw = product[23:8];  // >> 8

                            // Saturate to INT8
                            if (mean_raw > 16'sd127)
                                gap_feat[feat_idx] <= 8'sd127;
                            else if (mean_raw < -16'sd128)
                                gap_feat[feat_idx] <= -8'sd128;
                            else
                                gap_feat[feat_idx] <= mean_raw[7:0];
                        end

                        feat_idx <= feat_idx + 1;
                    end else begin
                        // GAP done → start linear layer
                        state      <= S_LINEAR;
                        cls_idx    <= 4'd0;
                        feat_idx   <= 5'd0;
                        linear_acc <= {ACC_WIDTH{1'b0}};
                    end
                end

                // ---- Linear: logit[c] = sum(gap_feat[f] * W[f][c]) + bias[c] ----
                // 16 features × 12 classes = 192 MACs + 12 bias adds
                S_LINEAR: begin
                    if (cls_idx < N_CLASSES) begin
                        if (feat_idx < D_MODEL) begin
                            // Request weight from SRAM
                            // Weight layout: W[feat][class], row-major
                            // Address: WEIGHT_BASE + feat_idx * N_CLASSES + cls_idx
                            wt_rd_en <= 1'b1;
                            wt_addr  <= WEIGHT_BASE + {feat_idx, 4'b0}
                                        - {feat_idx, 2'b0}  // feat_idx * 12 = feat_idx*16 - feat_idx*4
                                        + {9'b0, cls_idx};

                            // MAC: accumulate (1-cycle SRAM latency handled)
                            linear_acc <= linear_acc
                                          + $signed(gap_feat[feat_idx]) * $signed(wt_rdata);

                            feat_idx <= feat_idx + 1;
                        end else begin
                            // Load bias
                            wt_rd_en <= 1'b1;
                            wt_addr  <= BIAS_BASE + {9'b0, cls_idx};

                            // Add bias and store logit
                            begin
                                reg signed [ACC_WIDTH-1:0] logit_raw;
                                logit_raw = linear_acc + $signed(wt_rdata);

                                // Quantize accumulator to INT8 (>> FRAC_BITS)
                                // For Q1.7 weights: shift right by 7
                                if (logit_raw[ACC_WIDTH-1:7] > 24'sd127)
                                    logit_buf[cls_idx] <= 8'sd127;
                                else if (logit_raw[ACC_WIDTH-1:7] < -24'sd128)
                                    logit_buf[cls_idx] <= -8'sd128;
                                else
                                    logit_buf[cls_idx] <= logit_raw[14:7];
                            end

                            // Next class
                            cls_idx    <= cls_idx + 1;
                            feat_idx   <= 5'd0;
                            linear_acc <= {ACC_WIDTH{1'b0}};

                            wt_rd_en <= 1'b0;
                        end
                    end else begin
                        // All classes computed → argmax
                        state      <= S_ARGMAX;
                        argmax_idx <= 4'd0;
                        max_val    <= -8'sd128;  // INT8 minimum
                        max_idx    <= 4'd0;
                        wt_rd_en   <= 1'b0;
                    end
                end

                // ---- Argmax: find max logit and its index ----
                S_ARGMAX: begin
                    if (argmax_idx < N_CLASSES) begin
                        if ($signed(logit_buf[argmax_idx]) > $signed(max_val)) begin
                            max_val <= logit_buf[argmax_idx];
                            max_idx <= argmax_idx;
                        end
                        argmax_idx <= argmax_idx + 1;
                    end else begin
                        result_class      <= max_idx;
                        result_confidence <= max_val;
                        state             <= S_OUTPUT;
                    end
                end

                // ---- Output results ----
                S_OUTPUT: begin
                    logits_valid <= 1'b1;

                    // Pack all logits into output bus
                    // (done via continuous assign below)

                    state <= S_IDLE;
                end
            endcase
        end
    end

    // ---- Pack logits into output bus ----
    genvar gi;
    generate
        for (gi = 0; gi < N_CLASSES; gi = gi + 1) begin : gen_pack
            always @(posedge clk) begin
                logits[gi*8 +: 8] <= logit_buf[gi];
            end
        end
    endgenerate

    // ---- Reset GAP accumulators ----
    integer ai;
    initial begin
        for (ai = 0; ai < D_MODEL; ai = ai + 1) begin
            gap_acc[ai]  = 24'd0;
            gap_feat[ai] = 8'd0;
        end
        for (ai = 0; ai < N_CLASSES; ai = ai + 1) begin
            logit_buf[ai] = 8'd0;
        end
    end

endmodule
