// ============================================================================
// NanoMamba ASIC/FPGA Top-Level IP Block
// ============================================================================
// Description: Ultra-lightweight noise-robust KWS hardware accelerator
//   - DualPCEN MOE with Spectral Flatness routing
//   - SA-SSM with SNR-modulated dt and B
//   - INT8 datapath (4.5KB weight SRAM)
//   - AXI4-Lite register interface + AXI4-Stream audio I/O
//
// Target: BT Audio SoC (Qualcomm QCC517x, JieLi AC79, BES2700)
//         FPGA prototyping: Xilinx Artix-7 / Intel Cyclone V
//
// Author : Jin Ho Choi, Ph.D.
// Date   : 2026-02-26
// License: Patent pending — All rights reserved
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_top #(
    // ---- Architecture Parameters ----
    parameter N_MELS        = 40,       // Mel filterbank bands
    parameter N_FFT         = 512,      // FFT size
    parameter N_FREQ        = 257,      // FFT bins (N_FFT/2 + 1)
    parameter HOP_LENGTH    = 160,      // STFT hop (10ms @ 16kHz)
    parameter D_MODEL       = 16,       // Model dimension
    parameter D_INNER       = 24,       // Inner dimension (d_model * expand)
    parameter D_STATE       = 4,        // SSM state dimension
    parameter D_CONV        = 3,        // Depthwise conv kernel size
    parameter N_LAYERS      = 2,        // Number of SA-SSM blocks
    parameter N_CLASSES     = 12,       // Output classes
    parameter N_EXPERTS     = 2,        // DualPCEN experts

    // ---- INT8 Quantization ----
    parameter DATA_WIDTH    = 8,        // INT8 datapath
    parameter ACC_WIDTH     = 32,       // Accumulator width
    parameter FRAC_BITS     = 7,        // Q1.7 fixed-point for weights

    // ---- Memory ----
    parameter WEIGHT_DEPTH  = 4736,     // ~4.6KB INT8 weights
    parameter WEIGHT_ADDR_W = 13,       // ceil(log2(4736))

    // ---- Bus Interface ----
    parameter AXI_ADDR_W    = 12,       // 4KB register space
    parameter AXI_DATA_W    = 32        // 32-bit data bus
)(
    // ---- Clock & Reset ----
    input  wire                     clk,            // System clock (50-100MHz)
    input  wire                     rst_n,          // Active-low reset

    // ---- AXI4-Lite Slave (Register Access) ----
    input  wire [AXI_ADDR_W-1:0]   s_axi_awaddr,
    input  wire                     s_axi_awvalid,
    output wire                     s_axi_awready,
    input  wire [AXI_DATA_W-1:0]   s_axi_wdata,
    input  wire [3:0]              s_axi_wstrb,
    input  wire                     s_axi_wvalid,
    output wire                     s_axi_wready,
    output wire [1:0]              s_axi_bresp,
    output wire                     s_axi_bvalid,
    input  wire                     s_axi_bready,
    input  wire [AXI_ADDR_W-1:0]   s_axi_araddr,
    input  wire                     s_axi_arvalid,
    output wire                     s_axi_arready,
    output wire [AXI_DATA_W-1:0]   s_axi_rdata,
    output wire [1:0]              s_axi_rresp,
    output wire                     s_axi_rvalid,
    input  wire                     s_axi_rready,

    // ---- AXI4-Stream Audio Input ----
    input  wire [15:0]             s_axis_audio_tdata,   // 16-bit PCM
    input  wire                     s_axis_audio_tvalid,
    output wire                     s_axis_audio_tready,
    input  wire                     s_axis_audio_tlast,   // Frame boundary

    // ---- AXI4-Stream Result Output ----
    output wire [N_CLASSES*8-1:0]  m_axis_result_tdata,  // 12 × INT8 logits
    output wire                     m_axis_result_tvalid,
    input  wire                     m_axis_result_tready,
    output wire                     m_axis_result_tlast,

    // ---- Interrupt ----
    output wire                     irq_done,       // Utterance processed
    output wire                     irq_kw_detect   // Keyword detected (above threshold)
);

    // ========================================================================
    // Internal Wires
    // ========================================================================

    // Control/Status
    wire        ctrl_start;
    wire        ctrl_stop;
    wire        ctrl_reset;
    wire        status_busy;
    wire        status_done;
    wire [3:0]  result_class;       // argmax class index
    wire [7:0]  result_confidence;  // max logit value

    // Configuration registers
    wire [15:0] cfg_gate_temp;      // DualPCEN gate temperature (FP16)
    wire [15:0] cfg_delta_floor;    // SA-SSM delta floor (FP16)
    wire [15:0] cfg_epsilon;        // SA-SSM epsilon (FP16)
    wire [7:0]  cfg_kw_threshold;   // Keyword detection threshold

    // Weight memory interface
    wire [WEIGHT_ADDR_W-1:0]  wt_addr;
    wire [DATA_WIDTH-1:0]     wt_rdata;
    wire                      wt_rd_en;

    // Pipeline stage connections
    // STFT → SNR Estimator
    wire [15:0]  stft_mag [0:N_FREQ-1];   // Magnitude spectrum
    wire         stft_valid;
    wire         stft_frame_done;

    // SNR Estimator → MOE Router / SA-SSM
    wire [7:0]   snr_mel [0:N_MELS-1];    // Per-mel-band SNR (INT8)
    wire         snr_valid;

    // MOE Router → PCEN mux
    wire [7:0]   moe_gate;                 // Gate value (0-255 = 0.0-1.0)
    wire         moe_valid;
    wire         expert_sel;               // 0=nonstat, 1=stat (for clock gating)

    // PCEN → SSM
    wire [DATA_WIDTH-1:0] pcen_out [0:N_MELS-1];  // PCEN output (INT8)
    wire                  pcen_valid;

    // SSM → Classifier
    wire [DATA_WIDTH-1:0] ssm_out [0:D_MODEL-1];  // SSM output (INT8)
    wire                  ssm_valid;

    // Classifier → Output
    wire [7:0]   logits [0:N_CLASSES-1];
    wire         logits_valid;

    // Clock gating signals
    wire         clk_stft;
    wire         clk_pcen_expert0;  // Non-stationary expert
    wire         clk_pcen_expert1;  // Stationary expert
    wire         clk_ssm;
    wire         clk_en_stft;
    wire         clk_en_pcen0;
    wire         clk_en_pcen1;
    wire         clk_en_ssm;

    // ========================================================================
    // Module Instantiations
    // ========================================================================

    // ---- 1. AXI4-Lite Register File ----
    nanomamba_regfile #(
        .AXI_ADDR_W     (AXI_ADDR_W),
        .AXI_DATA_W     (AXI_DATA_W),
        .N_CLASSES       (N_CLASSES),
        .WEIGHT_ADDR_W   (WEIGHT_ADDR_W),
        .WEIGHT_DEPTH    (WEIGHT_DEPTH)
    ) u_regfile (
        .clk             (clk),
        .rst_n           (rst_n),
        // AXI4-Lite
        .s_axi_awaddr    (s_axi_awaddr),
        .s_axi_awvalid   (s_axi_awvalid),
        .s_axi_awready   (s_axi_awready),
        .s_axi_wdata     (s_axi_wdata),
        .s_axi_wstrb     (s_axi_wstrb),
        .s_axi_wvalid    (s_axi_wvalid),
        .s_axi_wready    (s_axi_wready),
        .s_axi_bresp     (s_axi_bresp),
        .s_axi_bvalid    (s_axi_bvalid),
        .s_axi_bready    (s_axi_bready),
        .s_axi_araddr    (s_axi_araddr),
        .s_axi_arvalid   (s_axi_arvalid),
        .s_axi_arready   (s_axi_arready),
        .s_axi_rdata     (s_axi_rdata),
        .s_axi_rresp     (s_axi_rresp),
        .s_axi_rvalid    (s_axi_rvalid),
        .s_axi_rready    (s_axi_rready),
        // Control/Status
        .ctrl_start      (ctrl_start),
        .ctrl_stop       (ctrl_stop),
        .ctrl_reset      (ctrl_reset),
        .status_busy     (status_busy),
        .status_done     (status_done),
        .result_class    (result_class),
        .result_confidence(result_confidence),
        // Configuration
        .cfg_gate_temp   (cfg_gate_temp),
        .cfg_delta_floor (cfg_delta_floor),
        .cfg_epsilon     (cfg_epsilon),
        .cfg_kw_threshold(cfg_kw_threshold),
        // Weight memory write port (for initialization)
        .wt_wr_addr      (),  // connected internally
        .wt_wr_data      (),
        .wt_wr_en        ()
    );

    // ---- 2. Weight SRAM (4.5KB INT8) ----
    nanomamba_weight_sram #(
        .DEPTH           (WEIGHT_DEPTH),
        .ADDR_W          (WEIGHT_ADDR_W),
        .DATA_W          (DATA_WIDTH)
    ) u_weight_sram (
        .clk             (clk),
        .rd_addr         (wt_addr),
        .rd_data         (wt_rdata),
        .rd_en           (wt_rd_en),
        .wr_addr         (),  // from regfile
        .wr_data         (),
        .wr_en           ()
    );

    // ---- 3. STFT Unit ----
    nanomamba_stft #(
        .N_FFT           (N_FFT),
        .HOP_LENGTH      (HOP_LENGTH),
        .DATA_WIDTH      (16)
    ) u_stft (
        .clk             (clk_stft),
        .rst_n           (rst_n),
        .audio_in        (s_axis_audio_tdata),
        .audio_valid     (s_axis_audio_tvalid),
        .audio_ready     (s_axis_audio_tready),
        .audio_last      (s_axis_audio_tlast),
        .mag_out         (),   // stft_mag bus
        .mag_valid       (stft_valid),
        .frame_done      (stft_frame_done)
    );

    // ---- 4. SNR Estimator ----
    nanomamba_snr_estimator #(
        .N_FREQ          (N_FREQ),
        .N_MELS          (N_MELS),
        .NOISE_FRAMES    (5),
        .DATA_WIDTH      (DATA_WIDTH)
    ) u_snr_est (
        .clk             (clk),
        .rst_n           (rst_n),
        .mag_in          (),   // from STFT
        .mag_valid       (stft_valid),
        .snr_mel_out     (),   // snr_mel bus
        .snr_valid       (snr_valid)
    );

    // ---- 5. MOE Router (Spectral Flatness) ----
    nanomamba_moe_router #(
        .N_MELS          (N_MELS),
        .DATA_WIDTH      (DATA_WIDTH)
    ) u_moe_router (
        .clk             (clk),
        .rst_n           (rst_n),
        .mel_in          (),   // mel-domain energy
        .mel_valid       (stft_valid),
        .gate_temp       (cfg_gate_temp),
        .gate_out        (moe_gate),
        .gate_valid      (moe_valid),
        .expert_sel      (expert_sel)
    );

    // ---- 6. DualPCEN Unit (2 Experts) ----
    nanomamba_dual_pcen #(
        .N_MELS          (N_MELS),
        .DATA_WIDTH      (DATA_WIDTH),
        .N_EXPERTS       (N_EXPERTS)
    ) u_dual_pcen (
        .clk_expert0     (clk_pcen_expert0),
        .clk_expert1     (clk_pcen_expert1),
        .rst_n           (rst_n),
        .mel_in          (),   // mel linear energy
        .mel_valid       (stft_valid),
        .gate            (moe_gate),
        .gate_valid      (moe_valid),
        .pcen_out        (),   // pcen_out bus
        .pcen_valid      (pcen_valid)
    );

    // ---- 7. SA-SSM Compute Unit ----
    nanomamba_ssm_compute #(
        .D_MODEL         (D_MODEL),
        .D_INNER         (D_INNER),
        .D_STATE         (D_STATE),
        .D_CONV          (D_CONV),
        .N_LAYERS        (N_LAYERS),
        .N_MELS          (N_MELS),
        .DATA_WIDTH      (DATA_WIDTH),
        .ACC_WIDTH       (ACC_WIDTH)
    ) u_ssm_compute (
        .clk             (clk_ssm),
        .rst_n           (rst_n),
        .feat_in         (),   // from PCEN (after patch_proj)
        .feat_valid      (pcen_valid),
        .snr_mel         (),   // from SNR estimator
        .delta_floor     (cfg_delta_floor),
        .epsilon         (cfg_epsilon),
        .wt_addr         (wt_addr),
        .wt_rdata        (wt_rdata),
        .wt_rd_en        (wt_rd_en),
        .ssm_out         (),   // ssm_out bus
        .ssm_valid       (ssm_valid)
    );

    // ---- 8. Classifier (GAP + Linear) ----
    nanomamba_classifier #(
        .D_MODEL         (D_MODEL),
        .N_CLASSES       (N_CLASSES),
        .DATA_WIDTH      (DATA_WIDTH),
        .ACC_WIDTH       (ACC_WIDTH)
    ) u_classifier (
        .clk             (clk),
        .rst_n           (rst_n),
        .feat_in         (),   // from SSM
        .feat_valid      (ssm_valid),
        .wt_addr         (),
        .wt_rdata        (wt_rdata),
        .logits          (),   // logits bus
        .logits_valid    (logits_valid),
        .result_class    (result_class),
        .result_confidence(result_confidence)
    );

    // ---- 9. Power Management ----
    nanomamba_power_mgmt u_power_mgmt (
        .clk             (clk),
        .rst_n           (rst_n),
        .ctrl_start      (ctrl_start),
        .ctrl_stop       (ctrl_stop),
        .status_busy     (status_busy),
        .status_done     (status_done),
        .expert_sel      (expert_sel),
        .frame_done      (stft_frame_done),
        .logits_valid    (logits_valid),
        // Clock gating outputs
        .clk_en_stft     (clk_en_stft),
        .clk_en_pcen0    (clk_en_pcen0),
        .clk_en_pcen1    (clk_en_pcen1),
        .clk_en_ssm      (clk_en_ssm),
        // Gated clocks
        .clk_stft        (clk_stft),
        .clk_pcen0       (clk_pcen_expert0),
        .clk_pcen1       (clk_pcen_expert1),
        .clk_ssm         (clk_ssm)
    );

    // ---- 10. Output & Interrupt ----
    assign irq_done      = status_done;
    assign irq_kw_detect = logits_valid && (result_confidence > cfg_kw_threshold);

    // AXI4-Stream result output
    assign m_axis_result_tvalid = logits_valid;
    assign m_axis_result_tlast  = logits_valid;
    // Pack 12 × INT8 logits into output bus
    genvar gi;
    generate
        for (gi = 0; gi < N_CLASSES; gi = gi + 1) begin : gen_pack_logits
            assign m_axis_result_tdata[gi*8 +: 8] = logits[gi];
        end
    endgenerate

endmodule
