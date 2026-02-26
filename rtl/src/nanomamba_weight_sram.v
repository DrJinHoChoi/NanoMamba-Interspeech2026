// ============================================================================
// NanoMamba Weight SRAM — Dual-Port (Read + Write)
// ============================================================================
// 4,736 × 8-bit = 4.625 KB — stores all INT8 weights for NanoMamba-Tiny
//
// Memory Map:
//   0x0000 - 0x020F : SNR estimator weights (~520 bytes)
//   0x0210 - 0x02CF : PCEN Expert 0 params (160 bytes, delta=2.0)
//   0x02D0 - 0x038F : PCEN Expert 1 params (160 bytes, delta=0.01)
//   0x0390 - 0x0391 : gate_temp (2 bytes, FP16)
//   0x0400 - 0x06FF : Block 0 weights (in_proj, conv1d, x_proj, snr_proj, ...)
//   0x0700 - 0x09FF : Block 1 weights
//   0x0A00 - 0x0A4F : patch_proj (40*16 = 640 bytes)
//   0x0A50 - 0x0AFF : classifier (16*12 = 192 bytes)
//   0x0B00 - 0x127F : reserved / LUT tables
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_weight_sram #(
    parameter DEPTH     = 4736,
    parameter ADDR_W    = 13,
    parameter DATA_W    = 8
)(
    input  wire                clk,

    // Read port (computation datapath)
    input  wire [ADDR_W-1:0]  rd_addr,
    output reg  [DATA_W-1:0]  rd_data,
    input  wire                rd_en,

    // Write port (initialization from AXI)
    input  wire [ADDR_W-1:0]  wr_addr,
    input  wire [DATA_W-1:0]  wr_data,
    input  wire                wr_en
);

    // Single-clock dual-port SRAM
    (* ram_style = "block" *)  // Xilinx: use BRAM
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // Initialize to zero
    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1)
            mem[i] = {DATA_W{1'b0}};
    end

    // Write port
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Read port (1-cycle latency)
    always @(posedge clk) begin
        if (rd_en) begin
            rd_data <= mem[rd_addr];
        end
    end

endmodule
