// ============================================================================
// NanoMamba Register File — AXI4-Lite Slave Interface
// ============================================================================
// Address Map:
//   0x000  CTRL       [RW]  Bit0=Start, Bit1=Stop, Bit2=Reset
//   0x004  STATUS     [RO]  Bit0=Busy, Bit1=Done, Bit2=Error
//   0x008  CONFIG     [RW]  {mode[7:0], n_layers[3:0], d_state[3:0], d_model[7:0]}
//   0x00C  GATE_TEMP  [RW]  FP16 gate temperature for DualPCEN
//   0x010  DELTA_FLOOR[RW]  FP16 delta floor for SA-SSM (default 0x3120 ≈ 0.15)
//   0x014  EPSILON    [RW]  FP16 epsilon for SA-SSM (default 0x2E66 ≈ 0.1)
//   0x018  KW_THRESH  [RW]  INT8 keyword detection threshold
//   0x01C  WEIGHT_CNT [RO]  Total weight count loaded
//   0x020  RESULT_CLS [RO]  Argmax class index (4 bits)
//   0x024  RESULT_CONF[RO]  Max logit value (INT8)
//   0x028  RESULT[0]  [RO]  Class 0 logit
//   ...
//   0x054  RESULT[11] [RO]  Class 11 logit
//   0x100  WEIGHT_LOAD[WO]  Write port: {addr[12:0], data[7:0]}
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_regfile #(
    parameter AXI_ADDR_W    = 12,
    parameter AXI_DATA_W    = 32,
    parameter N_CLASSES      = 12,
    parameter WEIGHT_ADDR_W  = 13,
    parameter WEIGHT_DEPTH   = 4736
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // AXI4-Lite Slave
    input  wire [AXI_ADDR_W-1:0]   s_axi_awaddr,
    input  wire                     s_axi_awvalid,
    output reg                      s_axi_awready,
    input  wire [AXI_DATA_W-1:0]   s_axi_wdata,
    input  wire [3:0]              s_axi_wstrb,
    input  wire                     s_axi_wvalid,
    output reg                      s_axi_wready,
    output reg  [1:0]              s_axi_bresp,
    output reg                      s_axi_bvalid,
    input  wire                     s_axi_bready,
    input  wire [AXI_ADDR_W-1:0]   s_axi_araddr,
    input  wire                     s_axi_arvalid,
    output reg                      s_axi_arready,
    output reg  [AXI_DATA_W-1:0]   s_axi_rdata,
    output reg  [1:0]              s_axi_rresp,
    output reg                      s_axi_rvalid,
    input  wire                     s_axi_rready,

    // Control / Status
    output wire                     ctrl_start,
    output wire                     ctrl_stop,
    output wire                     ctrl_reset,
    input  wire                     status_busy,
    input  wire                     status_done,
    input  wire [3:0]              result_class,
    input  wire [7:0]              result_confidence,

    // Configuration
    output wire [15:0]             cfg_gate_temp,
    output wire [15:0]             cfg_delta_floor,
    output wire [15:0]             cfg_epsilon,
    output wire [7:0]              cfg_kw_threshold,

    // Weight SRAM write port
    output reg  [WEIGHT_ADDR_W-1:0] wt_wr_addr,
    output reg  [7:0]              wt_wr_data,
    output reg                      wt_wr_en
);

    // ---- Register storage ----
    reg  [31:0] reg_ctrl;           // 0x000
    reg  [31:0] reg_config;         // 0x008
    reg  [15:0] reg_gate_temp;      // 0x00C
    reg  [15:0] reg_delta_floor;    // 0x010
    reg  [15:0] reg_epsilon;        // 0x014
    reg  [7:0]  reg_kw_threshold;   // 0x018
    reg  [31:0] reg_weight_cnt;     // 0x01C
    reg  [7:0]  reg_logits [0:N_CLASSES-1]; // 0x028-0x054

    // Status register (read-only, assembled from inputs)
    wire [31:0] reg_status = {29'b0, 1'b0, status_done, status_busy};

    // Control outputs (active for 1 cycle)
    reg ctrl_start_r, ctrl_stop_r, ctrl_reset_r;
    assign ctrl_start = ctrl_start_r;
    assign ctrl_stop  = ctrl_stop_r;
    assign ctrl_reset = ctrl_reset_r;

    // Config outputs
    assign cfg_gate_temp   = reg_gate_temp;
    assign cfg_delta_floor = reg_delta_floor;
    assign cfg_epsilon     = reg_epsilon;
    assign cfg_kw_threshold = reg_kw_threshold;

    // ---- Default values ----
    localparam FP16_GATE_TEMP_5_0  = 16'h4500;  // FP16(5.0)
    localparam FP16_DELTA_FLOOR    = 16'h3120;  // FP16(0.15)  (approx)
    localparam FP16_EPSILON        = 16'h2E66;  // FP16(0.1)   (approx)

    // ---- AXI4-Lite Write FSM ----
    localparam WR_IDLE  = 2'd0,
               WR_DATA  = 2'd1,
               WR_RESP  = 2'd2;

    reg [1:0]  wr_state;
    reg [AXI_ADDR_W-1:0] wr_addr_latch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state       <= WR_IDLE;
            s_axi_awready  <= 1'b0;
            s_axi_wready   <= 1'b0;
            s_axi_bvalid   <= 1'b0;
            s_axi_bresp    <= 2'b00;
            reg_ctrl       <= 32'b0;
            reg_config     <= 32'h0002_0410;  // n_layers=2, d_state=4, d_model=16
            reg_gate_temp  <= FP16_GATE_TEMP_5_0;
            reg_delta_floor <= FP16_DELTA_FLOOR;
            reg_epsilon    <= FP16_EPSILON;
            reg_kw_threshold <= 8'd128;       // 50% confidence
            reg_weight_cnt <= 32'b0;
            wt_wr_en       <= 1'b0;
            ctrl_start_r   <= 1'b0;
            ctrl_stop_r    <= 1'b0;
            ctrl_reset_r   <= 1'b0;
        end else begin
            // Clear one-shot control signals
            ctrl_start_r <= 1'b0;
            ctrl_stop_r  <= 1'b0;
            ctrl_reset_r <= 1'b0;
            wt_wr_en     <= 1'b0;

            case (wr_state)
                WR_IDLE: begin
                    s_axi_awready <= 1'b1;
                    s_axi_wready  <= 1'b0;
                    s_axi_bvalid  <= 1'b0;
                    if (s_axi_awvalid && s_axi_awready) begin
                        wr_addr_latch <= s_axi_awaddr;
                        s_axi_awready <= 1'b0;
                        s_axi_wready  <= 1'b1;
                        wr_state      <= WR_DATA;
                    end
                end

                WR_DATA: begin
                    if (s_axi_wvalid && s_axi_wready) begin
                        s_axi_wready <= 1'b0;

                        // Register write decode
                        case (wr_addr_latch[7:0])
                            8'h00: begin  // CTRL
                                reg_ctrl     <= s_axi_wdata;
                                ctrl_start_r <= s_axi_wdata[0];
                                ctrl_stop_r  <= s_axi_wdata[1];
                                ctrl_reset_r <= s_axi_wdata[2];
                            end
                            8'h08: reg_config       <= s_axi_wdata;
                            8'h0C: reg_gate_temp    <= s_axi_wdata[15:0];
                            8'h10: reg_delta_floor  <= s_axi_wdata[15:0];
                            8'h14: reg_epsilon      <= s_axi_wdata[15:0];
                            8'h18: reg_kw_threshold <= s_axi_wdata[7:0];
                            // Weight load port
                            8'h80: begin  // WEIGHT_LOAD: {addr[20:8], data[7:0]}
                                wt_wr_addr   <= s_axi_wdata[20:8];
                                wt_wr_data   <= s_axi_wdata[7:0];
                                wt_wr_en     <= 1'b1;
                                reg_weight_cnt <= reg_weight_cnt + 1;
                            end
                        endcase

                        s_axi_bvalid <= 1'b1;
                        s_axi_bresp  <= 2'b00;  // OKAY
                        wr_state     <= WR_RESP;
                    end
                end

                WR_RESP: begin
                    if (s_axi_bready && s_axi_bvalid) begin
                        s_axi_bvalid <= 1'b0;
                        wr_state     <= WR_IDLE;
                    end
                end
            endcase
        end
    end

    // ---- AXI4-Lite Read FSM ----
    localparam RD_IDLE = 1'b0,
               RD_DATA = 1'b1;

    reg        rd_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state      <= RD_IDLE;
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'b0;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    s_axi_arready <= 1'b1;
                    s_axi_rvalid  <= 1'b0;
                    if (s_axi_arvalid && s_axi_arready) begin
                        s_axi_arready <= 1'b0;

                        // Register read decode
                        case (s_axi_araddr[7:0])
                            8'h00: s_axi_rdata <= reg_ctrl;
                            8'h04: s_axi_rdata <= reg_status;
                            8'h08: s_axi_rdata <= reg_config;
                            8'h0C: s_axi_rdata <= {16'b0, reg_gate_temp};
                            8'h10: s_axi_rdata <= {16'b0, reg_delta_floor};
                            8'h14: s_axi_rdata <= {16'b0, reg_epsilon};
                            8'h18: s_axi_rdata <= {24'b0, reg_kw_threshold};
                            8'h1C: s_axi_rdata <= reg_weight_cnt;
                            8'h20: s_axi_rdata <= {24'b0, result_class};
                            8'h24: s_axi_rdata <= {24'b0, result_confidence};
                            default: begin
                                // Logits: 0x28 + i*4 for i=0..11
                                if (s_axi_araddr[7:0] >= 8'h28 &&
                                    s_axi_araddr[7:0] < 8'h28 + N_CLASSES*4) begin
                                    s_axi_rdata <= {24'b0,
                                        reg_logits[(s_axi_araddr[7:0] - 8'h28) >> 2]};
                                end else begin
                                    s_axi_rdata <= 32'hDEAD_BEEF;
                                end
                            end
                        endcase

                        s_axi_rvalid <= 1'b1;
                        s_axi_rresp  <= 2'b00;
                        rd_state     <= RD_DATA;
                    end
                end

                RD_DATA: begin
                    if (s_axi_rready && s_axi_rvalid) begin
                        s_axi_rvalid <= 1'b0;
                        rd_state     <= RD_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
