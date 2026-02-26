// ============================================================================
// NanoMamba FPGA Wrapper — Artix-7 Board-Level Integration
// ============================================================================
// Wraps the NanoMamba IP core with:
//   1. Clock management (MMCM: 100MHz → 50MHz)
//   2. I2S audio receiver (16kHz, 16-bit PCM)
//   3. UART debug interface (115200 baud)
//   4. LED status indicators
//   5. Button debouncing
//   6. SPI weight loader (optional)
//
// Target: Digilent Arty A7-35T (XC7A35TCPG236-1)
//
// Pin Usage:
//   PMOD JA: I2S audio input (MCLK, BCLK, LRCK, SDIN)
//   PMOD JB: SPI weight load (CS, SCLK, MOSI, MISO)
//   USB-UART: Debug output (115200 baud)
//   LED[3:0]: Status indicators
//   RGB LED0: Classification result color
//   BTN[3:0]: Control buttons
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_fpga_wrapper (
    // ---- Board Signals ----
    input  wire         clk_100mhz,     // 100MHz oscillator
    input  wire         rst_n,          // Active-low reset (button)

    // ---- I2S Audio Input (PMOD JA) ----
    output wire         i2s_mclk,       // Master clock output
    output wire         i2s_bclk,       // Bit clock
    output wire         i2s_lrck,       // Word select (L/R)
    input  wire         i2s_sdin,       // Serial data in

    // ---- UART Debug (USB-UART) ----
    output wire         uart_tx,
    input  wire         uart_rx,

    // ---- SPI Weight Load (PMOD JB) ----
    output wire         spi_cs_n,
    output wire         spi_sclk,
    output wire         spi_mosi,
    input  wire         spi_miso,

    // ---- LED Status ----
    output wire [3:0]   led,
    output wire         rgb0_r,
    output wire         rgb0_g,
    output wire         rgb0_b,

    // ---- Push Buttons ----
    input  wire [3:0]   btn
);

    // ========================================================================
    // Clock Management (MMCM)
    // ========================================================================
    wire clk_50mhz;
    wire clk_locked;

    // Simple clock divider (for prototyping)
    // In production: use Xilinx MMCM IP core
    reg clk_div2;
    always @(posedge clk_100mhz or negedge rst_n) begin
        if (!rst_n)
            clk_div2 <= 1'b0;
        else
            clk_div2 <= ~clk_div2;
    end
    assign clk_50mhz = clk_div2;
    assign clk_locked = rst_n;  // Simplified for prototyping

    // ========================================================================
    // Button Debounce (20ms debounce period)
    // ========================================================================
    reg [3:0] btn_sync0, btn_sync1, btn_db;
    reg [19:0] db_counter;

    always @(posedge clk_50mhz or negedge rst_n) begin
        if (!rst_n) begin
            btn_sync0  <= 4'b0;
            btn_sync1  <= 4'b0;
            btn_db     <= 4'b0;
            db_counter <= 20'd0;
        end else begin
            // 2-stage synchronizer
            btn_sync0 <= btn;
            btn_sync1 <= btn_sync0;

            // Debounce counter
            if (db_counter == 20'd999_999) begin  // 20ms @ 50MHz
                db_counter <= 20'd0;
                btn_db     <= btn_sync1;
            end else begin
                db_counter <= db_counter + 1;
            end
        end
    end

    wire btn_start = btn_db[0];   // BTN0: Start inference
    wire btn_stop  = btn_db[1];   // BTN1: Stop / deep sleep

    // ========================================================================
    // I2S Receiver
    // ========================================================================
    // I2S clock generation: MCLK = 256*fs = 256*16000 = 4.096MHz
    // BCLK = 32*fs = 32*16000 = 512kHz
    // LRCK = fs = 16kHz

    reg [6:0] i2s_clk_div;       // 50MHz / 97 ≈ 515kHz ≈ BCLK
    reg       i2s_bclk_reg;
    reg [4:0] i2s_bit_cnt;       // 0..31 per L/R channel
    reg       i2s_lrck_reg;
    reg [15:0] i2s_shift_reg;
    reg [15:0] i2s_data_reg;
    reg        i2s_data_valid;

    // MCLK generation (simplified: use BCLK × 8)
    reg [2:0] mclk_div;
    always @(posedge clk_50mhz) begin
        mclk_div <= mclk_div + 1;
    end
    assign i2s_mclk = mclk_div[2];  // ~6.25MHz (approx 4.096MHz)

    // BCLK generation: ~512kHz from 50MHz
    always @(posedge clk_50mhz or negedge rst_n) begin
        if (!rst_n) begin
            i2s_clk_div  <= 7'd0;
            i2s_bclk_reg <= 1'b0;
        end else begin
            if (i2s_clk_div == 7'd48) begin  // 50MHz / (49*2) ≈ 510kHz
                i2s_clk_div  <= 7'd0;
                i2s_bclk_reg <= ~i2s_bclk_reg;
            end else begin
                i2s_clk_div <= i2s_clk_div + 1;
            end
        end
    end
    assign i2s_bclk = i2s_bclk_reg;

    // I2S data reception
    always @(posedge i2s_bclk_reg or negedge rst_n) begin
        if (!rst_n) begin
            i2s_bit_cnt    <= 5'd0;
            i2s_lrck_reg   <= 1'b0;
            i2s_shift_reg  <= 16'd0;
            i2s_data_reg   <= 16'd0;
            i2s_data_valid <= 1'b0;
        end else begin
            i2s_data_valid <= 1'b0;

            // Shift in serial data (MSB first)
            i2s_shift_reg <= {i2s_shift_reg[14:0], i2s_sdin};
            i2s_bit_cnt   <= i2s_bit_cnt + 1;

            // Word boundary (every 16 bits on left channel)
            if (i2s_bit_cnt == 5'd15) begin
                if (!i2s_lrck_reg) begin  // Left channel only (mono)
                    i2s_data_reg   <= i2s_shift_reg;
                    i2s_data_valid <= 1'b1;
                end
            end

            // LRCK toggle every 16 bits
            if (i2s_bit_cnt == 5'd15) begin
                i2s_lrck_reg <= ~i2s_lrck_reg;
            end
        end
    end
    assign i2s_lrck = i2s_lrck_reg;

    // ========================================================================
    // NanoMamba IP Core
    // ========================================================================
    // Internal AXI-Lite signals (directly driven by control logic)
    reg  [11:0] axi_awaddr;
    reg         axi_awvalid;
    wire        axi_awready;
    reg  [31:0] axi_wdata;
    reg  [3:0]  axi_wstrb;
    reg         axi_wvalid;
    wire        axi_wready;
    wire [1:0]  axi_bresp;
    wire        axi_bvalid;
    reg         axi_bready;
    reg  [11:0] axi_araddr;
    reg         axi_arvalid;
    wire        axi_arready;
    wire [31:0] axi_rdata;
    wire [1:0]  axi_rresp;
    wire        axi_rvalid;
    reg         axi_rready;

    wire        irq_done;
    wire        irq_kw_detect;

    nanomamba_top #(
        .N_MELS       (40),
        .N_FFT        (512),
        .N_FREQ       (257),
        .HOP_LENGTH   (160),
        .D_MODEL      (16),
        .D_INNER      (24),
        .D_STATE      (4),
        .D_CONV       (3),
        .N_LAYERS     (2),
        .N_CLASSES    (12),
        .N_EXPERTS    (2),
        .DATA_WIDTH   (8),
        .ACC_WIDTH    (32),
        .WEIGHT_DEPTH (4736),
        .WEIGHT_ADDR_W(13),
        .AXI_ADDR_W   (12),
        .AXI_DATA_W   (32)
    ) u_nanomamba (
        .clk                   (clk_50mhz),
        .rst_n                 (rst_n & clk_locked),
        // AXI4-Lite
        .s_axi_awaddr          (axi_awaddr),
        .s_axi_awvalid         (axi_awvalid),
        .s_axi_awready         (axi_awready),
        .s_axi_wdata           (axi_wdata),
        .s_axi_wstrb           (axi_wstrb),
        .s_axi_wvalid          (axi_wvalid),
        .s_axi_wready          (axi_wready),
        .s_axi_bresp           (axi_bresp),
        .s_axi_bvalid          (axi_bvalid),
        .s_axi_bready          (axi_bready),
        .s_axi_araddr          (axi_araddr),
        .s_axi_arvalid         (axi_arvalid),
        .s_axi_arready         (axi_arready),
        .s_axi_rdata           (axi_rdata),
        .s_axi_rresp           (axi_rresp),
        .s_axi_rvalid          (axi_rvalid),
        .s_axi_rready          (axi_rready),
        // AXI4-Stream Audio Input
        .s_axis_audio_tdata    (i2s_data_reg),
        .s_axis_audio_tvalid   (i2s_data_valid),
        .s_axis_audio_tready   (),          // Always accept
        .s_axis_audio_tlast    (1'b0),      // Continuous stream
        // AXI4-Stream Result
        .m_axis_result_tdata   (),
        .m_axis_result_tvalid  (),
        .m_axis_result_tready  (1'b1),
        .m_axis_result_tlast   (),
        // IRQ
        .irq_done              (irq_done),
        .irq_kw_detect         (irq_kw_detect)
    );

    // ========================================================================
    // Control State Machine
    // ========================================================================
    localparam CTRL_IDLE   = 3'd0,
               CTRL_START  = 3'd1,
               CTRL_RUN    = 3'd2,
               CTRL_DONE   = 3'd3,
               CTRL_STOP   = 3'd4;

    reg [2:0] ctrl_state;
    reg       btn_start_prev;

    always @(posedge clk_50mhz or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state    <= CTRL_IDLE;
            axi_awaddr    <= 12'd0;
            axi_awvalid   <= 1'b0;
            axi_wdata     <= 32'd0;
            axi_wstrb     <= 4'h0;
            axi_wvalid    <= 1'b0;
            axi_bready    <= 1'b0;
            axi_araddr    <= 12'd0;
            axi_arvalid   <= 1'b0;
            axi_rready    <= 1'b0;
            btn_start_prev <= 1'b0;
        end else begin
            btn_start_prev <= btn_start;

            case (ctrl_state)
                CTRL_IDLE: begin
                    // Detect button press rising edge
                    if (btn_start && !btn_start_prev) begin
                        ctrl_state  <= CTRL_START;
                        axi_awaddr  <= 12'h000;  // CTRL register
                        axi_awvalid <= 1'b1;
                        axi_wdata   <= 32'h00000001;  // start=1
                        axi_wstrb   <= 4'hF;
                        axi_wvalid  <= 1'b1;
                        axi_bready  <= 1'b1;
                    end
                end

                CTRL_START: begin
                    if (axi_awready) axi_awvalid <= 1'b0;
                    if (axi_wready) axi_wvalid <= 1'b0;
                    if (axi_bvalid) begin
                        axi_bready <= 1'b0;
                        ctrl_state <= CTRL_RUN;
                    end
                end

                CTRL_RUN: begin
                    if (irq_done) begin
                        ctrl_state <= CTRL_DONE;
                    end
                    if (btn_stop) begin
                        ctrl_state  <= CTRL_STOP;
                        axi_awaddr  <= 12'h000;
                        axi_awvalid <= 1'b1;
                        axi_wdata   <= 32'h00000002;  // stop=1
                        axi_wstrb   <= 4'hF;
                        axi_wvalid  <= 1'b1;
                        axi_bready  <= 1'b1;
                    end
                end

                CTRL_DONE: begin
                    ctrl_state <= CTRL_IDLE;
                end

                CTRL_STOP: begin
                    if (axi_awready) axi_awvalid <= 1'b0;
                    if (axi_wready) axi_wvalid <= 1'b0;
                    if (axi_bvalid) begin
                        axi_bready <= 1'b0;
                        ctrl_state <= CTRL_IDLE;
                    end
                end
            endcase
        end
    end

    // ========================================================================
    // LED Status Display
    // ========================================================================
    assign led[0] = (ctrl_state == CTRL_RUN);     // Processing active
    assign led[1] = irq_kw_detect;                 // Keyword detected
    assign led[2] = u_nanomamba.clk_en_pcen0;      // Expert 0 (non-stat) active
    assign led[3] = u_nanomamba.clk_en_pcen1;      // Expert 1 (stat) active

    // RGB LED: classification result (simplified color mapping)
    // 12 classes → 8 colors (RGB 3-bit)
    reg [2:0] result_color;
    always @(posedge clk_50mhz) begin
        if (irq_done) begin
            result_color <= u_nanomamba.result_class[2:0];
        end
    end
    assign rgb0_r = result_color[0];
    assign rgb0_g = result_color[1];
    assign rgb0_b = result_color[2];

    // ========================================================================
    // UART Debug Output (115200 baud)
    // ========================================================================
    // Simplified: send classification result on irq_done
    reg [7:0]  uart_tx_data;
    reg        uart_tx_start;
    reg [3:0]  uart_bit_cnt;
    reg [15:0] uart_baud_cnt;
    reg        uart_tx_reg;
    reg [9:0]  uart_shift;

    localparam BAUD_DIV = 16'd434;  // 50MHz / 115200 ≈ 434

    assign uart_tx = uart_tx_reg;

    always @(posedge clk_50mhz or negedge rst_n) begin
        if (!rst_n) begin
            uart_tx_reg   <= 1'b1;  // Idle high
            uart_tx_start <= 1'b0;
            uart_bit_cnt  <= 4'd0;
            uart_baud_cnt <= 16'd0;
        end else begin
            // Trigger on classification done
            if (irq_done && !uart_tx_start) begin
                uart_tx_start <= 1'b1;
                uart_shift    <= {1'b1, u_nanomamba.result_class[3:0], 4'b0, 1'b0};  // Start + data + stop
                uart_tx_data  <= {4'b0, u_nanomamba.result_class};
                uart_bit_cnt  <= 4'd0;
                uart_baud_cnt <= 16'd0;
            end

            if (uart_tx_start) begin
                if (uart_baud_cnt == BAUD_DIV - 1) begin
                    uart_baud_cnt <= 16'd0;
                    uart_tx_reg   <= uart_shift[0];
                    uart_shift    <= {1'b1, uart_shift[9:1]};
                    uart_bit_cnt  <= uart_bit_cnt + 1;

                    if (uart_bit_cnt == 4'd10) begin
                        uart_tx_start <= 1'b0;
                        uart_tx_reg   <= 1'b1;  // Return to idle
                    end
                end else begin
                    uart_baud_cnt <= uart_baud_cnt + 1;
                end
            end
        end
    end

    // SPI interface (directly pass-through for weight loading via external controller)
    assign spi_cs_n = 1'b1;   // Inactive (no SPI transaction by default)
    assign spi_sclk = 1'b0;
    assign spi_mosi = 1'b0;

endmodule
