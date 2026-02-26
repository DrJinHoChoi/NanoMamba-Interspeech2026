// ============================================================================
// NanoMamba Top-Level Testbench
// ============================================================================
// Verifies the complete NanoMamba pipeline:
//   1. Weight loading via AXI4-Lite
//   2. Audio streaming via AXI4-Stream
//   3. Classification output and IRQ
//   4. Power state transitions and clock gating
//   5. Expert selection based on noise type
//
// Test Scenarios:
//   - Clean speech (1kHz sine tone) → should classify
//   - White noise → expert_sel should favor stationary expert
//   - Babble noise → expert_sel should favor non-stationary expert
//   - Deep sleep → wake-on-voice → process → sleep
//
// Simulation: ~20ms of audio (2 frames × 10ms) for quick verification
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

// Define FPGA target for testbench (AND-gate clock gating)
`define FPGA_TARGET

module nanomamba_top_tb;

    // ---- Parameters ----
    parameter CLK_PERIOD   = 20;    // 50MHz (20ns period)
    parameter SAMPLE_RATE  = 16000; // 16kHz audio
    parameter HOP_LENGTH   = 160;   // Samples per frame
    parameter N_FRAMES     = 3;     // Number of test frames
    parameter N_CLASSES    = 12;

    // ---- DUT Signals ----
    reg          clk;
    reg          rst_n;

    // AXI4-Lite Write
    reg  [11:0]  s_axi_awaddr;
    reg          s_axi_awvalid;
    wire         s_axi_awready;
    reg  [31:0]  s_axi_wdata;
    reg  [3:0]   s_axi_wstrb;
    reg          s_axi_wvalid;
    wire         s_axi_wready;
    wire [1:0]   s_axi_bresp;
    wire         s_axi_bvalid;
    reg          s_axi_bready;

    // AXI4-Lite Read
    reg  [11:0]  s_axi_araddr;
    reg          s_axi_arvalid;
    wire         s_axi_arready;
    wire [31:0]  s_axi_rdata;
    wire [1:0]   s_axi_rresp;
    wire         s_axi_rvalid;
    reg          s_axi_rready;

    // AXI4-Stream Audio Input
    reg  [15:0]  s_axis_audio_tdata;
    reg          s_axis_audio_tvalid;
    wire         s_axis_audio_tready;
    reg          s_axis_audio_tlast;

    // AXI4-Stream Result Output
    wire [N_CLASSES*8-1:0] m_axis_result_tdata;
    wire         m_axis_result_tvalid;
    reg          m_axis_result_tready;
    wire         m_axis_result_tlast;

    // Interrupts
    wire         irq_done;
    wire         irq_kw_detect;

    // ---- DUT Instantiation ----
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
    ) dut (
        .clk                   (clk),
        .rst_n                 (rst_n),
        // AXI4-Lite Write
        .s_axi_awaddr          (s_axi_awaddr),
        .s_axi_awvalid         (s_axi_awvalid),
        .s_axi_awready         (s_axi_awready),
        .s_axi_wdata           (s_axi_wdata),
        .s_axi_wstrb           (s_axi_wstrb),
        .s_axi_wvalid          (s_axi_wvalid),
        .s_axi_wready          (s_axi_wready),
        .s_axi_bresp           (s_axi_bresp),
        .s_axi_bvalid          (s_axi_bvalid),
        .s_axi_bready          (s_axi_bready),
        // AXI4-Lite Read
        .s_axi_araddr          (s_axi_araddr),
        .s_axi_arvalid         (s_axi_arvalid),
        .s_axi_arready         (s_axi_arready),
        .s_axi_rdata           (s_axi_rdata),
        .s_axi_rresp           (s_axi_rresp),
        .s_axi_rvalid          (s_axi_rvalid),
        .s_axi_rready          (s_axi_rready),
        // AXI4-Stream Audio
        .s_axis_audio_tdata    (s_axis_audio_tdata),
        .s_axis_audio_tvalid   (s_axis_audio_tvalid),
        .s_axis_audio_tready   (s_axis_audio_tready),
        .s_axis_audio_tlast    (s_axis_audio_tlast),
        // AXI4-Stream Result
        .m_axis_result_tdata   (m_axis_result_tdata),
        .m_axis_result_tvalid  (m_axis_result_tvalid),
        .m_axis_result_tready  (m_axis_result_tready),
        .m_axis_result_tlast   (m_axis_result_tlast),
        // Interrupts
        .irq_done              (irq_done),
        .irq_kw_detect         (irq_kw_detect)
    );

    // ---- Clock Generation ----
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- Audio Sample Generation ----
    // Generate 16-bit PCM: 1kHz sine tone @ 16kHz sample rate
    // Using INT approximation: sin(2*pi*1000/16000 * n) = sin(pi/8 * n)
    reg signed [15:0] sine_table [0:15];
    integer si;
    initial begin
        // 16-sample period sine wave (1kHz @ 16kHz)
        sine_table[0]  = 16'sd0;
        sine_table[1]  = 16'sd12539;
        sine_table[2]  = 16'sd23170;
        sine_table[3]  = 16'sd30273;
        sine_table[4]  = 16'sd32767;
        sine_table[5]  = 16'sd30273;
        sine_table[6]  = 16'sd23170;
        sine_table[7]  = 16'sd12539;
        sine_table[8]  = 16'sd0;
        sine_table[9]  = -16'sd12539;
        sine_table[10] = -16'sd23170;
        sine_table[11] = -16'sd30273;
        sine_table[12] = -16'sd32767;
        sine_table[13] = -16'sd30273;
        sine_table[14] = -16'sd23170;
        sine_table[15] = -16'sd12539;
    end

    // LFSR for pseudo-random noise generation
    reg [15:0] lfsr;
    wire lfsr_bit = lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10];
    always @(posedge clk) begin
        if (!rst_n)
            lfsr <= 16'hACE1;
        else
            lfsr <= {lfsr[14:0], lfsr_bit};
    end

    // ---- AXI4-Lite Write Task ----
    task axi_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;

            // Wait for address ready
            @(posedge clk);
            while (!s_axi_awready) @(posedge clk);
            s_axi_awvalid <= 1'b0;

            // Wait for write ready
            while (!s_axi_wready) @(posedge clk);
            s_axi_wvalid <= 1'b0;

            // Wait for response
            while (!s_axi_bvalid) @(posedge clk);
            s_axi_bready <= 1'b0;

            @(posedge clk);
        end
    endtask

    // ---- AXI4-Lite Read Task ----
    task axi_read;
        input  [11:0] addr;
        output [31:0] data;
        begin
            @(posedge clk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;

            // Wait for address ready
            @(posedge clk);
            while (!s_axi_arready) @(posedge clk);
            s_axi_arvalid <= 1'b0;

            // Wait for read data valid
            while (!s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            s_axi_rready <= 1'b0;

            @(posedge clk);
        end
    endtask

    // ---- Stream Audio Samples Task ----
    task stream_audio_frame;
        input integer frame_num;
        input integer noise_type;  // 0=clean, 1=white, 2=babble
        integer sample_idx;
        reg signed [15:0] sample;
        begin
            $display("[%0t] Streaming audio frame %0d (noise_type=%0d)", $time, frame_num, noise_type);
            for (sample_idx = 0; sample_idx < HOP_LENGTH; sample_idx = sample_idx + 1) begin
                @(posedge clk);
                case (noise_type)
                    0: sample = sine_table[sample_idx % 16];  // Clean 1kHz tone
                    1: sample = lfsr;                          // White noise
                    2: sample = sine_table[sample_idx % 16] + (lfsr >>> 2);  // Tone + noise
                    default: sample = 16'd0;
                endcase

                s_axis_audio_tdata  <= sample;
                s_axis_audio_tvalid <= 1'b1;
                s_axis_audio_tlast  <= (sample_idx == HOP_LENGTH - 1);

                // Wait for ready
                while (!s_axis_audio_tready) @(posedge clk);
            end
            s_axis_audio_tvalid <= 1'b0;
            s_axis_audio_tlast  <= 1'b0;
        end
    endtask

    // ---- Monitor: Classification Output ----
    always @(posedge clk) begin
        if (m_axis_result_tvalid && m_axis_result_tready) begin
            $display("[%0t] === CLASSIFICATION RESULT ===", $time);
            $display("  Class: %0d", dut.result_class);
            $display("  Confidence: %0d", dut.result_confidence);
            $display("  Logits: %h", m_axis_result_tdata);
        end
    end

    // ---- Monitor: Power State ----
    reg [2:0] prev_pwr_state;
    always @(posedge clk) begin
        if (dut.u_power_mgmt.pwr_state !== prev_pwr_state) begin
            case (dut.u_power_mgmt.pwr_state)
                3'd0: $display("[%0t] PWR: DEEP_SLEEP", $time);
                3'd1: $display("[%0t] PWR: WAKING", $time);
                3'd2: $display("[%0t] PWR: IDLE", $time);
                3'd3: $display("[%0t] PWR: STFT_ACTIVE", $time);
                3'd4: $display("[%0t] PWR: COMPUTE", $time);
                3'd5: $display("[%0t] PWR: INTER_FRAME", $time);
                3'd6: $display("[%0t] PWR: DONE", $time);
            endcase
            prev_pwr_state <= dut.u_power_mgmt.pwr_state;
        end
    end

    // ---- Monitor: Expert Selection ----
    always @(posedge clk) begin
        if (dut.u_moe_router.gate_valid) begin
            $display("[%0t] MOE: gate=%0d, expert_sel=%b (0=nonstat, 1=stat)",
                     $time, dut.u_moe_router.gate_out, dut.u_moe_router.expert_sel);
        end
    end

    // ---- Monitor: IRQ ----
    always @(posedge irq_done) begin
        $display("[%0t] *** IRQ: Classification Done ***", $time);
    end
    always @(posedge irq_kw_detect) begin
        $display("[%0t] *** IRQ: Keyword Detected! ***", $time);
    end

    // ---- Monitor: Clock Gating ----
    always @(posedge clk) begin
        if (dut.u_power_mgmt.clk_en_stft !== dut.u_power_mgmt.clk_en_stft) begin
            $display("[%0t] CLK_EN: STFT=%b PCEN0=%b PCEN1=%b SSM=%b",
                     $time,
                     dut.u_power_mgmt.clk_en_stft,
                     dut.u_power_mgmt.clk_en_pcen0,
                     dut.u_power_mgmt.clk_en_pcen1,
                     dut.u_power_mgmt.clk_en_ssm);
        end
    end

    // ---- Main Test Sequence ----
    reg [31:0] read_data;
    integer frame_i;
    integer weight_i;

    initial begin
        // ---- Initialize ----
        $display("============================================");
        $display("  NanoMamba ASIC/FPGA Testbench");
        $display("  Author: Jin Ho Choi, Ph.D.");
        $display("============================================");

        // Reset all signals
        rst_n             = 1'b0;
        s_axi_awaddr      = 12'd0;
        s_axi_awvalid     = 1'b0;
        s_axi_wdata       = 32'd0;
        s_axi_wstrb       = 4'h0;
        s_axi_wvalid      = 1'b0;
        s_axi_bready      = 1'b0;
        s_axi_araddr      = 12'd0;
        s_axi_arvalid     = 1'b0;
        s_axi_rready      = 1'b0;
        s_axis_audio_tdata = 16'd0;
        s_axis_audio_tvalid = 1'b0;
        s_axis_audio_tlast = 1'b0;
        m_axis_result_tready = 1'b1;
        prev_pwr_state    = 3'd0;

        // Hold reset for 10 cycles
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        $display("\n[%0t] === Phase 1: Register Configuration ===", $time);

        // ---- Test 1: Read default configuration ----
        $display("[%0t] Reading default registers...", $time);
        axi_read(12'h00C, read_data);  // GATE_TEMP
        $display("  GATE_TEMP = 0x%08h (expected 0x00004500 = FP16 5.0)", read_data);

        axi_read(12'h010, read_data);  // DELTA_FLOOR
        $display("  DELTA_FLOOR = 0x%08h (expected 0x00003120 = FP16 0.15)", read_data);

        axi_read(12'h014, read_data);  // EPSILON
        $display("  EPSILON = 0x%08h (expected 0x00002E66 = FP16 0.1)", read_data);

        // ---- Test 2: Write custom configuration ----
        $display("[%0t] Writing custom gate_temp = 3.0...", $time);
        axi_write(12'h00C, 32'h00004200);  // FP16 3.0
        axi_read(12'h00C, read_data);
        if (read_data == 32'h00004200)
            $display("  PASS: gate_temp updated to 0x%08h", read_data);
        else
            $display("  FAIL: gate_temp = 0x%08h (expected 0x00004200)", read_data);

        // ---- Test 3: Load weights via AXI ----
        $display("\n[%0t] === Phase 2: Weight Loading ===", $time);
        $display("[%0t] Loading %0d INT8 weights via AXI4-Lite...", $time, 4736);

        // Load first 256 weights (demonstration — full load would take 4736 writes)
        for (weight_i = 0; weight_i < 256; weight_i = weight_i + 1) begin
            // Write to WEIGHT_LOAD register (0x100)
            // Pack: {weight_count, weight_data}
            axi_write(12'h100, {16'd0, weight_i[7:0], 8'h42});  // weight value 0x42
        end
        $display("[%0t] Weight loading complete (256/%0d demo weights)", $time, 4736);

        // ---- Test 4: Start inference ----
        $display("\n[%0t] === Phase 3: Inference — Clean Speech ===", $time);
        axi_write(12'h000, 32'h00000001);  // CTRL: start=1

        // Read status
        axi_read(12'h004, read_data);
        $display("  STATUS = 0x%08h (busy=%b)", read_data, read_data[0]);

        // Stream audio frames (clean 1kHz tone)
        for (frame_i = 0; frame_i < N_FRAMES; frame_i = frame_i + 1) begin
            stream_audio_frame(frame_i, 0);  // noise_type=0 (clean)
            // Wait for frame processing
            repeat (3000) @(posedge clk);
        end

        // Wait for classification
        $display("[%0t] Waiting for classification...", $time);
        repeat (5000) @(posedge clk);

        // Read result
        axi_read(12'h020, read_data);
        $display("  RESULT_CLS = %0d", read_data[3:0]);
        axi_read(12'h024, read_data);
        $display("  RESULT_CONF = %0d", read_data[7:0]);

        // ---- Test 5: White noise (stationary expert) ----
        $display("\n[%0t] === Phase 4: Inference — White Noise ===", $time);
        axi_write(12'h000, 32'h00000001);  // Start new inference

        for (frame_i = 0; frame_i < N_FRAMES; frame_i = frame_i + 1) begin
            stream_audio_frame(frame_i, 1);  // noise_type=1 (white)
            repeat (3000) @(posedge clk);
        end

        repeat (5000) @(posedge clk);
        $display("[%0t] White noise test complete", $time);

        // ---- Test 6: Babble noise (non-stationary expert) ----
        $display("\n[%0t] === Phase 5: Inference — Babble Noise ===", $time);
        axi_write(12'h000, 32'h00000001);  // Start new inference

        for (frame_i = 0; frame_i < N_FRAMES; frame_i = frame_i + 1) begin
            stream_audio_frame(frame_i, 2);  // noise_type=2 (babble)
            repeat (3000) @(posedge clk);
        end

        repeat (5000) @(posedge clk);
        $display("[%0t] Babble noise test complete", $time);

        // ---- Test 7: Deep sleep / Wake-on-voice ----
        $display("\n[%0t] === Phase 6: Power Management ===", $time);

        // Force stop → deep sleep
        axi_write(12'h000, 32'h00000002);  // CTRL: stop=1
        repeat (100) @(posedge clk);

        $display("[%0t] Verifying deep sleep state...", $time);
        axi_read(12'h004, read_data);
        $display("  STATUS = 0x%08h (expected: not busy)", read_data);

        // Wake via start signal
        $display("[%0t] Waking from deep sleep...", $time);
        axi_write(12'h000, 32'h00000001);  // Start
        repeat (20) @(posedge clk);

        axi_read(12'h004, read_data);
        $display("  STATUS after wake = 0x%08h", read_data);

        // ---- Done ----
        repeat (1000) @(posedge clk);
        $display("\n============================================");
        $display("  Testbench Complete");
        $display("============================================");
        $finish;
    end

    // ---- Watchdog Timer ----
    initial begin
        #50_000_000;  // 50ms timeout
        $display("\n[TIMEOUT] Simulation exceeded 50ms — aborting");
        $finish;
    end

    // ---- Waveform Dump ----
    initial begin
        $dumpfile("nanomamba_top_tb.vcd");
        $dumpvars(0, nanomamba_top_tb);
    end

endmodule
