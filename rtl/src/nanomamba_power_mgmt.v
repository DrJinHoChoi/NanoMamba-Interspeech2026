// ============================================================================
// NanoMamba Power Management — Expert Clock Gating + Auto-Sleep
// ============================================================================
// Power optimization features:
//   1. Expert-level clock gating: disable unused PCEN expert
//   2. Inter-frame auto-sleep: gate all clocks between audio frames
//   3. Wake-on-voice: wake from deep sleep on VAD trigger
//   4. Dynamic frequency scaling ready (external PLL control)
//
// Power breakdown (estimated @ 50MHz, 28nm):
//   Active processing:  ~0.5mW (10ms frame, ~0.5ms active = 5% duty)
//   Inter-frame sleep:  ~0.05mW (clock-gated, only leakage)
//   Deep sleep (VAD):   ~0.01mW (all power domains off except VAD)
//   Average always-on:  ~0.08mW (with 95% sleep duty cycle)
//
// Author : Jin Ho Choi, Ph.D.
// ============================================================================

`timescale 1ns / 1ps

module nanomamba_power_mgmt (
    input  wire     clk,
    input  wire     rst_n,

    // Control
    input  wire     ctrl_start,
    input  wire     ctrl_stop,
    output reg      status_busy,
    output reg      status_done,

    // MOE routing feedback
    input  wire     expert_sel,      // 0=nonstat, 1=stat
    input  wire     frame_done,      // STFT frame complete
    input  wire     logits_valid,    // Classification complete

    // External VAD trigger (from always-on VAD module)
    input  wire     vad_trigger,

    // Clock enable outputs (active-high)
    output reg      clk_en_stft,
    output reg      clk_en_pcen0,    // Non-stationary expert
    output reg      clk_en_pcen1,    // Stationary expert
    output reg      clk_en_ssm,

    // Gated clocks (active when enabled)
    output wire     clk_stft,
    output wire     clk_pcen0,
    output wire     clk_pcen1,
    output wire     clk_ssm
);

    // ---- Power States ----
    localparam PWR_DEEP_SLEEP  = 3'd0,  // All off, wait for VAD
               PWR_WAKING      = 3'd1,  // Clock stabilization
               PWR_IDLE        = 3'd2,  // Clocks on, waiting for audio
               PWR_STFT_ACTIVE = 3'd3,  // STFT processing frame
               PWR_COMPUTE     = 3'd4,  // PCEN + SSM computation
               PWR_INTER_FRAME = 3'd5,  // Sleep between frames (10ms period)
               PWR_DONE        = 3'd6;  // Classification complete

    reg [2:0] pwr_state;

    // ---- Frame timing ----
    reg [19:0] sleep_timer;           // Counter for inter-frame sleep
    localparam SLEEP_CYCLES = 20'd475000;  // ~9.5ms @ 50MHz (10ms - 0.5ms compute)

    // ---- Wake-up latency counter ----
    reg [7:0] wake_counter;
    localparam WAKE_CYCLES = 8'd10;   // 10 cycles for clock stabilization

    // ---- Expert clock gating logic ----
    // Based on MOE router output: disable unused expert
    reg expert_sel_latch;
    reg [7:0] gate_threshold_high;   // gate > 192 → only expert 1
    reg [7:0] gate_threshold_low;    // gate < 64  → only expert 0

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pwr_state    <= PWR_DEEP_SLEEP;
            status_busy  <= 1'b0;
            status_done  <= 1'b0;
            clk_en_stft  <= 1'b0;
            clk_en_pcen0 <= 1'b0;
            clk_en_pcen1 <= 1'b0;
            clk_en_ssm   <= 1'b0;
            sleep_timer  <= 20'd0;
            wake_counter <= 8'd0;
            expert_sel_latch <= 1'b0;
            gate_threshold_high <= 8'd192;
            gate_threshold_low  <= 8'd64;
        end else begin

            case (pwr_state)
                // ---- Deep Sleep: Everything off ----
                PWR_DEEP_SLEEP: begin
                    clk_en_stft  <= 1'b0;
                    clk_en_pcen0 <= 1'b0;
                    clk_en_pcen1 <= 1'b0;
                    clk_en_ssm   <= 1'b0;
                    status_busy  <= 1'b0;
                    status_done  <= 1'b0;

                    if (ctrl_start || vad_trigger) begin
                        pwr_state    <= PWR_WAKING;
                        wake_counter <= WAKE_CYCLES;
                        status_busy  <= 1'b1;
                    end
                end

                // ---- Waking: Clock stabilization ----
                PWR_WAKING: begin
                    clk_en_stft <= 1'b1;  // Enable STFT clock first
                    if (wake_counter > 0) begin
                        wake_counter <= wake_counter - 1;
                    end else begin
                        pwr_state <= PWR_IDLE;
                    end
                end

                // ---- Idle: Ready for audio ----
                PWR_IDLE: begin
                    clk_en_stft  <= 1'b1;
                    clk_en_pcen0 <= 1'b0;  // PCEN/SSM off until needed
                    clk_en_pcen1 <= 1'b0;
                    clk_en_ssm   <= 1'b0;
                    pwr_state    <= PWR_STFT_ACTIVE;
                end

                // ---- STFT Active: Processing audio frame ----
                PWR_STFT_ACTIVE: begin
                    clk_en_stft <= 1'b1;

                    if (frame_done) begin
                        // STFT done → enable compute clocks
                        clk_en_ssm <= 1'b1;

                        // Expert clock gating based on routing
                        expert_sel_latch <= expert_sel;
                        if (expert_sel) begin
                            // Stationary dominant → can potentially gate expert 0
                            clk_en_pcen0 <= 1'b0;  // Gate non-stat expert
                            clk_en_pcen1 <= 1'b1;
                        end else begin
                            // Non-stationary dominant → can potentially gate expert 1
                            clk_en_pcen0 <= 1'b1;
                            clk_en_pcen1 <= 1'b0;  // Gate stat expert
                        end

                        // For safety: enable both during initial frames
                        // (until routing stabilizes)
                        // clk_en_pcen0 <= 1'b1;
                        // clk_en_pcen1 <= 1'b1;

                        pwr_state <= PWR_COMPUTE;
                    end

                    if (ctrl_stop) begin
                        pwr_state <= PWR_DEEP_SLEEP;
                    end
                end

                // ---- Compute: PCEN + SSM active ----
                PWR_COMPUTE: begin
                    if (logits_valid) begin
                        // Classification done for this utterance
                        pwr_state   <= PWR_DONE;
                        status_done <= 1'b1;
                    end else if (frame_done) begin
                        // Move to inter-frame sleep
                        pwr_state   <= PWR_INTER_FRAME;
                        sleep_timer <= SLEEP_CYCLES;
                    end
                end

                // ---- Inter-Frame Sleep: Gate clocks for ~9.5ms ----
                PWR_INTER_FRAME: begin
                    // Gate all compute clocks
                    clk_en_pcen0 <= 1'b0;
                    clk_en_pcen1 <= 1'b0;
                    clk_en_ssm   <= 1'b0;
                    clk_en_stft  <= 1'b0;

                    if (sleep_timer > 0) begin
                        sleep_timer <= sleep_timer - 1;
                    end else begin
                        // Wake up for next frame
                        clk_en_stft <= 1'b1;
                        pwr_state   <= PWR_STFT_ACTIVE;
                    end

                    if (ctrl_stop) begin
                        pwr_state <= PWR_DEEP_SLEEP;
                    end
                end

                // ---- Done: Classification complete ----
                PWR_DONE: begin
                    status_done <= 1'b1;
                    // Return to deep sleep or wait for next utterance
                    if (ctrl_start) begin
                        status_done <= 1'b0;
                        pwr_state   <= PWR_IDLE;
                    end else begin
                        // Auto-return to deep sleep after done
                        pwr_state <= PWR_DEEP_SLEEP;
                    end
                end
            endcase
        end
    end

    // ---- Clock Gating Cells ----
    // In ASIC: use ICG (Integrated Clock Gating) cells from standard cell library
    // In FPGA: use BUFGCE (Xilinx) or ALTCLKCTRL (Intel)
    //
    // Proper ICG implementation (latch-based, glitch-free):
    //   EN_latch = (clk == 0) ? EN : EN_latch;
    //   gated_clk = clk & EN_latch;

    // FPGA-compatible gating (using AND gate — for prototyping only)
    // In ASIC synthesis: replace with library ICG cell
    `ifdef FPGA_TARGET
        // Xilinx BUFGCE equivalent
        assign clk_stft  = clk & clk_en_stft;
        assign clk_pcen0 = clk & clk_en_pcen0;
        assign clk_pcen1 = clk & clk_en_pcen1;
        assign clk_ssm   = clk & clk_en_ssm;
    `else
        // ASIC: Latch-based ICG (synthesis tool will map to library cell)
        reg clk_en_stft_latch, clk_en_pcen0_latch, clk_en_pcen1_latch, clk_en_ssm_latch;

        always @(*) begin
            if (!clk) begin
                clk_en_stft_latch  = clk_en_stft;
                clk_en_pcen0_latch = clk_en_pcen0;
                clk_en_pcen1_latch = clk_en_pcen1;
                clk_en_ssm_latch   = clk_en_ssm;
            end
        end

        assign clk_stft  = clk & clk_en_stft_latch;
        assign clk_pcen0 = clk & clk_en_pcen0_latch;
        assign clk_pcen1 = clk & clk_en_pcen1_latch;
        assign clk_ssm   = clk & clk_en_ssm_latch;
    `endif

endmodule
