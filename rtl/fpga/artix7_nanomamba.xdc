## ============================================================================
## NanoMamba FPGA Constraints — Xilinx Artix-7 (XC7A35T / XC7A100T)
## ============================================================================
## Target Board: Digilent Arty A7-35T or similar Artix-7 board
## System Clock: 100MHz oscillator → PLL → 50MHz for NanoMamba
##
## Resource Estimates (Artix-7 35T):
##   LUTs:    ~2,500 / 20,800  (12%)
##   FFs:     ~1,800 / 41,600  (4.3%)
##   BRAM:    5 × 36Kb         / 50 (10%)
##   DSP48:   4                / 90 (4.4%)
##   Power:   ~15mW (FPGA prototype, ASIC target <0.1mW)
##
## Author : Jin Ho Choi, Ph.D.
## ============================================================================

## ============================================================================
## Clock Constraints
## ============================================================================

## 100MHz system clock (Arty A7 E3 pin)
set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports clk_100mhz]
create_clock -period 10.000 -name sys_clk [get_ports clk_100mhz]

## PLL-generated 50MHz NanoMamba clock
## (Assuming MMCM/PLL instantiated in top-level wrapper)
create_generated_clock -name nanomamba_clk -source [get_ports clk_100mhz] \
    -divide_by 2 [get_pins u_mmcm/CLKOUT0]

## Clock domain crossing constraints
set_clock_groups -asynchronous \
    -group [get_clocks sys_clk] \
    -group [get_clocks nanomamba_clk]

## ============================================================================
## Clock Gating Constraints
## ============================================================================
## CRITICAL: Ensure Vivado does not optimize away clock gating logic
## NanoMamba uses clock gating for expert-level power management

## Preserve clock enable signals
set_property DONT_TOUCH true [get_cells u_nanomamba_top/u_power_mgmt/*clk_en*]

## BUFGCE instantiation for gated clocks (Artix-7)
## In FPGA: use BUFGCE instead of AND-gate clock gating
## set_property CLOCK_BUFFER_TYPE BUFGCE [get_nets clk_stft]
## set_property CLOCK_BUFFER_TYPE BUFGCE [get_nets clk_pcen_expert0]
## set_property CLOCK_BUFFER_TYPE BUFGCE [get_nets clk_pcen_expert1]
## set_property CLOCK_BUFFER_TYPE BUFGCE [get_nets clk_ssm]

## ============================================================================
## Reset
## ============================================================================

## Active-low reset button (Active-low, directly from button with pull-up)
set_property -dict { PACKAGE_PIN C2 IOSTANDARD LVCMOS33 } [get_ports rst_n]
set_property PULLUP true [get_ports rst_n]

## Reset timing
set_input_delay -clock [get_clocks nanomamba_clk] -max 5.0 [get_ports rst_n]
set_input_delay -clock [get_clocks nanomamba_clk] -min 0.0 [get_ports rst_n]
set_false_path -from [get_ports rst_n]

## ============================================================================
## I2S Audio Input (PMOD JA — Top Row)
## ============================================================================
## For audio prototyping using PMOD I2S2 or similar I2S ADC

## I2S Master Clock Out (MCLK)
set_property -dict { PACKAGE_PIN G13 IOSTANDARD LVCMOS33 } [get_ports i2s_mclk]

## I2S Bit Clock (BCLK/SCK)
set_property -dict { PACKAGE_PIN B11 IOSTANDARD LVCMOS33 } [get_ports i2s_bclk]

## I2S Word Select (LRCK)
set_property -dict { PACKAGE_PIN A11 IOSTANDARD LVCMOS33 } [get_ports i2s_lrck]

## I2S Serial Data In (from ADC)
set_property -dict { PACKAGE_PIN D12 IOSTANDARD LVCMOS33 } [get_ports i2s_sdin]

## I2S timing constraints (16kHz sample rate × 16-bit × 2ch = 512kHz BCLK)
create_clock -period 1953.125 -name i2s_bclk_virtual
set_input_delay -clock i2s_bclk_virtual -max 10.0 [get_ports i2s_sdin]
set_input_delay -clock i2s_bclk_virtual -min 0.0  [get_ports i2s_sdin]

## ============================================================================
## UART Debug Interface (USB-UART, Arty A7 built-in)
## ============================================================================

## UART TX (FPGA → Host)
set_property -dict { PACKAGE_PIN D10 IOSTANDARD LVCMOS33 } [get_ports uart_tx]

## UART RX (Host → FPGA)
set_property -dict { PACKAGE_PIN A9  IOSTANDARD LVCMOS33 } [get_ports uart_rx]

## ============================================================================
## SPI for Weight Loading (PMOD JB — Top Row)
## ============================================================================
## Used to load INT8 weights from external flash/host

## SPI Chip Select
set_property -dict { PACKAGE_PIN E15 IOSTANDARD LVCMOS33 } [get_ports spi_cs_n]

## SPI Clock
set_property -dict { PACKAGE_PIN E16 IOSTANDARD LVCMOS33 } [get_ports spi_sclk]

## SPI MOSI (Host → FPGA)
set_property -dict { PACKAGE_PIN D15 IOSTANDARD LVCMOS33 } [get_ports spi_mosi]

## SPI MISO (FPGA → Host)
set_property -dict { PACKAGE_PIN C15 IOSTANDARD LVCMOS33 } [get_ports spi_miso]

## ============================================================================
## LED Status Indicators
## ============================================================================

## LED[0]: Power/Active (green when processing)
set_property -dict { PACKAGE_PIN H5  IOSTANDARD LVCMOS33 } [get_ports {led[0]}]

## LED[1]: Keyword detected (blinks on detection)
set_property -dict { PACKAGE_PIN J5  IOSTANDARD LVCMOS33 } [get_ports {led[1]}]

## LED[2]: Expert 0 active (non-stationary)
set_property -dict { PACKAGE_PIN T9  IOSTANDARD LVCMOS33 } [get_ports {led[2]}]

## LED[3]: Expert 1 active (stationary)
set_property -dict { PACKAGE_PIN T10 IOSTANDARD LVCMOS33 } [get_ports {led[3]}]

## RGB LED 0: Classification result display
set_property -dict { PACKAGE_PIN E1  IOSTANDARD LVCMOS33 } [get_ports rgb0_r]
set_property -dict { PACKAGE_PIN F6  IOSTANDARD LVCMOS33 } [get_ports rgb0_g]
set_property -dict { PACKAGE_PIN G6  IOSTANDARD LVCMOS33 } [get_ports rgb0_b]

## ============================================================================
## Push Buttons
## ============================================================================

## BTN[0]: Start inference / Wake from deep sleep
set_property -dict { PACKAGE_PIN D9  IOSTANDARD LVCMOS33 } [get_ports {btn[0]}]

## BTN[1]: Stop / Force deep sleep
set_property -dict { PACKAGE_PIN C9  IOSTANDARD LVCMOS33 } [get_ports {btn[1]}]

## BTN[2]: Manual reset (active high → inverted for rst_n)
set_property -dict { PACKAGE_PIN B9  IOSTANDARD LVCMOS33 } [get_ports {btn[2]}]

## BTN[3]: Mode select (debug / normal)
set_property -dict { PACKAGE_PIN B8  IOSTANDARD LVCMOS33 } [get_ports {btn[3]}]

## ============================================================================
## Timing Constraints
## ============================================================================

## Maximum input delay for AXI-Stream audio data
set_input_delay  -clock [get_clocks nanomamba_clk] -max 8.0  [get_ports s_axis_audio_tdata*]
set_input_delay  -clock [get_clocks nanomamba_clk] -min 0.0  [get_ports s_axis_audio_tdata*]
set_input_delay  -clock [get_clocks nanomamba_clk] -max 8.0  [get_ports s_axis_audio_tvalid]
set_input_delay  -clock [get_clocks nanomamba_clk] -min 0.0  [get_ports s_axis_audio_tvalid]

## Output delay for result AXI-Stream
set_output_delay -clock [get_clocks nanomamba_clk] -max 5.0  [get_ports m_axis_result_tdata*]
set_output_delay -clock [get_clocks nanomamba_clk] -min 0.0  [get_ports m_axis_result_tdata*]

## IRQ outputs
set_output_delay -clock [get_clocks nanomamba_clk] -max 5.0  [get_ports irq_*]
set_output_delay -clock [get_clocks nanomamba_clk] -min 0.0  [get_ports irq_*]

## ============================================================================
## BRAM Placement Hints
## ============================================================================
## Weight SRAM (4.5KB) → 3 × BRAM18K or 2 × BRAM36K
## FFT buffer (2KB) → 1 × BRAM36K
## LUT tables (3 × 256B) → distributed RAM or 1 × BRAM18K

## Force BRAM inference for weight SRAM
set_property RAM_STYLE BLOCK [get_cells u_nanomamba_top/u_weight_sram/mem_reg*]

## Force BRAM for FFT buffer
set_property RAM_STYLE BLOCK [get_cells u_nanomamba_top/u_stft/fft_re_reg*]
set_property RAM_STYLE BLOCK [get_cells u_nanomamba_top/u_stft/fft_im_reg*]

## ============================================================================
## DSP48 Usage Hints
## ============================================================================
## MAC operations should map to DSP48 slices automatically
## Ensure Vivado infers DSP48 for multiply-accumulate in:
##   - SSM compute (primary MAC)
##   - PCEN IIR smoother
##   - Classifier linear layer
##   - FFT butterfly

set_property USE_DSP48 YES [get_cells u_nanomamba_top/u_ssm_compute/*mac*]
set_property USE_DSP48 YES [get_cells u_nanomamba_top/u_classifier/*linear*]

## ============================================================================
## Power Optimization
## ============================================================================

## Enable power optimization in Vivado
# set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE ExploreWithRemap [get_runs impl_1]
# set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraPostPlacementOpt [get_runs impl_1]
# set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]

## Block-level power gating not available on Artix-7
## Clock gating via BUFGCE is the primary power strategy

## ============================================================================
## Configuration & Bitstream
## ============================================================================

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]

## ============================================================================
## Debug (ILA) — Optional
## ============================================================================
## Uncomment to add ILA debug cores for signal probing
## Useful for verifying clock gating, expert selection, SSM state

# set_property MARK_DEBUG true [get_nets u_nanomamba_top/u_power_mgmt/pwr_state*]
# set_property MARK_DEBUG true [get_nets u_nanomamba_top/u_moe_router/expert_sel]
# set_property MARK_DEBUG true [get_nets u_nanomamba_top/u_moe_router/gate_out*]
# set_property MARK_DEBUG true [get_nets u_nanomamba_top/u_classifier/result_class*]
# set_property MARK_DEBUG true [get_nets u_nanomamba_top/u_classifier/logits_valid]
