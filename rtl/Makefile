# ============================================================================
# NanoMamba ASIC/FPGA Synthesis & Simulation Makefile
# ============================================================================
# Targets:
#   sim          - Icarus Verilog simulation (testbench)
#   sim_view     - Open waveform viewer (GTKWave)
#   synth_yosys  - Open-source synthesis (Yosys + nextpnr)
#   synth_vivado - Xilinx Vivado synthesis (Artix-7)
#   lint         - Verilator lint check
#   clean        - Remove generated files
#
# Dependencies:
#   - Icarus Verilog (iverilog)
#   - GTKWave (gtkwave)
#   - Yosys (yosys)
#   - Verilator (verilator)
#   - Xilinx Vivado 2023.2+ (optional, for FPGA)
#
# Author : Jin Ho Choi, Ph.D.
# ============================================================================

# ---- Project Configuration ----
TOP_MODULE    = nanomamba_top
TB_MODULE     = nanomamba_top_tb
FPGA_PART     = xc7a35tcpg236-1
FPGA_BOARD    = arty-a7-35

# ---- Directories ----
SRC_DIR       = src
TB_DIR        = tb
FPGA_DIR      = fpga
SIM_DIR       = sim
BUILD_DIR     = build
MEM_DIR       = mem

# ---- Source Files ----
RTL_SRCS = \
    $(SRC_DIR)/nanomamba_top.v \
    $(SRC_DIR)/nanomamba_regfile.v \
    $(SRC_DIR)/nanomamba_weight_sram.v \
    $(SRC_DIR)/nanomamba_stft.v \
    $(SRC_DIR)/nanomamba_snr_estimator.v \
    $(SRC_DIR)/nanomamba_moe_router.v \
    $(SRC_DIR)/nanomamba_dual_pcen.v \
    $(SRC_DIR)/nanomamba_ssm_compute.v \
    $(SRC_DIR)/nanomamba_classifier.v \
    $(SRC_DIR)/nanomamba_power_mgmt.v

TB_SRCS = \
    $(TB_DIR)/nanomamba_top_tb.v

# ---- Simulation (Icarus Verilog) ----
IVERILOG      = iverilog
VVP           = vvp
IVFLAGS       = -g2012 -Wall -DFPGA_TARGET
SIM_OUT       = $(BUILD_DIR)/$(TB_MODULE).vvp
VCD_FILE      = $(BUILD_DIR)/$(TB_MODULE).vcd

# ---- Lint (Verilator) ----
VERILATOR     = verilator
VLINT_FLAGS   = --lint-only -Wall --top-module $(TOP_MODULE) -DFPGA_TARGET

# ---- Synthesis (Yosys) ----
YOSYS         = yosys
YOSYS_SCRIPT  = $(BUILD_DIR)/synth.ys

# ---- Vivado ----
VIVADO        = vivado
VIVADO_TCL    = $(BUILD_DIR)/vivado_synth.tcl
XDC_FILE      = $(FPGA_DIR)/artix7_nanomamba.xdc

# ============================================================================
# Default Target
# ============================================================================
.PHONY: all
all: lint sim

# ============================================================================
# Build Directory
# ============================================================================
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ============================================================================
# Simulation (Icarus Verilog)
# ============================================================================
.PHONY: sim sim_view sim_clean

sim: $(BUILD_DIR) $(SIM_OUT)
	@echo "=== Running NanoMamba Simulation ==="
	cd $(BUILD_DIR) && $(VVP) $(notdir $(SIM_OUT)) | tee sim.log
	@echo "=== Simulation Complete ==="
	@echo "Waveform: $(VCD_FILE)"

$(SIM_OUT): $(RTL_SRCS) $(TB_SRCS)
	@echo "=== Compiling Verilog ==="
	$(IVERILOG) $(IVFLAGS) -o $@ $(RTL_SRCS) $(TB_SRCS)

sim_view: sim
	gtkwave $(VCD_FILE) &

sim_clean:
	rm -f $(BUILD_DIR)/*.vvp $(BUILD_DIR)/*.vcd $(BUILD_DIR)/sim.log

# ============================================================================
# Lint (Verilator)
# ============================================================================
.PHONY: lint

lint:
	@echo "=== Verilator Lint Check ==="
	$(VERILATOR) $(VLINT_FLAGS) $(RTL_SRCS)
	@echo "=== Lint PASSED ==="

# ============================================================================
# Yosys Open-Source Synthesis
# ============================================================================
.PHONY: synth_yosys synth_report

synth_yosys: $(BUILD_DIR) $(YOSYS_SCRIPT)
	@echo "=== Yosys Synthesis ==="
	cd $(BUILD_DIR) && $(YOSYS) -s $(notdir $(YOSYS_SCRIPT)) | tee yosys.log
	@echo "=== Synthesis Complete ==="

$(YOSYS_SCRIPT): $(BUILD_DIR)
	@echo "# Auto-generated Yosys synthesis script" > $@
	@echo "# NanoMamba — Ultra-lightweight KWS Accelerator" >> $@
	@echo "" >> $@
	@echo "# Read Verilog sources" >> $@
	@for f in $(RTL_SRCS); do echo "read_verilog -DFPGA_TARGET ../$$f" >> $@; done
	@echo "" >> $@
	@echo "# Elaborate" >> $@
	@echo "hierarchy -top $(TOP_MODULE)" >> $@
	@echo "" >> $@
	@echo "# Synthesize" >> $@
	@echo "proc; opt; fsm; opt; memory; opt" >> $@
	@echo "techmap; opt" >> $@
	@echo "" >> $@
	@echo "# Technology mapping (generic)" >> $@
	@echo "synth -top $(TOP_MODULE) -flatten" >> $@
	@echo "" >> $@
	@echo "# Statistics" >> $@
	@echo "stat" >> $@
	@echo "" >> $@
	@echo "# Write output" >> $@
	@echo "write_verilog $(TOP_MODULE)_synth.v" >> $@
	@echo "write_json $(TOP_MODULE)_synth.json" >> $@

# Xilinx-specific Yosys synthesis
.PHONY: synth_yosys_xilinx

synth_yosys_xilinx: $(BUILD_DIR)
	@echo "=== Yosys Xilinx Synthesis ==="
	@echo "# Yosys Xilinx synthesis script" > $(BUILD_DIR)/synth_xilinx.ys
	@for f in $(RTL_SRCS); do echo "read_verilog -DFPGA_TARGET ../$$f" >> $(BUILD_DIR)/synth_xilinx.ys; done
	@echo "synth_xilinx -top $(TOP_MODULE) -family xc7 -flatten" >> $(BUILD_DIR)/synth_xilinx.ys
	@echo "stat" >> $(BUILD_DIR)/synth_xilinx.ys
	@echo "write_edif $(TOP_MODULE)_xilinx.edif" >> $(BUILD_DIR)/synth_xilinx.ys
	cd $(BUILD_DIR) && $(YOSYS) -s synth_xilinx.ys | tee yosys_xilinx.log

# ============================================================================
# Vivado Synthesis (Xilinx Artix-7)
# ============================================================================
.PHONY: synth_vivado vivado_project

$(VIVADO_TCL): $(BUILD_DIR)
	@echo "# Auto-generated Vivado TCL script" > $@
	@echo "# NanoMamba FPGA Synthesis for $(FPGA_PART)" >> $@
	@echo "" >> $@
	@echo "# Create project" >> $@
	@echo "create_project nanomamba_fpga $(BUILD_DIR)/vivado_project -part $(FPGA_PART) -force" >> $@
	@echo "" >> $@
	@echo "# Add sources" >> $@
	@for f in $(RTL_SRCS); do echo "add_files ../$$f" >> $@; done
	@echo "add_files ../$(XDC_FILE)" >> $@
	@echo "" >> $@
	@echo "# Set top module" >> $@
	@echo "set_property top $(TOP_MODULE) [current_fileset]" >> $@
	@echo "" >> $@
	@echo "# Set FPGA_TARGET define" >> $@
	@echo "set_property verilog_define {FPGA_TARGET=1} [current_fileset]" >> $@
	@echo "" >> $@
	@echo "# Run synthesis" >> $@
	@echo "launch_runs synth_1 -jobs 4" >> $@
	@echo "wait_on_run synth_1" >> $@
	@echo "" >> $@
	@echo "# Report utilization" >> $@
	@echo "open_run synth_1" >> $@
	@echo "report_utilization -file $(BUILD_DIR)/utilization_synth.rpt" >> $@
	@echo "report_timing_summary -file $(BUILD_DIR)/timing_synth.rpt" >> $@
	@echo "report_power -file $(BUILD_DIR)/power_synth.rpt" >> $@
	@echo "" >> $@
	@echo "# Run implementation" >> $@
	@echo "launch_runs impl_1 -jobs 4" >> $@
	@echo "wait_on_run impl_1" >> $@
	@echo "" >> $@
	@echo "# Report post-implementation" >> $@
	@echo "open_run impl_1" >> $@
	@echo "report_utilization -file $(BUILD_DIR)/utilization_impl.rpt" >> $@
	@echo "report_timing_summary -file $(BUILD_DIR)/timing_impl.rpt" >> $@
	@echo "report_power -file $(BUILD_DIR)/power_impl.rpt" >> $@
	@echo "" >> $@
	@echo "# Generate bitstream" >> $@
	@echo "launch_runs impl_1 -to_step write_bitstream -jobs 4" >> $@
	@echo "wait_on_run impl_1" >> $@
	@echo "" >> $@
	@echo "puts \"Synthesis complete! Check reports in $(BUILD_DIR)/\"" >> $@

synth_vivado: $(VIVADO_TCL)
	@echo "=== Vivado FPGA Synthesis ==="
	$(VIVADO) -mode batch -source $(VIVADO_TCL) | tee $(BUILD_DIR)/vivado.log
	@echo "=== Vivado Synthesis Complete ==="
	@echo "Reports:"
	@echo "  Utilization: $(BUILD_DIR)/utilization_impl.rpt"
	@echo "  Timing:      $(BUILD_DIR)/timing_impl.rpt"
	@echo "  Power:       $(BUILD_DIR)/power_impl.rpt"

# Open Vivado GUI with the project
vivado_project: $(VIVADO_TCL)
	$(VIVADO) -mode gui -source $(VIVADO_TCL) &

# ============================================================================
# ASIC Synthesis (Yosys + Liberty)
# ============================================================================
.PHONY: synth_asic

synth_asic: $(BUILD_DIR)
	@echo "=== ASIC Synthesis (Generic Liberty) ==="
	@echo "# ASIC synthesis script" > $(BUILD_DIR)/synth_asic.ys
	@for f in $(RTL_SRCS); do echo "read_verilog ../$$f" >> $(BUILD_DIR)/synth_asic.ys; done
	@echo "hierarchy -top $(TOP_MODULE)" >> $(BUILD_DIR)/synth_asic.ys
	@echo "proc; opt; fsm; opt; memory; opt" >> $(BUILD_DIR)/synth_asic.ys
	@echo "" >> $(BUILD_DIR)/synth_asic.ys
	@echo "# Map to generic standard cells" >> $(BUILD_DIR)/synth_asic.ys
	@echo "synth -top $(TOP_MODULE) -flatten" >> $(BUILD_DIR)/synth_asic.ys
	@echo "dfflibmap -liberty ../lib/nangate45_typ.lib" >> $(BUILD_DIR)/synth_asic.ys
	@echo "abc -liberty ../lib/nangate45_typ.lib" >> $(BUILD_DIR)/synth_asic.ys
	@echo "opt_clean" >> $(BUILD_DIR)/synth_asic.ys
	@echo "" >> $(BUILD_DIR)/synth_asic.ys
	@echo "# Reports" >> $(BUILD_DIR)/synth_asic.ys
	@echo "stat -liberty ../lib/nangate45_typ.lib" >> $(BUILD_DIR)/synth_asic.ys
	@echo "write_verilog $(TOP_MODULE)_asic.v" >> $(BUILD_DIR)/synth_asic.ys
	cd $(BUILD_DIR) && $(YOSYS) -s synth_asic.ys | tee yosys_asic.log
	@echo "=== ASIC Synthesis Complete ==="
	@echo "NOTE: Replace nangate45_typ.lib with target foundry library (TSMC28/Samsung28/GF22)"

# ============================================================================
# LUT Memory File Generation
# ============================================================================
.PHONY: gen_luts

gen_luts: $(BUILD_DIR)
	@echo "=== Generating LUT .mem files ==="
	@python3 ../scripts/gen_lut_mem.py --output $(MEM_DIR)/
	@echo "Generated LUT files in $(MEM_DIR)/"

# ============================================================================
# Weight Export (from PyTorch model)
# ============================================================================
.PHONY: export_weights

export_weights:
	@echo "=== Exporting INT8 Weights from PyTorch Model ==="
	@python3 ../scripts/export_weights.py \
		--checkpoint ../checkpoints/nanomamba_tiny_best.pt \
		--output $(MEM_DIR)/weights.mem \
		--format hex8
	@echo "Weight file: $(MEM_DIR)/weights.mem"

# ============================================================================
# Resource Estimation
# ============================================================================
.PHONY: estimate

estimate:
	@echo "============================================"
	@echo "  NanoMamba Resource Estimation"
	@echo "============================================"
	@echo ""
	@echo "  Architecture:"
	@echo "    Model:      NanoMamba-Tiny"
	@echo "    Params:     4,634 (INT8 = 4.5KB)"
	@echo "    Datapath:   INT8 (8-bit data, 32-bit accumulator)"
	@echo "    Experts:    2 (DualPCEN MOE)"
	@echo "    SSM blocks: 2 (weight-shared)"
	@echo ""
	@echo "  FPGA (Artix-7 XC7A35T @ 50MHz):"
	@echo "    LUTs:       ~2,500 / 20,800 (12%)"
	@echo "    FFs:        ~1,800 / 41,600 (4%)"
	@echo "    BRAM 36K:   3 / 50 (6%)"
	@echo "    DSP48:      4 / 90 (4%)"
	@echo "    Power:      ~15mW (FPGA)"
	@echo ""
	@echo "  ASIC (28nm, 50MHz):"
	@echo "    Area:       ~0.15 mm²"
	@echo "    Gate count: ~50K gates"
	@echo "    SRAM:       4.5KB (4736 × 8-bit)"
	@echo "    Power:"
	@echo "      Active:     ~0.5mW (5% duty cycle)"
	@echo "      Sleep:      ~0.05mW (clock-gated)"
	@echo "      Deep sleep: ~0.01mW (leakage only)"
	@echo "      Average:    ~0.08mW (always-on KWS)"
	@echo ""
	@echo "  Performance (50MHz):"
	@echo "    MAC/frame:    ~4,200"
	@echo "    MAC/utterance: ~424K (101 frames)"
	@echo "    Latency:      ~0.5ms per frame"
	@echo "    Throughput:    1 utterance per 1.01s"
	@echo "============================================"

# ============================================================================
# Clean
# ============================================================================
.PHONY: clean clean_all

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.vcd *.vvp *.log

clean_all: clean
	rm -rf $(BUILD_DIR)/vivado_project
	rm -f $(MEM_DIR)/*.mem

# ============================================================================
# Help
# ============================================================================
.PHONY: help

help:
	@echo "NanoMamba ASIC/FPGA Makefile"
	@echo "============================"
	@echo ""
	@echo "Simulation:"
	@echo "  make sim          - Run Icarus Verilog simulation"
	@echo "  make sim_view     - Open waveform in GTKWave"
	@echo ""
	@echo "Lint:"
	@echo "  make lint         - Verilator lint check"
	@echo ""
	@echo "Synthesis:"
	@echo "  make synth_yosys  - Open-source synthesis (Yosys)"
	@echo "  make synth_vivado - Xilinx Vivado synthesis (Artix-7)"
	@echo "  make synth_asic   - ASIC synthesis (Yosys + Liberty)"
	@echo ""
	@echo "Utilities:"
	@echo "  make gen_luts     - Generate LUT memory files"
	@echo "  make export_weights - Export INT8 weights from PyTorch"
	@echo "  make estimate     - Show resource estimation"
	@echo ""
	@echo "Clean:"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean_all    - Remove all generated files"
