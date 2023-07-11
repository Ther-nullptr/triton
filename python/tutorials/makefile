CUDA_HOME ?= /usr/local/cuda
CU=$(CUDA_HOME)/bin/nvcc

# for profiling
PYTHON_FILE=03-2-batched-matrix-multiplication-ncu-profiling
M=4096
N=4096
K=4096
BLOCK_M=64
BLOCK_N=256
BLOCK_K=32
GROUP_M=8
NUM_STAGES=4
NUM_WARPS=4
ARCH=sm_80
NCU=/opt/nvidia/nsight-compute/2023.1.0/ncu

FILE_NAME=M-$(M)_N-$(N)_K-$(K)-BM-$(BLOCK_M)_BN-$(BLOCK_N)_BK-$(BLOCK_K)_GM-$(GROUP_M)-S-$(NUM_STAGES)W-_$(NUM_WARPS)
CMD_OPT=--m $(M) --n $(N) --k $(K) --block-m $(BLOCK_M) --block-n $(BLOCK_N) --block-k $(BLOCK_K) --group-m $(GROUP_M) --num-stages $(NUM_STAGES) --num-warps $(NUM_WARPS)

# metrics for roofline
NCU_FLAG := --profile-from-start=ON
metrics_gpu = sm__cycles_elapsed.max,gpc__cycles_elapsed.avg.per_second,
metrics_compute = smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.avg.peak_sustained_elapsed,breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed,
metrics_memory = breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,
metrics_shared = smsp__sass_data_bytes_mem_shared_op_ld,smsp__sass_data_bytes_mem_shared_op_ldgsts,smsp__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass,smsp__sass_data_bytes_mem_shared_op_ldgsts_cache_access,smsp__sass_data_bytes_mem_shared_op_ldsm,smsp__sass_data_bytes_mem_shared_op_st,smsp__sass_data_bytes_mem_shared,smsp__sass_data_bytes_mem_shared.avg.peak_sustained,smsp__sass_data_bytes_mem_shared.max.peak_sustained,smsp__sass_data_bytes_mem_shared.avg.peak_sustained_elapsed,smsp__sass_data_bytes_mem_shared.avg.per_cycle_elapsed,smsp__sass_data_bytes_mem_shared.sum.pct_of_peak_sustained_elapsed,smsp__sass_data_bytes_mem_shared.sum.pct_of_peak_sustained_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.pct_of_peak_sustained_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.pct_of_peak_sustained_elapsed,sm__sass_data_bytes_mem_shared_op_ldsm.avg.peak_sustained,sm__sass_data_bytes_mem_shared_op_ldsm.avg.peak_sustained_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.peak_sustained_elapsed,sm__sass_data_bytes_mem_shared_op_ldsm.avg.per_cycle_active,sm__sass_data_bytes_mem_shared_op_ldsm.avg.per_cycle_elapsed,sm__sass_data_bytes_mem_shared_op_ld.avg.per_cycle_active,
metrics_time = gpu__time_duration.sum,gpc__cycles_elapsed.max,
metrics_throughput = sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,
metrics_occupancy = sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,launch__grid_size,launch__registers_per_thread,launch__block_size,launch__thread_count
metrics := $(metrics_time)$(metrics_throughput)$(metrics_occupancy)

roofline_section ?= --section SpeedOfLight --section LaunchStats --section SpeedOfLight_HierarchicalTensorRooflineChart --section Occupancy --section MemoryWorkloadAnalysis_Chart --section ComputeWorkloadAnalysis

ncu:
	$(NCU) $(NCU_FLAG) --target-processes all -f -o profile-$(PYTHON_FILE)-$(CMD_OPT) python $(PYTHON_FILE).py $(CMD_OPT)

ncu_metrics:
	$(NCU) $(NCU_FLAG) --target-processes all --metrics $(metrics) --csv --log-file x.csv python3 $(PYTHON_FILE).py $(CMD_OPT) | tee csv/nsight-compute-$(PYTHON_FILE)-$(FILE_NAME).csv

ncu_roofline:
	$(NCU) $(NCU_FLAG) --target-processes all $(roofline_section) -o ncu-rep/roofline-$(PYTHON_FILE)-$(FILE_NAME) -f python $(PYTHON_FILE).py $(CMD_OPT) 
