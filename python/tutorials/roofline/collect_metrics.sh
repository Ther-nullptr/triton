#!/bin/bash 

B=16
M=2048
N=2048
K=2048
BLOCK_M=128
BLOCK_N=128
BLOCK_K=32
GROUP_M=8
NUM_STAGES=5
NUM_WARPS=4

exec_name=/home/yujin/workspace/triton/python/tutorials/03-2-batched-matrix-multiplication-ncu-profiling.py
args="--b $B --m $M --n $N --k $K --block-m $BLOCK_M --block-n $BLOCK_N --block-k $BLOCK_K --group-m $GROUP_M --num-stages $NUM_STAGES --num-warps $NUM_WARPS"
name="[${B},${M},${N},${K}]_[${BLOCK_M},${BLOCK_N},${BLOCK_K}]_[${GROUP_M},${NUM_STAGES},${NUM_WARPS}]"

# baseline
metrics="dram__bytes.sum.peak_sustained,\
dram__cycles_elapsed.avg.per_second,\
lts__t_bytes.sum.peak_sustained,\
lts__cycles_elapsed.avg.per_second,\
l1tex__t_bytes.sum.peak_sustained,\
l1tex__cycles_elapsed.avg.per_second,\
smsp__sass_data_bytes_mem_shared.avg.peak_sustained,\
sm__inst_executed_pipe_tensor.sum.peak_sustained,\
sm__cycles_elapsed.avg.per_second,"
 
# Time
metrics+="sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,"
 
# DP
metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"
 
# SP
metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"
 
# HP
metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"
 
# Tensor Core
metrics+="sm__inst_executed_pipe_tensor.sum,"
 
# DRAM, L2 and L1
metrics+="dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum,"

# ld
metrics+="sm__sass_inst_executed_op_ld.avg.peak_sustained,\
sm__sass_inst_executed_op_ld.avg,\
sm__sass_inst_executed_op_ld.avg.peak_sustained_active,\
sm__sass_inst_executed_op_ld.max.per_cycle_active,\
sm__sass_inst_executed_op_ld.max.pct_of_peak_sustained_active,"

# st
metrics+="sm__sass_inst_executed_op_st.avg.peak_sustained,\
sm__sass_inst_executed_op_st.avg,\
sm__sass_inst_executed_op_st.avg.peak_sustained_active,\
sm__sass_inst_executed_op_st.max.per_cycle_active,\
sm__sass_inst_executed_op_st.max.pct_of_peak_sustained_active,"

# ldsm
metrics+="sm__sass_inst_executed_op_ldsm.avg.peak_sustained,\
sm__sass_inst_executed_op_ldsm.avg,\
sm__sass_inst_executed_op_ldsm.avg.peak_sustained_active,\
sm__sass_inst_executed_op_ldsm.max.per_cycle_active,\
sm__sass_inst_executed_op_ldsm.max.pct_of_peak_sustained_active,"

# stream multiprocessor
metrics+="sm__inst_executed_pipe_tensor.avg.peak_sustained,\
sm__inst_executed_pipe_tensor.avg.peak_sustained_active,\
sm__inst_executed_pipe_tensor.max.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.max.per_cycle_active,"

# cycles
metrics+="gpc__cycles_elapsed.avg.peak_sustained"

 
/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --csv --target-processes all python3 $exec_name $args > output.csv
sed -i '/^==/d' output.csv

python postprocess.py --name $name