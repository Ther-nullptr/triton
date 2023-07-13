#!/bin/bash 

exec_name=$1

if [ -z "$exec_name" ]; then
    echo "Usage: $0 <exec_name>"
    exit 1
fi

# baseline
metrics="dram__bytes.sum.peak_sustained,\
dram__cycles_elapsed.avg.per_second,\
lts__t_bytes.sum.peak_sustained,\
lts__cycles_elapsed.avg.per_second,\
l1tex__t_bytes.sum.peak_sustained,\
l1tex__cycles_elapsed.avg.per_second,\
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
l1tex__t_bytes.sum"
 
 
/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --csv --target-processes all python3 $exec_name > output.csv
sed -i '/^==/d' output.csv