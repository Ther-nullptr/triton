#!/bin/bash 

B=1
M=2048
N=32
K=8192
BLOCK_M=128
BLOCK_N=128
BLOCK_K=32
GROUP_M=8
NUM_STAGES=4
NUM_WARPS=4

exec_name=/home/yujin/workspace/triton/python/tutorials/03-2-batched-matrix-multiplication-ncu-profiling.py
args="--b $B --m $M --n $N --k $K --block-m $BLOCK_M --block-n $BLOCK_N --block-k $BLOCK_K --group-m $GROUP_M --num-stages $NUM_STAGES --num-warps $NUM_WARPS"
csv_name="[${B},${M},${N},${K}]_[${BLOCK_M},${BLOCK_N},${BLOCK_K}]_[${GROUP_M},${NUM_STAGES},${NUM_WARPS}]"
picture_name=group7

# if the first argument is "plot", then plot the csv file
if [ $# -eq 1 ]; then
    if [ $1=="plot" ]; then
        python postprocess.py --name $picture_name --dir csv/$picture_name
        exit 0
    fi
fi

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
l1tex__t_bytes.sum"

# mkdir 
if [ ! -d "csv/$picture_name" ]; then
    mkdir csv/$picture_name
fi

/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --csv --target-processes all python3 $exec_name $args > csv/$picture_name/$csv_name.csv
sed -i '/^==/d' csv/$picture_name/$csv_name.csv

