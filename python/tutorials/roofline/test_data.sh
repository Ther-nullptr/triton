#!/bin/bash 

B=16
M=2048
N=2048
K=2048
BLOCK_M=128
BLOCK_N=128
BLOCK_K=32
GROUP_M=8
NUM_STAGES=4
NUM_WARPS=4

exec_name=/home/yujin/workspace/triton/python/tutorials/03-2-batched-matrix-multiplication-ncu-profiling.py
args="--b $B --m $M --n $N --k $K --block-m $BLOCK_M --block-n $BLOCK_N --block-k $BLOCK_K --group-m $GROUP_M --num-stages $NUM_STAGES --num-warps $NUM_WARPS"
name="[${B},${M},${N},${K}]_[${BLOCK_M},${BLOCK_N},${BLOCK_K}]_[${GROUP_M},${NUM_STAGES},${NUM_WARPS}]"

# list method: ncu --query-metrics | grep sm__sass_inst_executed_op
# extract method: ncu --query-metrics | grep sm__sass_inst_executed_op | sed 's/^\([^[:blank:]]*\).*/\1,/'
# ncu --query-metrics | grep sm__inst_executed | sed 's/^\([^[:blank:]]*\).*/\1,/'
# ncu --query-metrics | grep sm__sass_data_bytes | sed 's/^\([^[:blank:]]*\).*/\1,/'
metrics="sm__sass_data_bytes_mem_global.avg,\
sm__sass_data_bytes_mem_global_op_atom.avg,\
sm__sass_data_bytes_mem_global_op_ld.avg,\
sm__sass_data_bytes_mem_global_op_ldgsts.avg,\
sm__sass_data_bytes_mem_global_op_ldgsts_cache_access.avg,\
sm__sass_data_bytes_mem_global_op_ldgsts_cache_bypass.avg,\
sm__sass_data_bytes_mem_global_op_red.avg,\
sm__sass_data_bytes_mem_global_op_st.avg,\
sm__sass_data_bytes_mem_local.avg,\
sm__sass_data_bytes_mem_local_op_ld.avg,\
sm__sass_data_bytes_mem_local_op_st.avg,\
sm__sass_data_bytes_mem_shared.avg,\
sm__sass_data_bytes_mem_shared_op_atom.avg,\
sm__sass_data_bytes_mem_shared_op_ld.avg,\
sm__sass_data_bytes_mem_shared_op_ldgsts.avg,\
sm__sass_data_bytes_mem_shared_op_ldgsts_cache_access.avg,\
sm__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass.avg,\
sm__sass_data_bytes_mem_shared_op_ldsm.avg,\
sm__sass_data_bytes_mem_shared_op_st.avg"
         
/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --target-processes all python3 $exec_name $args
