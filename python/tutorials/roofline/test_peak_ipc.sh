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

# list method: ncu --query-metrics | grep sm__sass_inst_executed_op
# extract method: ncu --query-metrics | grep sm__sass_inst_executed_op | sed 's/^\([^[:blank:]]*\).*/\1,/'
# ncu --query-metrics | grep sm__inst_executed | sed 's/^\([^[:blank:]]*\).*/\1,/'
metrics="sm__sass_inst_executed.max.peak_sustained,\
sm__sass_inst_executed_memdesc_explicit.max.peak_sustained,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_first.max.peak_sustained,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_last.max.peak_sustained,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_normal.max.peak_sustained,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_normal_demote.max.peak_sustained,\
sm__sass_inst_executed_memdesc_explicit_missprop_evict_first.max.peak_sustained,\
sm__sass_inst_executed_memdesc_explicit_missprop_evict_normal.max.peak_sustained,\
sm__sass_inst_executed_op_atom.max.peak_sustained,\
sm__sass_inst_executed_op_branch.max.peak_sustained,\
sm__sass_inst_executed_op_global.max.peak_sustained,\
sm__sass_inst_executed_op_global_atom.max.peak_sustained,\
sm__sass_inst_executed_op_global_ld.max.peak_sustained,\
sm__sass_inst_executed_op_global_red.max.peak_sustained,\
sm__sass_inst_executed_op_global_st.max.peak_sustained,\
sm__sass_inst_executed_op_ld.max.peak_sustained,\
sm__sass_inst_executed_op_ldgsts.max.peak_sustained,\
sm__sass_inst_executed_op_ldgsts_cache_access.max.peak_sustained,\
sm__sass_inst_executed_op_ldgsts_cache_bypass.max.peak_sustained,\
sm__sass_inst_executed_op_ldsm.max.peak_sustained,\
sm__sass_inst_executed_op_local.max.peak_sustained,\
sm__sass_inst_executed_op_local_ld.max.peak_sustained,\
sm__sass_inst_executed_op_local_st.max.peak_sustained,\
sm__sass_inst_executed_op_memory_128b.max.peak_sustained,\
sm__sass_inst_executed_op_memory_16b.max.peak_sustained,\
sm__sass_inst_executed_op_memory_32b.max.peak_sustained,\
sm__sass_inst_executed_op_memory_64b.max.peak_sustained,\
sm__sass_inst_executed_op_memory_8b.max.peak_sustained,\
sm__sass_inst_executed_op_shared.max.peak_sustained,\
sm__sass_inst_executed_op_shared_atom.max.peak_sustained,\
sm__sass_inst_executed_op_shared_ld.max.peak_sustained,\
sm__sass_inst_executed_op_shared_st.max.peak_sustained,\
sm__sass_inst_executed_op_st.max.peak_sustained,\
sm__sass_inst_executed_op_texture.max.peak_sustained,"

metrics+="sm__inst_executed_op_ldsm.max.peak_sustained,\
sm__inst_executed.max.peak_sustained,\
sm__inst_executed_op_ldsm.max.peak_sustained,\
sm__inst_executed_pipe_adu.max.peak_sustained,\
sm__inst_executed_pipe_alu.max.peak_sustained,\
sm__inst_executed_pipe_cbu.max.peak_sustained,\
sm__inst_executed_pipe_cbu_pred_off_all.max.peak_sustained,\
sm__inst_executed_pipe_cbu_pred_on_any.max.peak_sustained,\
sm__inst_executed_pipe_fma.max.peak_sustained,\
sm__inst_executed_pipe_fp16.max.peak_sustained,\
sm__inst_executed_pipe_fp64.max.peak_sustained,\
sm__inst_executed_pipe_ipa.max.peak_sustained,\
sm__inst_executed_pipe_lsu.max.peak_sustained,\
sm__inst_executed_pipe_tensor.max.peak_sustained,\
sm__inst_executed_pipe_tensor_op_dmma.max.peak_sustained,\
sm__inst_executed_pipe_tensor_op_hmma.max.peak_sustained,\
sm__inst_executed_pipe_tensor_op_hmma_type_hfma2.max.peak_sustained,\
sm__inst_executed_pipe_tensor_op_imma.max.peak_sustained,\
sm__inst_executed_pipe_tex.max.peak_sustained,\
sm__inst_executed_pipe_uniform.max.peak_sustained,\
sm__inst_executed_pipe_xu.max.peak_sustained"
         
/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --target-processes all python3 $exec_name $args
