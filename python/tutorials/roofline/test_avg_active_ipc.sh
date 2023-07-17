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
metrics="gpc__cycles_elapsed.avg,\
sm__sass_inst_executed.avg,\
sm__sass_inst_executed.avg.peak_sustained,\
sm__sass_inst_executed.avg.peak_sustained_active,\
sm__sass_inst_executed.avg.per_cycle_active,\
sm__pipe_shared_cycles_active.avg,\
sm__pipe_shared_cycles_active.avg.peak_sustained_active,\
sm__pipe_shared_cycles_active.avg.peak_sustained,\
sm__pipe_shared_cycles_active.avg.per_cycle_active,\
sm__sass_inst_executed_memdesc_explicit.avg.peak_sustained_active,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_first.avg.peak_sustained_active,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_last.avg.peak_sustained_active,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_normal.avg.peak_sustained_active,\
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_normal_demote.avg.peak_sustained_active,\
sm__sass_inst_executed_memdesc_explicit_missprop_evict_first.avg.peak_sustained_active,\
sm__sass_inst_executed_memdesc_explicit_missprop_evict_normal.avg.peak_sustained_active,\
sm__sass_inst_executed_op_atom.avg.peak_sustained_active,\
sm__sass_inst_executed_op_branch.avg.peak_sustained_active,\
sm__sass_inst_executed_op_global.avg.peak_sustained_active,\
sm__sass_inst_executed_op_global_atom.avg.peak_sustained_active,\
sm__sass_inst_executed_op_global_ld.avg.peak_sustained_active,\
sm__sass_inst_executed_op_global_red.avg.peak_sustained_active,\
sm__sass_inst_executed_op_global_st.avg.peak_sustained_active,\
sm__sass_inst_executed_op_ld.avg.peak_sustained_active,\
sm__sass_inst_executed_op_ldgsts.avg.peak_sustained_active,\
sm__sass_inst_executed_op_ldgsts_cache_access.avg.peak_sustained_active,\
sm__sass_inst_executed_op_ldgsts_cache_bypass.avg.peak_sustained_active,\
sm__sass_inst_executed_op_ldsm.avg.peak_sustained_active,\
sm__sass_inst_executed_op_local.avg.peak_sustained_active,\
sm__sass_inst_executed_op_local_ld.avg.peak_sustained_active,\
sm__sass_inst_executed_op_local_st.avg.peak_sustained_active,\
sm__sass_inst_executed_op_memory_128b.avg.peak_sustained_active,\
sm__sass_inst_executed_op_memory_16b.avg.peak_sustained_active,\
sm__sass_inst_executed_op_memory_32b.avg.peak_sustained_active,\
sm__sass_inst_executed_op_memory_64b.avg.peak_sustained_active,\
sm__sass_inst_executed_op_memory_8b.avg.peak_sustained_active,\
sm__sass_inst_executed_op_shared.avg.peak_sustained_active,\
sm__sass_inst_executed_op_shared_atom.avg.peak_sustained_active,\
sm__sass_inst_executed_op_shared_ld.avg.peak_sustained_active,\
sm__sass_inst_executed_op_shared_st.avg.peak_sustained_active,\
sm__sass_inst_executed_op_st.avg.peak_sustained_active,\
sm__sass_inst_executed_op_texture.avg.peak_sustained_active,"

metrics+="sm__inst_executed_op_ldsm.avg.peak_sustained_active,\
sm__inst_executed.avg.peak_sustained_active,\
sm__inst_executed_op_ldsm.avg.peak_sustained_active,\
sm__inst_executed_pipe_adu.avg.peak_sustained_active,\
sm__inst_executed_pipe_alu.avg.peak_sustained_active,\
sm__inst_executed_pipe_cbu.avg.peak_sustained_active,\
sm__inst_executed_pipe_cbu_pred_off_all.avg.peak_sustained_active,\
sm__inst_executed_pipe_cbu_pred_on_any.avg.peak_sustained_active,\
sm__inst_executed_pipe_fma.avg.peak_sustained_active,\
sm__inst_executed_pipe_fp16.avg.peak_sustained_active,\
sm__inst_executed_pipe_fp64.avg.peak_sustained_active,\
sm__inst_executed_pipe_ipa.avg.peak_sustained_active,\
sm__inst_executed_pipe_lsu.avg.peak_sustained_active,\
sm__inst_executed_pipe_tensor.avg.peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_dmma.avg.peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_hmma.avg.peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_hmma_type_hfma2.avg.peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_imma.avg.peak_sustained_active,\
sm__inst_executed_pipe_tex.avg.peak_sustained_active,\
sm__inst_executed_pipe_uniform.avg.peak_sustained_active,\
sm__inst_executed_pipe_xu.avg.peak_sustained_active"
         
/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --target-processes all python3 $exec_name $args
