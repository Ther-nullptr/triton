#!/bin/bash 

# list method: ncu --query-metrics | grep sm__sass_inst_executed_op
# extract method: ncu --query-metrics | grep sm__sass_inst_executed_op | sed 's/^\([^[:blank:]]*\).*/\1,/'
# ncu --query-metrics | grep sm__inst_executed | sed 's/^\([^[:blank:]]*\).*/\1,/'
# ncu --query-metrics | grep sm__sass_data_bytes | sed 's/^\([^[:blank:]]*\).*/\1,/'
choice="min"

metrics="gpc__cycles_elapsed.${choice},\
sm__sass_inst_executed.${choice},\
sm__sass_inst_executed.${choice}.peak_sustained,\
sm__sass_inst_executed.${choice}.peak_sustained_active,\
sm__sass_inst_executed.${choice}.per_cycle_active,\
sm__sass_inst_executed_op_shared_ld.${choice},\
sm__sass_inst_executed_op_shared_ld.${choice}.peak_sustained,\
sm__sass_inst_executed_op_shared_ld.${choice}.peak_sustained_active,\
sm__sass_inst_executed_op_shared_ld.${choice}.peak_sustained_active.per_second,\
sm__sass_inst_executed_op_shared_ld.${choice}.per_cycle_active,\
sm__sass_inst_executed_op_shared_ld.${choice}.per_second,\
sm__sass_inst_executed_op_shared_ld.${choice}.pct_of_peak_sustained_active,\
smsp__inst_executed_op_shared_ld.${choice},\
smsp__inst_executed_op_shared_ld.${choice}.peak_sustained,\
smsp__inst_executed_op_shared_ld.${choice}.peak_sustained_active,\
smsp__inst_executed_op_shared_ld.${choice}.peak_sustained_active.per_second,\
smsp__inst_executed_op_shared_ld.${choice}.per_cycle_active,\
smsp__inst_executed_op_shared_ld.${choice}.per_second,\
smsp__inst_executed_op_shared_ld.${choice}.pct_of_peak_sustained_active,\
smsp__inst_executed_op_shared_ld_pred_off_all.${choice},\
smsp__inst_executed_op_shared_ld_pred_off_all.${choice}.peak_sustained,\
smsp__inst_executed_op_shared_ld_pred_off_all.${choice}.peak_sustained_active,\
smsp__inst_executed_op_shared_ld_pred_off_all.${choice}.peak_sustained_active.per_second,\
smsp__inst_executed_op_shared_ld_pred_off_all.${choice}.per_cycle_active,\
smsp__inst_executed_op_shared_ld_pred_off_all.${choice}.per_second,\
smsp__inst_executed_op_shared_ld_pred_off_all.${choice}.pct_of_peak_sustained_active,\
smsp__inst_executed_op_shared_ld_pred_on_any.${choice},\
smsp__inst_executed_op_shared_ld_pred_on_any.${choice}.peak_sustained,\
smsp__inst_executed_op_shared_ld_pred_on_any.${choice}.peak_sustained_active,\
smsp__inst_executed_op_shared_ld_pred_on_any.${choice}.peak_sustained_active.per_second,\
smsp__inst_executed_op_shared_ld_pred_on_any.${choice}.per_cycle_active,\
smsp__inst_executed_op_shared_ld_pred_on_any.${choice}.per_second,\
smsp__inst_executed_op_shared_ld_pred_on_any.${choice}.pct_of_peak_sustained_active"
         
/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --target-processes all ./shared_lat
