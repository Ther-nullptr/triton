#!/bin/bash 

# list method: ncu --query-metrics | grep sm__sass_inst_executed_op
# extract method: ncu --query-metrics | grep sm__sass_inst_executed_op | sed 's/^\([^[:blank:]]*\).*/\1,/'
# ncu --query-metrics | grep sm__inst_executed | sed 's/^\([^[:blank:]]*\).*/\1,/'
# ncu --query-metrics | grep sm__sass_data_bytes | sed 's/^\([^[:blank:]]*\).*/\1,/'
choice="min"

metrics="gpc__cycles_elapsed.${choice},\
sm__sass_inst_executed.${choice}"

         
/opt/nvidia/nsight-compute/2023.1.0/ncu --metrics $metrics --csv --target-processes all ./template_shared_lat | tee test.csv
