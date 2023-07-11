import os

if __name__ == '__main__':
    b_list = [1] # [1, 16, 100]
    m_list = [2048] # [128 * i for i in range(10, 40)]
    n_list = [2048] # [128 * i for i in range(10, 40)]
    k_list = [2048] # [128 * i for i in range(10, 40)]
    block_m_list = [128] # [32, 64, 128]
    block_n_list = [128] # [64, 128, 256]
    block_k_list = [32] # [32, 64]
    group_m_list = [8]
    num_stage_list = [3, 4, 5]
    num_warps_list = [2, 4, 8]

    for b in b_list:
        for m in m_list:
            for n in n_list:
                for k in k_list:
                    for block_m in block_m_list:
                        for block_n in block_n_list:
                            for block_k in block_k_list:
                                for group_m in group_m_list:
                                    for num_stage in num_stage_list:
                                        for num_warps in num_warps_list:
                                            print(f"b: {b}, m: {m}, n: {n}, k: {k} block_m: {block_m}, block_n: {block_n}, block_k: {block_k}, group_m: {group_m}, num_stage: {num_stage}, num_warps: {num_warps}")
                                            # exectue /home/yujin/workspace/triton/python/tutorials/03-2-batched-matrix-multiplication-ncu-profiling.py
                                            os.system(f"/opt/nvidia/nsight-compute/2023.1.0/ncu \
                                                    --profile-from-start=ON --target-processes all \
                                                    --metrics gpu__time_duration.sum,gpc__cycles_elapsed.max,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,launch__grid_size,launch__registers_per_thread,launch__block_size,launch__thread_count \
                                                    --csv python3 03-2-batched-matrix-multiplication-ncu-profiling.py --b {b} --m {m} --n {n} --k {k} --block-m {block_m} --block-n {block_n} --block-k {block_k} --group-m {group_m} --num-stages {num_stage} --num-warps {num_warps} | \
                                                        tee csv/nsight-compute-03-2-batched-matrix-multiplication-ncu-profiling-{b}_{m}_{n}_{k}-{block_m}_{block_n}_{block_k}_{group_m}-{num_stage}_{num_warps}.csv")