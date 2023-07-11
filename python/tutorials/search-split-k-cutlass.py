import os
import argparse
import subprocess
import pandas as pd
from functools import partial

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--acc-dtype", type=str, choices=["f16", "f32"], default="f32")
    args.add_argument("--out-dtype", type=str, choices=["f16", "f32"], default="f16")
    args.add_argument("--profiling-iterations", type=int, default=5)
    args.add_argument("--warmup-iterations", type=int, default=1)
    args.add_argument("--cutlass-home", type=str, default='/home/yujin/workspace/cutlass')
    args.add_argument("--log-dir", type=str, default="logs/cutlass")
    parsed = args.parse_args()
    parsed.cutlass_home = parsed.cutlass_home or os.getenv("CUTLASS_HOME")
    assert (
        parsed.cutlass_home
    ), "Please specify 'CUTLASS_HOME', by either setting the environment variable or using --cutlass-home"
    parsed.profiler = f"{parsed.cutlass_home}/build/tools/profiler/cutlass_profiler"
    os.makedirs(parsed.log_dir, exist_ok=True)
    return parsed

ARGS = parse_args()

def _run_cutlass(instruction: str, workload: str):
    _ = subprocess.check_output(instruction, shell=True)
    df = pd.read_csv(f"{ARGS.log_dir}/{workload}.gemm.csv")
    df.sort_values('GFLOPs', inplace=True, ascending=False)
    df.reset_index(inplace=True)
    print(f"[{workload}] max TFLOPs: {df['GFLOPs'][0] / 1000}, operation: {df['Operation'][0]}, runtime: {df['Runtime'][0]}, GB/s: {df['GB/s'][0]}")


def _run_gemm(
    workload: str,
    b: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
    warmup_iterations: int,
    profiling_iterations: int,
    k_slices: int
):
    _run_cutlass(
        f"{ARGS.profiler} --operation=gemm --op_class=tensorop"
        f" --warmup_iterations={warmup_iterations}"
        f" --iterations={profiling_iterations}"
        f" --batch_count={b} --n={n} --m={m} --k={k}"
        f" --A=f16:row --B=f16:column --C={out_dtype}:column"
        f" --accumulator-type={acc_dtype}"
        f" --split-k-slices={k_slices}"
        f" --sort-results=true"
        f" --output={ARGS.log_dir}/{workload}.csv",
        workload=workload,
    )


def GMM(
    workload: str,
    b: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
    warmup_iterations: int,
    profiling_iterations: int,
    k_slices: int
):
    return _run_gemm(
        workload,
        b,
        n,
        m,
        k,
        acc_dtype,
        out_dtype,
        warmup_iterations,
        profiling_iterations,
        k_slices
    )


WORKLOADS = {}


def main():
    for n in [32, 64, 128, 256]:
        for m, k in [(2048, 8192), (2752, 8192)]:
            for k_slices in [1, 2, 4, 8, 16, 32]: 
                WORKLOADS.update(
                    {
                        f"{m}-{n}-{k}-{k_slices}": partial(
                            GMM, workload=f"m{m}-n{n}-k{k}-ksep{k_slices}", b=1, m=m, n=n, k=k, k_slices=k_slices
                        )
                    }
                )

    for workload_name in WORKLOADS.keys():
        WORKLOADS.get(workload_name)(
            acc_dtype=ARGS.acc_dtype,
            out_dtype=ARGS.out_dtype,
            warmup_iterations=ARGS.warmup_iterations,
            profiling_iterations=ARGS.profiling_iterations,
        )

if __name__ == "__main__":
    main()
