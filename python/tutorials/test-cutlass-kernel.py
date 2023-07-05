import os
import argparse
import subprocess
from functools import partial

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--acc-dtype", type=str, choices=["f16", "f32"], default="f16")
    args.add_argument("--out-dtype", type=str, choices=["f16", "f32"], default="f16")
    args.add_argument("--split-k-slices", type=int, default=1)
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
    print("Running:", workload)
    print("Instruction:", instruction)
    logs = subprocess.check_output(instruction, shell=True)
    logs = logs.decode("utf-8")
    logs = logs.split("\n")

    print("Instruction Done")


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
        f" --A=f16:row --B=f16:column --C={out_dtype}"
        f" --accumulator-type={acc_dtype}"
        f" --split-k-slices={k_slices}"
        f" --sort-results=true"
        f" --output={ARGS.log_dir}/{workload}.csv",
        workload=f"{workload}-{acc_dtype}-{out_dtype}",
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
    test_dict_train = {
        'XxQKVw': (1, 8192, 4608, 12288),
        'QxK^T': (192, 512, 512, 128),
        'QK^TxV': (192, 512, 128, 512),
        'Proj': (1, 8192, 12288, 1536),
        'FC1': (1, 8192, 6144, 12288),
        'FC2': (1, 8192, 12288, 6144),
        'QxK^TFlat': (1, 512, 512, 24576),
        'QK^TxVFlat': (1, 512, 24576, 512)
    }

    test_dict_inference_w_o_KV = {
        'XxQKVw': (1, 8704, 4608, 12288),
        'QxK^T': (192, 543, 543, 128),
        'QK^TxV': (192, 543, 128, 543),
        'Proj': (1, 8688, 12288, 1536),
        'FC1': (1, 8688, 6144, 12288),
        'FC2': (1, 8688, 12288, 6144),
        'QxK^TFlat': (1, 543, 543, 24576),
        'QK^TxVFlat': (1, 543, 24576, 543)
    }

    test_dict_inference_w_KV = {
        'XxQKVw': (1, 16, 4608, 12288),
        'QxK^T': (192, 1, 543, 128),
        'QK^TxV': (192, 1, 128, 543),
        'Proj': (1, 16, 12288, 1536),
        'FC1': (1, 16, 6144, 12288),
        'FC2': (1, 16, 12288, 6144),
        'QxK^TFlat': (1, 1, 543, 24576),
        'QK^TxVFlat': (1, 1, 24576, 543)
    }

    # for i, test_dict in enumerate([test_dict_train, test_dict_inference_w_o_KV, test_dict_inference_w_KV]):
    for i, test_dict in enumerate([test_dict_inference_w_KV]):
        for key, val in test_dict.items():
            WORKLOADS.update(
                {
                    f"{key}-{val}-{i}": partial(
                        GMM, workload=f"{key}-{val[0]}-{val[1]}-{val[2]}-{i}", b=val[0], m=val[1], n=val[2], k=val[3]
                    )
                }
            )

    print(f"Accumulator type: {ARGS.acc_dtype}")
    print(f"Output type: {ARGS.out_dtype}")
    print(f"Split K slices: {ARGS.split_k_slices}")

    for workload_name in WORKLOADS.keys():
        WORKLOADS.get(workload_name)(
            acc_dtype=ARGS.acc_dtype,
            out_dtype=ARGS.out_dtype,
            warmup_iterations=ARGS.warmup_iterations,
            profiling_iterations=ARGS.profiling_iterations,
            k_slices=ARGS.split_k_slices
        )


if __name__ == "__main__":
    main()
