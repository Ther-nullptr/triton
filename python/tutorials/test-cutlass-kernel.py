import os
import json
import argparse
import subprocess
from functools import partial


def load_config():
    cur_path = os.path.dirname(__file__)
    config_path = os.path.join(cur_path, "configs")
    with open(config_path) as f:
        return json.load(f)

CONFIGS = load_config()

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--acc-dtype", type=str, choices=["f16", "f32"], default="f16")
    args.add_argument("--out-dtype", type=str, choices=["f16", "f32"], default="f16")
    args.add_argument("--cutlass-home", type=str)
    args.add_argument("--log-dir", type=str, default="logs/cutlass/")
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
    logs = subprocess.check_output(instruction, shell=True)
    logs = logs.decode("utf-8")
    logs = logs.split("\n")
    csv_index = logs.index("CSV Results:")

    csv_file = os.path.join(ARGS.log_dir, f"{workload}.csv")
    with open(csv_file, "w") as f:
        f.write("\n".join(logs[csv_index + 2 :]))

    max_gflops = max(
        [float(log.split(",")[-1]) for log in logs[csv_index + 3 :] if log]
    )
    print(f"{workload}: {max_gflops} GFLOPS")
    print(f"Full results have been written to {csv_file}")


def _run_gemm(
    workload: str,
    b: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
):
    _run_cutlass(
        f"{ARGS.profiler} --operation=gemm"
        f" --batch_count={b} --n={n} --m={m} --k={k}"
        f" --A=f16:row --B=f16:column --C={out_dtype}"
        f" --accumulator-type={acc_dtype}",
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
):
    return _run_gemm(
        workload,
        b,
        n,
        m,
        k,
        acc_dtype,
        out_dtype,
    )


WORKLOADS = {}


def main():
    test_dict_train = {
        'X x QKV w': (1, 8192, 4608, 12288),
        'Q x K^T': (192, 512, 512, 128),
        'QK^T x V': (192, 512, 128, 512),
        'Proj': (1, 8192, 12288, 1536),
        'FC1': (1, 8192, 6144, 12288),
        'FC2': (1, 8192, 12288, 6144),
        'Q x K^T Flat': (1, 512, 512, 24576),
        'QK^T x V Flat': (1, 512, 24576, 512)
    }

    test_dict_inference_w_o_KV = {
        'X x QKV w': (1, 8704, 4608, 12288),
        'Q x K^T': (192, 543, 543, 128),
        'QK^T x V': (192, 543, 128, 543),
        'Proj': (1, 8688, 12288, 1536),
        'FC1': (1, 8688, 6144, 12288),
        'FC2': (1, 8688, 12288, 6144),
        'Q x K^T Flat': (1, 543, 543, 24576),
        'QK^T x V Flat': (1, 543, 24576, 543)
    }

    test_dict_inference_w_KV = {
        'X x QKV w': (1, 16, 4608, 12288),
        'Q x K^T': (192, 1, 543, 128),
        'QK^T x V': (192, 1, 128, 543),
        'Proj': (1, 16, 12288, 1536),
        'FC1': (1, 16, 6144, 12288),
        'FC2': (1, 16, 12288, 6144),
        'Q x K^T Flat': (1, 1, 543, 24576),
        'QK^T x V Flat': (1, 1, 24576, 543)
    }

    for i, test_dict in enumerate([test_dict_train, test_dict_inference_w_o_KV, test_dict_inference_w_KV]):
        for key, val in test_dict.items():
            WORKLOADS.update(
                {
                    f"{key}-{val}-{i}": partial(
                        GMM, workload=f"{key}-{val}-{i}", b=val[0], m=val[1], n=val[2], k=val[3]
                    )
                }
            )

    print(f"Accumulator type: {ARGS.acc_dtype}")
    print(f"Output type: {ARGS.out_dtype}")

    for workload_name in WORKLOADS.keys():
        WORKLOADS.get(workload_name)(
            acc_dtype=ARGS.acc_dtype,
            out_dtype=ARGS.out_dtype,
        )


if __name__ == "__main__":
    main()
