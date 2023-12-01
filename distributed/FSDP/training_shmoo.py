import os
import re
import subprocess
# TODO csv with outputs, sam_rmm with hugetlbfs on scv-hw13, CPU pointer chasing 64GiB footprint

ALLOC_TYPES = ["", "rmm", "fsdp", "fsdp_act_offload", "fsdp_cpu_offload"]
MODEL_SIZES = [
    (2048, 5120, 32),
    (2304, 5760, 32),
    (2560, 6400, 32),
    (2816, 7040, 32),
    (3072, 7680, 32),
    (3328, 8320, 48),
    (3584, 8960, 32),
    (3840, 9600, 32),
    (4096, 10240, 32),
    (4096, 10240, 64)
]
BATCH_SIZES = [2 ** x for x in range(5)]
TIMING_REGEX = re.compile(
    r"Train time: ([0-9-\.]+) s. Full epoch time: ([0-9-\.]+) s")
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def run_custom_config(config):
    cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", "1", "T5_training.py"]
    for k, v in config.items():
        cmd += ["--" + k, "'" + str(v) + "'"]
    return subprocess.run(
        " ".join(cmd), check=False, shell=True, capture_output=True,
        text=True, cwd=SCRIPT_DIR, timeout=3600
    )


def parse_result(result, config):
    # we can usually ignore errors, if we got at least 2 outputs of
    # timing, we know that the run has succeeded far enough
    try:
        timings = [float(x.group(1)) for x in TIMING_REGEX.finditer(result.stdout)]
    except (TypeError, ValueError):
        timings = []
    if len(timings) >= 2:
        return timings[-1]
    # we did not get at least 2 timing outputs: if the process has failed, this
    # is more serious, display the output
    if result.returncode != 0:
        print(f"Info: config {config} failed with code {result.returncode}")
        print(result.stdout)
        print(result.stderr)
    return 0.


def main():
    for alloc_type in ALLOC_TYPES:
        # first shmoo the model sizes
        for d_model, d_ff, num_heads in MODEL_SIZES:
            config = {
                "alloc_type": alloc_type, "d_model": d_model, "d_ff": d_ff,
                "num_heads": num_heads
            }

            # for debugging
            config["max_steps_per_epoch"] = 1

            result = run_custom_config(config)
            time = parse_result(result, config)
            print(f"config {config}: got time {time}")
        # shmoo batch sizes with base config otherwise
        for batch_size in BATCH_SIZES:
            config = {"batch_size_training": batch_size}
            if batch_size > 32:
                config["max_steps_per_epoch"] = 40 // (batch_size // 32)
    
            # for debugging
            config["max_steps_per_epoch"] = 1

            result = run_custom_config(config)
            time = parse_result(result, config)
            print(f"config {config}: got time {time}")


if __name__ == '__main__':
    main()
