import csv
import itertools
import os
import re
import subprocess
from configs import train_config
# TODO rmm with prefetch hints

ALLOC_TYPES = ["", "rmm", "fsdp", "fsdp_act_offload", "fsdp_cpu_offload"]
MODEL_SIZES = [
    (2048, 5120, 32, 96),
    (2304, 5760, 32, 96),
    (2560, 6400, 32, 96),
    (2816, 7040, 32, 112),
    (3072, 7680, 32, 128),
    (3328, 8320, 48, 144),
    (3584, 8960, 32, 144),
    (3840, 9600, 32, 160),
    (4096, 10240, 32, 176),
    (4096, 10240, 64, 256)
]
BATCH_SIZES = [(2 ** x, max(96, 96 + (x - 8) * 8)) for x in range(6)]
TIMING_REGEX = re.compile(
    r"Train time: ([0-9-\.]+) s. Full epoch time: ([0-9-\.]+) s")
STEPS_REGEX = re.compile(r", #epoch steps: (\d+),")
OOM_REGEX = re.compile(r"torch\.cuda\.OutOfMemoryError: CUDA out of memory")
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
FULL_LOG_FILE = next(
    f"shmoo_full_logs{x}.txt" for x in itertools.count()
    if not os.path.isfile(f"shmoo_full_logs{x}.txt")
)


def run_custom_config(config):
    cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", "1", "T5_training.py"]
    for k, v in config.items():
        cmd += ["--" + k, "'" + str(v) + "'"]
    return subprocess.run(
        " ".join(cmd), check=False, shell=True, capture_output=True,
        text=True, cwd=SCRIPT_DIR, timeout=3600
    )


def parse_result(result, config):
    # always log results
    with open(FULL_LOG_FILE, "a") as f:
        f.write("=" * 66 + os.linesep)
        f.write(str(config) + os.linesep)
        f.write("=" * 30 + "STDOUT" + "=" * 30 + os.linesep)
        f.write(result.stdout)
        f.write("=" * 30 + "STDERR" + "=" * 30 + os.linesep)
        f.write(result.stderr)
        f.write("=" * 66 + os.linesep)
    # we can usually ignore errors, if we got at least 2 outputs of
    # timing, we know that the run has succeeded far enough
    try:
        timings = [float(x.group(1)) for x in TIMING_REGEX.finditer(result.stdout)]
        steps = [int(x.group(1)) for x in STEPS_REGEX.finditer(result.stdout)]
    except (TypeError, ValueError):
        timings, steps = [], []
    if len(timings) >= 2 and len(steps) >= 2:
        return timings[-1], steps[-1]
    # we did not get at least 2 timing outputs: if the process has failed, this
    # is more serious.
    if result.returncode != 0:
        # First attempt to recognize a CUDA out of memory error,
        # then display full output otherwise
        if OOM_REGEX.search(result.stderr + result.stdout) is None:
            print(
                f"Info: config {config} failed with code {result.returncode} "
                f"(for reason see full logs in {FULL_LOG_FILE})"
            )
        else:
            print(f"Info: config {config} failed with OOM")
    else:
        print(f"Unable to parse output, see full logs in {FULL_LOG_FILE}")
    return 0., -1


def get_alloc_type_config(alloc_type):
    config = {"alloc_type": alloc_type}
    if "fsdp" in alloc_type:
        # use default allocator here !
        config["alloc_type"] = ""
        config["use_fsdp"] = True
        config["fsdp_activation_checkpointing"] = "act_offload" in alloc_type
        config["cpu_offload"] = "cpu_offload" in alloc_type
    else:
        config["use_fsdp"] = False
    return config


def run_model_sizes(out_row, alloc_type):
    for d_model, d_ff, num_heads, max_pool_gb in MODEL_SIZES:
        config = get_alloc_type_config(alloc_type)
        config.update({
            "d_model": d_model, "d_ff": d_ff, "num_heads": num_heads,
            "alloc_max_pool_size": max_pool_gb * (1024 ** 3)
        })
        result = run_custom_config(config)
        time, steps = parse_result(result, config)
        normed_time = time / (steps * train_config.batch_size_training)
        print(
            f"config {config}: got time {time} (#steps/batch size: "
            f"{steps}/{train_config.batch_size_training}, normed: {normed_time})"
        )
        out_row.append(normed_time)


def run_batch_sizes(out_row, alloc_type):
    for batch_size, max_pool_gb in BATCH_SIZES:
        config = get_alloc_type_config(alloc_type)
        config.update({
            "batch_size_training": batch_size,
            "alloc_max_pool_size": max_pool_gb * (1024 ** 3)
        })
        if batch_size > 32:
            config["max_steps_per_epoch"] = 40 // (batch_size // 32)
        result = run_custom_config(config)
        time, steps = parse_result(result, config)
        normed_time = time / (steps * batch_size)
        print(
            f"config {config}: got time {time} (#steps/batch size: "
            f"{steps}/{batch_size}, normed: {normed_time})"
        )
        out_row.append(normed_time)


def main():
    model_rows, batch_rows = [], []
    model_header = ["Model size (d_model/d_ff/num_heads)"]
    model_header += [f"{x[0]}/{x[1]}/{x[2]}" for x in MODEL_SIZES]
    batch_header = ["Batch size"] + [str(x[0]) for x in BATCH_SIZES]
    out_info = [
        (model_header, model_rows, "model.csv"),
        (batch_header, batch_rows, "batch.csv")
    ]
    try:
        for alloc_type in ALLOC_TYPES:
            # first shmoo the model sizes
            model_rows.append([alloc_type or "cuda"])
            run_model_sizes(model_rows[-1], alloc_type)
            # shmoo batch sizes with base config otherwise
            batch_rows.append([alloc_type or "cuda"])
            run_batch_sizes(batch_rows[-1], alloc_type)
    finally:
        print("Dumping CSVs")
        for header, rows, out_file in out_info:
            with open(out_file, "w", newline="") as csvfile:
                writer = csv.writer(
                    csvfile, delimiter=",", quotechar='"',
                    quoting=csv.QUOTE_ALL
                )
                writer.writerow(header)
                writer.writerows(rows)


if __name__ == '__main__':
    main()
