# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import csv
import itertools
import os
import re
import subprocess
from configs import train_config

NUM_GPUS = list(range(1, 9))
DIST_TYPES = ["interleaved-p0-a2", "fsdp", "fsdp_act_cpt", "fsdp_no_shard", "fsdp_no_shard_act_cpt"]
# for CG1:
# NUM_GPUS = [1]
# DIST_TYPES = ["interleaved-p0-a2", "interleaved-p0-a0", "interleaved-p2-a2"]
BATCH_SIZES = [2 ** x for x in range(8)]
TIMING_REGEX = re.compile(
    r"Train time: ([0-9-\.]+) s. Full epoch time: ([0-9-\.]+) s")
STEPS_REGEX = re.compile(r", #epoch steps: (\d+),")
OOM_REGEX = re.compile(r"torch\.cuda\.OutOfMemoryError: CUDA out of memory")
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
FULL_LOG_FILE = next(
    f"shmoo_full_logs{x}.txt" for x in itertools.count()
    if not os.path.isfile(f"shmoo_full_logs{x}.txt")
)


def run_custom_config(config, n_gpus):
    cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", str(n_gpus), "T5_training.py"]
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


def get_alloc_type_config(dist_type):
    config = {"alloc_type": ""}
    if "fsdp" in dist_type:
        config["use_fsdp"] = True
        config["fsdp_activation_checkpointing"] = "act_cpt" in dist_type
        config["fsdp_activation_offloading"] = "act_off" in dist_type
        config["fsdp_no_shard"] = "no_shard" in dist_type
    else:
        config["use_fsdp"] = False
    if "interleaved" in dist_type:
        _, param, act = dist_type.split("-")
        config["interleaved_offload_param"] = int(param[1:])
        config["interleaved_offload_act"] = int(act[1:])
    return config


def run_batch_sizes(out_row, dist_type, n_gpus):
    for batch_size in BATCH_SIZES:
        if batch_size * n_gpus > BATCH_SIZES[-1]:
            print(f"Skipping global batch size {batch_size * n_gpus} > {BATCH_SIZES[-1]}")
            out_row.append(-1)
            continue
        config = get_alloc_type_config(dist_type)
        config.update({
            "batch_size_training": batch_size
        })
        result = run_custom_config(config, n_gpus)
        time, steps = parse_result(result, config)
        normed_time = time / (steps * batch_size)
        print(
            f"Dist: {dist_type}, #GPUs {n_gpus}, config {config}: got time "
            f"{time} (#steps/batch size: {steps}/{batch_size}, "
            f"normed: {normed_time})"
        )
        out_row.append(normed_time)


def main():
    batch_rows = [[] for _ in DIST_TYPES]
    batch_header = ["Batch size"] + [str(b) for b in BATCH_SIZES]
    out_info = [
        (batch_header, batch_rows[i], f"{dist_type}.csv")
        for i, dist_type in enumerate(DIST_TYPES)
    ]
    try:
        for i, dist_type in enumerate(DIST_TYPES):
            for n_gpus in NUM_GPUS:
                if n_gpus > 1 and "interleaved" in dist_type:
                    continue
                batch_rows[i].append([f"#GPUs: {n_gpus}"])
                run_batch_sizes(batch_rows[i][-1], dist_type, n_gpus)
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
