"""
run_benchmark.py - Automated benchmark pipeline for Assignment 4.

Pipeline:
    1. Build the C programs of Assignments 2 and 3 if they are missing or
       out of date (benchmark_config.ensure_c_binaries).
    2. Generate the shared input files that are missing (common/data_generator.py).
    3. Run the planned configurations (below). For every configuration and
       every repetition, distributed_matmul.py is launched as a FRESH process
       and one CSV row is recorded.
    4. Every row is written and flushed immediately, so a crash or Ctrl+C
       never loses measurements already taken.

The plan (parameters in benchmark_config.py):
    strong   for n in STRONG_SIZES (2000, 4000), tile edge b = 500:
                 c_baseline              p = 1           (only n in C_COMPARISON_SIZES)
                 c_omp_tiled             p in WORKER_COUNTS  (only n in C_COMPARISON_SIZES)
                 seq_python              p = 1
                 distributed             p in WORKER_COUNTS
    weak     for p in WORKER_COUNTS: n_p = round(1000 * p^(1/3)), b_p = ceil(n_p / 4):
                 seq_python              p = 1 at n_p
                 distributed             p at n_p
    Worker counts above the number of logical CPUs are skipped unless
    --allow-oversubscription is given: more busy processes than hardware
    threads would measure the operating system's time slicing, not the program.

CSV columns (results/measurements.csv):
    experiment              strong | weak
    mode                    distributed | seq_python | c_baseline | c_omp_tiled
    kernel                  block kernel of the Python modes (blas); empty for C
    size                    matrix size n
    workers                 p: Dask worker processes, or OpenMP threads (c_omp_tiled)
    grid                    process grid pr x pc of the 2D block-cyclic placement
    block_size              tile edge b (Python modes; empty for C, whose
                            cache tile is fixed by Assignment 3: 64)
    tile_grid, tasks        tiles per side nb, number of block tasks nb^3
    repetition              1 .. REPETITIONS
    e2e_s                   distributed: pipelined end-to-end time on warm
                            workers (partition + scatter + compute + collect,
                            no barriers, cluster setup excluded); seq_python:
                            the whole block algorithm; C: the kernel time the
                            program prints
    setup_s                 distributed: cluster start + warm-up on 1x1 tiles
    cold_e2e_s              distributed: the same as e2e_s for the FIRST
                            full-size pass on fresh worker processes
    scatter_s, compute_s, gather_s
                            distributed: phases of the separate BARRIERED pass
                            (their sum is close to, not equal to, e2e_s);
                            seq_python: partition / multiply / assemble; C: compute only
    worker_peak_mb_max, worker_peak_mb_mean
                            peak RSS of the computing process(es): the Dask
                            workers (largest and mean), or the single process
    client_peak_mb          distributed: peak RSS of the driver (client + scheduler)
    scatter_bytes           client -> workers (all A and B tiles once)
    comm_bytes_theoretical  worker <-> worker, predicted by the placement plan
    comm_bytes_measured     worker <-> worker, recorded by Dask during the barriered pass
    gather_bytes            workers -> client (all C tiles once)
    checksum                sum of C (same order and format as the C programs)
    status                  ok | timeout | error | mismatch
  Empty cells mean "not applicable" or "no measurement" (timeout or error);
  they are deliberately never 0.

Why a fresh process per repetition:
    Each repetition starts from the same state: a new cluster with new
    worker processes, cold caches in the client, and a peak-memory counter
    that belongs to exactly one run. The cluster start-up itself is timed
    separately (setup_s) and is not part of e2e_s.

Consistency checks (cheap insurance, not a replacement for verify_correctness.py):
    - Checksum: the first checksum seen for each input file is remembered
      (normally the C baseline's); a later run whose checksum differs by
      more than CHECKSUM_REL_TOL is recorded as "mismatch". BLAS rounds
      differently from the C loops, so exact equality is not expected.
    - Labels: a run that reports a different mode, size, worker count, block
      size or kernel from the one requested is recorded as "error".

Timeouts:
    A run exceeding the budget is killed together with all of its Dask
    processes and recorded with status "timeout" and empty measurements.
    Larger sizes are then skipped for that (experiment, mode, workers),
    because they would only take longer.

Requirements: Linux (or WSL on Windows) with GNU time, GCC >= 14 (for the
Assignment 3 program) and the Python packages in requirements.txt.

Run (from anywhere):
    python3 assignment_4/run_benchmark.py
    python3 assignment_4/run_benchmark.py --experiments strong --sizes 2000 --workers 1 2 \\
        --repetitions 2 --output assignment_4/results/quick_test.csv
"""

import argparse
import csv
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

import benchmark_config as config

CSV_COLUMNS = [
    "experiment", "mode", "kernel", "size", "workers", "grid", "block_size", "tile_grid", "tasks",
    "repetition",
    "e2e_s", "setup_s", "cold_e2e_s", "scatter_s", "compute_s", "gather_s",
    "worker_peak_mb_max", "worker_peak_mb_mean", "client_peak_mb",
    "scatter_bytes", "comm_bytes_theoretical", "comm_bytes_measured", "gather_bytes",
    "checksum", "status",
]

# Columns copied from the program's result line (it calls the size "n").
MEASURED_COLUMNS = [
    "grid", "tile_grid", "tasks",
    "e2e_s", "setup_s", "cold_e2e_s", "scatter_s", "compute_s", "gather_s",
    "worker_peak_mb_max", "worker_peak_mb_mean", "client_peak_mb",
    "scatter_bytes", "comm_bytes_theoretical", "comm_bytes_measured", "gather_bytes",
    "checksum",
]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def configuration(experiment: str, mode: str, n: int, workers: int,
                  block_size: int | None = None) -> dict:
    """One configuration = everything that identifies a set of repetitions."""
    kernel = config.KERNEL if mode in config.PYTHON_MODES else None
    return {"experiment": experiment, "mode": mode, "kernel": kernel, "size": n,
            "workers": workers, "block_size": block_size}


def strong_scaling_plan(sizes: list[int], worker_counts: list[int]) -> list[dict]:
    """Fixed n, increasing p (see the module docstring)."""
    plan = []
    for n in sizes:
        if n in config.C_COMPARISON_SIZES:
            plan.append(configuration("strong", config.C_SEQUENTIAL_MODE, n, 1))
            for workers in worker_counts:
                plan.append(configuration("strong", config.C_PARALLEL_MODE, n, workers))
        plan.append(configuration("strong", config.SEQUENTIAL_PYTHON_MODE, n, 1,
                                  config.STRONG_BLOCK_SIZE))
        for workers in worker_counts:
            plan.append(configuration("strong", config.DISTRIBUTED_MODE, n, workers,
                                      config.STRONG_BLOCK_SIZE))
    return plan


def weak_scaling_plan(worker_counts: list[int]) -> list[dict]:
    """n grows with p so that the work per worker stays constant."""
    plan = []
    for workers in worker_counts:
        n = config.weak_size(workers)
        block_size = config.weak_block_size(n)
        plan.append(configuration("weak", config.SEQUENTIAL_PYTHON_MODE, n, 1, block_size))
        plan.append(configuration("weak", config.DISTRIBUTED_MODE, n, workers, block_size))
    return plan


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_inputs(sizes: list[int]) -> None:
    """Generate the shared dense input files that do not exist yet."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n in sorted(set(sizes)):
        path = config.input_file_for(n)
        if not path.exists():
            print(f"  generating {path.name} (pure Python generator: a few seconds per file)")
            config.data_generator.generate_for_size(n, config.DATA_SEED, config.DATA_DIR)


def select_worker_counts(requested: list[int], allow_oversubscription: bool) -> list[int]:
    """Worker counts to measure: by default, only those the hardware can run at once."""
    available = config.logical_cpu_count()
    if allow_oversubscription:
        return requested

    selected = [p for p in requested if p <= available]
    skipped = [p for p in requested if p > available]
    if skipped:
        print(f"  skipping workers={skipped}: more than the {available} logical CPUs "
              "(use --allow-oversubscription to run them anyway)")
    return selected


def describe_environment() -> str:
    """Machine and software versions, printed at the start (for the report)."""
    import dask
    import distributed
    import numpy
    import psutil

    memory_gb = psutil.virtual_memory().total / 1024 ** 3
    return (f"  machine: {platform.system()} {platform.release()} | "
            f"{config.physical_core_count()} physical cores, {config.logical_cpu_count()} logical CPUs, "
            f"{memory_gb:.1f} GB RAM\n"
            f"  software: Python {platform.python_version()}, dask {dask.__version__}, "
            f"distributed {distributed.__version__}, numpy {numpy.__version__}, psutil {psutil.__version__}")


# ---------------------------------------------------------------------------
# One measured run
# ---------------------------------------------------------------------------

def empty_measurement(status: str) -> dict:
    """A row without measurements, for runs that timed out or failed."""
    row = {column: "" for column in MEASURED_COLUMNS}
    row["status"] = status
    return row


def check_labels(fields: dict, setting: dict) -> None:
    """Raise ValueError if the program reports a configuration other than the requested one."""
    expected = {"mode": setting["mode"], "n": setting["size"], "workers": setting["workers"]}
    if setting["mode"] in config.PYTHON_MODES:
        expected["block_size"] = setting["block_size"]
        expected["kernel"] = setting["kernel"]
    for key, wanted in expected.items():
        if fields.get(key) != str(wanted):
            raise ValueError(f"program reports {key}={fields.get(key)}, requested {wanted}")


def run_once(setting: dict, timeout_s: int) -> dict:
    """
    Run one configuration once. Returns the measured columns plus "status".
    """
    command = config.run_command(setting["mode"], config.input_file_for(setting["size"]),
                                 setting["workers"], setting["block_size"], setting["kernel"])

    # start_new_session=True puts the program and every process it starts
    # (Dask nannies and workers, or GNU time and the C program) in one new
    # process group, so a timeout or Ctrl+C can stop all of them at once.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return empty_measurement("timeout")
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGKILL)   # do not leave a cluster running
        process.communicate()
        raise

    try:
        if process.returncode != 0:
            raise ValueError(f"exit code {process.returncode}")
        # The result line is the last line on stdout (logs go to stderr).
        lines = stdout.strip().splitlines()
        if not lines:
            raise ValueError("no result line on stdout")
        fields = config.parse_program_output(lines[-1])
        check_labels(fields, setting)

        row = empty_measurement("ok")
        for column in MEASURED_COLUMNS:
            row[column] = fields.get(column, "")   # absent = not applicable to this mode
        if row["checksum"] == "":
            raise ValueError("no checksum in the result line")
        return row

    except (ValueError, KeyError) as problem:
        print(f"    run failed ({problem}):\n{stderr.strip()}", file=sys.stderr)
        return empty_measurement("error")


# ---------------------------------------------------------------------------
# One configuration = all repetitions
# ---------------------------------------------------------------------------

def run_configuration(record, setting: dict, repetitions: int, timeout_s: int,
                      checksums: dict) -> bool:
    """
    Measure one configuration `repetitions` times and record every run.
    `checksums` maps input size -> first checksum seen (consistency check).
    Returns True if a run timed out (the remaining repetitions are skipped,
    because they would time out as well).
    """
    for repetition in range(1, repetitions + 1):
        measurement = run_once(setting, timeout_s)

        if measurement["status"] == "ok":
            expected = checksums.setdefault(setting["size"], measurement["checksum"])
            if not config.checksums_agree(measurement["checksum"], expected):
                measurement["status"] = "mismatch"
                print(f"    WARNING: checksum {measurement['checksum']} differs from "
                      f"{expected} for n={setting['size']}", file=sys.stderr)

        row = {**setting, "kernel": setting["kernel"] or "",
               "block_size": setting["block_size"] or "",
               "repetition": repetition, **measurement}
        record(row)
        print_progress(setting, repetition, repetitions, measurement)

        if measurement["status"] == "timeout":
            return True
    return False


def print_progress(setting: dict, repetition: int, repetitions: int, measurement: dict) -> None:
    """One console line per run, so a long benchmark shows where it is."""
    label = (f"{setting['experiment']:<6} n={setting['size']:<5} {setting['mode']:<12} "
             f"p={setting['workers']:<2} b={setting['block_size'] or '-':<4} "
             f"rep {repetition}/{repetitions}")
    status = measurement["status"]
    if status not in ("ok", "mismatch"):
        print(f"{label}  {status:<8} no measurement recorded")
        return

    details = f"e2e={float(measurement['e2e_s']):.4f}s"
    if measurement["cold_e2e_s"] != "":
        details += f" (cold {float(measurement['cold_e2e_s']):.4f}s)"
    if measurement["scatter_s"] != "":
        phases = [float(measurement[key]) for key in ("scatter_s", "compute_s", "gather_s")]
        details += (f"  phases={phases[0]:.3f}+{phases[1]:.3f}+{phases[2]:.3f}"
                    f"={sum(phases):.4f}s")
    if measurement["setup_s"] != "":
        details += f"  setup={float(measurement['setup_s']):.2f}s"
    details += f"  mem/worker={measurement['worker_peak_mb_max']}MB"
    print(f"{label}  {status:<8} {details}")


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace, plan: list[dict]) -> None:
    """Run every configuration of the plan and write the rows to args.output."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}      # size -> first checksum seen
    timed_out = set()   # (experiment, mode, workers) that exceeded the budget

    with open(args.output, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        def record(row: dict) -> None:
            writer.writerow(row)
            csv_file.flush()   # keep the CSV up to date after every run

        current_heading = None
        for setting in plan:
            heading = f"=== {setting['experiment']} scaling, n = {setting['size']} ==="
            if heading != current_heading:
                print(f"\n{heading}")
                current_heading = heading

            group = (setting["experiment"], setting["mode"], setting["workers"])
            if group in timed_out:
                print(f"{setting['experiment']:<6} n={setting['size']:<5} {setting['mode']:<12} "
                      f"p={setting['workers']:<2} skipped (a smaller size timed out)")
                continue
            if run_configuration(record, setting, args.repetitions, args.timeout, checksums):
                timed_out.add(group)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Assignment 4 benchmark (strong and weak scaling).")
    parser.add_argument("--experiments", nargs="+", choices=["strong", "weak"],
                        default=["strong", "weak"])
    parser.add_argument("--sizes", type=int, nargs="+", default=config.STRONG_SIZES,
                        help="matrix sizes for strong scaling")
    parser.add_argument("--workers", type=int, nargs="+", default=config.WORKER_COUNTS,
                        help="worker counts p (also the OpenMP thread counts of c_omp_tiled)")
    parser.add_argument("--allow-oversubscription", action="store_true",
                        help="also run worker counts above the number of logical CPUs")
    parser.add_argument("--repetitions", type=int, default=config.REPETITIONS)
    parser.add_argument("--timeout", type=int, default=config.TIMEOUT_S,
                        help="seconds allowed per run")
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "measurements.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    worker_counts = select_worker_counts(args.workers, args.allow_oversubscription)

    plan = []
    if "strong" in args.experiments:
        plan += strong_scaling_plan(args.sizes, worker_counts)
    if "weak" in args.experiments:
        plan += weak_scaling_plan(worker_counts)
    if not plan:
        print("Nothing to run.", file=sys.stderr)
        return 1

    print("Checking requirements and C programs...")
    try:
        config.c_baselines.check_requirements()
        config.ensure_c_binaries()
    except (RuntimeError, subprocess.CalledProcessError) as problem:
        print(f"Error: {problem}", file=sys.stderr)
        return 1

    print("Preparing inputs...")
    ensure_inputs([setting["size"] for setting in plan])

    print(f"Running {len(plan)} configurations x {args.repetitions} repetitions "
          f"(timeout {args.timeout}s per run), kernel={config.KERNEL}\n"
          f"  strong: sizes={args.sizes if 'strong' in args.experiments else '-'} "
          f"b={config.STRONG_BLOCK_SIZE}; "
          f"weak: n_1={config.WEAK_BASE_SIZE}, tile grid {config.WEAK_TILE_GRID}x{config.WEAK_TILE_GRID}; "
          f"workers={worker_counts}\n"
          f"{describe_environment()}")
    try:
        run_experiment(args, plan)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Measurements taken so far are in {args.output}")
        return 130

    print(f"\nRaw measurements saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
