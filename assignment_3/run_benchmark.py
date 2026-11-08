"""
run_benchmark.py - Automated benchmark pipeline for Assignment 3.

Pipeline:
    1. Build parallel_mult (benchmark_config.build_all); this also rewrites
       results/vectorization_report.txt.
    2. Generate the shared input files that are missing (common/data_generator.py).
    3. Sweep sizes x algorithms x thread counts. For every configuration and
       every repetition, the program is launched as a FRESH process under
       GNU time, and one CSV row is recorded.
    4. Every row is written and flushed immediately, so a crash or Ctrl+C
       never loses measurements already taken.

The sweep (defined in benchmark_config.py):
    for n in SIZES:
        seq_tiled                     p = 1 only (the sequential baseline)
        omp_tiled, omp_simd_tiled     p in THREAD_COUNTS
    Thread counts above the number of logical CPUs are skipped unless
    --allow-oversubscription is given: with more threads than hardware
    threads, the OS must time-slice them, which measures the scheduler
    rather than the kernel.

CSV columns (results/measurements.csv):
    size        matrix size n
    algo        seq_tiled | omp_tiled | omp_simd_tiled
    threads     number of OpenMP threads (as reported by the program)
    repetition  1 .. REPETITIONS
    time_s      multiplication kernel only, as printed by the program
    memory_mb   peak resident set size of the whole process (GNU time)
    status      ok | timeout | error | mismatch
  Empty time_s / memory_mb cells mean "no measurement" (timeout or error);
  they are deliberately not 0.0.
  The block size is fixed for the whole sweep (benchmark_config.BLOCK_SIZE).

Why a fresh process per repetition:
    Each repetition starts from the same state (cold process, new thread
    team), and peak RSS belongs to exactly one run. The program creates its
    threads before starting the timer, so this does not add start-up cost
    to time_s.

Memory metric:
    Peak RSS = the largest amount of physical RAM the process occupied,
    tracked by the operating system and reported by GNU time. It is
    dominated by A, B and C (3 * n^2 * 8 bytes); each extra thread only adds
    the few stack pages it actually touches.

Consistency checks (cheap insurance, not a replacement for verify_correctness.py):
    - Checksum: for the same input file, all kernels produce bit-identical
      results with any thread count (see the C source), so they must print
      the same checksum. The first checksum seen for each input file is
      remembered; a later run with a different one is recorded as
      "mismatch", so a wrong result (e.g. a race) cannot slip into the
      statistics unnoticed.
    - Thread count: the program reports the team size OpenMP really
      started. A run that reports a different count from the one requested
      is recorded as "error", never under the wrong label.

Timeouts:
    A run exceeding the budget is killed and recorded with status "timeout"
    and empty measurements; no numbers are invented. Larger sizes are then
    skipped for that (algorithm, threads) pair, because they would only take
    longer.

Requirements: Linux (or WSL on Windows) with GNU time installed
(Debian/Ubuntu: sudo apt install time), GCC >= 14 and Python 3.10+.

Run (from anywhere):
    python3 assignment_3/run_benchmark.py
    python3 assignment_3/run_benchmark.py --sizes 500 --threads 1 2 --repetitions 2 \\
        --output assignment_3/results/quick_test.csv
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
from pathlib import Path

import benchmark_config as config

GNU_TIME = "/usr/bin/time"

CSV_COLUMNS = ["size", "algo", "threads", "repetition", "time_s", "memory_mb", "status"]

# The line of GNU time's -v report (on stderr) that holds the peak memory.
RSS_LINE_PREFIX = "Maximum resident set size (kbytes):"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def check_requirements() -> None:
    """Fail early with a clear message instead of in the middle of a run."""
    if not Path(GNU_TIME).exists():
        sys.exit(f"Error: {GNU_TIME} not found. Install GNU time "
                 "(Debian/Ubuntu: sudo apt install time). On Windows, use WSL.")


def ensure_inputs(sizes: list[int]) -> None:
    """Generate the shared dense input files that do not exist yet."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        path = config.input_file_for(n)
        if not path.exists():
            print(f"  generating {path.name}")
            config.data_generator.generate_for_size(n, config.DATA_SEED, config.DATA_DIR)


def select_thread_counts(requested: list[int], allow_oversubscription: bool) -> list[int]:
    """Thread counts to measure: by default, only those the hardware can run at once."""
    available = config.logical_cpu_count()
    if allow_oversubscription:
        return requested

    selected = [p for p in requested if p <= available]
    skipped = [p for p in requested if p > available]
    if skipped:
        print(f"  skipping threads={skipped}: more than the {available} logical CPUs "
              "(use --allow-oversubscription to run them anyway)")
    return selected


# ---------------------------------------------------------------------------
# One measured run
# ---------------------------------------------------------------------------

def empty_measurement(status: str) -> dict:
    """A row without measurements, for runs that timed out or failed."""
    return {"time_s": "", "memory_mb": "", "status": status}


def peak_rss_mb(gnu_time_report: str) -> float | None:
    """Peak RSS in MB from GNU time's -v report, or None if it is missing."""
    for line in gnu_time_report.splitlines():
        line = line.strip()
        if line.startswith(RSS_LINE_PREFIX):
            kilobytes = int(line[len(RSS_LINE_PREFIX):])
            return kilobytes / 1024
    return None


def run_once(algorithm: str, input_file: Path, threads: int,
             timeout_s: int) -> tuple[dict, str | None]:
    """
    Run one algorithm once under GNU time.
    Returns (measurement, checksum). The measurement dict has the CSV metric
    columns plus "status"; checksum is None when the run did not succeed.
    """
    command = [GNU_TIME, "-v", *config.run_command(algorithm, input_file, threads)]

    # start_new_session=True puts the program in its own process group,
    # so on a timeout we can kill GNU time AND the program it started.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True,
                               env=config.run_environment())
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return empty_measurement("timeout"), None

    # stdout = the program's result line; stderr = GNU time's report
    # (plus any warning the program printed).
    try:
        if process.returncode != 0:
            raise ValueError(f"exit code {process.returncode}")
        fields = config.parse_program_output(stdout)
        if fields["threads"] != str(threads):
            raise ValueError(f"program reports threads={fields['threads']}, "
                             f"requested {threads}")
        memory_mb = peak_rss_mb(stderr)
        if memory_mb is None:
            raise ValueError("GNU time did not report the peak RSS")

        measurement = empty_measurement("ok")
        measurement["time_s"] = float(fields["time_s"])
        measurement["memory_mb"] = round(memory_mb, 2)
        return measurement, fields["checksum"]

    except (ValueError, KeyError) as problem:
        # KeyError: a field we need is missing from the program's output.
        print(f"    run failed ({problem!r}):\n{stderr.strip()}", file=sys.stderr)
        return empty_measurement("error"), None


# ---------------------------------------------------------------------------
# One configuration = all repetitions of (n, algorithm, threads)
# ---------------------------------------------------------------------------

def run_configuration(record, n: int, algorithm: str, threads: int,
                      repetitions: int, timeout_s: int, checksums: dict) -> bool:
    """
    Measure one configuration `repetitions` times and record every run.
    `checksums` maps input file -> first checksum seen (consistency check).
    Returns True if a run timed out (the remaining repetitions are skipped,
    because they would time out as well).
    """
    input_file = config.input_file_for(n)

    for repetition in range(1, repetitions + 1):
        measurement, checksum = run_once(algorithm, input_file, threads, timeout_s)

        if measurement["status"] == "ok":
            expected = checksums.setdefault(input_file, checksum)
            if checksum != expected:
                measurement["status"] = "mismatch"
                print(f"    WARNING: checksum {checksum} differs from {expected} "
                      f"for {input_file.name}", file=sys.stderr)

        record({"size": n, "algo": algorithm, "threads": threads,
                "repetition": repetition, **measurement})
        print_progress(n, algorithm, threads, repetition, repetitions, measurement)

        if measurement["status"] == "timeout":
            return True
    return False


def print_progress(n: int, algorithm: str, threads: int, repetition: int,
                   repetitions: int, measurement: dict) -> None:
    """One console line per run, so a long benchmark shows where it is."""
    status = measurement["status"]
    if status in ("ok", "mismatch"):
        details = f"time={measurement['time_s']:.4f}s  mem={measurement['memory_mb']}MB"
    else:
        details = "no measurement recorded"
    print(f"n={n:<5} {algorithm:<15} p={threads:<3} rep {repetition}/{repetitions}  "
          f"{status:<8} {details}")


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace, thread_counts: list[int]) -> None:
    """Run every (n, algorithm, threads) configuration and write it to args.output."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}      # input file -> first checksum seen
    timed_out = set()   # (algorithm, threads) pairs that exceeded the budget

    with open(args.output, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        def record(row: dict) -> None:
            writer.writerow(row)
            csv_file.flush()   # keep the CSV up to date after every run

        for n in args.sizes:
            print(f"\n=== n = {n} ===")
            for algorithm in args.algorithms:
                for threads in config.thread_counts_for(algorithm, thread_counts):
                    if (algorithm, threads) in timed_out:
                        print(f"n={n:<5} {algorithm:<15} p={threads:<3} "
                              "skipped (a smaller size timed out)")
                        continue
                    if run_configuration(record, n, algorithm, threads,
                                         args.repetitions, args.timeout, checksums):
                        timed_out.add((algorithm, threads))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Assignment 3 benchmark (threads and SIMD).")
    parser.add_argument("--sizes", type=int, nargs="+", default=config.SIZES)
    parser.add_argument("--algorithms", nargs="+", choices=config.ALGORITHMS,
                        default=config.ALGORITHMS)
    parser.add_argument("--threads", type=int, nargs="+", default=config.THREAD_COUNTS,
                        help="Thread counts for the omp_* kernels")
    parser.add_argument("--allow-oversubscription", action="store_true",
                        help="Also run thread counts above the number of logical CPUs")
    parser.add_argument("--repetitions", type=int, default=config.REPETITIONS)
    parser.add_argument("--timeout", type=int, default=config.TIMEOUT_S,
                        help="Seconds allowed per run")
    parser.add_argument("--output", type=Path,
                        default=config.RESULTS_DIR / "measurements.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    check_requirements()

    print("Building program...")
    config.build_all()
    print(f"  vectorisation report: {config.VECTORIZATION_REPORT}")

    print("Preparing inputs...")
    ensure_inputs(args.sizes)
    thread_counts = select_thread_counts(args.threads, args.allow_oversubscription)

    omp_settings = " ".join(f"{key}={value}" for key, value in config.OMP_ENVIRONMENT.items())
    print(f"Running sizes={args.sizes} algorithms={args.algorithms} threads={thread_counts}\n"
          f"  repetitions={args.repetitions} timeout={args.timeout}s "
          f"block_size={config.BLOCK_SIZE} logical_cpus={config.logical_cpu_count()}\n"
          f"  flags={' '.join(config.C_FLAGS)}\n"
          f"  {omp_settings}")
    run_experiment(args, thread_counts)
    print(f"\nRaw measurements saved to {args.output}")


if __name__ == "__main__":
    main()
