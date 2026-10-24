"""
run_benchmark.py - Automated benchmark pipeline for Assignment 2.

Pipeline:
    1. Build dense_mult and sparse_mult (benchmark_config.build_all).
    2. Generate the shared input files that are missing: dense inputs for
       Part A, sparse inputs for Part B (common/data_generator.py).
    3. Run Part A, then Part B. The two parts never mix (see below). For
       every configuration and every repetition, the program is launched as
       a FRESH process under GNU time, and one CSV row is recorded.
    4. Every row is written and flushed immediately, so a crash or Ctrl+C
       never loses measurements already taken.

The two experiments (defined in benchmark_config.py):
    Part A - dense optimisation : sizes PART_A_SIZES, sparsity 0.0,
                                  algorithms baseline, reordered, tiled
    Part B - sparse storage     : size PART_B_SIZE, sparsities PART_B_SPARSITIES,
                                  algorithms tiled (best dense) and csr
    Each part is self-contained: the configuration (n=1000, sparsity 0.0,
    tiled) appears in both, and is measured in both, so each part's results
    can be analysed without borrowing numbers from the other.

CSV columns (results/measurements.csv):
    experiment_part    "A" (dense optimisation) or "B" (sparse storage)
    size, sparsity     the input: n, and the fraction of zeros in A and B
    algorithm          baseline | reordered | tiled | csr
    repetition         1 .. REPETITIONS
    time_s             multiplication kernel only, as printed by the program
    conversion_time_s  dense -> CSR conversion of A and B (csr only)
    memory_mb          peak resident set size of the whole process (GNU time)
    csr_bytes          storage of the CSR arrays of A, B and C (csr only)
    nnz_C              number of non-zeros in the result C (csr only)
    status             ok | timeout | error | mismatch
  An EMPTY cell means "does not apply" (e.g. the conversion time of a dense
  kernel) or "no measurement" (timeout/error). It is deliberately not 0.0:
  a dense kernel has no conversion step, which is not the same as a free
  one. An analysis that needs total time uses time_s + conversion_time_s,
  treating an empty conversion time as 0.

Why a fresh process per repetition:
    Each repetition starts from the same state (cold process, same memory
    layout), and peak RSS belongs to exactly one run.

Memory metric:
    Peak RSS = the largest amount of physical RAM the process occupied,
    tracked by the operating system and reported by GNU time. For csr it
    includes the dense input buffers read at the start (the shared inputs
    are stored densely), so it overstates the memory of the CSR format
    itself. csr_bytes reports that storage directly; the report uses both.

Consistency check (cheap insurance, not a replacement for verify_correctness.py):
    For the same input file, all four kernels produce bit-identical results
    (see the C sources), so they must print the same checksum. The first
    checksum seen for each input file is remembered; a later run that prints
    a different one is recorded with status "mismatch" instead of "ok", so a
    wrong result can never slip into the statistics unnoticed.

Timeouts:
    A run exceeding the budget is killed and recorded with status "timeout"
    and empty measurements; no numbers are invented.
    Part A: larger sizes are then skipped for that algorithm (they would
            only take longer). The largest size with status "ok" is the
            largest feasible n for the report.
    Part B: nothing is skipped. n is fixed and csr gets faster as sparsity
            grows, so a slow level says nothing about the next one.

Requirements: Linux (or WSL on Windows) with GNU time installed
(Debian/Ubuntu: sudo apt install time), gcc and Python 3.10+.

Run (from anywhere):
    python3 assignment_2/run_benchmark.py
    python3 assignment_2/run_benchmark.py --parts B --sparsities 0.5 0.9 --repetitions 3 \\
        --output assignment_2/results/quick_test.csv
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

CSV_COLUMNS = ["experiment_part", "size", "sparsity", "algorithm", "repetition",
               "time_s", "conversion_time_s", "memory_mb", "csr_bytes", "nnz_C", "status"]

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


def ensure_inputs(inputs: list[tuple[int, float]]) -> None:
    """Generate the shared input files (n, sparsity) that do not exist yet."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n, sparsity in inputs:
        path = config.input_file_for(n, sparsity)
        if not path.exists():
            print(f"  generating {path.name}")
            config.data_generator.generate_for_size(n, config.DATA_SEED, config.DATA_DIR, sparsity)


# ---------------------------------------------------------------------------
# One measured run
# ---------------------------------------------------------------------------

def empty_measurement(status: str) -> dict:
    """A row without measurements, for runs that timed out or failed."""
    return {"time_s": "", "conversion_time_s": "", "memory_mb": "",
            "csr_bytes": "", "nnz_C": "", "status": status}


def peak_rss_mb(gnu_time_report: str) -> float | None:
    """Peak RSS in MB from GNU time's -v report, or None if it is missing."""
    for line in gnu_time_report.splitlines():
        line = line.strip()
        if line.startswith(RSS_LINE_PREFIX):
            kilobytes = int(line[len(RSS_LINE_PREFIX):])
            return kilobytes / 1024
    return None


def run_once(algorithm: str, input_file: Path, timeout_s: int) -> tuple[dict, str | None]:
    """
    Run one algorithm once under GNU time.
    Returns (measurement, checksum). The measurement dict has the CSV metric
    columns plus "status"; checksum is None when the run did not succeed.
    """
    command = [GNU_TIME, "-v", *config.run_command(algorithm, input_file)]

    # start_new_session=True puts the program in its own process group,
    # so on a timeout we can kill GNU time AND the program it started.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return empty_measurement("timeout"), None

    # stdout = the program's result line; stderr = GNU time's report
    # (plus any message the program printed).
    try:
        if process.returncode != 0:
            raise ValueError(f"exit code {process.returncode}")
        fields = config.parse_program_output(stdout)
        memory_mb = peak_rss_mb(stderr)
        if memory_mb is None:
            raise ValueError("GNU time did not report the peak RSS")

        measurement = empty_measurement("ok")
        measurement["time_s"] = float(fields["time_s"])
        measurement["memory_mb"] = round(memory_mb, 2)
        if algorithm == "csr":
            measurement["conversion_time_s"] = float(fields["conversion_time_s"])
            measurement["csr_bytes"] = int(fields["csr_bytes"])
            measurement["nnz_C"] = int(fields["nnz_C"])
        return measurement, fields["checksum"]

    except (ValueError, KeyError) as problem:
        # KeyError: a field we need is missing from the program's output.
        print(f"    run failed ({problem!r}):\n{stderr.strip()}", file=sys.stderr)
        return empty_measurement("error"), None


# ---------------------------------------------------------------------------
# One configuration = all repetitions of (part, n, sparsity, algorithm)
# ---------------------------------------------------------------------------

def run_configuration(record, part: str, n: int, sparsity: float, algorithm: str,
                      repetitions: int, timeout_s: int, checksums: dict) -> bool:
    """
    Measure one configuration `repetitions` times and record every run.
    `checksums` maps input file -> first checksum seen (consistency check).
    Returns True if a run timed out (the remaining repetitions are skipped,
    because they would time out as well).
    """
    input_file = config.input_file_for(n, sparsity)

    for repetition in range(1, repetitions + 1):
        measurement, checksum = run_once(algorithm, input_file, timeout_s)

        if measurement["status"] == "ok":
            expected = checksums.setdefault(input_file, checksum)
            if checksum != expected:
                measurement["status"] = "mismatch"
                print(f"    WARNING: checksum {checksum} differs from {expected} "
                      f"for {input_file.name}", file=sys.stderr)

        record({"experiment_part": part, "size": n, "sparsity": sparsity,
                "algorithm": algorithm, "repetition": repetition, **measurement})
        print_progress(part, n, sparsity, algorithm, repetition, repetitions, measurement)

        if measurement["status"] == "timeout":
            return True
    return False


def print_progress(part: str, n: int, sparsity: float, algorithm: str,
                   repetition: int, repetitions: int, measurement: dict) -> None:
    """One console line per run, so a long benchmark shows where it is."""
    status = measurement["status"]
    if status in ("ok", "mismatch"):
        details = f"time={measurement['time_s']:.4f}s  mem={measurement['memory_mb']}MB"
        if algorithm == "csr":
            details += (f"  conv={measurement['conversion_time_s']:.4f}s"
                        f"  nnz_C={measurement['nnz_C']}")
    else:
        details = "no measurement recorded"
    print(f"[{part}] n={n:<5} sparsity={sparsity:<6g} {algorithm:<10} "
          f"rep {repetition}/{repetitions}  {status:<8} {details}")


# ---------------------------------------------------------------------------
# The two experiments
# ---------------------------------------------------------------------------

def run_part_a(record, sizes: list[int], repetitions: int, timeout_s: int,
               checksums: dict) -> None:
    """Part A - dense optimisation: increasing n, dense inputs (sparsity 0.0)."""
    print("\n=== Part A: dense optimisation (sparsity 0.0, increasing n) ===")
    timed_out = set()   # algorithms that already exceeded the time budget

    for n in sizes:
        for algorithm in config.PART_A_ALGORITHMS:
            if algorithm in timed_out:
                print(f"[A] n={n:<5} {algorithm:<10} skipped (a smaller size timed out)")
                continue
            if run_configuration(record, "A", n, 0.0, algorithm,
                                 repetitions, timeout_s, checksums):
                timed_out.add(algorithm)


def run_part_b(record, n: int, sparsities: list[float], repetitions: int,
               timeout_s: int, checksums: dict) -> None:
    """Part B - sparse storage: fixed n, increasing sparsity, tiled vs csr."""
    print(f"\n=== Part B: sparse storage (n = {n}, increasing sparsity) ===")
    for sparsity in sparsities:
        for algorithm in config.PART_B_ALGORITHMS:
            run_configuration(record, "B", n, sparsity, algorithm,
                              repetitions, timeout_s, checksums)


def run_experiment(args: argparse.Namespace) -> None:
    """Run the selected parts and write every run to args.output."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}   # shared by both parts: same input file -> same checksum

    with open(args.output, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        def record(row: dict) -> None:
            writer.writerow(row)
            csv_file.flush()   # keep the CSV up to date after every run

        if "A" in args.parts:
            run_part_a(record, args.part_a_sizes, args.repetitions, args.timeout, checksums)
        if "B" in args.parts:
            run_part_b(record, args.part_b_size, args.sparsities, args.repetitions,
                       args.timeout, checksums)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Assignment 2 benchmark (Part A: dense, Part B: sparse).")
    parser.add_argument("--parts", nargs="+", choices=["A", "B"], default=["A", "B"],
                        help="Which experiments to run (default: both). The output "
                             "CSV is overwritten, so use --output when running one part.")
    parser.add_argument("--part-a-sizes", type=int, nargs="+", default=config.PART_A_SIZES)
    parser.add_argument("--part-b-size", type=int, default=config.PART_B_SIZE)
    parser.add_argument("--sparsities", type=float, nargs="+", default=config.PART_B_SPARSITIES,
                        help="Part B sparsity levels (fractions between 0.0 and 1.0)")
    parser.add_argument("--repetitions", type=int, default=config.REPETITIONS)
    parser.add_argument("--timeout", type=int, default=config.TIMEOUT_S,
                        help="Seconds allowed per run")
    parser.add_argument("--output", type=Path,
                        default=config.RESULTS_DIR / "measurements.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    check_requirements()

    print("Building programs...")
    config.build_all()

    print("Preparing inputs...")
    inputs = []
    if "A" in args.parts:
        inputs += [(n, 0.0) for n in args.part_a_sizes]
    if "B" in args.parts:
        inputs += [(args.part_b_size, sparsity) for sparsity in args.sparsities]
    ensure_inputs(inputs)

    print(f"Running parts={args.parts} repetitions={args.repetitions} "
          f"timeout={args.timeout}s flags={' '.join(config.C_FLAGS)} "
          f"block_size={config.TILED_BLOCK_SIZE}")
    run_experiment(args)
    print(f"\nRaw measurements saved to {args.output}")


if __name__ == "__main__":
    main()
