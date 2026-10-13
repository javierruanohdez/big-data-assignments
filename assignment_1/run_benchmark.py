"""
run_benchmark.py - Automated benchmark pipeline for Assignment 1.

Pipeline:
    1. Build the C and Java programs.
    2. Generate the shared input files (common/data_generator.py) if missing.
    3. For every size n, every language and every repetition, launch the
       program as a FRESH process and record:
         - time_s    : kernel time printed by the program itself (monotonic
                       clock, only the multiplication - see each source file)
         - memory_mb : peak resident set size (max RSS) of the whole process,
                       measured by GNU time (/usr/bin/time -v)
    4. Append one row per run to results/measurements.csv immediately, so a
       crash or Ctrl+C never loses measurements already taken.

Why a fresh process per repetition:
    Each repetition then starts from the same state (cold process, same
    memory layout), and peak RSS belongs to exactly one run. For Java this
    means every repetition also performs its own JIT warm-up inside
    MatMul.java, and those warm-up runs are never part of time_s.

Memory metric (the same one for all three languages):
    Peak RSS = the largest amount of physical RAM the process occupied.
    The operating system tracks it, so it needs no code inside the programs.
    For Java it includes the JVM runtime itself (JIT compiler, class data,
    GC structures), not only the matrices - this is the real cost of running
    the program and must be discussed in the report. Heap-only numbers could
    be added later with -Xlog:gc if needed.

Timeouts (guidelines 3):
    A run exceeding TIMEOUT_S is killed and recorded with status "timeout"
    and EMPTY time/memory - no invented numbers. Larger sizes are then
    skipped for that language, because they would only take longer.

Requirements: Linux (or WSL on Windows) with GNU time installed
(Debian/Ubuntu: sudo apt install time), gcc, a JDK and Python 3.10+.

Run (from anywhere):
    python3 assignment_1/run_benchmark.py
    python3 assignment_1/run_benchmark.py --sizes 100 250 --repetitions 5 --languages c java
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import benchmark_config as config

GNU_TIME = "/usr/bin/time"
CSV_COLUMNS = ["size", "language", "repetition", "time_s", "memory_mb", "status"]

# Patterns used to read the measurements out of the programs' output.
TIME_PATTERN = re.compile(r"time_s=([0-9.eE+-]+)")                       # program stdout
RSS_PATTERN = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")  # GNU time stderr


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def check_requirements() -> None:
    """Fail early with a clear message instead of in the middle of a run."""
    if not Path(GNU_TIME).exists():
        sys.exit(f"Error: {GNU_TIME} not found. Install GNU time "
                 "(Debian/Ubuntu: sudo apt install time). On Windows, use WSL.")


def ensure_inputs(sizes: list[int]) -> None:
    """Generate the shared input files that do not exist yet."""
    missing = [n for n in sizes if not config.input_file_for(n).exists()]
    if missing:
        subprocess.run(
            [config.PYTHON_EXECUTABLE, str(config.DATA_GENERATOR),
             "--sizes", *map(str, missing),
             "--seed", str(config.DATA_SEED),
             "--output-dir", str(config.DATA_DIR)],
            check=True,
        )


# ---------------------------------------------------------------------------
# One measured run
# ---------------------------------------------------------------------------

def run_once(language: str, n: int, timeout_s: int) -> dict:
    """
    Run one program once under GNU time.
    Returns a dict with time_s, memory_mb and status ("ok", "timeout", "error").
    """
    command = [GNU_TIME, "-v", *config.run_command(language, config.input_file_for(n))]

    # start_new_session=True puts the program in its own process group,
    # so on a timeout we can kill GNU time AND the program it started.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {"time_s": "", "memory_mb": "", "status": "timeout"}

    time_match = TIME_PATTERN.search(stdout)
    rss_match = RSS_PATTERN.search(stderr)
    if process.returncode != 0 or time_match is None or rss_match is None:
        print(f"    program failed:\n{stderr.strip()}", file=sys.stderr)
        return {"time_s": "", "memory_mb": "", "status": "error"}

    memory_mb = int(rss_match.group(1)) / 1024   # GNU time reports kilobytes
    return {"time_s": float(time_match.group(1)), "memory_mb": round(memory_mb, 2), "status": "ok"}


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment(sizes: list[int], languages: list[str], repetitions: int,
                   timeout_s: int, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    timed_out = set()   # languages that already exceeded the time budget

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for n in sizes:
            for language in languages:
                if language in timed_out:
                    print(f"n={n:<5} {language:<7} skipped (smaller size already timed out)")
                    continue

                for repetition in range(1, repetitions + 1):
                    result = run_once(language, n, timeout_s)
                    writer.writerow({"size": n, "language": language,
                                     "repetition": repetition, **result})
                    f.flush()   # keep the CSV up to date after every run

                    if result["status"] == "ok":
                        details = f"time={result['time_s']}s  mem={result['memory_mb']}MB"
                    else:
                        details = "no measurement recorded"
                    print(f"n={n:<5} {language:<7} rep {repetition}/{repetitions}  "
                          f"{result['status']:<8} {details}")

                    if result["status"] == "timeout":
                        timed_out.add(language)
                        break   # remaining repetitions would also time out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full matmul benchmark.")
    parser.add_argument("--sizes", type=int, nargs="+", default=config.SIZES)
    parser.add_argument("--languages", nargs="+", default=config.LANGUAGES,
                        choices=config.LANGUAGES)
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
    ensure_inputs(args.sizes)

    print(f"Running: sizes={args.sizes} languages={args.languages} "
          f"repetitions={args.repetitions} timeout={args.timeout}s\n")
    run_experiment(args.sizes, args.languages, args.repetitions, args.timeout, args.output)
    print(f"\nRaw measurements saved to {args.output}")


if __name__ == "__main__":
    main()
