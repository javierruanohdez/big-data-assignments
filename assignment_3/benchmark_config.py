"""
benchmark_config.py - Central configuration for Assignment 3.

Every fixed parameter of the experiment lives here, so the README and the
report can point to ONE place. verify_correctness.py and run_benchmark.py
both import this module, so the way the program is built, run and parsed
is defined exactly once.

The experiment:
    Question : how does the time of the tiled kernel change with the number
               of threads p, and what does explicit SIMD add on top?
    Varies   : matrix size n, thread count p
    Compares : seq_tiled       sequential baseline (Assignment 2 kernel), p = 1 only
               omp_tiled       OpenMP threads, scalar inner loop
               omp_simd_tiled  OpenMP threads + '#pragma omp simd' (optional part)
    Fixed    : block size, dense inputs, compiler flags, OpenMP binding

Metrics for the report (computed later from the medians in the CSV):
    speedup          S(p) = T(seq_tiled) / T(algo, p)
    efficiency       E(p) = S(p) / p
    OpenMP overhead  T(omp_tiled, p=1) - T(seq_tiled)
    SIMD gain        T(omp_tiled, p) / T(omp_simd_tiled, p)
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all derived from this file's location, so scripts can be launched
# from any working directory)
# ---------------------------------------------------------------------------

ASSIGNMENT_DIR = Path(__file__).resolve().parent          # assignment_3/
REPO_ROOT = ASSIGNMENT_DIR.parent                          # repository root
SOURCE_RELATIVE = Path("src") / "c" / "parallel_mult.c"    # relative to ASSIGNMENT_DIR
SOURCE_FILE = ASSIGNMENT_DIR / SOURCE_RELATIVE
BUILD_DIR = ASSIGNMENT_DIR / "build"                       # compiled program (git-ignored)
RESULTS_DIR = ASSIGNMENT_DIR / "results"
DATA_DIR = REPO_ROOT / "data"                              # shared .bin inputs (git-ignored)
DATA_GENERATOR = REPO_ROOT / "common" / "data_generator.py"

BINARY = BUILD_DIR / "c" / "parallel_mult"

# GCC's vectorisation report, written at every build. It is the evidence
# that only omp_simd_tiled was vectorised, so it lives in results/ and is
# committed with the measurements.
VECTORIZATION_REPORT = RESULTS_DIR / "vectorization_report.txt"

# The generator defines the input file names; importing it (instead of
# repeating the naming rule here) keeps a single source of truth.
sys.path.insert(0, str(DATA_GENERATOR.parent))
import data_generator  # noqa: E402

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

SEQUENTIAL_ALGORITHM = "seq_tiled"
PARALLEL_ALGORITHMS = ["omp_tiled", "omp_simd_tiled"]
ALGORITHMS = [SEQUENTIAL_ALGORITHM] + PARALLEL_ALGORITHMS   # parallel_mult --algo ...

# Tile edge for all three kernels. 64 -> one tile = 32 KiB (about one L1
# data cache), three tiles = 96 KiB (fit in L2). It is also a multiple of 8,
# which makes every thread boundary a cache-line boundary (no false
# sharing; see multiply_omp_tiled in the C source).
BLOCK_SIZE = 64

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# 500: all three matrices (6 MB) fit in the 8 MiB L3 -> little work per
#      thread, so start-up and synchronisation costs are most visible.
# 1000, 2000: the matrices (24 MB, 96 MB) far exceed the L3 -> the regime
#      where tiling matters and where threads compete for memory bandwidth.
SIZES = [500, 1000, 2000]

# Thread counts p for the omp_* kernels (seq_tiled always runs once, p = 1).
#   p = 1  measures the cost of the OpenMP machinery against seq_tiled.
#   Counts above the number of logical CPUs are skipped by run_benchmark.py
#   unless --allow-oversubscription is given (more threads than hardware
#   threads only adds context switches). On the reference machine
#   (4 cores / 8 hardware threads) that means 16 is skipped by default.
THREAD_COUNTS = [1, 2, 4, 8, 16]

REPETITIONS = 5                 # measured runs per configuration
TIMEOUT_S = 300                 # hard budget per run: 5 minutes (as before)
DATA_SEED = 42                  # passed to data_generator.py

# ---------------------------------------------------------------------------
# OpenMP runtime settings (passed as environment variables to every run)
# ---------------------------------------------------------------------------

# Thread binding ("pinning"). Without it, the OS may move a thread to
# another core in the middle of a run, which empties its caches and adds
# run-to-run variability.
#   OMP_PLACES=cores    one place = one physical core (both of its hardware
#                       threads).
#   OMP_PROC_BIND=spread spread the threads over the places as evenly as
#                       possible: p = 2 and p = 4 use separate physical cores;
#                       p = 8 puts two threads on each of the 4 cores (SMT).
# This makes the scaling curve easy to read: up to 4 threads, every new
# thread brings a whole core; beyond that, threads share a core's FP units.
# The thread COUNT itself is not set here: the program fixes it with
# omp_set_num_threads(--threads), which overrides OMP_NUM_THREADS anyway.
OMP_ENVIRONMENT = {
    "OMP_PLACES": "cores",
    "OMP_PROC_BIND": "spread",
}

# ---------------------------------------------------------------------------
# Toolchain settings
# ---------------------------------------------------------------------------

# GCC >= 14 is required ('#pragma GCC novector', see the C source).
C_COMPILER = "gcc"

# The SAME flags for all three kernels:
#   -O3               full optimisation (includes auto-vectorisation)
#   -march=native     use every instruction set of THIS CPU (AVX2/AVX-512),
#                     so SIMD has its full width. The binary is meant to run
#                     only on the machine that built it.
#   -ftree-vectorize  already implied by -O3; listed to document that the
#                     vectoriser is ON. The scalar kernels opt out loop by
#                     loop with '#pragma GCC novector'.
#   -fopenmp          enables the OpenMP pragmas and links the runtime (libgomp)
#   -std=c11          strict ISO C. Side effect we rely on: it implies
#                     -ffp-contract=off, i.e. no FMA contraction, so the scalar
#                     and the SIMD kernels round identically and produce
#                     bit-identical results (checked through the checksum).
#   -Wall -Wextra     warnings, as in the previous assignments
C_FLAGS = ["-O3", "-march=native", "-ftree-vectorize", "-fopenmp",
           "-std=c11", "-Wall", "-Wextra"]

# -fopt-info-vec-optimized: list every loop the vectoriser transformed
# (the same as -fopt-info-vec), written to a file instead of the terminal.
REPORT_FLAG = f"-fopt-info-vec-optimized={VECTORIZATION_REPORT}"

# The interpreter running these scripts (used for the reference checks).
PYTHON_EXECUTABLE = sys.executable


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def input_file_for(n: int) -> Path:
    """Path of the shared dense input file for size n, as named by data_generator.py."""
    return DATA_DIR / data_generator.matrix_filename(n, 0.0)


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------

def logical_cpu_count() -> int:
    """
    Number of logical CPUs this process may run on.
    sched_getaffinity respects restrictions (e.g. taskset, containers);
    os.cpu_count() is the fallback where it does not exist.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def thread_counts_for(algorithm: str, thread_counts: list[int]) -> list[int]:
    """Thread counts to measure for one algorithm: seq_tiled is sequential by definition."""
    if algorithm == SEQUENTIAL_ALGORITHM:
        return [1]
    return thread_counts


# ---------------------------------------------------------------------------
# Building and running the program
# ---------------------------------------------------------------------------

def build_command() -> list[str]:
    """
    Command that compiles parallel_mult. It is run from ASSIGNMENT_DIR with
    a relative source path, so the report reads "src/c/parallel_mult.c:LINE"
    instead of a machine-specific absolute path.
    """
    return [C_COMPILER, *C_FLAGS, REPORT_FLAG,
            "-o", str(BINARY), str(SOURCE_RELATIVE)]


def build_all() -> None:
    """Compile the program. Raises CalledProcessError if the compiler fails."""
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # GCC APPENDS to an existing report file, so start from an empty one.
    VECTORIZATION_REPORT.unlink(missing_ok=True)
    subprocess.run(build_command(), check=True, cwd=ASSIGNMENT_DIR)


def run_command(algorithm: str, input_file: Path, threads: int,
                output_file: Path | None = None,
                block_size: int = BLOCK_SIZE) -> list[str]:
    """Command that runs one algorithm with `threads` threads on one input file."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if algorithm == SEQUENTIAL_ALGORITHM and threads != 1:
        # The program would ignore it, and the run would be recorded
        # under the wrong thread count.
        raise ValueError(f"{algorithm} runs on 1 thread only, got threads={threads}")

    command = [str(BINARY), str(input_file), "--algo", algorithm,
               "--threads", str(threads), "--block-size", str(block_size)]
    if output_file is not None:
        command += ["--output", str(output_file)]
    return command


def run_environment() -> dict[str, str]:
    """Environment for every run: the current one plus the OpenMP settings above."""
    environment = os.environ.copy()
    environment.update(OMP_ENVIRONMENT)
    return environment


def parse_program_output(stdout: str) -> dict[str, str]:
    """
    Turn the result line of the program into a dict, e.g.
        "algo=omp_tiled n=500 threads=4 block_size=64 time_s=0.0123 checksum=..."
        -> {"algo": "omp_tiled", "n": "500", "threads": "4", ...}

    Splitting into whole key=value tokens makes every key an exact match
    (a regular expression for "time_s=" could also match inside a longer
    key, as we learned in Assignment 2).

    Values stay strings; callers convert the ones they need. Raises
    ValueError if the line is malformed (a token without '=').
    """
    fields = {}
    for token in stdout.split():
        key, separator, value = token.partition("=")
        if separator != "=" or key == "":
            raise ValueError(f"Unexpected token in program output: {token!r}")
        fields[key] = value
    return fields
