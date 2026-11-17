"""
benchmark_config.py - Central configuration for Assignment 4.

Every fixed parameter of the experiment lives here, so the README and the
report can point to ONE place. verify_correctness.py and run_benchmark.py
both import this module, so the way the program is run and its output
parsed is defined exactly once.

The experiment:
    Question : how does a distributed (multi-process, message-passing) block
               matrix multiplication scale with the number of worker
               processes p, what do partitioning, data transfer and
               scheduling cost, and how does it compare with the sequential
               (Assignment 2) and shared-memory parallel (Assignment 3) C
               programs on the same input files?

    Strong scaling (fixed problem, more workers)
        sizes n = 2000, 4000; workers p = 1, 2, 4, 8; tile edge b = 500
        references: seq_python (same algorithm, no Dask) at every n;
                    c_baseline and c_omp_tiled (p = 1..8 threads) at n = 2000

    Weak scaling (work per worker constant)
        The work grows as n^3, so p workers get p times the work of one
        worker when n_p = round(n_1 * p^(1/3)):
            p = 1 -> 1000, p = 2 -> 1260, p = 4 -> 1587, p = 8 -> 2000
        The tile GRID is fixed at 4 x 4, so the tile edge grows with n
        (b_p = ceil(n_p / 4) = 250, 315, 397, 500): every run has the same
        16 C tiles and 64 tasks, the work per task grows like the work per
        worker, and 16 tiles split evenly over 1, 2, 4 or 8 workers.
        reference: seq_python at every n_p

Metrics for the report (computed by plot_results.py from the medians in the CSV):
    speedup              S(p)   = T(distributed, 1, n) / T(distributed, p, n)   (self-relative)
    efficiency           E(p)   = S(p) / p
    gain vs. NumPy              T(seq_python, n) / T(distributed, p, n)
                         (same kernel and tiles, no Dask: > 1 means distribution paid off)
    weak efficiency      E_w(p) = T(distributed, 1, n_1) / T(distributed, p, n_p)   (ideal: 1)
    overhead breakdown   setup_s, scatter_s, compute_s, gather_s
    communication        scatter, inter-worker (theoretical and measured) and gather bytes
    cross-paradigm       T(c_baseline), T(c_omp_tiled, p) vs T(distributed, p) at n = 2000
"""

import math
import os
import sys
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# Paths (all derived from this file's location, so scripts can be launched
# from any working directory)
# ---------------------------------------------------------------------------

ASSIGNMENT_DIR = Path(__file__).resolve().parent          # assignment_4/
REPO_ROOT = ASSIGNMENT_DIR.parent                          # repository root
SRC_DIR = ASSIGNMENT_DIR / "src"
PROGRAM = SRC_DIR / "distributed_matmul.py"
BUILD_DIR = ASSIGNMENT_DIR / "build"                       # verification files (git-ignored)
RESULTS_DIR = ASSIGNMENT_DIR / "results"
DATA_DIR = REPO_ROOT / "data"                              # shared .bin inputs (git-ignored)
DATA_GENERATOR = REPO_ROOT / "common" / "data_generator.py"

# The generator defines the input file names, and c_baselines.py knows how
# to build and run the Assignment 2/3 programs. Importing both (instead of
# repeating their rules here) keeps a single source of truth.
sys.path.insert(0, str(DATA_GENERATOR.parent))
sys.path.insert(0, str(SRC_DIR))
import data_generator  # noqa: E402
import c_baselines     # noqa: E402

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

DISTRIBUTED_MODE = "distributed"
SEQUENTIAL_PYTHON_MODE = "seq_python"
C_SEQUENTIAL_MODE = "c_baseline"        # Assignment 2: dense_mult --algo baseline
C_PARALLEL_MODE = "c_omp_tiled"         # Assignment 3: parallel_mult --algo omp_tiled
PYTHON_MODES = [DISTRIBUTED_MODE, SEQUENTIAL_PYTHON_MODE]
C_MODES = [C_SEQUENTIAL_MODE, C_PARALLEL_MODE]
MODES = PYTHON_MODES + C_MODES

# Block kernel of the Python modes. "blas" (NumPy matmul) is what a real
# distributed system runs on each node; "ordered" reproduces the C results
# bit for bit but is ~20x slower and is used by verify_correctness.py.
# See multiply_accumulate() in src/distributed_matmul.py.
KERNEL = "blas"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Worker processes p (or OpenMP threads for c_omp_tiled). On the reference
# machine (4 physical cores, 8 logical CPUs with SMT): up to 4, every worker
# gets a core of its own; at 8, two workers share each core. Counts above
# the number of logical CPUs are skipped by run_benchmark.py unless
# --allow-oversubscription is given.
WORKER_COUNTS = [1, 2, 4, 8]

# --- Strong scaling ---------------------------------------------------------
STRONG_SIZES = [2000, 4000]

# Tile edge b = 500 for both sizes:
#   - n = 2000 -> 4 x 4 tiles, n = 4000 -> 8 x 8 tiles. Both tile grids
#     split evenly over the process grids 1x1, 1x2, 2x2 and 2x4, so every
#     worker owns the same number of C tiles (no load imbalance).
#   - One task multiplies two 500 x 500 tiles: 2 * 500^3 = 0.25 GFLOP,
#     7-9 ms with single-threaded BLAS on the reference machine. That is far above Dask's
#     scheduling cost per task (well under 1 ms), so the measurement is
#     not dominated by bookkeeping.
#   - Keeping b fixed across p means the same tasks are only distributed
#     differently: the per-task work does not change with p.
STRONG_BLOCK_SIZE = 500

# Cross-paradigm comparison with the C programs (the baseline needs ~16 s
# at n = 2000, so larger sizes would mostly measure waiting).
C_COMPARISON_SIZES = [2000]

# --- Weak scaling -----------------------------------------------------------
WEAK_BASE_SIZE = 1000      # n_1: the problem of a single worker
WEAK_TILE_GRID = 4         # tiles per side at every p (see the module docstring)

# --- Repetitions and limits -------------------------------------------------
REPETITIONS = 5            # measured runs per configuration
TIMEOUT_S = 300            # hard budget per run: 5 minutes (as in Assignments 1-3)
DATA_SEED = 42             # passed to data_generator.py

# Checksums of different kernels agree only up to rounding: BLAS adds the
# terms of each dot product in its own order (with FMA), the C kernels in
# order k = 0..n-1. The difference in the checksum is around 1e-13
# relative; a real error (missing or duplicated tile) changes it by far
# more than 1e-9. (The "ordered" kernel and the C programs agree exactly.)
CHECKSUM_REL_TOL = 1e-9

# The interpreter running these scripts also runs the program.
PYTHON_EXECUTABLE = sys.executable


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def input_file_for(n: int) -> Path:
    """Path of the shared dense input file for size n, as named by data_generator.py."""
    return DATA_DIR / data_generator.matrix_filename(n, 0.0)


def weak_size(workers: int) -> int:
    """n_p = round(n_1 * p^(1/3)): p times the O(n^3) work of n_1."""
    return round(WEAK_BASE_SIZE * workers ** (1.0 / 3.0))


def weak_block_size(n: int) -> int:
    """Tile edge that cuts n into WEAK_TILE_GRID tiles per side (the last one may be shorter)."""
    return math.ceil(n / WEAK_TILE_GRID)


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


def physical_core_count() -> int:
    """Number of physical cores (psutil); falls back to the logical count."""
    return psutil.cpu_count(logical=False) or logical_cpu_count()


# ---------------------------------------------------------------------------
# Running the program
# ---------------------------------------------------------------------------

def ensure_c_binaries() -> None:
    """Compile dense_mult and parallel_mult if they are missing or out of date."""
    for mode in C_MODES:
        c_baselines.ensure_built(mode)


def run_command(mode: str, input_file: Path, workers: int, block_size: int | None = None,
                kernel: str | None = None, output_file: Path | None = None) -> list[str]:
    """Command that runs distributed_matmul.py in one mode on one input file."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")

    command = [PYTHON_EXECUTABLE, str(PROGRAM), str(input_file), "--mode", mode]
    if mode in (DISTRIBUTED_MODE, C_PARALLEL_MODE):
        command += ["--workers", str(workers)]
    elif workers != 1:
        # The program would reject it anyway; fail here with a clearer message.
        raise ValueError(f"{mode} is sequential, got workers={workers}")
    if mode in PYTHON_MODES:
        command += ["--block-size", str(block_size), "--kernel", kernel or KERNEL]
    if output_file is not None:
        command += ["--output", str(output_file)]
    return command


# Both the C programs and distributed_matmul.py print one line of
# key=value pairs; Assignment 3's parser handles all of them.
parse_program_output = c_baselines.parse_program_output


def checksums_agree(checksum: str, reference: str) -> bool:
    """True if two printed checksums are equal up to CHECKSUM_REL_TOL (see above)."""
    return math.isclose(float(checksum), float(reference),
                        rel_tol=CHECKSUM_REL_TOL, abs_tol=CHECKSUM_REL_TOL)
