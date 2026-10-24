"""
benchmark_config.py - Central configuration for Assignment 2.

Every fixed parameter of the experiment lives here, so the README and the
report can point to ONE place. verify_correctness.py and run_benchmark.py
both import this module, so the way each program is built, run and parsed
is defined exactly once.

The experiment has two strictly separate parts (the report discusses them
separately, as the guidelines require):

    Part A - dense optimisation
        Question : how much do loop reordering and tiling speed up the
                   dense kernel?
        Varies   : matrix size n          (sparsity fixed at 0.0)
        Compares : baseline, reordered, tiled

    Part B - sparse storage
        Question : from which sparsity does CSR beat the best dense kernel,
                   once conversion and indexing overheads are counted?
        Varies   : sparsity               (size fixed at PART_B_SIZE)
        Compares : tiled (best dense kernel from Part A) vs csr
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all derived from this file's location, so scripts can be launched
# from any working directory)
# ---------------------------------------------------------------------------

ASSIGNMENT_DIR = Path(__file__).resolve().parent          # assignment_2/
REPO_ROOT = ASSIGNMENT_DIR.parent                          # repository root
SRC_DIR = ASSIGNMENT_DIR / "src" / "c"
BUILD_DIR = ASSIGNMENT_DIR / "build"                       # compiled programs (git-ignored)
RESULTS_DIR = ASSIGNMENT_DIR / "results"
DATA_DIR = REPO_ROOT / "data"                              # shared .bin inputs (git-ignored)
DATA_GENERATOR = REPO_ROOT / "common" / "data_generator.py"

DENSE_BINARY = BUILD_DIR / "c" / "dense_mult"
SPARSE_BINARY = BUILD_DIR / "c" / "sparse_mult"

# The generator defines the input file names; importing it (instead of
# repeating the naming rule here) keeps a single source of truth.
sys.path.insert(0, str(DATA_GENERATOR.parent))
import data_generator  # noqa: E402

# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

DENSE_ALGORITHMS = ["baseline", "reordered", "tiled"]     # dense_mult --algo ...
SPARSE_ALGORITHMS = ["csr"]                                # sparse_mult
ALGORITHMS = DENSE_ALGORITHMS + SPARSE_ALGORITHMS

# Tile edge for --algo tiled. 64 -> one tile = 32 KiB (about one L1 data
# cache), three tiles = 96 KiB (fit in L2). See dense_mult.c for details.
TILED_BLOCK_SIZE = 64

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Part A: increasing n, dense inputs. 2000 is included because B (32 MB)
# is then far larger than the last-level cache, which is exactly the regime
# that reordering and tiling target. Larger sizes for one algorithm are
# skipped after it times out, so the list can be extended safely.
PART_A_SIZES = [100, 250, 500, 1000, 2000]
PART_A_ALGORITHMS = ["baseline", "reordered", "tiled"]

# Part B: fixed n, increasing sparsity (fraction of exact zeros in A and B).
# The guideline levels (50/90/95/99 %) plus extra points where the curves
# change fastest: 0.3-0.6 around the crossover with the dense kernel, and
# 0.999 where conversion dominates. One run at n=1000 takes < 1 s, so the
# extra levels cost about a minute in total.
PART_B_SIZE = 1000
PART_B_SPARSITIES = [0.0, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
PART_B_ALGORITHMS = ["tiled", "csr"]

REPETITIONS = 5                 # measured runs per configuration
TIMEOUT_S = 300                 # hard budget per run: 5 minutes (as in Assignment 1)
DATA_SEED = 42                  # passed to data_generator.py

# ---------------------------------------------------------------------------
# Toolchain settings
# ---------------------------------------------------------------------------

# Same flags as Assignment 1, for all four kernels. -O2 is the standard
# "release" level; it keeps the comparison about algorithms and memory
# access patterns. (-O3 adds auto-vectorisation, which helps the stride-1
# kernels much more than the baseline; see the report's discussion.)
C_COMPILER = "gcc"
C_FLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra"]

# The interpreter running these scripts (used for the reference checks).
PYTHON_EXECUTABLE = sys.executable


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def input_file_for(n: int, sparsity: float = 0.0) -> Path:
    """Path of the shared input file for (n, sparsity), as named by data_generator.py."""
    return DATA_DIR / data_generator.matrix_filename(n, sparsity)


# ---------------------------------------------------------------------------
# Building and running the programs
# ---------------------------------------------------------------------------

def build_commands() -> list[list[str]]:
    """Commands that compile both C programs into BUILD_DIR."""
    return [
        [C_COMPILER, *C_FLAGS, "-o", str(DENSE_BINARY), str(SRC_DIR / "dense_mult.c")],
        [C_COMPILER, *C_FLAGS, "-o", str(SPARSE_BINARY), str(SRC_DIR / "sparse_mult.c")],
    ]


def build_all() -> None:
    """Compile both programs. Raises CalledProcessError if the compiler fails."""
    DENSE_BINARY.parent.mkdir(parents=True, exist_ok=True)
    for command in build_commands():
        subprocess.run(command, check=True)


def run_command(algorithm: str, input_file: Path, output_file: Path | None = None) -> list[str]:
    """Command that runs one algorithm on one input file."""
    if algorithm in DENSE_ALGORITHMS:
        command = [str(DENSE_BINARY), str(input_file), "--algo", algorithm]
        if algorithm == "tiled":
            command += ["--block-size", str(TILED_BLOCK_SIZE)]
    elif algorithm == "csr":
        command = [str(SPARSE_BINARY), str(input_file)]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if output_file is not None:
        command += ["--output", str(output_file)]
    return command


def parse_program_output(stdout: str) -> dict[str, str]:
    """
    Turn the result line of either program into a dict, e.g.
        "algo=csr n=500 nnz_C=118000 conversion_time_s=0.0012 time_s=0.0045 ..."
        -> {"algo": "csr", "n": "500", "nnz_C": "118000", ...}

    Why not a regular expression: keys can contain other keys as a
    substring. The Assignment 1 pattern r"time_s=..." would also match
    inside "conversion_time_s=..." and silently return the conversion time.
    Splitting into whole key=value tokens makes every key an exact match.

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
