"""
c_baselines.py - Runs the compiled C programs of Assignments 2 and 3 on the
same binary input files as the distributed version (Assignment 4).

Two reference points for the cross-paradigm comparison:
    c_baseline   dense_mult    --algo baseline   (Assignment 2: sequential i-j-k loop)
    c_omp_tiled  parallel_mult --algo omp_tiled  (Assignment 3: OpenMP, shared memory)

Why this module loads the Assignment 2/3 configuration files instead of
repeating their paths, flags and command lines here:
    Those files are the single source of truth for how each program is
    built and run (compiler, flags, OpenMP thread pinning, output format).
    Loading them guarantees that Assignment 4 measures exactly the programs
    measured in Assignments 2 and 3, under the same settings.

    Both files are called benchmark_config.py, so a plain "import" would
    load only one of them. importlib loads each from its path under its
    own module name.

Memory metric:
    Peak RSS of the C process, measured with GNU time exactly as in
    Assignments 2 and 3, so the numbers are directly comparable.

    Why not resource.getrusage(RUSAGE_CHILDREN) from Python: a child
    process starts as a copy of its parent and Linux keeps the larger peak
    across exec(). Started from Python (~30 MB), every C program would
    report at least ~30 MB, even for n = 100. GNU time is a small C
    program (~2 MB), so its child starts small.

Platform: the C programs are Linux binaries (built with GCC, run on Linux or
WSL) and GNU time is required, as in Assignments 2 and 3.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

GNU_TIME = Path("/usr/bin/time")

# The line of GNU time's -v report (on stderr) that holds the peak memory.
RSS_LINE_PREFIX = "Maximum resident set size (kbytes):"

ASSIGNMENT_4_DIR = Path(__file__).resolve().parent.parent   # assignment_4/
REPO_ROOT = ASSIGNMENT_4_DIR.parent                         # repository root

C_MODES = ["c_baseline", "c_omp_tiled"]

# The thread-count variables that distributed_matmul.py sets to 1 for NumPy
# (see its first lines). They are removed again for the C programs, so these
# run in exactly the environment of Assignments 2 and 3.
BLAS_THREAD_VARIABLES = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]


# ---------------------------------------------------------------------------
# Loading the Assignment 2 and 3 configurations
# ---------------------------------------------------------------------------

def load_assignment_config(number: int):
    """Load assignment_<number>/benchmark_config.py as module 'assignment_<number>_config'."""
    path = REPO_ROOT / f"assignment_{number}" / "benchmark_config.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: Assignment {number} is needed for the C baselines")
    module_name = f"assignment_{number}_config"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


A2_CONFIG = load_assignment_config(2)   # dense_mult (baseline)
A3_CONFIG = load_assignment_config(3)   # parallel_mult (omp_tiled)

SEQUENTIAL_ALGORITHM = "baseline"       # dense_mult --algo baseline
PARALLEL_ALGORITHM = "omp_tiled"        # parallel_mult --algo omp_tiled

# Both programs print the same "key=value key=value" line; Assignment 3's
# parser handles it (exact key matching, no regular expressions).
parse_program_output = A3_CONFIG.parse_program_output


# ---------------------------------------------------------------------------
# Building (only when needed)
# ---------------------------------------------------------------------------

def _build_settings(mode: str) -> tuple[Path, Path, list[str]]:
    """(binary, source file, compile command) for one C mode."""
    if mode == "c_baseline":
        binary = A2_CONFIG.DENSE_BINARY
        source = A2_CONFIG.SRC_DIR / "dense_mult.c"
        command = [A2_CONFIG.C_COMPILER, *A2_CONFIG.C_FLAGS, "-o", str(binary), str(source)]
    elif mode == "c_omp_tiled":
        binary = A3_CONFIG.BINARY
        source = A3_CONFIG.SOURCE_FILE
        # Same compiler and flags as Assignment 3, but WITHOUT its
        # vectorisation-report flag: that report is a committed result of
        # Assignment 3 and must not be rewritten by Assignment 4.
        command = [A3_CONFIG.C_COMPILER, *A3_CONFIG.C_FLAGS, "-o", str(binary), str(source)]
    else:
        raise ValueError(f"Unknown C mode: {mode}")
    return binary, source, command


def ensure_built(mode: str) -> Path:
    """
    Make sure the binary for `mode` exists and is newer than its source.
    The binaries live in the (git-ignored) build/ folders of Assignments 2
    and 3, so a fresh clone has to compile them once.
    Returns the binary path. Raises CalledProcessError if compilation fails.
    """
    binary, source, command = _build_settings(mode)
    is_stale = binary.exists() and binary.stat().st_mtime < source.stat().st_mtime
    if not binary.exists() or is_stale:
        print(f"  compiling {binary.relative_to(REPO_ROOT)}: {' '.join(command)}", file=sys.stderr)
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    return binary


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_command(mode: str, input_file: Path, workers: int,
                output_file: Path | None = None) -> list[str]:
    """Command line of the C program, built by the Assignment 2/3 configuration."""
    if mode == "c_baseline":
        if workers != 1:
            raise ValueError(f"c_baseline is sequential, got workers={workers}")
        return A2_CONFIG.run_command(SEQUENTIAL_ALGORITHM, input_file, output_file)
    if mode == "c_omp_tiled":
        # Cache tile edge: Assignment 3's benchmark value (BLOCK_SIZE = 64).
        return A3_CONFIG.run_command(PARALLEL_ALGORITHM, input_file, workers, output_file)
    raise ValueError(f"Unknown C mode: {mode}")


def run_environment(mode: str) -> dict[str, str]:
    """Environment of the C program: as in Assignment 2/3 (OpenMP pinning for omp_tiled)."""
    if mode == "c_omp_tiled":
        environment = A3_CONFIG.run_environment()   # adds OMP_PLACES / OMP_PROC_BIND
    else:
        environment = os.environ.copy()
    for variable in BLAS_THREAD_VARIABLES:
        environment.pop(variable, None)
    return environment


def check_requirements() -> None:
    """Fail early with a clear message (the same requirements as Assignments 2 and 3)."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The C baselines are Linux programs: run them on Linux or WSL")
    if not GNU_TIME.exists():
        raise RuntimeError(f"{GNU_TIME} not found. Install GNU time "
                           "(Debian/Ubuntu: sudo apt install time)")


def peak_rss_mb(gnu_time_report: str) -> float:
    """Peak RSS in MB from GNU time's -v report (the same parsing as Assignment 3)."""
    for line in gnu_time_report.splitlines():
        line = line.strip()
        if line.startswith(RSS_LINE_PREFIX):
            kilobytes = int(line[len(RSS_LINE_PREFIX):])
            return kilobytes / 1024
    raise RuntimeError("GNU time did not report the peak RSS")


def run(mode: str, input_file: Path, workers: int, output_file: Path | None = None) -> dict:
    """
    Run one C program once (under GNU time) and return its measurements:
        {"n", "workers", "compute_s", "worker_peak_mb", "checksum"}
    compute_s is the kernel time printed by the C program itself (loading
    and allocation are outside its timer, as in Assignments 2 and 3).
    Raises RuntimeError if the program fails or reports something unexpected.
    """
    check_requirements()
    ensure_built(mode)

    command = [str(GNU_TIME), "-v", *run_command(mode, input_file, workers, output_file)]
    completed = subprocess.run(command, capture_output=True, text=True, env=run_environment(mode))
    # stdout = the program's result line; stderr = GNU time's report
    # (plus any message the program printed).
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} failed with exit code {completed.returncode}: "
                           f"{completed.stderr.strip()}")

    fields = parse_program_output(completed.stdout)
    # dense_mult prints no "threads" field (it is always sequential).
    reported_threads = int(fields.get("threads", "1"))
    if reported_threads != workers:
        raise RuntimeError(f"{mode} reports threads={reported_threads}, requested {workers}")

    return {
        "n": int(fields["n"]),
        "workers": reported_threads,
        "compute_s": float(fields["time_s"]),
        "worker_peak_mb": peak_rss_mb(completed.stderr),
        "checksum": fields["checksum"],          # kept as text: the exact %.10e string
    }
