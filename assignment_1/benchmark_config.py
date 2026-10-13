"""
benchmark_config.py - Central configuration for Assignment 1.

Every fixed parameter of the experiment lives here, so the README and the
report can point to ONE place (guidelines, section 4: "Record and fix the
following parameters in a central configuration file"). Both
verify_correctness.py and run_benchmark.py import this module, so the way
each language is built and run is defined exactly once.
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all derived from this file's location, so scripts can be launched
# from any working directory)
# ---------------------------------------------------------------------------

ASSIGNMENT_DIR = Path(__file__).resolve().parent          # assignment_1/
REPO_ROOT = ASSIGNMENT_DIR.parent                          # repository root
SRC_DIR = ASSIGNMENT_DIR / "src"
BUILD_DIR = ASSIGNMENT_DIR / "build"                       # compiled programs (git-ignored)
RESULTS_DIR = ASSIGNMENT_DIR / "results"
DATA_DIR = REPO_ROOT / "data"                              # shared .bin inputs (git-ignored)
DATA_GENERATOR = REPO_ROOT / "common" / "data_generator.py"

C_BINARY = BUILD_DIR / "c" / "matmul"
JAVA_CLASS_DIR = BUILD_DIR / "java"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SIZES = [100, 250, 500, 1000]   # increasing n; stop at the first timeout
REPETITIONS = 5                 # measured runs per (size, language)
TIMEOUT_S = 300                 # hard budget per run: 5 minutes (guidelines 3)
DATA_SEED = 42                  # passed to data_generator.py
LANGUAGES = ["c", "java", "python"]

# ---------------------------------------------------------------------------
# Toolchain settings
# ---------------------------------------------------------------------------

# -O2: the standard "release" optimization level. -O0 would keep every
# variable in memory and measure an artificially slow program that nobody
# would ship; -O3 mainly adds auto-vectorization, which the i-j-k loop with
# its strided access to B and its sequential sum cannot use without
# -ffast-math anyway. -O2 is therefore the fair "normal compiled C" baseline.
C_COMPILER = "gcc"
C_FLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra"]

# -Xmx2g fixes the maximum heap so results do not depend on the JVM's
# automatic sizing (which varies with the machine's RAM). 2 GB is far more
# than n=1000 needs (3 matrices x 8 MB), so the GC never limits the kernel.
JAVA_COMPILER = "javac"
JAVA_RUNTIME = "java"
JVM_FLAGS = ["-Xmx2g"]
JAVA_WARMUP_RUNS = 3            # untimed kernel runs inside MatMul.java

# The interpreter running these scripts is also the one benchmarked,
# so the recorded Python version is always the one actually used.
PYTHON_EXECUTABLE = sys.executable


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def build_commands() -> list[list[str]]:
    """Commands that compile the C and Java programs into BUILD_DIR."""
    return [
        [C_COMPILER, *C_FLAGS, "-o", str(C_BINARY), str(SRC_DIR / "c" / "matmul.c")],
        [JAVA_COMPILER, "-d", str(JAVA_CLASS_DIR), str(SRC_DIR / "java" / "MatMul.java")],
    ]


def build_all() -> None:
    """Compile C and Java. Raises CalledProcessError if a compiler fails."""
    C_BINARY.parent.mkdir(parents=True, exist_ok=True)
    JAVA_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    for command in build_commands():
        subprocess.run(command, check=True)


def run_command(language: str, input_file: Path, output_file: Path | None = None) -> list[str]:
    """Command that runs one language's program on one input file."""
    if language == "c":
        command = [str(C_BINARY), str(input_file)]
    elif language == "java":
        command = [JAVA_RUNTIME, *JVM_FLAGS, "-cp", str(JAVA_CLASS_DIR), "MatMul",
                   str(input_file), "--warmup", str(JAVA_WARMUP_RUNS)]
    elif language == "python":
        command = [PYTHON_EXECUTABLE, str(SRC_DIR / "python" / "matmul.py"), str(input_file)]
    else:
        raise ValueError(f"Unknown language: {language}")

    if output_file is not None:
        command += ["--output", str(output_file)]
    return command


def input_file_for(n: int) -> Path:
    """Path of the shared input file for size n (as named by data_generator.py)."""
    return DATA_DIR / f"matrices_n{n}.bin"
