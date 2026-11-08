"""
verify_correctness.py - Correctness validation for Assignment 3.

What it does:
    1. Builds parallel_mult (benchmark_config.build_all), which also writes
       GCC's vectorisation report.
    2. Checks that report: the inner loop of omp_simd_tiled must be
       vectorised, and the scalar kernels seq_tiled and omp_tiled must have
       NO vectorised loop. This is the proof that the SIMD comparison
       compares scalar code with vector code.
    3. Creates the test cases:
         identity        C = A x I must equal A (exactly)
         zero            C = A x 0 must be all zeros
         known_3x3       a product computed by hand (exact integers)
         random_n<size>  shared generator input, small
         random_n100     larger than one 64-tile: full + partial edge tiles
         random_n131     odd prime size: every tile edge is uneven
    4. Runs every kernel on every case with two block sizes and, for the
       omp_* kernels, several thread counts:
         block sizes 64  the benchmark value
                     7   small and odd: many tiles per thread, and inner
                         loops of 7 iterations, which are not a multiple of
                         the SIMD width, so the vector "epilogue" code runs
         threads 1, 2, 3, 4, 8, 16
                     3   does not divide the number of tiles: uneven split
                     16  for small n there are fewer row tiles than threads,
                         so some threads get no work; the result must still
                         be complete. (Using more threads than the machine
                         has is fine for a correctness check.)
    5. Compares every result element-wise against an independent REFERENCE
       computed with math.fsum, using absolute + relative tolerances.
    6. Checks that the program reports the thread count and block size it
       was asked for (so the benchmark can trust those labels).
    7. Reports, for information, whether each result is bit-identical to
       seq_tiled's. With our flags they all are (same summation order per
       element, no FMA), but this is not required: tolerances decide.

Why a separate reference instead of comparing the kernels to each other:
    If all kernels shared the same bug, they would agree with each other and
    the bug would go unnoticed. math.fsum returns the correctly rounded sum,
    i.e. the most accurate double result possible.

Run (from anywhere):
    python3 assignment_3/verify_correctness.py [--size 10]
Exit code 0 = all checks passed, 1 = at least one failure.
"""

import argparse
import math
import random
import re
import subprocess
import sys
from array import array
from pathlib import Path

import benchmark_config as config

# The generator module (imported by benchmark_config) writes the test inputs
# in exactly the format the program reads.
data_generator = config.data_generator

VERIFY_DIR = config.BUILD_DIR / "verify"

TILE_TEST_SIZES = [100, 131]
VERIFY_BLOCK_SIZES = [config.BLOCK_SIZE, 7]
VERIFY_THREAD_COUNTS = [1, 2, 3, 4, 8, 16]

# ---------------------------------------------------------------------------
# Tolerances (same rule and values as Assignments 1 and 2)
# ---------------------------------------------------------------------------
# An element passes if   |result - reference| <= ABS_TOL + REL_TOL * |reference|
#
# Why tolerances at all: floating-point addition is not associative. Our
# kernels add the terms in the same order and normally agree bit for bit,
# but a different compiler or flag (e.g. FMA contraction) changes the last
# bits. That is rounding, not a bug.
#
# Values: unit roundoff u = 2^-53. The error of a length-n dot product is at
# most n * u * sum_k |a_ik * b_kj| <= n^2 * u for inputs in [-1, 1)
# (Higham, "Accuracy and Stability of Numerical Algorithms", ch. 3).
#   ABS_TOL = 100 * n^2 * u  -> 100x margin; essential for values near zero.
#   REL_TOL = 1e-12          -> accepts rounding differences, while a real
#                               bug (wrong index, missing tile, race) causes
#                               errors around 0.1 for our data.
UNIT_ROUNDOFF = 2.0 ** -53
SAFETY_FACTOR = 100
REL_TOL = 1e-12


def absolute_tolerance(n: int) -> float:
    """Absolute tolerance derived from the dot-product error bound (see above)."""
    return SAFETY_FACTOR * n * n * UNIT_ROUNDOFF


# ---------------------------------------------------------------------------
# Matrix helpers (flat row-major lists, the same layout as the program)
# ---------------------------------------------------------------------------

def identity_matrix(n: int) -> list[float]:
    matrix = [0.0] * (n * n)
    for i in range(n):
        matrix[i * n + i] = 1.0
    return matrix


def random_matrix(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n * n)]


def reference_product(a: list[float], b: list[float], n: int) -> list[float]:
    """Correctly rounded C = A x B, computed independently with math.fsum()."""
    c = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            c[i * n + j] = math.fsum(a[i * n + k] * b[k * n + j] for k in range(n))
    return c


def read_input_file(path: Path) -> tuple[int, list[float], list[float]]:
    """Read n, A and B from a shared input file."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little", signed=True)
        a = array("d")
        b = array("d")
        a.fromfile(f, n * n)
        b.fromfile(f, n * n)
    if sys.byteorder != "little":
        a.byteswap()
        b.byteswap()
    return n, a.tolist(), b.tolist()


def read_result_file(path: Path) -> tuple[int, list[float]]:
    """Read n and C from a result file written by the program's --output."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little", signed=True)
        c = array("d")
        c.fromfile(f, n * n)
    if sys.byteorder != "little":
        c.byteswap()
    return n, c.tolist()


# ---------------------------------------------------------------------------
# Step 2: the vectorisation report
# ---------------------------------------------------------------------------

# A report line looks like:
#   src/c/parallel_mult.c:465:34: optimized: loop vectorized using 32 byte vectors
# Only "loop vectorized" lines matter here. The report may also contain
# "basic block part vectorized" lines at the '#pragma omp parallel' line of
# the omp_* kernels: that is GCC packing the shared variables (a, b, c, n,
# block_size) into the small struct handed to the threads, done once per
# call and outside every loop, so it says nothing about the kernels' loops.
LOOP_VECTORIZED = re.compile(r":(\d+):\d+: optimized: loop vectorized using (\d+) byte vectors")


def kernel_line_ranges() -> dict[str, tuple[int, int]]:
    """
    First and last source line of each kernel function multiply_<algo>().
    The C file's style makes this easy: a definition starts with "static"
    and contains "multiply_<algo>(", and the function ends at the next line
    that is exactly "}" (a closing brace in column 0).
    """
    lines = config.SOURCE_FILE.read_text().splitlines()
    ranges = {}
    for algorithm in config.ALGORITHMS:
        function_name = f"multiply_{algorithm}("
        start = None
        end = None
        for number, line in enumerate(lines, start=1):
            if start is None:
                if line.startswith("static") and function_name in line:
                    start = number
            elif line == "}":
                end = number
                break
        if start is None or end is None:
            raise RuntimeError(f"Kernel {function_name}...) not found in {config.SOURCE_FILE}")
        ranges[algorithm] = (start, end)
    return ranges


def check_vectorization_report() -> bool:
    """Step 2. Print one line per kernel; return True if all expectations hold."""
    report = config.VECTORIZATION_REPORT.read_text()
    ranges = kernel_line_ranges()

    # algorithm -> list of (line, vector width in bytes) of vectorised loops
    vectorized = {algorithm: [] for algorithm in config.ALGORITHMS}
    for match in LOOP_VECTORIZED.finditer(report):
        line = int(match.group(1))
        width_bytes = int(match.group(2))
        for algorithm, (start, end) in ranges.items():
            if start <= line <= end:
                vectorized[algorithm].append((line, width_bytes))

    report_name = config.VECTORIZATION_REPORT.relative_to(config.ASSIGNMENT_DIR)
    print(f"Vectorisation report ({report_name}):")
    all_passed = True
    for algorithm in config.ALGORITHMS:
        loops = vectorized[algorithm]
        should_be_vectorized = (algorithm == "omp_simd_tiled")
        passed = (len(loops) > 0) == should_be_vectorized
        all_passed = all_passed and passed

        # GCC may list the same loop twice: the main vector loop and a
        # narrower vectorised "epilogue" for the last few iterations.
        widths_by_line = {}
        for line, width_bytes in loops:
            widths_by_line.setdefault(line, []).append(width_bytes)

        if widths_by_line:
            parts = []
            for line, widths in widths_by_line.items():
                text = f"line {line}: {max(widths)}-byte vectors = {max(widths) // 8} doubles"
                if len(widths) > 1:
                    text += f" (epilogue: {min(widths)}-byte)"
                parts.append(text)
            details = "vectorised  " + "; ".join(parts)
        else:
            details = "scalar      no vectorised loop"
        expected = "vector" if should_be_vectorized else "scalar"
        status = "PASS" if passed else f"FAIL (expected {expected})"
        print(f"  {algorithm:<16}{status:<6}{details}")
    print()
    return all_passed


# ---------------------------------------------------------------------------
# Step 3: test cases
# ---------------------------------------------------------------------------

def build_test_cases(size: int) -> list[tuple[str, Path, list[float], int]]:
    """
    Write every test input to VERIFY_DIR.
    Returns a list of (name, input_file, expected_result, n).
    """
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    cases = []

    # Identity: A x I = A. Multiplying by 1.0 and adding 0.0 is exact in
    # IEEE 754, so the expected result is A itself, bit for bit.
    a = random_matrix(size, seed=1)
    path = VERIFY_DIR / "identity.bin"
    data_generator.write_matrices(path, size, array("d", a), array("d", identity_matrix(size)))
    cases.append(("identity", path, a, size))

    # Zero: A x 0 = 0 (exactly).
    a = random_matrix(size, seed=2)
    path = VERIFY_DIR / "zero.bin"
    data_generator.write_matrices(path, size, array("d", a), array("d", [0.0] * (size * size)))
    cases.append(("zero", path, [0.0] * (size * size), size))

    # Known 3x3 product, computed by hand:
    #   [1 2 3]   [9 8 7]   [ 30  24  18]
    #   [4 5 6] x [6 5 4] = [ 84  69  54]
    #   [7 8 9]   [3 2 1]   [138 114  90]
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    expected = [30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0]
    path = VERIFY_DIR / "known_3x3.bin"
    data_generator.write_matrices(path, 3, array("d", a), array("d", b))
    cases.append(("known_3x3", path, expected, 3))

    # Random inputs from the shared generator (the one the benchmark uses).
    for n in [size] + TILE_TEST_SIZES:
        path = data_generator.generate_for_size(n, config.DATA_SEED, VERIFY_DIR)
        file_n, a, b = read_input_file(path)
        name = path.stem.replace("matrices_", "random_")     # e.g. random_n131
        cases.append((name, path, reference_product(a, b, file_n), file_n))

    return cases


# ---------------------------------------------------------------------------
# Steps 4-7: running and checking
# ---------------------------------------------------------------------------

def compare(result: list[float], expected: list[float], n: int) -> tuple[bool, float]:
    """Element-wise tolerance check. Returns (passed, largest absolute error)."""
    abs_tol = absolute_tolerance(n)
    max_error = 0.0
    passed = True
    for value, reference in zip(result, expected):
        error = abs(value - reference)
        max_error = max(max_error, error)
        if not error <= abs_tol + REL_TOL * abs(reference):   # also catches NaN
            passed = False
    return passed, max_error


def check_one_run(algorithm: str, input_file: Path, expected: list[float], n: int,
                  threads: int, block_size: int, output_file: Path) -> tuple[bool, float, str]:
    """
    Run one (algorithm, threads, block_size) on one case and check it.
    Returns (passed, max_error, problem); problem is "" when passed.
    """
    command = config.run_command(algorithm, input_file, threads, output_file, block_size)
    completed = subprocess.run(command, capture_output=True, text=True,
                               env=config.run_environment())
    if completed.returncode != 0:
        return False, float("nan"), f"exit code {completed.returncode}: {completed.stderr.strip()}"

    # The labels the benchmark will record must be the ones we asked for.
    fields = config.parse_program_output(completed.stdout)
    for key, wanted in [("algo", algorithm), ("n", n), ("threads", threads),
                        ("block_size", block_size)]:
        if fields.get(key) != str(wanted):
            return False, float("nan"), f"reported {key}={fields.get(key)} (expected {wanted})"

    result_n, result = read_result_file(output_file)
    if result_n != n:
        return False, float("nan"), f"result has n={result_n}"

    passed, max_error = compare(result, expected, n)
    return passed, max_error, "" if passed else "outside tolerance"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the three Assignment 3 kernels.")
    parser.add_argument("--size", type=int, default=10,
                        help="n for the identity/zero/small random cases")
    args = parser.parse_args()

    config.build_all()
    all_passed = check_vectorization_report()

    cases = build_test_cases(args.size)
    threads_text = ",".join(str(p) for p in VERIFY_THREAD_COUNTS)

    print(f"Tolerances: REL_TOL={REL_TOL:.0e}, ABS_TOL=100*n^2*u "
          f"(n={args.size}: {absolute_tolerance(args.size):.2e}, "
          f"n={max(TILE_TEST_SIZES)}: {absolute_tolerance(max(TILE_TEST_SIZES)):.2e})")
    print(f"Block sizes: {VERIFY_BLOCK_SIZES}   threads for omp_*: {threads_text}\n")
    print(f"{'case':<14}{'bs':>4}  {'algorithm':<16}{'threads':<16}{'result':<8}"
          f"{'max |error|':>12}  {'same bits as seq_tiled':<24}notes")

    for name, input_file, expected, n in cases:
        reference_bytes = None    # seq_tiled's result at the first block size

        for block_size in VERIFY_BLOCK_SIZES:
            for algorithm in config.ALGORITHMS:
                thread_counts = config.thread_counts_for(algorithm, VERIFY_THREAD_COUNTS)
                is_reference_line = reference_bytes is None

                # One printed line per (case, block size, algorithm),
                # summarising all of its thread counts.
                line_passed = True
                line_max_error = 0.0
                identical = 0
                problems = []

                for threads in thread_counts:
                    output_file = VERIFY_DIR / f"result_{name}_{algorithm}_bs{block_size}_p{threads}.bin"
                    passed, max_error, problem = check_one_run(
                        algorithm, input_file, expected, n, threads, block_size, output_file)

                    line_passed = line_passed and passed
                    line_max_error = max(line_max_error, max_error)   # broken runs: see notes
                    if not passed:
                        problems.append(f"p={threads}: {problem}")
                        continue

                    # Informative only: bit-for-bit equal to seq_tiled?
                    result_bytes = output_file.read_bytes()
                    if reference_bytes is None:
                        reference_bytes = result_bytes
                    if result_bytes == reference_bytes:
                        identical += 1

                all_passed = all_passed and line_passed
                status = "PASS" if line_passed else "FAIL"
                threads_label = ",".join(str(p) for p in thread_counts)
                same_bits = "(reference)" if is_reference_line else f"{identical}/{len(thread_counts)}"
                print(f"{name:<14}{block_size:>4}  {algorithm:<16}{threads_label:<16}{status:<8}"
                      f"{line_max_error:>12.3e}  {same_bits:<24}{'; '.join(problems)}")

    print("\nAll checks passed." if all_passed else "\nSOME CHECKS FAILED.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
