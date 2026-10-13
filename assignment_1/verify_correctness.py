"""
verify_correctness.py - Correctness validation for Assignment 1.

What it does (guidelines, section 2 "Validation"):
    1. Builds the C and Java programs.
    2. Creates four test cases:
         - identity : C = A x I must equal A
         - zero     : C = A x 0 must be all zeros
         - known    : a 3x3 product whose result was computed by hand
         - random   : the shared n=10 input from common/data_generator.py
    3. Runs the C, Java and Python programs on every case. Each program
       saves its result matrix with --output.
    4. Compares every result element-wise against a REFERENCE using
       absolute + relative tolerances, and reports the largest error.

Why a separate reference instead of comparing the languages to each other:
    If all three programs had the same bug (e.g. a wrong index), they would
    agree with each other and the bug would go unnoticed. The reference is
    computed independently with math.fsum(), which returns the correctly
    rounded sum, i.e. the most accurate double result possible.

Run (from anywhere):
    python3 assignment_1/verify_correctness.py [--size 10]
Exit code 0 = all checks passed, 1 = at least one failure.
"""

import argparse
import math
import random
import subprocess
import sys
from array import array
from pathlib import Path

import benchmark_config as config

# Reuse the generator's exact file writer so test inputs use the same format.
sys.path.insert(0, str(config.DATA_GENERATOR.parent))
import data_generator  # noqa: E402

VERIFY_DIR = config.BUILD_DIR / "verify"

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
# An element passes if   |result - reference| <= ABS_TOL + REL_TOL * |reference|
# (the same rule as Python's math.isclose / NumPy's isclose).
#
# Why tolerances at all?
#   Floating-point addition is not associative. Our three programs use the
#   same loop order, so they normally produce bit-identical results, but a
#   compiler is allowed to use fused multiply-add (FMA) instructions or a
#   different summation order, which changes the last few bits. Those
#   differences are rounding, not bugs, so an exact == test would be wrong.
#
# How the values were chosen:
#   Unit roundoff of a 64-bit double: u = 2^-53 ~ 1.11e-16.
#   Standard error bound for a dot product of length n (Higham, "Accuracy and
#   Stability of Numerical Algorithms", ch. 3):
#       |error| <= n * u * sum_k |a_ik * b_kj|
#   Our inputs lie in [-1, 1), so sum_k |a_ik * b_kj| <= n, giving
#       |error| <= n^2 * u      (n=10 -> ~1.1e-14)
#
#   ABS_TOL = 100 * n^2 * u
#       A 100x safety margin over the worst-case bound. The absolute term is
#       essential for elements close to zero (e.g. the zero-matrix case, or
#       random sums that cancel), where any relative test would fail.
#       It grows with n because the error bound grows with n.
#   REL_TOL = 1e-12
#       About 4500 units in the last place: loose enough to accept FMA or
#       reordering effects, yet a real bug (wrong index, missing term) causes
#       errors of order 0.1 for our data - eleven orders of magnitude larger.
UNIT_ROUNDOFF = 2.0 ** -53
SAFETY_FACTOR = 100
REL_TOL = 1e-12


def absolute_tolerance(n: int) -> float:
    """Absolute tolerance derived from the dot-product error bound (see above)."""
    return SAFETY_FACTOR * n * n * UNIT_ROUNDOFF


# ---------------------------------------------------------------------------
# Matrix helpers (flat row-major lists, the same layout as the programs)
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
    """Read n and C from a result file written by a program's --output."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little", signed=True)
        c = array("d")
        c.fromfile(f, n * n)
    if sys.byteorder != "little":
        c.byteswap()
    return n, c.tolist()


# ---------------------------------------------------------------------------
# Test cases
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
    # Small integers are exact in a double, so the result is exact too.
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    expected = [30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0]
    path = VERIFY_DIR / "known_3x3.bin"
    data_generator.write_matrices(path, 3, array("d", a), array("d", b))
    cases.append(("known_3x3", path, expected, 3))

    # Random: the same shared input file that the benchmark uses.
    path = data_generator.generate_for_size(size, config.DATA_SEED, VERIFY_DIR)
    n, a, b = read_input_file(path)
    cases.append((f"random_n{size}", path, reference_product(a, b, n), n))

    return cases


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(result: list[float], expected: list[float], n: int) -> tuple[bool, float, int]:
    """
    Element-wise tolerance check.
    Returns (passed, largest absolute error, number of failing elements).
    """
    abs_tol = absolute_tolerance(n)
    max_error = 0.0
    failures = 0
    for value, reference in zip(result, expected):
        error = abs(value - reference)
        max_error = max(max_error, error)
        if error > abs_tol + REL_TOL * abs(reference):
            failures += 1
    return failures == 0, max_error, failures


def run_program(language: str, input_file: Path, output_file: Path) -> None:
    """Run one language's program so it saves its result to output_file."""
    command = config.run_command(language, input_file, output_file)
    subprocess.run(command, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate C, Java and Python results.")
    parser.add_argument("--size", type=int, default=10, help="n for the identity/zero/random cases")
    args = parser.parse_args()

    config.build_all()
    cases = build_test_cases(args.size)

    print(f"Tolerances: REL_TOL={REL_TOL:.0e}, ABS_TOL=100*n^2*u "
          f"(n={args.size}: {absolute_tolerance(args.size):.2e})\n")
    print(f"{'case':<14}{'language':<10}{'result':<8}{'max |error|':>14}")

    all_passed = True
    for name, input_file, expected, n in cases:
        for language in config.LANGUAGES:
            output_file = VERIFY_DIR / f"result_{name}_{language}.bin"
            run_program(language, input_file, output_file)
            result_n, result = read_result_file(output_file)

            if result_n != n:
                passed, max_error = False, float("nan")
            else:
                passed, max_error, _ = compare(result, expected, n)

            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"{name:<14}{language:<10}{status:<8}{max_error:>14.3e}")

    print("\nAll checks passed." if all_passed else "\nSOME CHECKS FAILED.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
