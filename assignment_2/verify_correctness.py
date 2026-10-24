"""
verify_correctness.py - Correctness validation for Assignment 2.

What it does:
    1. Builds both C programs (dense_mult.c and sparse_mult.c).
    2. Creates dense and sparse test cases:
         identity        C = A x I must equal A          (I is itself sparse)
         zero            C = A x 0 must be all zeros     (B has no non-zeros)
         known_3x3       a product computed by hand      (exact integers)
         random_n<size>  shared generator input, dense
         random_n100, random_n100_sp50 / _sp90 / _sp99
                         n = 100 is larger than the tile size (64), so the
                         tiled kernel runs full AND partial edge tiles; the
                         99 % case has many empty rows, an edge case for CSR.
    3. Runs all four algorithms (baseline, reordered, tiled, csr) on every
       case. Each program saves its result matrix with --output (sparse_mult
       writes the same dense output format as dense_mult).
    4. Compares every result element-wise against an independent REFERENCE
       computed with math.fsum, using absolute + relative tolerances.
    5. For csr, also checks the counts it reports: nnz_A and nnz_B must match
       the input, and nnz_C must match the non-zeros of its own result matrix
       (this validates the fill-in tracking).
    6. Reports, for information, whether each result is bit-identical to the
       baseline's. With our flags they all are (same summation order), but
       this is not required: tolerances decide PASS/FAIL.

Why a separate reference instead of comparing the algorithms to each other:
    If all algorithms shared the same bug, they would agree with each other
    and the bug would go unnoticed. math.fsum returns the correctly rounded
    sum, i.e. the most accurate double result possible.

Run (from anywhere):
    python3 assignment_2/verify_correctness.py [--size 10]
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

# The generator module (imported by benchmark_config) writes the test inputs
# in exactly the format the programs read.
data_generator = config.data_generator

VERIFY_DIR = config.BUILD_DIR / "verify"

# n and sparsity levels of the multi-tile / sparse test cases.
TILE_TEST_SIZE = 100
TILE_TEST_SPARSITIES = [0.0, 0.5, 0.9, 0.99]

# ---------------------------------------------------------------------------
# Tolerances (same rule and values as Assignment 1)
# ---------------------------------------------------------------------------
# An element passes if   |result - reference| <= ABS_TOL + REL_TOL * |reference|
#
# Why tolerances at all: floating-point addition is not associative. Our
# kernels add the terms in the same order and normally agree bit for bit,
# but a compiler may legally use FMA instructions or reorder sums, which
# changes the last bits. That is rounding, not a bug.
#
# Values: unit roundoff u = 2^-53. The error of a length-n dot product is at
# most n * u * sum_k |a_ik * b_kj| <= n^2 * u for inputs in [-1, 1)
# (Higham, "Accuracy and Stability of Numerical Algorithms", ch. 3).
#   ABS_TOL = 100 * n^2 * u  -> 100x margin; essential for values near zero.
#   REL_TOL = 1e-12          -> accepts rounding differences, while a real
#                               bug (wrong index, missing term) causes errors
#                               around 0.1 for our data.
# Sparse inputs only remove terms from each sum, so the same bound holds.
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


def count_nonzeros(matrix: list[float]) -> int:
    return sum(1 for value in matrix if value != 0.0)


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

    # Zero: A x 0 = 0 (exactly). For csr, B has nnz = 0, so C must too.
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

    # Random inputs from the shared generator: small dense, then the
    # multi-tile size, dense and sparse (the same generator the benchmark uses).
    random_inputs = [(size, 0.0)] + [(TILE_TEST_SIZE, s) for s in TILE_TEST_SPARSITIES]
    for n, sparsity in random_inputs:
        path = data_generator.generate_for_size(n, config.DATA_SEED, VERIFY_DIR, sparsity)
        file_n, a, b = read_input_file(path)
        name = path.stem.replace("matrices_", "random_")     # e.g. random_n100_sp90
        cases.append((name, path, reference_product(a, b, file_n), file_n))

    return cases


# ---------------------------------------------------------------------------
# Checks
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


def check_csr_counts(fields: dict[str, str], input_file: Path,
                     result: list[float]) -> list[str]:
    """Compare the nnz values printed by sparse_mult with the actual counts."""
    _, a, b = read_input_file(input_file)
    expected = {"nnz_A": count_nonzeros(a),
                "nnz_B": count_nonzeros(b),
                "nnz_C": count_nonzeros(result)}
    problems = []
    for key, value in expected.items():
        if int(fields[key]) != value:
            problems.append(f"{key}={fields[key]} (expected {value})")
    return problems


def run_program(algorithm: str, input_file: Path, output_file: Path) -> dict[str, str]:
    """Run one algorithm so it saves its result; return its parsed output line."""
    command = config.run_command(algorithm, input_file, output_file)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return config.parse_program_output(completed.stdout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the four Assignment 2 kernels.")
    parser.add_argument("--size", type=int, default=10,
                        help="n for the identity/zero/small random cases")
    args = parser.parse_args()

    config.build_all()
    cases = build_test_cases(args.size)

    print(f"Tolerances: REL_TOL={REL_TOL:.0e}, ABS_TOL=100*n^2*u "
          f"(n={args.size}: {absolute_tolerance(args.size):.2e}, "
          f"n={TILE_TEST_SIZE}: {absolute_tolerance(TILE_TEST_SIZE):.2e})")
    print(f"Tiled block size: {config.TILED_BLOCK_SIZE}\n")
    print(f"{'case':<20}{'algorithm':<11}{'result':<8}{'max |error|':>12}  "
          f"{'same bits as baseline':<23}notes")

    all_passed = True
    for name, input_file, expected, n in cases:
        baseline_bytes = None

        for algorithm in config.ALGORITHMS:
            output_file = VERIFY_DIR / f"result_{name}_{algorithm}.bin"
            fields = run_program(algorithm, input_file, output_file)
            result_n, result = read_result_file(output_file)

            notes = ""
            if result_n != n:
                passed, max_error = False, float("nan")
                notes = f"result has n={result_n}"
            else:
                passed, max_error = compare(result, expected, n)

            if algorithm == "csr" and result_n == n:
                problems = check_csr_counts(fields, input_file, result)
                if problems:
                    passed = False
                    notes = "wrong counts: " + ", ".join(problems)
                else:
                    notes = f"nnz_A={fields['nnz_A']} nnz_B={fields['nnz_B']} nnz_C={fields['nnz_C']}"

            # Informative only: is the result bit-for-bit the baseline's?
            result_bytes = output_file.read_bytes()
            if algorithm == "baseline":
                baseline_bytes = result_bytes
                same_bits = "(reference)"
            else:
                same_bits = "yes" if result_bytes == baseline_bytes else "no"

            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"{name:<20}{algorithm:<11}{status:<8}{max_error:>12.3e}  "
                  f"{same_bits:<23}{notes}")

    print("\nAll checks passed." if all_passed else "\nSOME CHECKS FAILED.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
