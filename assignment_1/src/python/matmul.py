"""
matmul.py - Basic O(n^3) dense matrix multiplication (Assignment 1, Python version)

Responsibilities are split into small functions (SOLID-lite):
    1. load_matrices()  -> read A and B from the shared binary file
    2. multiply()       -> the O(n^3) triple-loop kernel (the ONLY timed part)
    3. checksum()       -> a simple sum of C, used to cross-check languages
    4. save_matrix()    -> optionally write C to disk for verify_correctness.py
    5. main()           -> orchestration + the timer boundary

Matrix file format (shared by all languages, see common/data_generator.py):
    Input : int64 n | n*n float64 (A, row-major) | n*n float64 (B, row-major)
    Output: int64 n | n*n float64 (C, row-major)
    All values little-endian.

Memory layout:
    Each matrix is ONE flat Python list of n*n floats in row-major order,
    element (i, j) at index i*n + j - exactly the same indexing as C and Java.
    A Python float is a 64-bit IEEE 754 double, so the numeric type matches.
    Unavoidable difference (document in the report): a list stores pointers
    to boxed float objects, not raw doubles, so memory use and access cost
    are higher than in C/Java.

Run:
    python3 assignment_1/src/python/matmul.py data/matrices_n100.bin [--output result_py.bin]

Output (one line on stdout, parsed by run_benchmark.py):
    n=<n> time_s=<seconds> checksum=<sum of C>
"""

import argparse
import struct
import sys
import time
from array import array


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matrices(path: str) -> tuple[int, list[float], list[float]]:
    """Read n, A and B from the shared binary file. Returns (n, a, b)."""
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid header in '{path}'")
        (n,) = struct.unpack("<q", header)   # '<q' = little-endian int64
        if n <= 0:
            raise ValueError(f"Invalid matrix size n={n} in '{path}'")

        count = n * n
        a = array("d")
        b = array("d")
        # fromfile() raises EOFError if the file is truncated.
        a.fromfile(f, count)
        b.fromfile(f, count)

    # array.fromfile() reads in the machine's native byte order.
    # The file is little-endian, so swap only on a big-endian machine.
    if sys.byteorder != "little":
        a.byteswap()
        b.byteswap()

    # Convert to plain lists: indexing a list is faster than indexing an
    # array('d') in CPython (no float object is created on every read).
    return n, a.tolist(), b.tolist()


# ---------------------------------------------------------------------------
# Multiplication kernel
# ---------------------------------------------------------------------------

def multiply(a: list[float], b: list[float], c: list[float], n: int) -> None:
    """
    C = A x B using the textbook i-j-k triple loop.

    Same loop order and same running-sum accumulator as the C and Java
    versions, so all three languages perform identical arithmetic.
    No NumPy or other optimized library is used, as the guidelines require.
    """
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i * n + k] * b[k * n + j]
            c[i * n + j] = total


# ---------------------------------------------------------------------------
# Result check
# ---------------------------------------------------------------------------

def checksum(c: list[float]) -> float:
    """
    Sum of all elements of C, used to compare results across languages.

    We use an explicit loop instead of the built-in sum(): since Python 3.12,
    sum() uses compensated (Neumaier) summation for floats, which would give
    a slightly different result from the plain left-to-right sum in C/Java.
    """
    total = 0.0
    for value in c:
        total += value
    return total


def save_matrix(path: str, c: list[float], n: int) -> None:
    """Write n and C to disk (same little-endian format as the input)."""
    values = array("d", c)
    if sys.byteorder != "little":
        values.byteswap()
    with open(path, "wb") as f:
        f.write(struct.pack("<q", n))
        values.tofile(f)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="O(n^3) matrix multiplication (Python).")
    parser.add_argument("input", help="Shared input file (data/matrices_n<N>.bin)")
    parser.add_argument("--output", help="Optional path to save the result matrix C")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # 1. Load inputs - NOT timed (guidelines: inputs prepared outside).
    n, a, b = load_matrices(args.input)

    # 2. Allocate the output - NOT timed, so we measure only arithmetic.
    c = [0.0] * (n * n)

    # ============ TIMED REGION START ============
    # perf_counter_ns() is monotonic (never jumps backwards when the system
    # clock is adjusted) and has the highest available resolution.
    # Integer nanoseconds avoid float rounding in the timer itself.
    start_ns = time.perf_counter_ns()

    multiply(a, b, c, n)

    end_ns = time.perf_counter_ns()
    # ============ TIMED REGION END ==============

    # 3. Report and (optionally) save - NOT timed.
    elapsed_s = (end_ns - start_ns) / 1e9
    print(f"n={n} time_s={elapsed_s:.6f} checksum={checksum(c):.10e}")

    if args.output:
        save_matrix(args.output, c, n)


if __name__ == "__main__":
    main()
