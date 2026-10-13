"""
Shared input generator for Assignment 1 (dense matrix multiplication benchmark).

Why this script exists:
    The guidelines require that Python, Java and C multiply EXACTLY the same
    matrices. Instead of reproducing one random generator in three languages
    (which is error-prone), we generate the data once here and save it to a
    simple binary file that every language can read.

File format (one file per matrix size, e.g. data/matrices_n100.bin):
    Offset 0          : int64, little-endian -> n (matrix dimension)
    Offset 8          : n*n float64, little-endian, row-major -> matrix A
    Offset 8 + 8*n*n  : n*n float64, little-endian, row-major -> matrix B

Why binary instead of CSV:
    - Exact: a float64 is stored bit-for-bit, so there are no rounding
      differences from text parsing between languages.
    - Fast: reading is a single block copy, so loading time stays small
      (it is outside the timed region anyway, but it keeps runs short).
    - Simple: C reads it with fread(), Java with a little-endian ByteBuffer,
      Python with the built-in `array` module.

Note: this script does NOT use NumPy. Only the standard library is needed,
so the generator runs anywhere Python 3 is installed.
"""

import argparse
import random
import struct
import sys
from array import array
from pathlib import Path

# Values are drawn from [-1.0, 1.0). A small symmetric range keeps the sums
# in C = A x B well-scaled, which makes floating-point comparisons between
# languages easier to reason about.
VALUE_MIN = -1.0
VALUE_MAX = 1.0

DEFAULT_SIZES = [10, 100, 500, 1000]
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "data"


def generate_matrix(n: int, rng: random.Random) -> array:
    """Return an n x n matrix as a flat, row-major array of float64 ('d')."""
    matrix = array("d")
    for _ in range(n * n):
        matrix.append(rng.uniform(VALUE_MIN, VALUE_MAX))
    return matrix


def write_matrices(path: Path, n: int, a: array, b: array) -> None:
    """Write the header (n) followed by A and B in little-endian format."""
    # array.tofile() writes in the machine's native byte order.
    # We force little-endian so the file is identical on every machine.
    if sys.byteorder != "little":
        a = array("d", a)
        b = array("d", b)
        a.byteswap()
        b.byteswap()

    with open(path, "wb") as f:
        f.write(struct.pack("<q", n))  # '<q' = little-endian signed int64
        a.tofile(f)
        b.tofile(f)


def generate_for_size(n: int, seed: int, output_dir: Path) -> Path:
    """Generate A and B for one size and save them. Returns the file path."""
    # Each size gets its own generator seeded with (seed, n).
    # This means the matrices for n=500 are the same whether or not
    # you also generate n=100 in the same run -> fully reproducible.
    rng = random.Random(f"{seed}-{n}")

    a = generate_matrix(n, rng)
    b = generate_matrix(n, rng)

    path = output_dir / f"matrices_n{n}.bin"
    write_matrices(path, n, a, b)
    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate shared input matrices for the matmul benchmark."
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help=f"Matrix sizes n to generate (default: {DEFAULT_SIZES})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Folder for the .bin files (default: {DEFAULT_OUTPUT_DIR}/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for n in args.sizes:
        if n <= 0:
            raise ValueError(f"Matrix size must be positive, got {n}")
        path = generate_for_size(n, args.seed, args.output_dir)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"n={n:>5}  ->  {path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
