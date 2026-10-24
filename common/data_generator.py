"""
Shared input generator for the matrix multiplication assignments.

Why this script exists:
    Every implementation (Python, Java and C in Assignment 1; the dense and
    sparse kernels in Assignment 2) must multiply EXACTLY the same matrices.
    Instead of reproducing one random generator in several languages
    (which is error-prone), we generate the data once here and save it to a
    simple binary file that every language can read.

File format (one file per matrix size and sparsity level):
    Offset 0          : int64, little-endian -> n (matrix dimension)
    Offset 8          : n*n float64, little-endian, row-major -> matrix A
    Offset 8 + 8*n*n  : n*n float64, little-endian, row-major -> matrix B

    The format is the SAME for dense and sparse inputs: zeros are simply
    stored as 0.0. The dense kernels (Assignment 2, Part A) read the file
    exactly as in Assignment 1, and the sparse kernels (Part B) read it and
    convert it to CSR themselves, so the conversion cost can be timed
    separately from the multiplication.

File names:
    matrices_n500.bin        -> sparsity 0.0 (identical to Assignment 1)
    matrices_n500_sp90.bin   -> 90 % of the entries of A and of B are 0.0
    matrices_n500_sp99.5.bin -> 99.5 % zeros

Sparsity (added for Assignment 2):
    --sparsity s puts EXACTLY round(s * n * n) zeros in each matrix, at
    random positions. The non-zero values stay in the range [-1, 1).

    How a sparse matrix is built, and why:
      1. The dense values are drawn with the same seed and in the same order
         as in Assignment 1, so --sparsity 0 reproduces those files
         bit-for-bit.
      2. A SEPARATE random generator then decides which positions become
         0.0: it puts all positions in a random order and zeroes the first k.
    Consequences for the experiments:
      - A sparse matrix is "the dense matrix with some entries removed":
        every non-zero that survives keeps its original value.
      - The sparsity levels are nested: the zeros of the 90 % matrix are a
        subset of the zeros of the 95 % matrix, which are a subset of the
        zeros of the 99 % matrix.
    So when we compare sparsity levels, the only thing that changes is the
    number of zeros, not the values or the luck of the random draw.

Seeds (to record in the report):
    values of A and B     : random.Random(f"{seed}-{n}")          (A first, then B)
    zero positions in A   : random.Random(f"{seed}-{n}-zeros-A")
    zero positions in B   : random.Random(f"{seed}-{n}-zeros-B")
    Every seed depends only on (seed, n), never on the other sizes or
    sparsity levels generated in the same run.

Why binary instead of CSV:
    - Exact: a float64 is stored bit-for-bit, so there are no rounding
      differences from text parsing between languages.
    - Fast: reading is a single block copy, so loading time stays small
      (it is outside the timed region anyway, but it keeps runs short).
    - Simple: C reads it with fread(), Java with a little-endian ByteBuffer,
      Python with the built-in `array` module.

Note: this script does NOT use NumPy. Only the standard library is needed,
so the generator runs anywhere Python 3 is installed.

Examples:
    python common/data_generator.py --sizes 100 500 1000
    python common/data_generator.py --sizes 500 1000 --sparsity 0.5 0.9 0.95 0.99
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
DEFAULT_SPARSITIES = [0.0]
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "data"


# ---------------------------------------------------------------------------
# Dense values
# ---------------------------------------------------------------------------

def generate_matrix(n: int, rng: random.Random) -> array:
    """Return an n x n matrix as a flat, row-major array of float64 ('d')."""
    matrix = array("d")
    for _ in range(n * n):
        matrix.append(rng.uniform(VALUE_MIN, VALUE_MAX))
    return matrix


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------

def zeros_per_matrix(n: int, sparsity: float) -> int:
    """Number of entries set to 0.0 in each n x n matrix for this sparsity."""
    return round(sparsity * n * n)


def choose_zero_positions(count: int, num_zeros: int, rng: random.Random) -> array:
    """
    Return `num_zeros` distinct positions out of range(count), in random order.

    This is a partial Fisher-Yates shuffle: step i swaps a randomly chosen,
    not-yet-used position into slot i. After num_zeros steps, the first
    num_zeros slots are a uniform random sample without repetition.

    Why not random.sample()? Its internal algorithm (and therefore its
    result for a given seed) changes with the number of items requested.
    Here, step i only depends on the steps before it, so asking for more
    zeros simply continues the same sequence. That is what makes the
    sparsity levels nested (see the module docstring).
    """
    # 'q' = int64. An array of int64 uses ~4x less memory than a list of
    # Python ints, which matters for large n (n=2000 -> 4 million positions).
    positions = array("q", range(count))
    for i in range(num_zeros):
        j = rng.randrange(i, count)          # random slot in the unused part [i, count)
        positions[i], positions[j] = positions[j], positions[i]
    return positions[:num_zeros]


def inject_zeros(matrix: array, num_zeros: int, rng: random.Random) -> None:
    """Set exactly `num_zeros` randomly chosen entries of `matrix` to 0.0 (in place)."""
    if num_zeros == 0:
        return   # dense case: nothing to do, and no need to build the position list
    for position in choose_zero_positions(len(matrix), num_zeros, rng):
        matrix[position] = 0.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def matrix_filename(n: int, sparsity: float) -> str:
    """
    File name for one (n, sparsity) input.

    Dense inputs keep the Assignment 1 name, so the existing Assignment 1
    scripts (and the Part A dense runs) find them without any change.
    """
    if sparsity == 0.0:
        return f"matrices_n{n}.bin"
    percent = f"{sparsity * 100:g}"          # 0.9 -> "90", 0.995 -> "99.5"
    return f"matrices_n{n}_sp{percent}.bin"


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


# ---------------------------------------------------------------------------
# One input file
# ---------------------------------------------------------------------------

def generate_for_size(n: int, seed: int, output_dir: Path, sparsity: float = 0.0) -> Path:
    """
    Generate A and B for one (n, sparsity) pair and save them.
    Returns the file path. With the default sparsity=0.0 this behaves
    exactly like the Assignment 1 version of this function.
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {sparsity}")

    # 1. Dense values: same seed string and draw order as Assignment 1.
    #    The seed depends on n only, so the matrices for n=500 are the same
    #    whether or not other sizes are generated in the same run.
    value_rng = random.Random(f"{seed}-{n}")
    a = generate_matrix(n, value_rng)
    b = generate_matrix(n, value_rng)

    # 2. Zero positions: one SEPARATE generator per matrix.
    #    - Separate from value_rng, so adding zeros never changes the values.
    #    - Separate for A and B, so the zeros of B do not depend on how many
    #      random draws A needed (otherwise B's levels would not be nested).
    num_zeros = zeros_per_matrix(n, sparsity)
    inject_zeros(a, num_zeros, random.Random(f"{seed}-{n}-zeros-A"))
    inject_zeros(b, num_zeros, random.Random(f"{seed}-{n}-zeros-B"))

    path = output_dir / matrix_filename(n, sparsity)
    write_matrices(path, n, a, b)
    return path


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate shared input matrices for the matmul benchmarks."
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help=f"Matrix sizes n to generate (default: {DEFAULT_SIZES})",
    )
    parser.add_argument(
        "--sparsity", type=float, nargs="+", default=DEFAULT_SPARSITIES,
        help="Fraction of exact 0.0 entries in each matrix, between 0.0 and 1.0. "
             "Several values may be given, e.g. --sparsity 0.5 0.9 0.95 0.99 "
             f"(default: {DEFAULT_SPARSITIES})",
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

    # Validate everything first, so a typo does not stop the run halfway.
    for n in args.sizes:
        if n <= 0:
            raise ValueError(f"Matrix size must be positive, got {n}")
    for sparsity in args.sparsity:
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {sparsity}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"seed={args.seed}")

    for n in args.sizes:
        for sparsity in args.sparsity:
            path = generate_for_size(n, args.seed, args.output_dir, sparsity)
            num_zeros = zeros_per_matrix(n, sparsity)
            achieved = num_zeros / (n * n)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"n={n:>5}  sparsity={sparsity:<6g} zeros/matrix={num_zeros:>9} "
                  f"(achieved {achieved:.4%})  ->  {path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
