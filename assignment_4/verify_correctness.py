"""
verify_correctness.py - Correctness validation for Assignment 4.

What it does:
    1. Builds the C programs of Assignments 2 and 3 if needed.
    2. Creates the test cases (as in Assignment 3):
         identity        C = A x I must equal A (exactly)
         zero            C = A x 0 must be all zeros
         known_3x3       a product computed by hand (exact integers)
         random_n<size>  shared generator input, small
         random_n100     several tiles per side
         random_n131     odd prime size: uneven tiles for most block sizes
       The expected result is an independent REFERENCE computed with
       math.fsum (the correctly rounded value of every element).
    3. Runs the C programs on every case through the wrapper modes of
       distributed_matmul.py (the same command path as the benchmark):
         c_baseline                 the bit-level reference for step 6
         c_omp_tiled  p = 1, 2, 4
    4. Runs seq_python on every case, three block sizes, both kernels.
    5. Runs the distributed mode on every case with p = 1, 2, 4 workers,
       three block sizes, both kernels and BOTH passes (pipelined and
       barriered). One cluster per p is reused for all of its runs.
       Block sizes per case (n = matrix size):
         n // 3      four tiles per side, the last one shorter (uneven split);
                     with p = 4 (2x2 grid) the tile grid wraps around the
                     process grid, which exercises the block-cyclic placement
         ceil(n / 2) two tiles per side
         n + 5       one tile larger than the matrix: a single task chain,
                     and with p >= 2 some workers get no work at all
    6. Checks for every result:
         - every element matches the reference within tolerance;
         - the "ordered" kernel reproduces c_baseline bit for bit, and its
           checksum is the same string (this proves that partitioning,
           placement and collection lose or duplicate nothing);
         - the checksum agrees with c_baseline's within CHECKSUM_REL_TOL
           (the rule run_benchmark.py applies to every run);
         - distributed only: the inter-worker bytes recorded by Dask equal
           the bytes the placement plan predicts (each needed tile crosses
           to each worker exactly once).
       For blas and c_omp_tiled, "same bits as c_baseline" is reported for
       information only: tolerances decide.
    7. Runs distributed_matmul.py exactly as the benchmark does (separate
       process, --output) for distributed and seq_python, and checks that
       the reported labels (mode, n, workers, block size, kernel) are the
       ones requested, so the benchmark can trust them.

Why a separate reference instead of comparing the modes to each other:
    If all modes shared the same bug, they would agree with each other and
    the bug would go unnoticed. math.fsum gives the most accurate double
    result possible, computed from the input file by an independent reader.

Run (from anywhere):
    python3 assignment_4/verify_correctness.py [--size 10]
Exit code 0 = all checks passed, 1 = at least one failure.
"""

import argparse
import math
import random
import subprocess
import sys
from array import array
from pathlib import Path

import numpy as np

import benchmark_config as config
import distributed_matmul as program   # importable: benchmark_config put src/ on sys.path

data_generator = config.data_generator

VERIFY_DIR = config.BUILD_DIR / "verify"

LARGER_TEST_SIZES = [100, 131]
VERIFY_WORKER_COUNTS = [1, 2, 4]
PASSES = ["pipelined", "barriered"]

# ---------------------------------------------------------------------------
# Tolerances (same rule and values as Assignments 1-3)
# ---------------------------------------------------------------------------
# An element passes if   |result - reference| <= ABS_TOL + REL_TOL * |reference|
#
# Values: unit roundoff u = 2^-53. The error of a length-n dot product is at
# most n * u * sum_k |a_ik * b_kj| <= n^2 * u for inputs in [-1, 1)
# (Higham, "Accuracy and Stability of Numerical Algorithms", ch. 3).
#   ABS_TOL = 100 * n^2 * u  -> 100x margin; essential for values near zero.
#   REL_TOL = 1e-12          -> accepts rounding differences (e.g. BLAS
#                               summation order), while a real bug (wrong
#                               tile, missing step) causes errors around 0.1.
UNIT_ROUNDOFF = 2.0 ** -53
SAFETY_FACTOR = 100
REL_TOL = 1e-12


def absolute_tolerance(n: int) -> float:
    """Absolute tolerance derived from the dot-product error bound (see above)."""
    return SAFETY_FACTOR * n * n * UNIT_ROUNDOFF


# ---------------------------------------------------------------------------
# Test inputs and the reference (plain Python, independent of the program)
# ---------------------------------------------------------------------------

def identity_matrix(n: int) -> list[float]:
    matrix = [0.0] * (n * n)
    for i in range(n):
        matrix[i * n + i] = 1.0
    return matrix


def random_matrix(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n * n)]


def reference_product(a: list[float], b: list[float], n: int) -> np.ndarray:
    """Correctly rounded C = A x B, computed independently with math.fsum()."""
    c = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            c[i * n + j] = math.fsum(a[i * n + k] * b[k * n + j] for k in range(n))
    return np.array(c).reshape(n, n)


def read_input_file(path: Path) -> tuple[int, list[float], list[float]]:
    """Read n, A and B with the standard library (NOT the program's own loader)."""
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


def read_result_file(path: Path) -> np.ndarray:
    """Read C from a result file written with --output (C programs or distributed_matmul.py)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little", signed=True)
        c = np.fromfile(f, dtype="<f8", count=n * n)
    if c.size != n * n:
        raise ValueError(f"{path}: truncated result file")
    return c.reshape(n, n)


def build_test_cases(size: int) -> list[dict]:
    """Write every test input to VERIFY_DIR; return [{"name", "path", "n", "expected"}]."""
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    cases = []

    def add_case(name: str, path: Path, n: int, a: list[float], b: list[float],
                 expected: np.ndarray | None = None) -> None:
        data_generator.write_matrices(path, n, array("d", a), array("d", b))
        if expected is None:
            expected = reference_product(a, b, n)
        cases.append({"name": name, "path": path, "n": n, "expected": expected})

    # Identity: A x I = A exactly (multiplying by 1.0 and adding 0.0 is exact).
    a = random_matrix(size, seed=1)
    add_case("identity", VERIFY_DIR / "identity.bin", size, a, identity_matrix(size),
             expected=np.array(a).reshape(size, size))

    # Zero: A x 0 = 0 exactly.
    a = random_matrix(size, seed=2)
    add_case("zero", VERIFY_DIR / "zero.bin", size, a, [0.0] * (size * size),
             expected=np.zeros((size, size)))

    # Known 3x3 product, computed by hand:
    #   [1 2 3]   [9 8 7]   [ 30  24  18]
    #   [4 5 6] x [6 5 4] = [ 84  69  54]
    #   [7 8 9]   [3 2 1]   [138 114  90]
    add_case("known_3x3", VERIFY_DIR / "known_3x3.bin", 3,
             [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
             [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
             expected=np.array([[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]]))

    # Random inputs from the shared generator (the one the benchmark uses).
    for n in [size] + LARGER_TEST_SIZES:
        path = data_generator.generate_for_size(n, config.DATA_SEED, VERIFY_DIR)
        file_n, a, b = read_input_file(path)
        cases.append({"name": path.stem.replace("matrices_", "random_"),   # e.g. random_n131
                      "path": path, "n": file_n, "expected": reference_product(a, b, file_n)})
    return cases


def block_sizes_for(n: int) -> list[int]:
    """Uneven split, two tiles, one oversized tile (see the module docstring)."""
    return [max(1, n // 3), math.ceil(n / 2), n + 5]


# ---------------------------------------------------------------------------
# Checking one result
# ---------------------------------------------------------------------------

def within_tolerance(result: np.ndarray, expected: np.ndarray) -> tuple[bool, float]:
    """Element-wise tolerance check. Returns (passed, largest absolute error)."""
    if result.shape != expected.shape:
        return False, float("nan")
    n = expected.shape[0]
    error = np.abs(result - expected)
    limit = absolute_tolerance(n) + REL_TOL * np.abs(expected)
    passed = bool(np.all(error <= limit))      # False for NaN as well
    return passed, float(np.max(error)) if error.size else 0.0


def check_result(result: np.ndarray, checksum: str, case: dict, kernel: str | None) -> dict:
    """
    Check one result against the reference and against c_baseline (whose
    result and checksum are stored in the case by run_c_programs).
    Returns {"passed", "max_error", "same_bits", "problems"}.
    """
    problems = []
    passed, max_error = within_tolerance(result, case["expected"])
    if not passed:
        problems.append("outside tolerance")

    same_bits = result.tobytes() == case["c_baseline_result"].tobytes()
    reference_checksum = case["c_baseline_checksum"]
    if kernel == "ordered":
        # The exactness claim: the same bits and the very same printed checksum.
        if not same_bits:
            problems.append("not bit-identical to c_baseline")
        if checksum != reference_checksum:
            problems.append(f"checksum {checksum} != c_baseline {reference_checksum}")
    elif not config.checksums_agree(checksum, reference_checksum):
        problems.append(f"checksum {checksum} not within tolerance of {reference_checksum}")

    return {"passed": not problems, "max_error": max_error, "same_bits": same_bits,
            "problems": problems}


# ---------------------------------------------------------------------------
# Steps 3-5: the three kinds of runs
# ---------------------------------------------------------------------------

def run_program(mode: str, case: dict, workers: int, output_file: Path,
                block_size: int | None = None, kernel: str | None = None) -> tuple[np.ndarray, dict]:
    """Run distributed_matmul.py as a separate process (as the benchmark does)."""
    command = config.run_command(mode, case["path"], workers, block_size, kernel, output_file)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"exit code {completed.returncode}: {completed.stderr.strip()}")
    fields = config.parse_program_output(completed.stdout.strip().splitlines()[-1])
    return read_result_file(output_file), fields


def run_c_programs(cases: list[dict], record) -> None:
    """Step 3. c_baseline first: its result becomes the bit-level reference of the case."""
    for case in cases:
        output_file = VERIFY_DIR / f"result_{case['name']}_c_baseline.bin"
        result, fields = run_program("c_baseline", case, 1, output_file)
        case["c_baseline_result"] = result
        case["c_baseline_checksum"] = fields["checksum"]
        record(case, "-", "c_baseline", "-", 1, check_result(result, fields["checksum"], case, None))

        for workers in VERIFY_WORKER_COUNTS:
            output_file = VERIFY_DIR / f"result_{case['name']}_c_omp_tiled_p{workers}.bin"
            result, fields = run_program("c_omp_tiled", case, workers, output_file)
            outcome = check_result(result, fields["checksum"], case, None)
            if fields.get("workers") != str(workers):
                outcome["passed"] = False
                outcome["problems"].append(f"reported workers={fields.get('workers')}")
            record(case, "-", "c_omp_tiled", "-", workers, outcome)


def run_seq_python(cases: list[dict], record) -> None:
    """Step 4. In-process calls of the program's own seq_python function."""
    for case in cases:
        _n, a, b = program.load_matrices(case["path"])
        for block_size in block_sizes_for(case["n"]):
            for kernel in program.KERNELS:
                result, metrics = program.run_seq_python(a, b, block_size, kernel)
                outcome = check_result(result, metrics["checksum"], case, kernel)
                record(case, block_size, "seq_python", kernel, 1, outcome)


def run_distributed(cases: list[dict], record) -> None:
    """
    Step 5. One cluster per worker count, reused for every case, block
    size, kernel and pass (starting a cluster takes about a second).
    """
    for workers in VERIFY_WORKER_COUNTS:
        print(f"  distributed: starting a cluster with {workers} worker(s)...")
        grid = program.process_grid(workers)
        cluster, client, addresses = program.start_cluster(workers)
        try:
            for case in cases:
                _n, a, b = program.load_matrices(case["path"])
                for block_size in block_sizes_for(case["n"]):
                    for kernel in program.KERNELS:
                        for pass_name in PASSES:
                            outcome = run_one_distributed_pass(client, addresses, grid, a, b,
                                                               block_size, kernel, pass_name, case)
                            record(case, block_size, f"distributed/{pass_name}", kernel,
                                   workers, outcome)
        finally:
            program.stop_cluster(cluster, client)


def run_one_distributed_pass(client, addresses, grid, a, b, block_size, kernel, pass_name, case) -> dict:
    """One pass on an existing cluster, checked like every other result, plus the traffic check."""
    client.run(program.clear_transfer_log)
    if pass_name == "pipelined":
        result, _elapsed = program.run_pipelined_pass(client, addresses, grid, a, b, block_size, kernel)
    else:
        result, _phases = program.run_barriered_pass(client, addresses, grid, a, b, block_size, kernel)
    measured = sum(client.run(program.incoming_transfer_bytes).values())
    program.wait_until_released(client)

    checksum = program.format_checksum(program.sequential_checksum(result))
    outcome = check_result(result, checksum, case, kernel)

    expected_traffic = program.theoretical_transfer_bytes(case["n"], block_size, grid)
    outcome["traffic"] = (measured, expected_traffic)
    if measured != expected_traffic:
        outcome["passed"] = False
        outcome["problems"].append(f"transferred {measured} B, plan says {expected_traffic} B")
    return outcome


# ---------------------------------------------------------------------------
# Step 7: the program as the benchmark runs it
# ---------------------------------------------------------------------------

def check_command_line(case: dict) -> bool:
    """Run both Python modes as separate processes; check labels and results."""
    print(f"\nCommand-line runs (as run_benchmark.py starts them), case {case['name']}:")
    all_passed = True
    block_size = block_sizes_for(case["n"])[0]
    runs = [("seq_python", 1, "blas"), ("distributed", 2, "blas"), ("distributed", 4, "ordered")]

    for mode, workers, kernel in runs:
        output_file = VERIFY_DIR / f"result_{case['name']}_cli_{mode}_p{workers}_{kernel}.bin"
        problems = []
        try:
            result, fields = run_program(mode, case, workers, output_file, block_size, kernel)
            for key, wanted in [("mode", mode), ("n", case["n"]), ("workers", workers),
                                ("block_size", block_size), ("kernel", kernel)]:
                if fields.get(key) != str(wanted):
                    problems.append(f"reported {key}={fields.get(key)} (expected {wanted})")
            problems += check_result(result, fields["checksum"], case, kernel)["problems"]
        except (RuntimeError, ValueError, KeyError) as error:
            problems.append(repr(error))

        passed = not problems
        all_passed = all_passed and passed
        label = f"{mode} p={workers} b={block_size} kernel={kernel}"
        print(f"  {label:<44}{'PASS' if passed else 'FAIL'}  {'; '.join(problems)}")
    return all_passed


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_table(lines: dict) -> bool:
    """One line per (case, block size, mode, kernel), summarising its worker counts."""
    print(f"\n{'case':<13}{'b':>5}  {'mode':<23}{'kernel':<9}{'workers':<9}{'result':<8}"
          f"{'max |error|':>12}  {'same bits as C':<16}{'traffic = plan':<16}notes")
    all_passed = True
    for (case_name, block_size, mode, kernel), outcomes in lines.items():
        passed = all(outcome["passed"] for _workers, outcome in outcomes)
        all_passed = all_passed and passed
        workers_label = ",".join(str(workers) for workers, _outcome in outcomes)
        max_error = max(outcome["max_error"] for _workers, outcome in outcomes)
        same_bits = sum(outcome["same_bits"] for _workers, outcome in outcomes)
        traffic = [outcome["traffic"] for _workers, outcome in outcomes if "traffic" in outcome]
        if traffic:
            matching = sum(measured == planned for measured, planned in traffic)
            traffic_label = f"{matching}/{len(traffic)}"
        else:
            traffic_label = "-"
        notes = "; ".join(f"p={workers}: {', '.join(outcome['problems'])}"
                          for workers, outcome in outcomes if outcome["problems"])
        if mode == "c_baseline":
            bits_label = "(reference)"
        else:
            bits_label = f"{same_bits}/{len(outcomes)}"
        print(f"{case_name:<13}{block_size:>5}  {mode:<23}{kernel:<9}{workers_label:<9}"
              f"{'PASS' if passed else 'FAIL':<8}{max_error:>12.3e}  {bits_label:<16}"
              f"{traffic_label:<16}{notes}")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Assignment 4 modes.")
    parser.add_argument("--size", type=int, default=10,
                        help="n for the identity/zero/small random cases")
    args = parser.parse_args()

    print("Preparing C programs and test cases (reference with math.fsum)...")
    config.ensure_c_binaries()
    cases = build_test_cases(args.size)

    print(f"Tolerances: REL_TOL={REL_TOL:.0e}, ABS_TOL=100*n^2*u "
          f"(n={max(LARGER_TEST_SIZES)}: {absolute_tolerance(max(LARGER_TEST_SIZES)):.2e}); "
          f"checksums: rel. {config.CHECKSUM_REL_TOL:.0e}")
    print(f"Block sizes per case: n//3, ceil(n/2), n+5   workers: {VERIFY_WORKER_COUNTS}   "
          f"kernels: {program.KERNELS}   passes: {PASSES}")

    # lines[(case, block size, mode, kernel)] = [(workers, outcome), ...]
    lines = {}

    def record(case, block_size, mode, kernel, workers, outcome):
        lines.setdefault((case["name"], block_size, mode, kernel), []).append((workers, outcome))

    run_c_programs(cases, record)
    run_seq_python(cases, record)
    run_distributed(cases, record)

    # Group the lines by case for readability (the runs above loop in different orders).
    case_order = {case["name"]: index for index, case in enumerate(cases)}
    mode_order = {"c_baseline": 0, "c_omp_tiled": 1, "seq_python": 2,
                  "distributed/pipelined": 3, "distributed/barriered": 4}

    def table_order(item):
        case_name, block_size, mode, kernel = item[0]
        block_number = -1 if block_size == "-" else block_size   # C lines (no tiles) first
        return case_order[case_name], block_number, mode_order[mode], kernel

    lines = dict(sorted(lines.items(), key=table_order))

    all_passed = print_table(lines)
    all_passed = check_command_line(cases[-1]) and all_passed

    print("\nAll checks passed." if all_passed else "\nSOME CHECKS FAILED.")
    return 0 if all_passed else 1


# Guard required: the Dask workers re-import modules with the "spawn" method.
if __name__ == "__main__":
    sys.exit(main())
