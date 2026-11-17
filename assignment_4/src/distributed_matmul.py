"""
distributed_matmul.py - Distributed dense matrix multiplication (Assignment 4)

Computes C = A x B for the shared binary input files written by
common/data_generator.py, in one of four modes:

    distributed  2D block-partitioned multiplication on a Dask cluster of
                 p worker processes (this file's main subject)
    seq_python   the SAME block algorithm and block kernel, run in one
                 process without Dask: the fair sequential baseline, so
                 T(seq_python) / T(distributed, p) measures only what the
                 distribution adds or costs
    c_baseline   runs Assignment 2's  dense_mult --algo baseline   on the same file
    c_omp_tiled  runs Assignment 3's  parallel_mult --algo omp_tiled  on the same file
                 (both through c_baselines.py, for the cross-paradigm comparison)

-----------------------------------------------------------------------------
Execution model: workers, processes and machines
-----------------------------------------------------------------------------
    Machines   : ONE physical machine. Everything runs locally (Dask
                 LocalCluster); processes talk over TCP on 127.0.0.1.
    Driver     : this script = the Dask CLIENT, plus the Dask SCHEDULER
                 (LocalCluster runs the scheduler inside the process that
                 creates it). The driver never multiplies anything.
    Workers    : p separate OS processes (processes=True), each with exactly
                 ONE thread (threads_per_worker=1). A worker has its own
                 Python interpreter and its own memory: tiles move between
                 workers only as serialised network messages, exactly as
                 they would between machines.
    Nannies    : Dask starts every worker through a small supervisor process
                 (the "nanny"), so the run uses 1 + 2p OS processes, of
                 which p do the arithmetic.
    BLAS       : NumPy's BLAS library is limited to one thread in every
                 process (first lines of this file + WORKER_ENVIRONMENT), so
                 p workers use p cores and no hidden extra threads.

-----------------------------------------------------------------------------
Partitioning: 2D blocks (tiles)
-----------------------------------------------------------------------------
    Each n x n matrix is cut into b x b tiles, nb = ceil(n / b) per side.
    When b does not divide n, the last row/column of tiles is smaller.
    Tile (I, J) of C is the sum over K of tile (I, K) of A times tile (K, J)
    of B: the textbook formula, applied to sub-matrices instead of numbers:

        C(I,J) = A(I,0) x B(0,J) + A(I,1) x B(1,J) + ... + A(I,nb-1) x B(nb-1,J)

    One task = one "multiply-accumulate" C(I,J) += A(I,K) x B(K,J), so a run
    has nb^3 tasks. The nb tasks of one output tile form a chain over
    K = 0, 1, ..., nb-1 (each step needs the previous partial sum); the nb^2
    chains are independent of each other, which is the parallelism.

-----------------------------------------------------------------------------
Placement: 2D block-cyclic distribution (as in ScaLAPACK and SUMMA)
-----------------------------------------------------------------------------
    The p workers form a pr x pc process grid, as square as possible:
        p = 1 -> 1x1    p = 2 -> 1x2    p = 4 -> 2x2    p = 8 -> 2x4
    Tile (I, J) of every matrix (A, B and C) belongs to worker
        owner(I, J) = (I mod pr) * pc + (J mod pc)
    - Inputs: each A and B tile is scattered once to its owner (its "home").
    - Tasks: every step of the chain for C(I,J) runs on owner(I,J)
      ("owner computes"), so partial sums of C never leave their worker
      until the final tile is collected.
    - Balance: when nb is a multiple of pr and pc (all our benchmark runs),
      every worker owns the same number of C tiles, i.e. the same work.
    Dask's own placement heuristics (work stealing, the active memory
    manager) are switched off, so the placement is exactly this plan.

-----------------------------------------------------------------------------
Communication
-----------------------------------------------------------------------------
    1. Scatter (client -> workers): every A and B tile goes to its home
       worker once: 2 * n^2 * 8 bytes.
    2. Between workers: C(I,J) needs A(I,K) and B(K,J) for all K. The home
       of A(I,K) is in the same process ROW as owner(I,J), and the home of
       B(K,J) in the same process COLUMN. So A tiles travel along process
       rows and B tiles along process columns: the SUMMA pattern. Dask
       fetches a missing tile directly from the worker that holds it (peer to
       peer, not through the client) and keeps the copy while tasks on that
       worker still need it, so each (tile, worker) pair crosses the network
       at most once. The program reports the bytes this plan implies
       ("theoretical") and the bytes Dask's transfer logs actually recorded
       ("measured").
    3. Gather (workers -> client): every C tile once: n^2 * 8 bytes.

-----------------------------------------------------------------------------
Timing (time.perf_counter, a monotonic clock; loading the input file is
outside every timer, as in Assignments 1-3)
-----------------------------------------------------------------------------
    setup_s     start the cluster and the client, then a warm-up run of the
                same task graph on 1x1 tiles (see warm_up). Like the C
                programs, which create their threads before starting their
                timer, this one-time cost is reported separately and is NOT
                part of e2e_s.
    cold_e2e_s  the first full-size pipelined pass, on fresh worker
                processes. It is measurably slower (about 20 % in our tests)
                because every process touches its memory for the first time
                (page faults, heap growth). Reported for the discussion of
                cold vs warm runs, but it is not the main metric.
    e2e_s       PIPELINED pass on warm workers, no barriers: from the first
                input tile handed to the client until the last result tile is
                in C on the client: partition + scatter + submit + compute +
                collect.
    The same multiplication is then repeated as a BARRIERED pass, with
    wait() between phases, to see where the time goes:
      scatter_s  partition A and B into tiles, send each to its home worker
      compute_s  submit all nb^3 tasks and run them, including the
                 worker-to-worker tile transfers
      gather_s   fetch all C tiles to the client and assemble C
    scatter_s + compute_s + gather_s and e2e_s measure the same work but
    are not equal; in our tests they differed by a few percent, in either
    direction:
      - pipelining can make e2e_s SHORTER: task submission overlaps the
        first computations, finished C tiles travel to the client while
        other tiles are still being computed, and there are no barrier
        round trips;
      - it can also make e2e_s LONGER: the tiles are collected one by one
        instead of in one batch, and the client (which also hosts the
        scheduler) receives and unpacks them while the workers compute,
        competing with them for CPU when p is close to the number of cores.
    Which effect wins depends on n, p and the machine. Both timed passes run
    after the cold pass, so neither pays the first-touch costs and the two
    can be compared fairly.
    All three passes must produce the same C; the program checks it.

-----------------------------------------------------------------------------
Output (one line on stdout, space-separated key=value pairs, as in A1-A3)
-----------------------------------------------------------------------------
    mode=distributed kernel=blas n=2000 workers=4 grid=2x2 block_size=500
    tile_grid=4 tasks=64 e2e_s=... setup_s=... cold_e2e_s=... scatter_s=... compute_s=...
    gather_s=... worker_peak_mb_max=... worker_peak_mb_mean=...
    client_peak_mb=... scatter_bytes=... comm_bytes_theoretical=...
    comm_bytes_measured=... gather_bytes=... checksum=1.2345678901e+02
    Fields that do not apply to a mode are left out (e.g. no setup_s for C).
    checksum is the sum of all elements of C, added in the same order and
    printed in the same format (%.10e) as by the C programs.

Usage:
    python distributed_matmul.py <matrices.bin> --mode distributed --workers P --block-size B
           [--kernel blas|ordered] [--output result.bin]
    python distributed_matmul.py <matrices.bin> --mode seq_python --block-size B [--kernel ...]
    python distributed_matmul.py <matrices.bin> --mode c_baseline
    python distributed_matmul.py <matrices.bin> --mode c_omp_tiled --workers P
"""

import os

# NumPy's BLAS library starts its own thread pool when NumPy is imported.
# Limit it to ONE thread before that happens, so that the sequential
# baseline uses one core and every Dask worker uses one core. Otherwise
# "p = 1" could silently use all cores and the scaling curves would be
# meaningless. (The worker processes get the same setting through
# WORKER_ENVIRONMENT below.)
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse  # noqa: E402  (imports after the environment setup on purpose)
import logging   # noqa: E402
import math      # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil       # noqa: E402

# Dask is imported only inside the functions of the distributed mode: the
# sequential baseline and the C wrappers should not pay its import time or
# carry its ~25 MB of libraries in their memory measurement.

MODES = ["distributed", "seq_python", "c_baseline", "c_omp_tiled"]
PYTHON_MODES = ["distributed", "seq_python"]
C_MODES = ["c_baseline", "c_omp_tiled"]

KERNELS = ["blas", "ordered"]
DEFAULT_KERNEL = "blas"

BYTES_PER_ELEMENT = 8                 # float64
MEGABYTE = 1024 * 1024
CHECKSUM_CHUNK = 1 << 20              # elements per np.cumsum call in sequential_checksum

# Environment of every worker process (passed to Dask's nanny, which sets it
# before the worker starts and imports NumPy): one BLAS thread per worker.
WORKER_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

# Dask settings, applied before the cluster starts (the nanny copies them
# into every worker process).
DASK_SETTINGS = {
    # Our placement plan must be the one that runs. Work stealing would move
    # queued tasks to idle workers, and the active memory manager would add
    # or drop copies of tiles; both would change who computes what and which
    # tiles travel. (Tasks are also pinned with workers=[...] below.)
    "distributed.scheduler.work-stealing": False,
    "distributed.scheduler.active-memory-manager.start": False,
    # No spilling to disk: every tile stays in RAM, so disk I/O never enters
    # the timings. A worker that ran out of memory would fail loudly instead.
    "distributed.worker.memory.target": False,
    "distributed.worker.memory.spill": False,
    # Send raw bytes: random float64 data does not compress, and without
    # compression the bytes on the wire equal the array bytes we report.
    "distributed.comm.compression": False,
}


# ---------------------------------------------------------------------------
# Input and output (format: common/data_generator.py)
# ---------------------------------------------------------------------------

def load_matrices(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Read n, A and B from a shared input file:
        int64 n | n*n float64 (A, row-major) | n*n float64 (B, row-major), little-endian.
    The header is checked against the file size before any array is
    allocated, as in the C programs: a corrupt n must not trigger a huge read.
    """
    with open(path, "rb") as file:
        header = file.read(BYTES_PER_ELEMENT)
        if len(header) != BYTES_PER_ELEMENT:
            raise ValueError(f"{path}: file too short for the header")
        n = int.from_bytes(header, "little", signed=True)
        if n <= 0:
            raise ValueError(f"{path}: invalid header n={n}")

        expected_bytes = BYTES_PER_ELEMENT + 2 * n * n * BYTES_PER_ELEMENT
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"{path}: n={n} needs {expected_bytes} bytes, file has {actual_bytes}")

        # '<f8' = little-endian float64, i.e. exactly the bytes in the file.
        a = np.fromfile(file, dtype="<f8", count=n * n).reshape(n, n)
        b = np.fromfile(file, dtype="<f8", count=n * n).reshape(n, n)
    return n, a, b


def save_matrix(path: Path, c: np.ndarray) -> None:
    """Write n and C in the result format of the C programs (for verify_correctness.py)."""
    n = c.shape[0]
    with open(path, "wb") as file:
        file.write(n.to_bytes(BYTES_PER_ELEMENT, "little", signed=True))
        np.ascontiguousarray(c, dtype="<f8").tofile(file)


# ---------------------------------------------------------------------------
# Partitioning into tiles
# ---------------------------------------------------------------------------

def tiles_per_side(n: int, block_size: int) -> int:
    """nb = ceil(n / b): the number of tile rows (= tile columns)."""
    return math.ceil(n / block_size)


def tile_range(index: int, n: int, block_size: int) -> slice:
    """Rows (or columns) covered by tile number `index`. The last tile may be shorter."""
    start = index * block_size
    return slice(start, min(start + block_size, n))


def tile_length(index: int, n: int, block_size: int) -> int:
    """Number of rows (or columns) of tile number `index`."""
    return min(block_size, n - index * block_size)


def split_into_tiles(matrix: np.ndarray, block_size: int) -> dict[tuple[int, int], np.ndarray]:
    """
    Cut a square matrix into tiles: {(I, J): tile}.

    Why .copy(): a slice of a row-major matrix is a strided VIEW (its rows
    are n elements apart in memory). Copying makes every tile one contiguous
    block, so it is serialised and sent as a single buffer, and the block
    kernel reads it with stride 1.
    """
    n = matrix.shape[0]
    count = tiles_per_side(n, block_size)
    tiles = {}
    for row in range(count):
        for col in range(count):
            rows = tile_range(row, n, block_size)
            cols = tile_range(col, n, block_size)
            tiles[(row, col)] = matrix[rows, cols].copy()
    return tiles


def assemble_tiles(tiles: dict[tuple[int, int], np.ndarray], n: int, block_size: int) -> np.ndarray:
    """
    Put the C tiles back into one n x n matrix.
    C starts filled with NaN (as in the C programs): a tile that went
    missing leaves NaNs behind, so the checksum becomes nan and the error
    cannot go unnoticed.
    """
    expected = tiles_per_side(n, block_size) ** 2
    if len(tiles) != expected:
        raise RuntimeError(f"expected {expected} result tiles, received {len(tiles)}")

    c = np.full((n, n), np.nan)
    for (row, col), tile in tiles.items():
        c[tile_range(row, n, block_size), tile_range(col, n, block_size)] = tile
    return c


# ---------------------------------------------------------------------------
# The block kernel: the body of every task
# ---------------------------------------------------------------------------

def multiply_accumulate(c_tile: np.ndarray | None, a_tile: np.ndarray,
                        b_tile: np.ndarray, kernel: str) -> np.ndarray:
    """
    Return C_tile + A_tile x B_tile (C_tile is None at the first step, K = 0).

    The inputs are never modified: Dask may still hold them (tasks must be
    pure functions), so a new array is returned. Copying a partial C tile
    costs b^2 operations against the b^3 of the product: negligible.

    kernel = "blas" (the benchmark kernel)
        NumPy's matmul, i.e. the optimised BLAS routine dgemm (one thread).
        This is what a real distributed system runs on each node. BLAS adds
        the terms of each dot product in its own blocked order, with FMA,
        so the last bits differ from the C kernels: same value within the
        rounding tolerance, not the same bits.

    kernel = "ordered" (the exactness proof)
        Rank-1 updates for k = 0, 1, ..., in increasing order. Every element
        of C then receives its terms in the order k = 0..n-1 (the K steps of
        the chain also run in increasing order), with the product and the sum
        rounded separately, exactly like the statement
            c_row[j] += a_ik * b_row[j];
        in the C kernels compiled without FMA. The result is therefore
        bit-identical to dense_mult --algo baseline, which proves that the
        tiling, placement and collection lose or duplicate nothing. It is
        about 20x slower than BLAS, so it is used for verification only.
    """
    if kernel == "blas":
        product = a_tile @ b_tile
        if c_tile is None:
            return product
        return c_tile + product

    if kernel == "ordered":
        if c_tile is None:
            result = np.zeros((a_tile.shape[0], b_tile.shape[1]))
        else:
            result = c_tile.copy()
        for k in range(a_tile.shape[1]):
            # Outer product of column k of A_tile and row k of B_tile:
            # one multiplication, then one addition, each rounded.
            result += np.multiply.outer(a_tile[:, k], b_tile[k, :])
        return result

    raise ValueError(f"Unknown kernel: {kernel}")


# ---------------------------------------------------------------------------
# Checksum (identical to checksum() in the C programs)
# ---------------------------------------------------------------------------

def sequential_checksum(c: np.ndarray) -> float:
    """
    Sum of all elements of C, added strictly left to right in row-major
    order, like checksum() in the C programs, so the values are comparable
    digit by digit.

    Why not sum() or np.sum(): both are more accurate, and that is the
    problem. Python >= 3.12 uses compensated summation in sum(), and np.sum
    uses pairwise summation: a different order gives different last bits.
    np.cumsum is a plain running sum. It is applied one chunk at a time
    (carrying the running total into the next chunk) so that it never
    allocates a second n x n array.
    """
    flat = c.ravel()
    total = 0.0
    for start in range(0, flat.size, CHECKSUM_CHUNK):
        chunk = flat[start:start + CHECKSUM_CHUNK]
        running = np.cumsum(np.concatenate(([total], chunk)))
        total = float(running[-1])
    return total


def format_checksum(value: float) -> str:
    """Same format as the C programs' printf("%.10e")."""
    return f"{value:.10e}"


# ---------------------------------------------------------------------------
# Placement plan (2D block-cyclic) and its communication volume
# ---------------------------------------------------------------------------

def process_grid(workers: int) -> tuple[int, int]:
    """
    Arrange p workers in a pr x pc grid, as square as possible, with pr <= pc:
    1 -> 1x1, 2 -> 1x2, 4 -> 2x2, 8 -> 2x4 (and 6 -> 2x3, 3 -> 1x3).
    A square grid minimises communication: a worker needs the A tiles of
    its process row and the B tiles of its process column.
    """
    grid_rows = math.isqrt(workers)
    while workers % grid_rows != 0:
        grid_rows -= 1
    return grid_rows, workers // grid_rows


def owner(row: int, col: int, grid: tuple[int, int]) -> int:
    """Worker index that holds tile (row, col) of A, B and C (2D block-cyclic)."""
    grid_rows, grid_cols = grid
    return (row % grid_rows) * grid_cols + (col % grid_cols)


def theoretical_transfer_bytes(n: int, block_size: int, grid: tuple[int, int]) -> int:
    """
    Bytes that must cross between workers under the placement plan.

    For every task C(I,J) += A(I,K) x B(K,J), running on owner(I,J), each
    input tile whose home is a different worker must be fetched. A worker
    keeps a fetched copy while it still needs it, so each (tile, worker)
    pair is counted once: that is why a set is used.
    """
    count = tiles_per_side(n, block_size)
    needed = set()   # entries ("A" or "B", tile row, tile col, destination worker)
    for row in range(count):
        for col in range(count):
            destination = owner(row, col, grid)
            for k in range(count):
                if owner(row, k, grid) != destination:
                    needed.add(("A", row, k, destination))
                if owner(k, col, grid) != destination:
                    needed.add(("B", k, col, destination))

    total = 0
    for _matrix, tile_row, tile_col, _destination in needed:
        rows = tile_length(tile_row, n, block_size)
        cols = tile_length(tile_col, n, block_size)
        total += rows * cols * BYTES_PER_ELEMENT
    return total


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def peak_rss_bytes() -> int:
    """
    Peak resident set size (high-water mark) of THIS process, in bytes:
    the largest amount of physical RAM it has occupied since it started.

    psutil gives the process handle and, on Windows, the peak directly
    (peak_wset). On Linux psutil exposes only the current RSS, so the
    kernel's high-water mark VmHWM is read from /proc. VmHWM belongs to the
    process's own memory: unlike getrusage(), it does not include the
    memory of the parent that started this process (the nanny, for a
    worker). macOS fallback: getrusage (in bytes).
    """
    process = psutil.Process()
    memory = process.memory_info()
    if hasattr(memory, "peak_wset"):
        return memory.peak_wset

    status_file = Path(f"/proc/{process.pid}/status")
    if status_file.exists():
        for line in status_file.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024    # the value is in kB

    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def worker_memory_report() -> dict:
    """Runs INSIDE each worker process (via client.run): its pid and peak RSS."""
    return {"pid": os.getpid(), "peak_bytes": peak_rss_bytes()}


# ---------------------------------------------------------------------------
# Small functions that run inside the workers or the scheduler.
# Dask passes the worker / scheduler object to a parameter with this name.
# ---------------------------------------------------------------------------

def clear_transfer_log(dask_worker) -> None:
    """Forget the worker's record of earlier incoming transfers."""
    dask_worker.transfer_incoming_log.clear()


def incoming_transfer_bytes(dask_worker) -> int:
    """
    Bytes this worker fetched from OTHER workers since the log was cleared.
    The log records only worker-to-worker fetches: data scattered by the
    client arrives through a different path and is not in it.
    """
    return sum(entry["total"] for entry in dask_worker.transfer_incoming_log)


def scheduler_task_count(dask_scheduler) -> int:
    """Number of tasks (data or computation) the scheduler still tracks."""
    return len(dask_scheduler.tasks)


def stored_item_count(dask_worker) -> int:
    """Number of results/tiles the worker still holds in memory."""
    return len(dask_worker.data)


# ---------------------------------------------------------------------------
# Mode seq_python: the same block algorithm in one process
# ---------------------------------------------------------------------------

def run_seq_python(a: np.ndarray, b: np.ndarray, block_size: int, kernel: str) -> tuple[np.ndarray, dict]:
    """
    Same tiles, same kernel calls, same K order as the distributed mode, in
    a plain loop: no processes, no serialisation, no network. The phases
    use the same names; "scatter" is only the partitioning here and
    "gather" only the assembly.
    """
    n = a.shape[0]
    count = tiles_per_side(n, block_size)

    start = time.perf_counter()
    a_tiles = split_into_tiles(a, block_size)
    b_tiles = split_into_tiles(b, block_size)
    partitioned = time.perf_counter()

    c_tiles = {}
    for row in range(count):
        for col in range(count):
            partial = None
            for k in range(count):
                partial = multiply_accumulate(partial, a_tiles[(row, k)], b_tiles[(k, col)], kernel)
            c_tiles[(row, col)] = partial
    computed = time.perf_counter()

    c = assemble_tiles(c_tiles, n, block_size)
    end = time.perf_counter()

    peak_mb = peak_rss_bytes() / MEGABYTE
    metrics = {
        "mode": "seq_python", "kernel": kernel, "n": n, "workers": 1, "grid": "1x1",
        "block_size": block_size, "tile_grid": count, "tasks": count ** 3,
        "e2e_s": end - start,
        "scatter_s": partitioned - start,
        "compute_s": computed - partitioned,
        "gather_s": end - computed,
        "worker_peak_mb_max": peak_mb,      # the single computing process
        "worker_peak_mb_mean": peak_mb,
        "scatter_bytes": 0, "comm_bytes_theoretical": 0, "comm_bytes_measured": 0, "gather_bytes": 0,
        "checksum": format_checksum(sequential_checksum(c)),
    }
    return c, metrics


# ---------------------------------------------------------------------------
# Mode distributed: cluster lifecycle
# ---------------------------------------------------------------------------

def start_cluster(workers: int):
    """
    Start a local Dask cluster of `workers` single-threaded worker processes
    and connect a client to it. Returns (cluster, client, addresses), where
    addresses[w] is the network address of worker number w in the
    placement plan.
    """
    import dask
    from dask.distributed import Client, LocalCluster

    dask.config.set(DASK_SETTINGS)

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=1,          # one task at a time per worker ...
        processes=True,                # ... and each worker is its own OS process
        host="127.0.0.1",              # local machine only
        dashboard_address=None,        # no web dashboard: avoids port clashes and its overhead
        silence_logs=logging.ERROR,    # keep stderr readable; errors still appear
        env=WORKER_ENVIRONMENT,
    )
    try:
        # direct_to_workers: the client sends tiles straight to their worker
        # and fetches results straight from it. The scheduler only records
        # where each tile lives; data never flows through it.
        client = Client(cluster, direct_to_workers=True)
    except Exception:
        cluster.close()
        raise

    try:
        client.wait_for_workers(workers, timeout=120)
        # client.run executes a function on every worker and returns
        # {address: result}. Sorting the addresses fixes the numbering
        # worker 0, 1, ..., p-1 used by owner().
        addresses = sorted(client.run(os.getpid))
    except Exception:
        stop_cluster(cluster, client)
        raise
    return cluster, client, addresses


def stop_cluster(cluster, client) -> None:
    """Close the client first, then the cluster (which stops all worker processes)."""
    try:
        client.close()
    finally:
        cluster.close()


def wait_until_released(client, timeout_s: float = 60.0) -> None:
    """
    Block until the previous pass has been released everywhere, so the next
    pass starts on empty workers and their memory and transfer logs are not
    mixed. Dropping our references to the futures tells Dask to release
    them, but asynchronously: first the scheduler forgets the tasks, then
    it tells each worker to delete its copies. Both are checked.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        scheduler_tasks = client.run_on_scheduler(scheduler_task_count)
        worker_items = sum(client.run(stored_item_count).values())
        if scheduler_tasks == 0 and worker_items == 0:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"Dask did not release the previous pass in time "
                               f"({scheduler_tasks} tasks, {worker_items} stored items left)")
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Mode distributed: the building blocks of one pass
# ---------------------------------------------------------------------------

def scatter_tiles(client, addresses: list[str], grid: tuple[int, int],
                  tiles: dict[tuple[int, int], np.ndarray], name: str) -> dict:
    """
    Send every tile to its home worker (owner of its position).
    Returns {(row, col): Future}; a Future is the client's handle to a
    piece of data that lives on a worker.

    Tiles are grouped per destination, so there is one scatter call (one
    message) per worker instead of one per tile. The key names, e.g.
    "barriered-A-1-2", make the task graph readable when debugging.
    """
    batches = {address: {} for address in addresses}   # address -> {key: tile}
    position_of_key = {}
    for (row, col), tile in tiles.items():
        key = f"{name}-{row}-{col}"
        home = addresses[owner(row, col, grid)]
        batches[home][key] = tile
        position_of_key[key] = (row, col)

    futures = {}
    for address, batch in batches.items():
        if not batch:
            continue   # more workers than tiles: this worker holds no tile of this matrix
        scattered = client.scatter(batch, workers=[address])
        for key, future in scattered.items():
            futures[position_of_key[key]] = future
    return futures


def submit_block_products(client, addresses: list[str], grid: tuple[int, int],
                          a_futures: dict, b_futures: dict, kernel: str, label: str) -> dict:
    """
    Submit the nb^3 multiply-accumulate tasks. Returns {(I, J): Future of
    the final C tile}.

    For each output tile, the K steps form a chain: step K receives the
    partial sum of step K-1 (a Future; Dask passes it the actual array).
    Every step is pinned to owner(I, J), so the partial sum stays on that
    worker; the input tiles it lacks are fetched by Dask from their home
    worker (see the module docstring, "Communication").
    The K steps are submitted in increasing order, which the "ordered"
    kernel needs for bit-identical results.
    """
    count = int(math.isqrt(len(a_futures)))
    c_futures = {}
    for row in range(count):
        for col in range(count):
            address = addresses[owner(row, col, grid)]
            partial = None
            for k in range(count):
                partial = client.submit(
                    multiply_accumulate, partial, a_futures[(row, k)], b_futures[(k, col)], kernel,
                    key=f"{label}-C-{row}-{col}-step{k}",
                    workers=[address],
                    allow_other_workers=False,   # a hard placement, never a hint
                )
            c_futures[(row, col)] = partial
    return c_futures


# ---------------------------------------------------------------------------
# Mode distributed: the two passes
# ---------------------------------------------------------------------------

def run_pipelined_pass(client, addresses, grid, a, b, block_size, kernel, label="pipelined"):
    """
    End-to-end pass WITHOUT barriers. Returns (C, elapsed seconds).

    Nothing waits for a phase to finish as a whole: the tasks start running
    as soon as their inputs exist, and as_completed() hands over each C tile
    the moment it is ready, so collecting early tiles overlaps computing
    the later ones. (client.gather would first wait for ALL tiles.)
    """
    from dask.distributed import as_completed

    n = a.shape[0]
    start = time.perf_counter()

    a_futures = scatter_tiles(client, addresses, grid, split_into_tiles(a, block_size), f"{label}-A")
    b_futures = scatter_tiles(client, addresses, grid, split_into_tiles(b, block_size), f"{label}-B")
    c_futures = submit_block_products(client, addresses, grid, a_futures, b_futures, kernel, label)

    position_of_key = {future.key: position for position, future in c_futures.items()}
    c_tiles = {}
    for future, tile in as_completed(list(c_futures.values()), with_results=True):
        c_tiles[position_of_key[future.key]] = tile
    c = assemble_tiles(c_tiles, n, block_size)

    return c, time.perf_counter() - start


def run_barriered_pass(client, addresses, grid, a, b, block_size, kernel, label="barriered"):
    """
    The same multiplication with a barrier (wait) after each phase, so each
    phase is timed in isolation. Returns (C, {"scatter_s", "compute_s", "gather_s"}).
    """
    from dask.distributed import wait

    n = a.shape[0]

    # (b) Partition and scatter. client.scatter already returns only when
    # the data has arrived; the wait() makes the barrier explicit.
    start = time.perf_counter()
    a_futures = scatter_tiles(client, addresses, grid, split_into_tiles(a, block_size), f"{label}-A")
    b_futures = scatter_tiles(client, addresses, grid, split_into_tiles(b, block_size), f"{label}-B")
    wait(list(a_futures.values()) + list(b_futures.values()))
    scattered = time.perf_counter()

    # (c) Compute: all tasks, including the worker-to-worker tile transfers.
    c_futures = submit_block_products(client, addresses, grid, a_futures, b_futures, kernel, label)
    wait(list(c_futures.values()))
    computed = time.perf_counter()

    # (d) Gather: every C tile to the client, then assemble C.
    c_tiles = client.gather(c_futures)   # a dict of futures -> a dict of arrays
    c = assemble_tiles(c_tiles, n, block_size)
    end = time.perf_counter()

    phases = {"scatter_s": scattered - start, "compute_s": computed - scattered, "gather_s": end - computed}
    return c, phases


def warm_up(client, addresses, grid, count: int, kernel: str) -> None:
    """
    Run the complete task graph once on 1x1 tiles (an nb x nb matrix with
    block size 1): the same tasks, placement and communication pattern as
    the real run, with negligible data.

    Why: the first use of a path pays one-time costs: opening the TCP
    connections (client -> workers, worker <-> worker), sending the task
    function to each worker, first-call initialisation inside NumPy. These
    costs belong to starting the cluster, so they are part of setup_s.
    (The first-touch cost of full-size memory is a different one-time cost:
    it is what the cold pass in run_distributed measures.)
    """
    tiny = np.ones((count, count))
    run_pipelined_pass(client, addresses, grid, tiny, tiny, 1, kernel, label="warmup")
    wait_until_released(client)


def run_distributed(a: np.ndarray, b: np.ndarray, workers: int, block_size: int,
                    kernel: str) -> tuple[np.ndarray, dict]:
    """Setup, cold pass, pipelined pass, barriered pass, measurements, shutdown."""
    n = a.shape[0]
    count = tiles_per_side(n, block_size)
    grid = process_grid(workers)

    setup_start = time.perf_counter()
    cluster, client, addresses = start_cluster(workers)
    try:
        warm_up(client, addresses, grid, count, kernel)
        setup_s = time.perf_counter() - setup_start

        # Pass 1: cold. The first full-size pass on fresh processes.
        c_cold, cold_e2e_s = run_pipelined_pass(client, addresses, grid, a, b, block_size, kernel,
                                                label="cold")
        checksums = {"cold": sequential_checksum(c_cold)}
        del c_cold                           # free the client's copy before the next pass
        wait_until_released(client)

        # Pass 2: pipelined end-to-end time, on warm workers.
        c_pipelined, e2e_s = run_pipelined_pass(client, addresses, grid, a, b, block_size, kernel)
        checksums["pipelined"] = sequential_checksum(c_pipelined)
        del c_pipelined
        wait_until_released(client)

        # Pass 3: phase breakdown. The transfer logs are cleared first, so
        # they then hold exactly the worker-to-worker traffic of this pass
        # (neither scatter nor gather is recorded in them).
        client.run(clear_transfer_log)
        c, phases = run_barriered_pass(client, addresses, grid, a, b, block_size, kernel)
        comm_bytes_measured = sum(client.run(incoming_transfer_bytes).values())
        checksums["barriered"] = sequential_checksum(c)

        # The same tasks with the same kernel must give the same bits.
        if len(set(checksums.values())) != 1:
            raise RuntimeError(f"the passes gave different results: checksums {checksums}")

        worker_memory = client.run(worker_memory_report)   # {address: {"pid", "peak_bytes"}}
    finally:
        stop_cluster(cluster, client)

    worker_peaks_mb = [report["peak_bytes"] / MEGABYTE for report in worker_memory.values()]
    matrix_bytes = n * n * BYTES_PER_ELEMENT
    metrics = {
        "mode": "distributed", "kernel": kernel, "n": n, "workers": workers,
        "grid": f"{grid[0]}x{grid[1]}", "block_size": block_size,
        "tile_grid": count, "tasks": count ** 3,
        "e2e_s": e2e_s, "setup_s": setup_s, "cold_e2e_s": cold_e2e_s, **phases,
        "worker_peak_mb_max": max(worker_peaks_mb),
        "worker_peak_mb_mean": sum(worker_peaks_mb) / len(worker_peaks_mb),
        "client_peak_mb": peak_rss_bytes() / MEGABYTE,     # driver: client + scheduler
        "scatter_bytes": 2 * matrix_bytes,                 # every A and B tile, once
        "comm_bytes_theoretical": theoretical_transfer_bytes(n, block_size, grid),
        "comm_bytes_measured": comm_bytes_measured,
        "gather_bytes": matrix_bytes,                      # every C tile, once
        "checksum": format_checksum(checksums["barriered"]),
    }
    return c, metrics


# ---------------------------------------------------------------------------
# Modes c_baseline and c_omp_tiled: wrapper around the Assignment 2/3 programs
# ---------------------------------------------------------------------------

def run_c_program(mode: str, input_file: Path, workers: int, output_file: Path | None) -> dict:
    """
    Run the compiled C program on the same input file. Its own timer covers
    the kernel only (there is no distribution step), so e2e_s = compute_s.
    """
    import c_baselines   # loads the Assignment 2/3 configurations; only needed here

    result = c_baselines.run(mode, input_file, workers, output_file)
    return {
        "mode": mode, "n": result["n"], "workers": result["workers"],
        "e2e_s": result["compute_s"], "compute_s": result["compute_s"],
        "worker_peak_mb_max": result["worker_peak_mb"],   # the single C process
        "worker_peak_mb_mean": result["worker_peak_mb"],
        "checksum": result["checksum"],
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

# Order and format of the fields in the result line. Times keep nanosecond
# digits like the C programs (%.9f); memory in MB with 2 decimals.
OUTPUT_FIELDS = [
    ("mode", "{}"), ("kernel", "{}"), ("n", "{}"), ("workers", "{}"), ("grid", "{}"),
    ("block_size", "{}"), ("tile_grid", "{}"), ("tasks", "{}"),
    ("e2e_s", "{:.9f}"), ("setup_s", "{:.9f}"), ("cold_e2e_s", "{:.9f}"), ("scatter_s", "{:.9f}"),
    ("compute_s", "{:.9f}"), ("gather_s", "{:.9f}"),
    ("worker_peak_mb_max", "{:.2f}"), ("worker_peak_mb_mean", "{:.2f}"), ("client_peak_mb", "{:.2f}"),
    ("scatter_bytes", "{}"), ("comm_bytes_theoretical", "{}"), ("comm_bytes_measured", "{}"),
    ("gather_bytes", "{}"),
    ("checksum", "{}"),
]


def format_result_line(metrics: dict) -> str:
    """One line of key=value pairs, in OUTPUT_FIELDS order, skipping absent fields."""
    parts = []
    for key, value_format in OUTPUT_FIELDS:
        if key in metrics:
            parts.append(f"{key}={value_format.format(metrics[key])}")
    return " ".join(parts)


def positive_integer(text: str) -> int:
    """argparse type: an integer >= 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def parse_arguments() -> argparse.Namespace:
    """
    Parse and cross-check the options. Options that do not apply to a mode
    are rejected rather than ignored, so a run can never be recorded under
    a label (workers, block size, kernel) that it did not use.
    """
    parser = argparse.ArgumentParser(
        description="Distributed (Dask) matrix multiplication and its baselines.")
    parser.add_argument("input_file", type=Path, help="shared input file (common/data_generator.py)")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--workers", type=positive_integer,
                        help="worker processes (distributed) or OpenMP threads (c_omp_tiled)")
    parser.add_argument("--block-size", type=positive_integer,
                        help="tile edge b for distributed and seq_python")
    parser.add_argument("--kernel", choices=KERNELS,
                        help=f"block kernel for distributed and seq_python (default: {DEFAULT_KERNEL})")
    parser.add_argument("--output", type=Path, help="write C to this file (result format of the C programs)")
    args = parser.parse_args()

    if args.mode in ("distributed", "c_omp_tiled") and args.workers is None:
        parser.error(f"--workers is required for --mode {args.mode}")
    if args.mode in ("seq_python", "c_baseline"):
        if args.workers not in (None, 1):
            parser.error(f"--mode {args.mode} is sequential: --workers must be 1 or omitted")
        args.workers = 1

    if args.mode in PYTHON_MODES:
        if args.block_size is None:
            parser.error(f"--block-size is required for --mode {args.mode}")
        if args.kernel is None:
            args.kernel = DEFAULT_KERNEL
    else:
        if args.block_size is not None or args.kernel is not None:
            parser.error(f"--block-size and --kernel do not apply to --mode {args.mode}")

    if not args.input_file.exists():
        parser.error(f"input file not found: {args.input_file}")
    return args


def main() -> int:
    args = parse_arguments()
    try:
        if args.mode in C_MODES:
            metrics = run_c_program(args.mode, args.input_file, args.workers, args.output)
        else:
            _n, a, b = load_matrices(args.input_file)    # not timed (as in A1-A3)
            if args.mode == "seq_python":
                c, metrics = run_seq_python(a, b, args.block_size, args.kernel)
            else:
                c, metrics = run_distributed(a, b, args.workers, args.block_size, args.kernel)
            if args.output is not None:
                save_matrix(args.output, c)
    except Exception as error:   # report any failure as one clean line; the cluster is already closed
        print(f"Error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1

    print(format_result_line(metrics))
    return 0


# The guard is required, not just good style: Dask starts its worker
# processes with the "spawn" method, which imports this file again in every
# new process. Without the guard, each worker would start a cluster of its own.
if __name__ == "__main__":
    sys.exit(main())
