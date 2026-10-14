# Assignment 1 — Basic Matrix Multiplication in C, Java and Python

**Research question:** how does the matrix size $n$ affect the execution time and
memory usage of the textbook $O(n^3)$ dense matrix multiplication, and how do an
ahead-of-time compiled language (C), a JIT-compiled language (Java) and an
interpreted language (Python) compare on the same hardware?

All three implementations use the same algorithm (i-j-k triple loop), the same
numeric type (64-bit IEEE 754 double), the same flat row-major memory layout and
the same input matrices, read from a shared binary file.

---

## 1. Repository layout

```
common/
├── data_generator.py        # shared input matrices (binary, fixed seed)
└── plotting_style.md        # project-wide figure style
assignment_1/
├── README.md                # this file
├── requirements.txt         # pandas, matplotlib (analysis only)
├── benchmark_config.py      # ALL fixed experiment parameters
├── verify_correctness.py    # correctness checks (run before benchmarking)
├── run_benchmark.py         # full benchmark pipeline -> results/measurements.csv
├── plot_results.py          # statistics, figures, results_summary.md
├── src/
│   ├── c/matmul.c
│   ├── java/MatMul.java
│   └── python/matmul.py
├── build/                   # compiled programs (generated, git-ignored)
└── results/                 # measurements, figures, summary (generated)
```

---

## 2. Test environment

> All results and conclusions apply **only** to this machine and configuration.

| Item | Value | How to obtain it |
|---|---|---|
| CPU model | **TODO** | `lscpu \| grep "Model name"` |
| Physical cores / logical threads | **TODO** / **TODO** | `lscpu \| grep -E "^Core|^Socket|^CPU\(s\)"` |
| Base / max clock | **TODO** GHz / **TODO** GHz | Windows: Task Manager → Performance → CPU |
| Cache sizes (L1d / L2 / L3) | **TODO** | `lscpu \| grep cache` |
| RAM | **TODO** GB, **TODO** (type/speed) | `free -h`; Windows: Task Manager → Memory |
| Host OS | **TODO** (e.g. Windows 11 23H2) | `winver` |
| Linux environment | **TODO** (e.g. WSL 2, Ubuntu 24.04) | `wsl --version` (Windows), `lsb_release -d` |
| Kernel | **TODO** | `uname -r` |
| Power settings | **TODO** (e.g. plugged in, "Best performance") | Windows: Settings → Power |
| C compiler | **TODO** | `gcc --version` |
| Java (vendor and version) | **TODO** | `java -version` |
| Python (implementation and version) | **TODO** (e.g. CPython 3.12.3) | `python3 -VV` |
| pandas / matplotlib | **TODO** / **TODO** | `pip show pandas matplotlib` |

During the measurements: other applications closed, laptop plugged in, no other
heavy process running.

---

## 3. Fixed experiment parameters

All values are defined once in [`benchmark_config.py`](benchmark_config.py).

| Parameter | Value | Justification |
|---|---|---|
| Matrix sizes $n$ | 100, 250, 500, 1000 | Roughly ×2 steps; 1000 is the largest size where Python stays under the time budget |
| Measured repetitions | 5 per (size, language) | Minimum required; median and IQR are robust with 5 values |
| Time budget | 300 s per run | Hard limit; a run that exceeds it is recorded as `timeout` with no time |
| Input seed | 42 | Same matrices in every language and every run |
| Value range | uniform in $[-1, 1)$ | Keeps sums well-scaled for floating-point comparisons |
| C flags | `gcc -O2 -std=c11 -Wall -Wextra` | `-O2` is the standard release level. `-O0` would measure an artificially slow program; `-O3` adds auto-vectorization, which this loop (strided access to B, sequential sum) cannot use without `-ffast-math` |
| JVM flags | `-Xmx2g` | Fixed maximum heap, so results do not depend on automatic heap sizing; far above what $n=1000$ needs |
| Java warm-up | 3 untimed kernel runs before the timed run | Lets the JIT compile the kernel to native code; excluded from the measurement |
| Python | CPython, standard library only | No NumPy or other optimized library |

---

## 4. Measurement method

- **Time:** each program measures **only the multiplication kernel**, with a
  monotonic clock (`clock_gettime(CLOCK_MONOTONIC)` in C, `System.nanoTime()` in
  Java, `time.perf_counter_ns()` in Python). Loading the input, allocating the
  result, process start-up and JVM start-up are outside the timed region.
- **Memory:** peak resident set size (max RSS) of the whole process, measured by
  GNU time (`/usr/bin/time -v`, field *Maximum resident set size*). The same
  metric is used for all three languages. For Java it includes the JVM itself
  (JIT compiler, class metadata, garbage collector), not only the matrices.
- **Repetitions:** every repetition is a fresh process, so each one starts from
  the same state and its peak RSS belongs to exactly one run.
- **Statistics:** median and interquartile range (IQR = Q3 − Q1, pandas linear
  interpolation) per (size, language). Runs with status `timeout` or `error`
  are excluded from the statistics and listed separately in the summary.

### Correctness checks

Before benchmarking, `verify_correctness.py` compares the result matrix of each
language against an independent, correctly rounded reference (`math.fsum`):

| Case | Expected result |
|---|---|
| Identity: $A \times I$ | exactly $A$ |
| Zero: $A \times 0$ | exactly $0$ |
| Known 3×3 product | hand-computed integer matrix |
| Random $n = 10$ (shared input) | `math.fsum` reference |

An element passes if `|x - ref| <= ABS_TOL + REL_TOL * |ref|`, with
`REL_TOL = 1e-12` and `ABS_TOL = 100 * n^2 * 2^-53` (1.1e-12 for $n = 10$),
derived from the standard dot-product rounding-error bound. The full
justification is in the comments of the script.

### Language-specific differences (unavoidable)

| Language | Matrix storage |
|---|---|
| C | one contiguous `double` block of $n^2$ elements |
| Java | one flat `double[]` (not `double[][]`, which stores each row as a separate object) |
| Python | one flat `list` of $n^2$ elements; a list stores pointers to boxed `float` objects (24 B each), not raw doubles |

---

## 5. How to reproduce

All commands are run from the **repository root**. A Linux shell is required
(GNU time does not exist on native Windows): use WSL on Windows. From WSL, the
repository is under `/mnt/c/...`, e.g.
`cd "/mnt/c/Users/<user>/Desktop/.../Assignments"`.

### 5.1 One-time setup (Ubuntu / WSL)

```bash
sudo apt update
sudo apt install build-essential default-jdk python3 python3-venv time fonts-liberation
rm -rf ~/.cache/matplotlib   # let Matplotlib find the newly installed serif font

python3 -m venv venv
source venv/bin/activate
pip install -r assignment_1/requirements.txt
```

Requirements: gcc, JDK 11 or newer, Python 3.10 or newer, GNU time.

### 5.2 Correctness check

```bash
python3 assignment_1/verify_correctness.py
```

Builds the C and Java programs, runs all three languages on the four test
cases and prints `PASS`/`FAIL` with the largest error. It ends with
`All checks passed.` (exit code 0).

### 5.3 Full benchmark pipeline

```bash
python3 assignment_1/run_benchmark.py
```

Builds the programs, generates any missing input files in `data/`, runs
4 sizes × 3 languages × 5 repetitions and writes one row per run to
`assignment_1/results/measurements.csv`. Total duration on the test machine:
about **TODO** minutes (dominated by Python at $n = 1000$).

Optional arguments, e.g. for a quick test run:

```bash
python3 assignment_1/run_benchmark.py --sizes 100 250 --repetitions 5 --languages c java python --timeout 300
```

### 5.4 Statistics and figures

```bash
python3 assignment_1/plot_results.py
```

Reads `results/measurements.csv` and writes the figures and the summary table
(section 6). Add `--titles` to get versions with a title for slides; the paper
figures have no title because the LaTeX caption serves that role.

### 5.5 Building and running each program manually (optional)

```bash
# Shared input file (A and B for n = 500)
python3 common/data_generator.py --sizes 500 --seed 42 --output-dir data

# C
mkdir -p assignment_1/build/c
gcc -O2 -std=c11 -Wall -Wextra -o assignment_1/build/c/matmul assignment_1/src/c/matmul.c
assignment_1/build/c/matmul data/matrices_n500.bin

# Java
javac -d assignment_1/build/java assignment_1/src/java/MatMul.java
java -Xmx2g -cp assignment_1/build/java MatMul data/matrices_n500.bin --warmup 3

# Python
python3 assignment_1/src/python/matmul.py data/matrices_n500.bin
```

Each program prints one line, e.g. `n=500 time_s=0.079447 checksum=...`.
Add `--output result.bin` to save the result matrix.

---

## 6. Outputs

| File | Content |
|---|---|
| `results/measurements.csv` | Raw data (the required `results.csv`): `size, language, repetition, time_s, memory_mb, status` — one row per run |
| `results/results_summary.md` | Median and IQR of time and memory, empirical scaling exponents, excluded runs |
| `results/time_vs_size.pdf` / `.png` | Kernel time vs. $n$ (log-log) with an $O(n^3)$ reference line |
| `results/memory_vs_size.pdf` / `.png` | Peak RSS vs. $n$ (log-log) with the theoretical size of the three matrices |

Use the PDF files in the LaTeX report (vector, embedded fonts) and the PNG files
for quick viewing and slides. The figures are drawn on a 5.6 in canvas and are
meant to be included at the column width, which gives the intended printed text
sizes (8.75 pt labels, 7.5 pt ticks):

```latex
\includegraphics[width=\columnwidth]{results/time_vs_size.pdf}
```

The figure style is defined in [`common/plotting_style.md`](../common/plotting_style.md).
