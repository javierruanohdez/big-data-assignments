<div align="center">

# Optimised Matrix Multiplication and Sparse Matrices

**Cache-aware dense kernels and CSR sparse multiplication in C, measured on identical inputs**

![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[Overview](#overview) •
[Prerequisites](#prerequisites--dependencies) •
[Quick Start](#quick-start) •
[Protocol](#experimental-protocol--parameters) •
[Environment](#reproducibility--test-environment) •
[Layout](#repository-layout)

</div>

---

## Overview

> **Research questions.**
> **Part A (dense optimisation):** how much do loop reordering (i-k-j) and cache
> blocking (tiling) speed up the textbook $O(n^3)$ dense multiplication, and why?
> **Part B (sparse storage):** from which sparsity does a Compressed Sparse Row (CSR)
> multiplication beat the best dense kernel, once the conversion and indexing
> overheads are counted?

This project implements three dense kernels and one sparse kernel in C and measures
them under a common, fully automated protocol:

- **Two separate experiments.** Part A varies the matrix size $n$ on dense inputs;
  Part B fixes $n = 1000$ and varies the sparsity. The two parts are run, stored and
  analysed separately, so the effect of dense optimisations never mixes with the
  effect of sparse storage.
- **Fair by construction.** All four kernels are sequential C compiled with the same
  flags, use 64-bit IEEE 754 doubles and read the **same binary input files**. Sparse
  inputs are the dense matrices with entries removed, and all four kernels produce
  **bit-identical results**, so only the memory access pattern differs between them.
- **Validated first.** Every kernel is checked against a correctly rounded reference
  on dense and sparse cases, and the CSR kernel's non-zero counts are checked exactly.
- **Measured carefully.** Only kernels are timed, with a monotonic clock. The dense to
  CSR conversion is timed **separately** from the CSR multiplication. Each repetition
  runs in a fresh process, and peak memory is measured with GNU `time`.
- **Analysed by script.** Medians, interquartile ranges, speed-ups, scaling exponents,
  crossover points and publication-ready plots are generated straight from the raw data.

### Results at a glance

<p align="center">
  <img src="results/part_a_time_vs_size.png" alt="Part A: kernel time vs. matrix size" width="48%">
  &nbsp;
  <img src="results/part_a_memory_vs_size.png" alt="Part A: peak memory vs. matrix size" width="48%">
</p>
<p align="center">
  <img src="results/part_b_time_vs_sparsity.png" alt="Part B: time vs. sparsity, dense tiled vs. CSR" width="48%">
  &nbsp;
  <img src="results/part_b_fill_in.png" alt="Part B: density of the result vs. sparsity" width="48%">
</p>

| Part A: median kernel time | Baseline (i-j-k) | Reordered (i-k-j) | Tiled ($b = 64$) |
|---|---:|---:|---:|
| $n = 1000$ | 1.64 s | 0.506 s (3.2×) | 0.428 s (3.8×) |
| $n = 2000$ | 15.6 s | 3.47 s (4.5×) | 3.55 s (4.4×) |
| Fitted time exponent (model: 3) | 3.48 | 2.84 | 2.90 |

| Part B: CSR vs. dense tiled, $n = 1000$ | Value |
|---|---:|
| Crossover sparsity, CSR total (multiply + conversion) | $s^* \approx 0.44$ |
| Crossover sparsity, CSR multiply alone | $s^* \approx 0.43$ |
| CSR speed-up at $s = 0.9$ / $0.99$ / $0.999$ | 11.9× / 74× / 132× |
| Conversion costs more than the CSR multiply from | $s \approx 0.97$ |
| Density of the result $C$ at $s = 0.95$ (fill-in) | 91.7 % |

Speed-ups in parentheses are relative to the baseline at the same size. Full statistics
are in [`results/results_summary.md`](results/results_summary.md). These numbers describe
one machine and configuration ([Test environment](#reproducibility--test-environment)):
the crossover in particular depends on the relative speed of the dense kernel, so it
moves from machine to machine.

---

## Prerequisites & Dependencies

A **Linux shell** is required: GNU `time` does not exist on native Windows, so use
WSL on Windows (see the notes under [Quick Start](#quick-start)).

| Tool | Minimum version | Used in this study | Purpose | Verify with |
|---|---|---|---|---|
| **Python** (CPython) | 3.10 | [TODO] | Scripts and analysis | `python3 --version` |
| **GCC** | C11 support | [TODO] | Compiles the C kernels | `gcc --version` |
| **GNU time** | any | system package | Measures peak memory (max RSS) | `/usr/bin/time --version` |
| **Git** | any | system package | Clones the repository | `git --version` |
| **pandas / Matplotlib** | 2.0 / 3.7 | [TODO] | Statistics and figures (`requirements.txt`) | `pip show pandas matplotlib` |
| *LaTeX (optional)* | TeX Live with `IEEEtran` | — | Builds the PDF report | `pdflatex --version` |

The benchmark itself (kernels, data generator, harness, correctness checks) uses **only
the Python standard library**; pandas and Matplotlib are needed only for the analysis step.

---

## Quick Start

All commands run from the **repository root**.

### 1. Set up the environment (Ubuntu / WSL)

```bash
git clone https://github.com/javierruanohdez/big-data-assignments.git
cd big-data-assignments

sudo apt update
sudo apt install build-essential python3 python3-venv time fonts-liberation
rm -rf ~/.cache/matplotlib   # let Matplotlib find the newly installed serif font

python3 -m venv venv
source venv/bin/activate
pip install -r assignment_2/requirements.txt
```

### 2. Check correctness

```bash
python3 assignment_2/verify_correctness.py
```

This builds both C programs, runs the four kernels on eight dense and sparse test cases
and prints `PASS`/`FAIL` with the largest error, the CSR non-zero counts and whether each
result is bit-identical to the baseline. It ends with `All checks passed.` (exit code 0).

### 3. Run the full benchmark pipeline

```bash
python3 assignment_2/run_benchmark.py
```

This builds the programs, generates any missing input files in `data/` (about 250 MB),
and runs both experiments, writing one row per run to
`assignment_2/results/measurements.csv`:

- **Part A:** 5 sizes × 3 dense kernels × 5 repetitions = 75 runs
- **Part B:** 12 sparsity levels × 2 kernels (tiled, CSR) × 5 repetitions = 120 runs

Total duration on the test machine: roughly **4 minutes**. This is an estimate: the
measured kernels and conversions alone add up to 2.8 minutes (most of it the baseline
at $n = 2000$), and generating the inputs the first time adds about half a minute.

Optional arguments, e.g. for a quick test run (the CSV is overwritten, so write it
elsewhere):

```bash
python3 assignment_2/run_benchmark.py --parts B --sparsities 0.5 0.9 --repetitions 3 \
    --output assignment_2/results/quick_test.csv
python3 assignment_2/run_benchmark.py --part-a-sizes 100 250 --part-b-size 500 --timeout 300
```

### 4. Generate statistics and figures

```bash
python3 assignment_2/plot_results.py            # figures for the paper (no titles)
python3 assignment_2/plot_results.py --titles   # same figures with titles, for slides
```

This reads `results/measurements.csv` and writes the four figures (PDF + PNG) and
`results/results_summary.md` (see [Outputs](#outputs)).

### 5. Build the report (optional)

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-publishers latexmk
cd assignment_2/report && latexmk -pdf main.tex
```

<details>
<summary><b>Windows / WSL notes</b></summary>

<br>

- **Install WSL** from PowerShell with `wsl --install -d Ubuntu`, then run every
  command above inside the Ubuntu terminal.
- **Repository on the Windows drive:** it is reachable under `/mnt/c/...`, e.g.
  `cd "/mnt/c/Users/<user>/Desktop/.../big-data-assignments"`. Paths with spaces
  or accents need quotes. File access through `/mnt/c` is slower, but file loading
  is outside every timed region.
- **Memory:** by default WSL 2 can use about half of the host RAM. Change it
  with `memory=` in `%UserProfile%\.wslconfig` if needed.
- **Power:** keep the laptop plugged in and use the *Best performance* power mode
  while benchmarking, and avoid other heavy work on the machine during the run.
- **Line endings:** edit configuration files (e.g. `.gitignore`) from WSL or an
  editor. PowerShell's `echo ... >> file` writes UTF-16 text, which Git
  cannot read.

</details>

<details>
<summary><b>Building and running each program manually</b></summary>

<br>

```bash
# Shared input files: dense n = 500, and n = 1000 with 90 % zeros
python3 common/data_generator.py --sizes 500 --seed 42 --output-dir data
python3 common/data_generator.py --sizes 1000 --sparsity 0.9 --seed 42 --output-dir data

# Build
mkdir -p assignment_2/build/c
gcc -O2 -std=c11 -Wall -Wextra -o assignment_2/build/c/dense_mult  assignment_2/src/c/dense_mult.c
gcc -O2 -std=c11 -Wall -Wextra -o assignment_2/build/c/sparse_mult assignment_2/src/c/sparse_mult.c

# Part A: the three dense kernels
assignment_2/build/c/dense_mult data/matrices_n500.bin --algo baseline
assignment_2/build/c/dense_mult data/matrices_n500.bin --algo reordered
assignment_2/build/c/dense_mult data/matrices_n500.bin --algo tiled --block-size 64

# Part B: dense tiled vs. CSR on the same sparse input
assignment_2/build/c/dense_mult  data/matrices_n1000_sp90.bin --algo tiled
assignment_2/build/c/sparse_mult data/matrices_n1000_sp90.bin
```

Each program prints one line of `key=value` pairs, e.g.

```
algo=tiled n=500 block_size=64 time_s=0.058110000 checksum=...
algo=csr n=1000 nnz_A=100000 nnz_B=100000 nnz_C=999948 csr_bytes=14423400 conversion_time_s=0.009702000 time_s=0.025760000 checksum=...
```

For the same input file, all four kernels print the same checksum. Add
`--output result.bin` to save the result matrix (both programs write the same dense format).

</details>

### Outputs

| File | Content |
|---|---|
| `results/measurements.csv` | Raw data, one row per run: `experiment_part, size, sparsity, algorithm, repetition, time_s, conversion_time_s, memory_mb, csr_bytes, nnz_C, status`. An empty cell means "does not apply" (e.g. the conversion time of a dense kernel) or "no measurement" (timeout or error) |
| `results/results_summary.md` | Key results; Part A: median and IQR of time and memory, speed-ups, scaling exponents, largest feasible $n$; Part B: multiply, conversion and total times, crossover points, fill-in and memory; excluded runs |
| `results/part_a_time_vs_size.pdf` / `.png` | Part A: kernel time vs. $n$ (log-log) with an $O(n^3)$ reference line |
| `results/part_a_memory_vs_size.pdf` / `.png` | Part A: peak RSS vs. $n$ (log-log) with the size of the three matrices ($24n^2$ bytes) |
| `results/part_b_time_vs_sparsity.pdf` / `.png` | Part B: dense tiled vs. CSR, with the CSR conversion stacked on top of the CSR multiplication and the crossover marked |
| `results/part_b_fill_in.pdf` / `.png` | Part B: density of the result, $\mathrm{nnz}(C)/n^2$, vs. sparsity, with the random-matrix model and the input density |

Use the PDF figures in LaTeX at column width (`\includegraphics[width=\columnwidth]{...}`),
which gives the intended printed text sizes. The figure style is defined in
[`common/plotting_style.md`](../common/plotting_style.md).

---

## Experimental Protocol & Parameters

### Fixed experiment parameters

All values are defined once in [`benchmark_config.py`](benchmark_config.py).

| Parameter | Value | Justification |
|---|---|---|
| Part A sizes $n$ | 100, 250, 500, 1000, 2000 | Roughly ×2 steps. At $n = 2000$, one matrix (32 MB) is far larger than a typical last-level cache, the regime the optimisations target. After a timeout, larger sizes are skipped for that kernel, so the list can be extended to find the largest feasible $n$ |
| Part A kernels | baseline, reordered, tiled | The Assignment 1 baseline plus two distinct optimisations |
| Part B size | $n = 1000$ | The dense kernel takes about 0.4 s: long enough to time reliably, short enough for 12 levels × 5 repetitions |
| Part B sparsity levels | 0, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999 | The guideline levels (50, 90, 95, 99 %) plus a finer grid around the crossover and the conversion-dominated end |
| Part B kernels | tiled vs. CSR | Tiled is the fastest dense kernel at $n = 1000$ in Part A, so CSR is compared with the best dense alternative |
| Measured repetitions | 5 per configuration | Minimum required; median and IQR are robust with 5 values |
| Time budget | 300 s per run | Hard limit; a run that exceeds it is recorded as `timeout` with no time |
| Input seed | 42 | Same matrices for every kernel and every run |
| Values | uniform in $[-1, 1)$, plus exact zeros | Keeps sums well scaled for floating-point comparisons |
| Zeros | exactly $\mathrm{round}(s \cdot n^2)$ per matrix, at random positions | Exact sparsity axis. A separate random generator chooses the positions, so the non-zero values never change, the levels are nested (the zeros at 90 % are a subset of those at 95 %) and $s = 0$ reproduces the Assignment 1 inputs bit for bit |
| Tile size $b$ | 64 | One tile = 32 KiB (about one L1 data cache); the three tiles in use (96 KiB) fit in L2 |
| C flags | `gcc -O2 -std=c11 -Wall -Wextra` | The same flags as Assignment 1, for all four kernels. `-O3` adds auto-vectorisation, which helps the stride-1 kernels much more than the baseline and would mix a compiler effect into the comparison |

### Part A: dense optimisations

All three kernels perform the same $n^3$ multiply-adds, and every $C_{ij}$ even receives
its terms in the same order ($k = 0, 1, \dots, n-1$). The results are therefore
bit-identical, and any speed difference comes from the **memory hierarchy** alone.
Matrices are flat, row-major arrays: moving along a row is the next address, moving
down a column jumps $8n$ bytes.

| Kernel | Loop order | Innermost loop | Why it is faster (or slower) |
|---|---|---|---|
| Baseline | i-j-k | walks a row of $A$ (stride 1) and a **column** of $B$ (stride $n$) | The Assignment 1 kernel. Each 64-byte cache line of $B$ delivers one useful double out of eight, and for $n \geq 512$ every step lands on a new 4 KiB page |
| Reordered | i-k-j | walks a row of $B$ and a row of $C$ (both stride 1); $A_{ik}$ stays in a register | Row $i$ of $C$ = $\sum_k A_{ik} \cdot$ (row $k$ of $B$). Every cache line is fully used and the hardware prefetcher sees two sequential streams. Remaining weakness: all of $B$ is streamed again for every row $i$ |
| Tiled | tiles $ii$-$jj$-$kk$ of $b \times b$, then i-k-j inside | row segments of one tile of $B$ and one tile of $C$ | Each tile of $B$ is reused $b$ times while it is still in cache, which cuts the memory traffic from about $n^3$ to about $2n^3/b$ elements. The $C$ tile stays in cache until it is complete |

### Part B: sparse storage (CSR, Gustavson's algorithm)

- **Format.** CSR stores only the non-zeros, row by row, in three arrays: `values`,
  `col_idx` (32-bit) and `row_ptr` (row $i$ owns positions `row_ptr[i]` to
  `row_ptr[i+1] - 1`). That is 12 bytes per non-zero plus $8(n+1)$ bytes, versus $8n^2$
  for a dense matrix.
- **Conversion** (timed as `conversion_time_s`, for $A$ and $B$ together). Two passes over
  the dense matrix: count the non-zeros of each row into `row_ptr`, allocate exactly, then
  copy. It costs $O(n^2)$ **whatever the sparsity**, because every zero must still be
  looked at.
- **Multiplication** (timed as `time_s`). Gustavson's row-by-row algorithm with a dense
  sparse accumulator: row $i$ of $C$ is the sum of the rows $k$ of $B$ selected by the
  non-zeros $A_{ik}$. The work is proportional to the number of non-zero products,
  about $d^2 n^3$ for density $d = 1 - s$ (e.g. 100× fewer multiply-adds than dense at
  $s = 0.9$), but each product pays for an indirect access and a branch. The output size is
  not known in advance; it is bounded cheaply from the row lengths, allocated once inside
  the timer, and exact cancellations to $0.0$ are dropped, so `nnz_C` is the true number
  of non-zeros.
- **Fill-in.** A non-zero $A_{ik}$ spreads a whole row of $B$ into row $i$ of $C$, so the
  result is much denser than the inputs. For random inputs,
  $P(C_{ij} \neq 0) = 1 - (1 - d^2)^n$; the fill-in figure compares this model with the
  measured $\mathrm{nnz}(C)$.
- **Crossover.** The sparsity $s^*$ from which the CSR median stays below the dense tiled
  median at every higher level, interpolated (on the log of the time ratio) between the
  two measured levels around it. It is reported for the CSR total (multiply + conversion),
  for the multiply alone, and for the point where the conversion outweighs the multiply.

### Measurement method

- **Time:** each program measures only its kernel with `clock_gettime(CLOCK_MONOTONIC)` and
  prints it with nanosecond resolution. Loading the input, allocating and pre-faulting
  the dense result, and process start-up are outside the timed regions. `sparse_mult` has
  **two separate timed regions**: the conversion of $A$ and $B$, then the multiplication.
- **Memory:** peak resident set size (max RSS) of the whole process, measured by GNU time
  (`/usr/bin/time -v`). For CSR it includes the dense input buffers read at the start,
  because the shared inputs are stored densely; `csr_bytes` therefore reports the storage
  of the CSR arrays of $A$, $B$ and $C$ separately.
- **Repetitions:** every repetition is a fresh process, so each one starts from the same
  state and its peak RSS belongs to exactly one run.
- **Statistics:** median and interquartile range (IQR = Q3 − Q1, pandas linear
  interpolation) per configuration. The CSR total is added per run before taking the
  median. Runs with status `timeout`, `error` or `mismatch` are excluded from the
  statistics and listed separately in the summary.
- **Consistency guard:** for the same input file, all kernels must print the same
  checksum; a run that disagrees is recorded as `mismatch`, so a wrong result can never
  enter the statistics.

### Correctness checks

`verify_correctness.py` compares the result of every kernel against an independent,
correctly rounded reference (`math.fsum`):

| Case | Expected result |
|---|---|
| Identity: $A \times I$ ($I$ is itself sparse) | exactly $A$ |
| Zero: $A \times 0$ ($B$ has no non-zeros) | exactly $0$, and `nnz_C = 0` |
| Known 3×3 product | hand-computed integer matrix |
| Random $n = 10$, dense | `math.fsum` reference |
| Random $n = 100$ at sparsity 0, 0.5, 0.9, 0.99 | `math.fsum` reference; $n > b$ exercises full and partial tiles, 0.99 has many empty rows |

An element passes if `|x - ref| <= ABS_TOL + REL_TOL * |ref|`, with `REL_TOL = 1e-12` and
`ABS_TOL = 100 * n^2 * 2^-53`, derived from the standard dot-product rounding-error bound.
For CSR, the printed `nnz_A`, `nnz_B` and `nnz_C` must match the actual counts exactly.
All 32 checks pass with a largest error of 1.1e-14, and every kernel's result is
bit-identical to the baseline's.

### Implementation notes

| Topic | Decision |
|---|---|
| Inlining | Every kernel is marked `noinline`. When GCC inlined the six-loop tiled kernel into `main()`, it ran out of registers and reloaded a pointer from the stack in every inner iteration, which made it about 2.5× slower |
| Dense output | Filled with NaN before the timer: the memory pages are mapped outside the timed region, and any element a kernel forgot to write would turn the checksum into `nan` |
| CSR indices | 32-bit column indices (4 bytes per non-zero instead of 8); the header check enforces $n \leq 2^{31} - 1$ |
| CSR output | Columns inside a row of $C$ are left in first-touch order (CSR does not require sorting); the checksum is still computed in row-major order, so it matches the dense kernels exactly |
| Robustness | Both programs validate the file header before allocating, check every allocation, and free all memory on every error path |

---

## Reproducibility & Test Environment

> All results and conclusions apply **only** to this machine and configuration.

| Item | Value |
|---|---|
| CPU model | [TODO] |
| Cores / logical threads | [TODO] |
| Clock | [TODO] |
| Caches | [TODO] |
| RAM | [TODO] |
| Host OS | [TODO] |
| Linux environment | [TODO] |
| Kernel | [TODO] |
| C compiler | [TODO] |
| Python (implementation and version) | [TODO] |
| pandas / Matplotlib | [TODO] |

---

## Repository Layout

```
big-data-assignments/
├── README.md
├── common/
│   ├── data_generator.py        # shared input matrices (binary, fixed seed, --sparsity)
│   └── plotting_style.md        # project-wide figure style
├── data/                        # generated .bin inputs (git-ignored)
├── assignment_1/                # basic matrix multiplication in C, Java and Python
└── assignment_2/
    ├── README.md                # this file
    ├── requirements.txt         # pandas, matplotlib, numpy (analysis only)
    ├── benchmark_config.py      # ALL fixed experiment parameters (Part A and Part B)
    ├── verify_correctness.py    # correctness checks (run before benchmarking)
    ├── run_benchmark.py         # full benchmark pipeline -> results/measurements.csv
    ├── plot_results.py          # statistics, figures, results_summary.md
    ├── src/
    │   └── c/
    │       ├── dense_mult.c     # Part A: baseline, reordered and tiled kernels
    │       └── sparse_mult.c    # Part B: dense -> CSR conversion and CSR x CSR kernel
    ├── results/                 # measurements, figures, summary (generated)
    ├── report/
    │   └── main.tex             # research report (IEEE two-column, LaTeX) [TODO]
    └── build/                   # compiled programs (generated, git-ignored)
```

---

## Author

| | |
|---|---|
| **Name** | Javier Ruano Hernández |
| **GitHub** | [@javierruanohdez](https://github.com/javierruanohdez) |
| **Course** | Big Data (ULPGC) |

<div align="center">
<sub>Built with C and Python · Measured, not guessed.</sub>
</div>
