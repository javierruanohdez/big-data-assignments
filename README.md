<div align="center">

# Big Data — Individual Assignments

**Matrix multiplication from a single core to a distributed cluster: four benchmarking studies for the Big Data course at ULPGC (2026–2027)**

![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=openjdk&logoColor=white)
![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)

[Assignments](#assignments) •
[Structure](#repository-structure) •
[Workflow](#general-workflow--reproducibility) •
[Author](#author)

</div>

---

## Assignments

The four assignments study the same problem, dense matrix multiplication $C = A \times B$,
under a different execution model each time: one core in three languages, one core with a
cache-aware memory layout, several cores sharing memory, and several processes exchanging
messages.

| Assignment | Topic & Paradigm | Core Techniques | Documentation |
|---|---|---|---|
| [**Assignment 1**](assignment_1/) | Execution time and memory scaling: sequential, cross-language (C ahead-of-time compiled, Java JIT, Python interpreted) | Identical $O(n^3)$ triple loop in three languages; growth of time and peak memory with $n$ | [README](assignment_1/README.md) · [Report](assignment_1/report/) |
| [**Assignment 2**](assignment_2/) | Dense cache optimisations and sparse matrices: sequential C | Loop reordering (`i-k-j`), cache tiling, CSR format with Gustavson's algorithm | [README](assignment_2/README.md) · [Report](assignment_2/report/) |
| [**Assignment 3**](assignment_3/) | Shared-memory parallelism and SIMD: multithreaded C | OpenMP multithreading, speedup $S(p)$ and efficiency $E(p)$, explicit vectorisation (`#pragma omp simd`) | [README](assignment_3/README.md) · [Report](assignment_3/report/) |
| [**Assignment 4**](assignment_4/) | Distributed matrix multiplication: multi-process message passing (Python, Dask) | 2D block partitioning, computation vs. communication overhead, strong and cubic weak scaling ($n_p = n_1 \cdot p^{1/3}$) | [README](assignment_4/README.md) · [Report](assignment_4/report/) |

Each assignment's README states its research question, the exact commands to reproduce it,
its results and the test environment.

---

## Repository Structure

```
big-data-assignments/
├── README.md                  # this index
├── common/
│   ├── data_generator.py      # shared binary input matrices (fixed seed, dense and sparse)
│   └── plotting_style.md      # unified figure style for every assignment
├── data/                      # generated input files (git-ignored)
├── assignment_1/              # cross-language baseline (C, Java, Python)
├── assignment_2/              # cache-aware dense kernels and CSR sparse multiplication (C)
├── assignment_3/              # OpenMP multithreading and SIMD (C)
└── assignment_4/              # distributed multiplication on a Dask cluster (Python, C baselines)
```

Every assignment folder is self-contained and follows the same layout:

```
assignment_N/
├── README.md                  # research question, reproduction steps, results, environment
├── requirements.txt           # Python packages for the harness and analysis
├── benchmark_config.py        # every fixed experiment parameter, in one place
├── verify_correctness.py      # correctness checks, run before benchmarking
├── run_benchmark.py           # automated measurement sweep -> results/measurements.csv
├── plot_results.py            # statistics, figures and results_summary.md
├── src/                       # implementations under test
├── results/                   # raw measurements, figures and summary
├── report/                    # IEEE-style LaTeX report
└── presentation/              # LaTeX slides (Assignments 1-3)
```

---

## General Workflow & Reproducibility

All four studies share one experimental methodology. Every implementation reads the same
binary input files, written once by `common/data_generator.py` (64-bit little-endian
floats, fixed seed), so differences in time come from the implementations and not from the
data. Each assignment follows the same pipeline: `verify_correctness.py` checks every
implementation against a correctly rounded reference (`math.fsum`), `run_benchmark.py`
runs every configuration five times, each in a fresh process, and `plot_results.py` reports
medians and interquartile ranges, with figures that follow `common/plotting_style.md`.
Peak memory is measured with GNU `time` (and from each process's own high-water mark for the
Dask workers of Assignment 4). The findings of each study are written up as an IEEE-style
LaTeX report.

The benchmarks need a Linux shell (Ubuntu or WSL on Windows). Exact build, verification,
execution and plotting commands, together with the hardware and software of each run, are
in each assignment's own README.

---

## Author

| | |
|---|---|
| **Name** | Javier Ruano Hernández |
| **GitHub** | [@javierruanohdez](https://github.com/javierruanohdez) |
| **Course** | Big Data (ULPGC) |

<div align="center">
<sub>Built with C, Java, Python and LaTeX · Measured, not guessed.</sub>
</div>
