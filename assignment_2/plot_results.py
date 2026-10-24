"""
plot_results.py - Analysis and figures for Assignment 2.

Pipeline:
    1. Load results/measurements.csv (written by run_benchmark.py). Only runs
       with status "ok" enter the statistics. Timeouts, errors and checksum
       mismatches have no valid measurement, so they are excluded, but they
       are counted and listed in the summary so nothing is hidden.
    2. Split the runs by experiment_part. Part A and Part B are analysed by
       separate functions and never mixed, matching the report structure.
    3. Part A - dense optimisation. Per (size, algorithm): median and IQR of
       the kernel time and of the peak memory, speed-up over the baseline,
       empirical scaling exponent, largest feasible size.
         part_a_time_vs_size     - log-log, with an O(n^3) reference line
         part_a_memory_vs_size   - log-log, with the size of the three matrices
    4. Part B - sparse storage. Per (sparsity, algorithm): median and IQR of
       the multiplication, conversion and total time (total = multiplication
       + conversion, added per run), peak memory, CSR storage and nnz(C).
         part_b_time_vs_sparsity - tiled vs CSR. The CSR conversion time is
                                   drawn stacked on top of the CSR
                                   multiplication time, and the crossover
                                   with the dense kernel is annotated.
         part_b_fill_in          - nnz(C) / n^2 vs sparsity, with the
                                   random-matrix model 1 - (1 - d^2)^n
    5. results_summary.md - all statistics as Markdown tables, with Part A
       and Part B in separate sections.

Crossover (see find_crossover): the sparsity at which the CSR median falls
below the dense tiled median and stays below for every higher level,
interpolated between the two measured levels around it.

Style: every visual choice follows common/plotting_style.md and is applied
in ONE place, apply_plot_style(). Figures are drawn on a 5.6 in canvas and
meant to be included at \\columnwidth (3.5 in) in LaTeX, i.e. scaled by 0.625.

Run (from anywhere):
    python3 assignment_2/plot_results.py               # paper figures (no titles)
    python3 assignment_2/plot_results.py --titles      # with titles, for slides
    python3 assignment_2/plot_results.py --input other.csv --output-dir somewhere/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # render to files only; no window/display is needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

import benchmark_config as config  # noqa: E402

# ---------------------------------------------------------------------------
# Style constants (see common/plotting_style.md)
# ---------------------------------------------------------------------------

# Fixed order and color per algorithm: color follows the series, never its
# rank, and stays the same in every figure (tiled is crimson in Part A and
# Part B). The baseline is the Assignment 1 C kernel, so it keeps the
# Assignment 1 C color. The four colors are the validated palette of the
# style guide (amber is its 4th series color).
# "label" is used in tables; "legend" is the short name that keeps every
# one-row legend inside the 5.6 in canvas (loop orders go in the caption).
ALGORITHMS = {
    "baseline":  {"label": "Baseline (i-j-k)",  "legend": "Baseline",  "color": "#1f43b2"},   # deep blue
    "reordered": {"label": "Reordered (i-k-j)", "legend": "Reordered", "color": "#9333ea"},   # vibrant purple
    "tiled":     {"label": "Tiled",             "legend": "Tiled",     "color": "#dc143c"},   # crimson
    "csr":       {"label": "CSR",               "legend": "CSR",       "color": "#c98500"},   # amber
}
PART_A_ALGORITHMS = ["baseline", "reordered", "tiled"]

REFERENCE_COLOR = "#3d3d3d"   # charcoal
TEXT_COLOR = "#1f1f1e"
GRID_COLOR = "#c8c8c5"

FIGURE_SIZE = (5.6, 4.2)      # inches; printed at 3.5 in wide (scale 0.625)
MEDIAN_LINEWIDTH = 2.5
BAND_ALPHA = 0.22             # transparency of the IQR band
REFERENCE_LINEWIDTH = 1.75    # thinner than the data lines
REFERENCE_DASHES = (0, (5, 6))  # "loosely dashed": 5 pt dash, 6 pt gap
TITLE_FONT = {"size": 16, "weight": "bold"}   # slides only (--titles)
PNG_DPI = 300

# Extension of the style guide for Part B, where one series (CSR) has two
# stacked components. The total keeps the normal median line; the
# multiplication-only component is a thinner, densely dashed line in the
# same color; the conversion between them is hatched, so it can never be
# confused with an IQR band (which is a flat, transparent fill).
COMPONENT_LINEWIDTH = 1.75
COMPONENT_DASHES = (0, (3, 2))
STACK_HATCH = "////"

BYTES_PER_MB = 1024 * 1024
BYTES_PER_DOUBLE = 8


def apply_plot_style() -> None:
    """Configure Matplotlib globally to match common/plotting_style.md."""
    plt.rcParams.update({
        # Typography: large on the canvas, 8.75 / 7.5 pt once scaled into the column.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.labelpad": 6,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,

        # Clean white background, despined axes.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": TEXT_COLOR,
        "axes.linewidth": 1.0,

        # Outward ticks.
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,

        # Subtle grid: dashed, light, major ticks only, behind the data.
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,

        # Thick, smooth median lines without markers.
        "lines.linewidth": MEDIAN_LINEWIDTH,
        "lines.marker": "None",
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",

        # Hatching (the stacked conversion band in Part B).
        "hatch.linewidth": 1.0,

        # Frameless legend.
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.columnspacing": 1.6,

        # Export: embedded TrueType fonts in the PDF (IEEE rejects Type 3 fonts),
        # 300 dpi PNG on a white background.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": PNG_DPI,
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Data loading and statistics
# ---------------------------------------------------------------------------

def load_measurements(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the raw CSV and split it into successful and failed runs.
    Returns (ok_runs, failed_runs).

    Empty cells (metrics that do not apply, e.g. the conversion time of a
    dense kernel) are read as NaN. total_time_s adds the conversion time to
    the multiplication time PER RUN, counting a missing conversion as 0
    (a dense kernel has no conversion step).
    """
    df = pd.read_csv(path)
    ok_runs = df[df["status"] == "ok"].copy()
    failed_runs = df[df["status"] != "ok"].copy()   # "timeout", "error" or "mismatch"
    ok_runs["total_time_s"] = ok_runs["time_s"] + ok_runs["conversion_time_s"].fillna(0.0)
    return ok_runs, failed_runs


def summarize(runs: pd.DataFrame, keys: list[str], metrics: dict[str, str]) -> pd.DataFrame:
    """
    Median, first/third quartile and IQR of each metric per group.
    `metrics` maps an output prefix to a CSV column, e.g. {"time": "time_s"}
    produces time_median, time_q1, time_q3 and time_iqr.

    Quartiles use pandas' default linear interpolation. With 5 repetitions,
    Q1 and Q3 are exactly the 2nd and 4th smallest values, so the IQR is
    the spread of the middle three runs - robust to one slow outlier run.
    """
    grouped = runs.groupby(keys)
    columns = {"runs": grouped["time_s"].count()}
    for prefix, column in metrics.items():
        columns[f"{prefix}_median"] = grouped[column].median()
        columns[f"{prefix}_q1"] = grouped[column].quantile(0.25)
        columns[f"{prefix}_q3"] = grouped[column].quantile(0.75)
    summary = pd.DataFrame(columns).reset_index()

    for prefix in metrics:
        summary[f"{prefix}_iqr"] = summary[f"{prefix}_q3"] - summary[f"{prefix}_q1"]
    return summary


def summarize_part_a(runs: pd.DataFrame) -> pd.DataFrame:
    """Part A statistics per (size, algorithm), plus the speed-up over the baseline."""
    summary = summarize(runs, ["size", "algorithm"],
                        {"time": "time_s", "memory": "memory_mb"})

    # Speed-up = median baseline time / median time, at the same size.
    baseline = summary[summary["algorithm"] == "baseline"].set_index("size")["time_median"]
    summary["speedup"] = summary["size"].map(baseline) / summary["time_median"]
    return summary


def summarize_part_b(runs: pd.DataFrame) -> pd.DataFrame:
    """Part B statistics per (sparsity, algorithm); nnz_C and csr_bytes are deterministic."""
    summary = summarize(runs, ["size", "sparsity", "algorithm"],
                        {"time": "time_s", "conversion": "conversion_time_s",
                         "total": "total_time_s", "memory": "memory_mb"})
    grouped = runs.groupby(["size", "sparsity", "algorithm"])
    extra = pd.DataFrame({"nnz_C": grouped["nnz_C"].median(),
                          "csr_bytes": grouped["csr_bytes"].median()}).reset_index()
    return summary.merge(extra, on=["size", "sparsity", "algorithm"])


def scaling_exponent(sizes: np.ndarray, times: np.ndarray) -> float:
    """
    Slope of a straight-line fit of log(time) against log(n).

    If time = k * n^b, then log(time) = log(k) + b * log(n), so the slope is
    the empirical exponent b. The O(n^3) model predicts b = 3; a larger value
    means the kernel slows down faster than the operation count alone
    explains (typically cache misses once the matrices no longer fit in cache).
    """
    slope, _intercept = np.polyfit(np.log(sizes), np.log(times), deg=1)
    return slope


def find_crossover(sparsity: np.ndarray, slower_at_start: np.ndarray,
                   faster_at_end: np.ndarray) -> float | None:
    """
    Sparsity at which the curve `slower_at_start` drops below `faster_at_end`
    for good, e.g. CSR time vs dense tiled time.

      1. r(s) = log(slower_at_start(s) / faster_at_end(s)) at every measured
         level: r > 0 means the first curve is still slower.
      2. Find the lowest measured level from which r < 0 at that level AND at
         every higher level, so one noisy point at low sparsity cannot
         create a false crossover.
      3. Interpolate linearly in r between that level and the one below it.
         Working with the log ratio follows the steep, roughly exponential
         drop of the CSR time better than interpolating raw seconds.

    Returns None if there is no crossing inside the measured range (the
    first curve is never below, or already below at the lowest level).
    """
    log_ratio = np.log(slower_at_start / faster_at_end)
    below = log_ratio < 0
    if not below[-1]:
        return None

    first = len(below) - 1
    while first > 0 and below[first - 1]:
        first -= 1
    if first == 0:
        return None

    s0, s1 = sparsity[first - 1], sparsity[first]
    r0, r1 = log_ratio[first - 1], log_ratio[first]
    return s0 + (s1 - s0) * r0 / (r0 - r1)


def describe_crossover(value: float | None, sparsity: np.ndarray,
                       slower_at_start: np.ndarray, faster_at_end: np.ndarray) -> str:
    """Text for the summary: the crossover, or why there is none in the measured range."""
    if value is not None:
        return f"s* = {value:.3f}"
    if slower_at_start[-1] < faster_at_end[-1]:
        # Below at the highest level but no crossing -> below at every level.
        return f"no crossing: already below at the lowest measured sparsity ({sparsity[0]:g})"
    return f"no crossing: still above at the highest measured sparsity ({sparsity[-1]:g})"


def fill_in_model(sparsity: np.ndarray, n: int) -> np.ndarray:
    """
    Expected density of C = A x B for random A and B with density d = 1 - s.

    C[i][j] is zero only if all n products A[i][k] * B[k][j] are zero. Each
    product is non-zero with probability d^2, so
        P(C[i][j] != 0) = 1 - (1 - d^2)^n.
    Computed as -expm1(n * log1p(-d^2)) to stay accurate when d^2 is tiny.
    (Our generator places an exact number of zeros, not independent ones,
    but for large n the difference is negligible.)
    """
    d_squared = (1.0 - sparsity) ** 2
    # At s = 0, log1p(-1) = -inf and the formula correctly gives 1.0;
    # silence NumPy's divide-by-zero warning for that exact case.
    with np.errstate(divide="ignore"):
        return -np.expm1(n * np.log1p(-d_squared))


# ---------------------------------------------------------------------------
# Plotting building blocks
# ---------------------------------------------------------------------------

def plot_median_with_iqr(ax, x, median, q1, q3, color: str, label: str) -> None:
    """
    One shaded band (Q1..Q3) under one thick median line.
    The band gets a lower zorder so every median line stays on top of it.
    """
    ax.fill_between(x, q1, q3, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, median, color=color, label=label, zorder=3)


def plot_reference(ax, x, values, label: str) -> None:
    """Draw a theoretical curve: charcoal, loosely dashed, thinner than the data."""
    ax.plot(x, values, color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH,
            linestyle=REFERENCE_DASHES, label=label, zorder=2.5)


def format_size_axis(ax, sizes: np.ndarray) -> None:
    """Log x-axis with a tick at every measured size, shown as plain numbers."""
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())   # hide the 200, 300, ... minor ticks
    ax.set_xlabel("Matrix size $n$")


def format_sparsity_axis(ax) -> None:
    """Linear x-axis from 0 to 1 (fraction of zero entries in A and B)."""
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_xlabel("Sparsity $s$ (fraction of zeros in $A$ and $B$)")


def add_legend(fig: plt.Figure, ax, title: str | None = None,
               columnspacing: float | None = None) -> None:
    """
    One-row, frameless legend ABOVE the plot area, so it can never cover data.
    "outside upper center" lets the constrained layout reserve room for it.

    The optional figure title (slides only) is drawn as the legend's title:
    it then sits directly above the legend row, and the layout engine keeps
    the two from overlapping (a separate suptitle would collide with it).
    `columnspacing` overrides the default gap between entries (rcParams)
    for a legend that would otherwise not fit the canvas width.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(handles),
               title=title, title_fontproperties=TITLE_FONT, columnspacing=columnspacing)


def series(summary: pd.DataFrame, algorithm: str, x_column: str) -> pd.DataFrame:
    """Rows of one algorithm, sorted along the x-axis."""
    return summary[summary["algorithm"] == algorithm].sort_values(x_column)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save the same figure as vector PDF (for LaTeX) and 300-dpi PNG."""
    for extension in ("pdf", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part A figures - dense optimisation
# ---------------------------------------------------------------------------

def plot_part_a_time(summary: pd.DataFrame, title: str | None) -> plt.Figure:
    """Kernel time vs. matrix size, log-log, with an O(n^3) reference."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    for algorithm in PART_A_ALGORITHMS:
        data = series(summary, algorithm, "size")
        if not data.empty:
            plot_median_with_iqr(ax, data["size"], data["time_median"], data["time_q1"],
                                 data["time_q3"], ALGORITHMS[algorithm]["color"],
                                 ALGORITHMS[algorithm]["legend"])

    # O(n^3) reference: on a log-log plot a straight line with slope 3. Only
    # the SLOPE carries meaning; it is anchored at the geometric mean of the
    # fastest and slowest medians at the smallest size, so it runs between
    # the curves without hiding any of them.
    sizes = np.sort(summary["size"].unique())
    dense_sizes = np.geomspace(sizes[0], sizes[-1], 100)
    smallest = summary[summary["size"] == sizes[0]]
    anchor = np.sqrt(smallest["time_median"].min() * smallest["time_median"].max())
    plot_reference(ax, dense_sizes, anchor * (dense_sizes / sizes[0]) ** 3, r"$O(n^3)$")

    format_size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("Kernel time (s)")
    add_legend(fig, ax, title)
    return fig


def plot_part_a_memory(summary: pd.DataFrame, title: str | None) -> plt.Figure:
    """
    Peak memory vs. matrix size, log-log, with the size of the three matrices.

    All three kernels allocate exactly the same arrays (A, B, C), so their
    curves are expected to coincide. To keep all three visible, they are
    drawn from widest (baseline) to narrowest (tiled): where they overlap,
    the result looks like one line with colored edges.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    widths = {"baseline": 3.0 * MEDIAN_LINEWIDTH,
              "reordered": 2.0 * MEDIAN_LINEWIDTH,
              "tiled": MEDIAN_LINEWIDTH}
    for algorithm in PART_A_ALGORITHMS:
        data = series(summary, algorithm, "size")
        if data.empty:
            continue
        color = ALGORITHMS[algorithm]["color"]
        ax.fill_between(data["size"], data["memory_q1"], data["memory_q3"],
                        color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(data["size"], data["memory_median"], color=color,
                linewidth=widths[algorithm], label=ALGORITHMS[algorithm]["legend"], zorder=3)

    # Theoretical minimum: three n x n matrices of doubles (24 n^2 bytes).
    # Anything above this line is process overhead (C runtime, page tables).
    sizes = np.sort(summary["size"].unique())
    dense_sizes = np.geomspace(sizes[0], sizes[-1], 100)
    matrix_mb = 3 * dense_sizes ** 2 * BYTES_PER_DOUBLE / BYTES_PER_MB
    plot_reference(ax, dense_sizes, matrix_mb, r"$24n^2$ bytes")

    format_size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_ylabel("Peak memory, RSS (MB)")
    add_legend(fig, ax, title)
    return fig


# ---------------------------------------------------------------------------
# Part B figures - sparse storage
# ---------------------------------------------------------------------------

def plot_part_b_time(summary: pd.DataFrame, crossover: float | None,
                     title: str | None) -> plt.Figure:
    """
    Time vs. sparsity: dense tiled vs CSR, with the CSR conversion stacked.

    - Tiled (dense): kernel time, median + IQR band. It does not depend on
      sparsity, because a dense kernel multiplies zeros like any other value.
    - CSR kernel: thinner dashed amber line (multiplication only, time_s).
    - CSR conversion: hatched band stacked ON TOP of the multiplication,
      reaching up to the CSR total. On the log axis its height shows the
      conversion share: it is as tall as the gap between 1 s and 2 s once
      conversion costs as much as the multiplication itself.
    - CSR total = multiplication + conversion: solid amber line + IQR band.
      This is the fair line to compare with the dense kernel, because a CSR
      program fed with dense input must pay for the conversion.
    The crossover with the dense kernel (CSR total vs tiled) is marked with
    a vertical reference line.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    tiled = series(summary, "tiled", "sparsity")
    csr = series(summary, "csr", "sparsity")
    amber = ALGORITHMS["csr"]["color"]

    if not tiled.empty:
        plot_median_with_iqr(ax, tiled["sparsity"], tiled["time_median"], tiled["time_q1"],
                             tiled["time_q3"], ALGORITHMS["tiled"]["color"], "Tiled")
    if not csr.empty:
        plot_median_with_iqr(ax, csr["sparsity"], csr["total_median"], csr["total_q1"],
                             csr["total_q3"], amber, "CSR total")
        ax.plot(csr["sparsity"], csr["time_median"], color=amber,
                linewidth=COMPONENT_LINEWIDTH, linestyle=COMPONENT_DASHES,
                label="CSR kernel", zorder=3)
        ax.fill_between(csr["sparsity"], csr["time_median"], csr["total_median"],
                        facecolor="none", edgecolor=amber, hatch=STACK_HATCH,
                        linewidth=0, label="Conversion", zorder=2.2)

    if crossover is not None:
        ax.axvline(crossover, color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH,
                   linestyle=REFERENCE_DASHES, zorder=2.5)
        # Label at the lower left of the line: at low sparsity both curves
        # are at their slowest (top of the plot), so this corner is empty.
        ax.text(crossover - 0.025, 0.05, f"crossover\n$s^* \\approx {crossover:.2f}$",
                transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                fontsize=12, color=REFERENCE_COLOR)

    format_sparsity_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Time (s)")
    # Four entries, three with two-word labels: a slightly tighter gap
    # (1.2 instead of 1.6 em) keeps the single legend row inside the canvas.
    add_legend(fig, ax, title, columnspacing=1.2)
    return fig


def plot_part_b_fill_in(summary: pd.DataFrame, n: int, title: str | None) -> plt.Figure:
    """
    Output density nnz(C) / n^2 vs. sparsity, with two references:
      - the random-matrix model 1 - (1 - d^2)^n (see fill_in_model), and
      - the input density d = 1 - s, so the gap shows the fill-in: C is
        denser than A and B whenever the measured line is above this one.
    Log y-axis: at high sparsity the density spans several decades.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    csr = series(summary, "csr", "sparsity")
    density = csr["nnz_C"] / n ** 2
    ax.plot(csr["sparsity"], density, color=ALGORITHMS["csr"]["color"],
            label=r"Measured $C$", zorder=3)

    # Dense grid for the smooth model curves, finer close to s = 1 where
    # they change fastest.
    s_max = csr["sparsity"].max()
    grid = np.unique(np.concatenate([np.linspace(0.0, min(0.99, s_max), 400),
                                     np.linspace(min(0.99, s_max), s_max, 400)]))
    plot_reference(ax, grid, fill_in_model(grid, n), r"Model $1-(1-d^2)^n$")
    ax.plot(grid, 1.0 - grid, color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH,
            linestyle=(0, (1, 3)), label=r"Inputs, $d=1-s$", zorder=2.5)

    format_sparsity_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel(r"Density of $C$, $\mathrm{nnz}(C)/n^2$")
    add_legend(fig, ax, title)
    return fig


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def markdown_table(header: list[str], rows: list[list[str]], text_columns: int = 1) -> list[str]:
    """
    Build a Markdown table as a list of lines (no extra dependency needed).
    The first `text_columns` columns are left-aligned, numbers right-aligned.
    """
    alignment = [":---" if i < text_columns else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(alignment) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def fmt(value, spec: str) -> str:
    """Format a number, or '-' for a metric that does not apply (NaN)."""
    return "-" if pd.isna(value) else format(value, spec)


def ordered(summary: pd.DataFrame, x_column: str) -> pd.DataFrame:
    """Sort by the x column, then by the fixed algorithm order."""
    order = list(ALGORITHMS)
    return (summary.assign(order=summary["algorithm"].map(order.index))
                   .sort_values([x_column, "order"]))


def part_a_markdown(summary: pd.DataFrame, failed_runs: pd.DataFrame) -> list[str]:
    """Part A section of results_summary.md."""
    lines = ["## Part A - Dense optimisation (sparsity 0)", "",
             "Speed-up = median baseline time / median time at the same size.", ""]
    rows = []
    for _, row in ordered(summary, "size").iterrows():
        rows.append([str(row["size"]), ALGORITHMS[row["algorithm"]]["label"], str(row["runs"]),
                     f"{row['time_median']:.4g}", f"{row['time_iqr']:.2g}",
                     f"{row['speedup']:.2f}x" if not pd.isna(row["speedup"]) else "-",
                     f"{row['memory_median']:.2f}", f"{row['memory_iqr']:.2f}"])
    lines += markdown_table(["Size n", "Algorithm", "Runs", "Median time (s)", "IQR time (s)",
                             "Speed-up", "Median memory (MB)", "IQR memory (MB)"],
                            rows, text_columns=2)

    lines += ["", "### Empirical scaling exponent", "",
              "Slope of a least-squares fit of log(median time) vs. log(n). "
              "The O(n^3) model predicts 3.", ""]
    rows = []
    for algorithm in PART_A_ALGORITHMS:
        data = series(summary, algorithm, "size")
        if len(data) >= 2:
            exponent = scaling_exponent(data["size"].to_numpy(), data["time_median"].to_numpy())
            rows.append([ALGORITHMS[algorithm]["label"], f"{exponent:.2f}",
                         f"{data['size'].min()}-{data['size'].max()}"])
    lines += markdown_table(["Algorithm", "Exponent", "Sizes used"], rows)

    lines += ["", "### Largest feasible size", "",
              "Largest n with successful runs within the time budget "
              f"({config.TIMEOUT_S} s per run unless overridden).", ""]
    rows = []
    part_a_failed = failed_runs[failed_runs["experiment_part"] == "A"]
    for algorithm in PART_A_ALGORITHMS:
        data = series(summary, algorithm, "size")
        if data.empty:
            continue
        timeouts = part_a_failed[(part_a_failed["algorithm"] == algorithm)
                                 & (part_a_failed["status"] == "timeout")]
        timeout_sizes = ", ".join(str(n) for n in sorted(timeouts["size"].unique())) or "none"
        rows.append([ALGORITHMS[algorithm]["label"], str(data["size"].max()), timeout_sizes])
    lines += markdown_table(["Algorithm", "Largest n", "Timed out at n"], rows)
    return lines


def part_b_markdown(summary: pd.DataFrame, n: int, crossover_text: dict[str, str]) -> list[str]:
    """Part B section of results_summary.md."""
    lines = [f"## Part B - Sparse storage (n = {n})", "",
             "Multiply = kernel time (time_s). Conversion = dense -> CSR for A and B. "
             "Total = multiply + conversion, added per run before taking the median. "
             "Speed-up = median tiled time / median CSR total at the same sparsity.", ""]

    # B1: time
    tiled_time = series(summary, "tiled", "sparsity").set_index("sparsity")["time_median"]
    rows = []
    for _, row in ordered(summary, "sparsity").iterrows():
        speedup = "-"
        if row["algorithm"] == "csr" and row["sparsity"] in tiled_time.index:
            speedup = f"{tiled_time[row['sparsity']] / row['total_median']:.3g}x"
        rows.append([f"{row['sparsity']:g}", ALGORITHMS[row["algorithm"]]["label"],
                     str(row["runs"]),
                     fmt(row["time_median"], ".4g"), fmt(row["time_iqr"], ".2g"),
                     fmt(row["conversion_median"], ".4g"), fmt(row["conversion_iqr"], ".2g"),
                     fmt(row["total_median"], ".4g"), fmt(row["total_iqr"], ".2g"), speedup])
    lines += ["### Time", ""]
    lines += markdown_table(["Sparsity", "Algorithm", "Runs", "Median multiply (s)",
                             "IQR multiply (s)", "Median conversion (s)", "IQR conversion (s)",
                             "Median total (s)", "IQR total (s)", "Speed-up vs. tiled"],
                            rows, text_columns=2)

    # B2: crossover points
    lines += ["", "### Crossover points", "",
              "Interpolated sparsity s* from which the first curve stays below the second "
              "(log-ratio interpolation between the two measured levels around it).", ""]
    lines += markdown_table(["Comparison", "Crossover"], [
        ["CSR total (multiply + conversion) faster than tiled", crossover_text["total"]],
        ["CSR multiply alone faster than tiled", crossover_text["multiply"]],
        ["CSR conversion costs more than the CSR multiply", crossover_text["conversion"]],
    ])

    # B3: fill-in and memory
    csr = series(summary, "csr", "sparsity")
    tiled_memory = series(summary, "tiled", "sparsity").set_index("sparsity")["memory_median"]
    dense_storage_mb = 3 * n * n * BYTES_PER_DOUBLE / BYTES_PER_MB
    rows = []
    for _, row in csr.iterrows():
        density = row["nnz_C"] / n ** 2
        model = fill_in_model(np.array([row["sparsity"]]), n)[0]
        rows.append([f"{row['sparsity']:g}", f"{row['nnz_C']:.0f}", f"{density:.5g}",
                     f"{model:.5g}", f"{density / (1 - row['sparsity']):.3g}"
                     if row["sparsity"] < 1 else "-",
                     fmt(tiled_memory.get(row["sparsity"]), ".2f"),
                     f"{row['memory_median']:.2f}",
                     f"{row['csr_bytes'] / BYTES_PER_MB:.2f}", f"{dense_storage_mb:.2f}"])
    lines += ["", "### Fill-in and memory", "",
              "Density of C = nnz(C) / n^2; model = 1 - (1 - d^2)^n with d = 1 - s. "
              "Fill-in ratio = density of C / density of the inputs. "
              "Peak RSS of csr includes the dense input buffers (the shared inputs are dense), "
              "so CSR storage (arrays of A, B and C) is listed separately; dense storage = 3 n^2 doubles.",
              ""]
    lines += markdown_table(["Sparsity", "nnz(C)", "Density of C", "Model density",
                             "Fill-in ratio", "Tiled RSS (MB)", "CSR RSS (MB)",
                             "CSR storage (MB)", "Dense storage (MB)"], rows)
    return lines


def excluded_markdown(failed_runs: pd.DataFrame) -> list[str]:
    """Runs without a valid measurement: reported, never silently dropped."""
    lines = ["## Excluded runs", ""]
    if failed_runs.empty:
        lines.append("None - every run finished successfully within the time budget.")
        return lines
    counts = (failed_runs.groupby(["experiment_part", "size", "sparsity", "algorithm", "status"])
              .size().reset_index(name="runs"))
    rows = [[r["experiment_part"], str(r["size"]), f"{r['sparsity']:g}",
             ALGORITHMS[r["algorithm"]]["label"], r["status"], str(r["runs"])]
            for _, r in counts.iterrows()]
    lines += markdown_table(["Part", "Size n", "Sparsity", "Algorithm", "Status", "Runs"],
                            rows, text_columns=5)
    return lines


def key_results_markdown(summary_a: pd.DataFrame | None, summary_b: pd.DataFrame | None,
                         crossover_text: dict[str, str]) -> list[str]:
    """Headline numbers (also the ones quoted in the README)."""
    lines = ["## Key results", ""]
    if summary_a is not None:
        largest = summary_a.groupby("size")["algorithm"].nunique()
        common = largest[largest == len(PART_A_ALGORITHMS)].index
        if len(common) > 0:
            n = common.max()
            at_n = summary_a[summary_a["size"] == n].set_index("algorithm")
            lines.append(f"- Part A, n = {n}: reordered {at_n.loc['reordered', 'speedup']:.2f}x and "
                         f"tiled {at_n.loc['tiled', 'speedup']:.2f}x faster than the baseline "
                         f"(medians: baseline {at_n.loc['baseline', 'time_median']:.4g} s, "
                         f"reordered {at_n.loc['reordered', 'time_median']:.4g} s, "
                         f"tiled {at_n.loc['tiled', 'time_median']:.4g} s).")
    if summary_b is not None:
        lines.append(f"- Part B: CSR total time (multiplication + conversion) vs. the dense "
                     f"tiled kernel: {crossover_text['total']}.")
    return lines + [""]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot the Assignment 2 results.")
    parser.add_argument("--input", type=Path, default=config.RESULTS_DIR / "measurements.csv")
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--titles", action="store_true",
                        help="Add titles to the figures (for slides; paper figures use captions)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    ok_runs, failed_runs = load_measurements(args.input)
    if ok_runs.empty:
        raise SystemExit(f"No successful runs in {args.input}; nothing to plot.")
    print(f"Loaded {len(ok_runs)} successful runs ({len(failed_runs)} excluded).")

    part_a = ok_runs[ok_runs["experiment_part"] == "A"]
    part_b = ok_runs[ok_runs["experiment_part"] == "B"]
    summary_a = summary_b = None
    crossovers = {"total": None, "multiply": None, "conversion": None}
    crossover_text = {key: "not available (missing tiled or csr runs)" for key in crossovers}
    sections = []

    # ---- Part A: dense optimisation ---------------------------------------
    if part_a.empty:
        print("No Part A runs found; skipping the Part A figures.")
    else:
        summary_a = summarize_part_a(part_a)
        save_figure(plot_part_a_time(summary_a, "Part A: kernel time vs. size" if args.titles else None),
                    args.output_dir, "part_a_time_vs_size")
        save_figure(plot_part_a_memory(summary_a, "Part A: peak memory vs. size" if args.titles else None),
                    args.output_dir, "part_a_memory_vs_size")
        sections += part_a_markdown(summary_a, failed_runs) + [""]

    # ---- Part B: sparse storage --------------------------------------------
    if part_b.empty:
        print("No Part B runs found; skipping the Part B figures.")
    else:
        sizes = part_b["size"].unique()
        if len(sizes) != 1:
            raise SystemExit(f"Part B should use one fixed size, found {sorted(sizes)}.")
        n = int(sizes[0])
        summary_b = summarize_part_b(part_b)

        tiled = series(summary_b, "tiled", "sparsity")
        csr = series(summary_b, "csr", "sparsity")
        # The three comparisons: (name, x values, first curve, second curve).
        comparisons = []
        if not tiled.empty and not csr.empty:
            both = csr.merge(tiled, on="sparsity", suffixes=("_csr", "_tiled"))
            s = both["sparsity"].to_numpy()
            comparisons += [("total", s, both["total_median_csr"].to_numpy(),
                             both["time_median_tiled"].to_numpy()),
                            ("multiply", s, both["time_median_csr"].to_numpy(),
                             both["time_median_tiled"].to_numpy())]
        if not csr.empty:
            comparisons.append(("conversion", csr["sparsity"].to_numpy(),
                                csr["time_median"].to_numpy(), csr["conversion_median"].to_numpy()))
        for name, s, first_curve, second_curve in comparisons:
            crossovers[name] = find_crossover(s, first_curve, second_curve)
            crossover_text[name] = describe_crossover(crossovers[name], s, first_curve, second_curve)

        save_figure(plot_part_b_time(summary_b, crossovers["total"],
                                     "Part B: time vs. sparsity" if args.titles else None),
                    args.output_dir, "part_b_time_vs_sparsity")
        if not csr.empty:
            save_figure(plot_part_b_fill_in(summary_b, n,
                                            "Part B: fill-in" if args.titles else None),
                        args.output_dir, "part_b_fill_in")
        sections += part_b_markdown(summary_b, n, crossover_text) + [""]

    # ---- Summary ------------------------------------------------------------
    lines = ["# Assignment 2 - Results Summary", "",
             "Generated by `plot_results.py` from `results/measurements.csv`. "
             "Statistics use successful runs only. Time = kernel only (monotonic clock); "
             "memory = peak resident set size of the whole process (GNU time); IQR = Q3 - Q1.", ""]
    lines += key_results_markdown(summary_a, summary_b, crossover_text)
    lines += sections
    lines += excluded_markdown(failed_runs)

    path = args.output_dir / "results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
