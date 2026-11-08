"""
plot_results.py - Analysis and figures for Assignment 3.

Pipeline:
    1. Load results/measurements.csv (written by run_benchmark.py). Only runs
       with status "ok" enter the statistics. Timeouts, errors and checksum
       mismatches have no valid measurement, so they are excluded, but they
       are counted and listed in the summary so nothing is hidden.
    2. Per (size, algorithm, threads): median and IQR of the kernel time and
       of the peak memory.
    3. Parallel metrics, all computed from MEDIAN times at the same size n:
         speedup                S(p) = T_seq(n) / T_p(n)
         efficiency             E(p) = S(p) / p
         self-relative speedup  T(algo, n, 1) / T(algo, n, p)
         SIMD gain              T(omp_tiled, n, p) / T(omp_simd_tiled, n, p)
         OpenMP cost at p = 1   T(omp_tiled, n, 1) - T_seq(n)
       T_seq(n) is the median time of seq_tiled, the sequential (and scalar)
       baseline. Consequence: S(p) of omp_simd_tiled contains BOTH the gain
       from threads and the gain from SIMD, so it can exceed the ideal p.
       The self-relative speedup measures the threading part alone.
    4. Three figures for one matrix size (--size, default: the largest):
         time_vs_threads        kernel time vs. p, with seq_tiled as a
                                horizontal reference line
         speedup_vs_threads     S(p) vs. p, with the ideal S(p) = p
         efficiency_vs_threads  E(p) vs. p, with the ideal E(p) = 1 (100 %)
    5. results_summary.md - all statistics, for every size, as Markdown tables.

Variability bands in the speedup and efficiency figures:
    The band runs from T_seq / Q3(T_p) to T_seq / Q1(T_p): it shows the
    spread of the parallel runs, with T_seq fixed at its median. (A slower
    parallel run, Q3, gives the lower speedup.)

Style: every visual choice follows common/plotting_style.md and is applied
in ONE place, apply_plot_style(). Figures are drawn on a 5.6 in canvas and
meant to be included at \\columnwidth (3.5 in) in LaTeX, i.e. scaled by 0.625.

Run (from anywhere):
    python3 assignment_3/plot_results.py                      # paper figures (no titles)
    python3 assignment_3/plot_results.py --titles             # with titles, for slides
    python3 assignment_3/plot_results.py --physical-cores 4   # also mark the core count
    python3 assignment_3/plot_results.py --size 1000 --output-dir somewhere/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # render to files only; no window/display is needed

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import (FuncFormatter, LogLocator, MaxNLocator,  # noqa: E402
                               NullLocator, PercentFormatter)

import benchmark_config as config  # noqa: E402

# ---------------------------------------------------------------------------
# Style constants (see common/plotting_style.md)
# ---------------------------------------------------------------------------

# Fixed order and color per algorithm; color follows the series, never its
# rank, and is the same in every figure.
#   seq_tiled keeps crimson: it is the Assignment 2 tiled kernel, which was
#       crimson there, so the same kernel has the same color across reports.
#   omp_tiled and omp_simd_tiled are compared in EVERY figure, so they get
#       the most distinct pair of the validated palette: deep blue and amber
#       (also clearly different in grayscale: dark vs. light).
# "label" is used in tables (the --algo names, as in the CSV); "legend" is
# the short name that keeps every one-row legend inside the 5.6 in canvas.
ALGORITHMS = {
    "seq_tiled":      {"label": "seq_tiled",      "legend": "Sequential",    "color": "#dc143c"},  # crimson
    "omp_tiled":      {"label": "omp_tiled",      "legend": "OpenMP",        "color": "#1f43b2"},  # deep blue
    "omp_simd_tiled": {"label": "omp_simd_tiled", "legend": "OpenMP + SIMD", "color": "#c98500"},  # amber
}
SEQUENTIAL = "seq_tiled"
PARALLEL_ALGORITHMS = ["omp_tiled", "omp_simd_tiled"]

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

# Theoretical ideals (S = p, E = 1): the style guide's reference curve.
REFERENCE_STYLE = {"color": REFERENCE_COLOR, "linewidth": REFERENCE_LINEWIDTH,
                   "linestyle": REFERENCE_DASHES, "zorder": 2.5}

# Extensions of the style guide for this assignment:
#   - The seq_tiled reference in the time figure is MEASURED, not
#     theoretical, so it keeps its series color (crimson) and its IQR band,
#     but it is dashed: it is one value drawn across all p, not a curve over p.
#   - The optional physical-core marker is a charcoal dotted vertical line
#     (dotted, so it cannot be mistaken for the dashed ideal curves).
SEQUENTIAL_LINEWIDTH = 2.0
SEQUENTIAL_DASHES = (0, (5, 3))
CORES_LINESTYLE = (0, (1, 3))


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
    """
    df = pd.read_csv(path)
    ok_runs = df[df["status"] == "ok"].copy()
    failed_runs = df[df["status"] != "ok"].copy()   # "timeout", "error" or "mismatch"
    return ok_runs, failed_runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Median, first/third quartile and IQR of time and memory per
    (size, algo, threads).

    Quartiles use pandas' default linear interpolation. With 5 repetitions,
    Q1 and Q3 are exactly the 2nd and 4th smallest values, so the IQR is
    the spread of the middle three runs - robust to one slow outlier run.
    """
    grouped = runs.groupby(["size", "algo", "threads"])
    summary = pd.DataFrame({
        "runs": grouped["time_s"].count(),
        "time_median": grouped["time_s"].median(),
        "time_q1": grouped["time_s"].quantile(0.25),
        "time_q3": grouped["time_s"].quantile(0.75),
        "memory_median": grouped["memory_mb"].median(),
        "memory_q1": grouped["memory_mb"].quantile(0.25),
        "memory_q3": grouped["memory_mb"].quantile(0.75),
    }).reset_index()
    summary["time_iqr"] = summary["time_q3"] - summary["time_q1"]
    summary["memory_iqr"] = summary["memory_q3"] - summary["memory_q1"]
    return summary


def add_parallel_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add speedup, efficiency and self-relative speedup to every row.

    T_seq(n) = median time of seq_tiled at size n (NaN if it was not run).
        speedup      S(p) = T_seq / T_p             (medians)
        efficiency   E(p) = S(p) / p
        band         T_seq / Q3(T_p) .. T_seq / Q1(T_p), and the same / p
    Self-relative speedup = T(algo, n, p=1) / T(algo, n, p): how well each
    kernel scales with threads compared with ITSELF on one thread. For
    omp_simd_tiled this separates the threading gain from the SIMD gain.
    """
    sequential = summary[summary["algo"] == SEQUENTIAL].set_index("size")["time_median"]
    summary = summary.copy()
    summary["t_seq"] = summary["size"].map(sequential)

    summary["speedup"] = summary["t_seq"] / summary["time_median"]
    summary["speedup_low"] = summary["t_seq"] / summary["time_q3"]
    summary["speedup_high"] = summary["t_seq"] / summary["time_q1"]
    summary["efficiency"] = summary["speedup"] / summary["threads"]
    summary["efficiency_low"] = summary["speedup_low"] / summary["threads"]
    summary["efficiency_high"] = summary["speedup_high"] / summary["threads"]

    one_thread = summary[summary["threads"] == 1][["size", "algo", "time_median"]]
    one_thread = one_thread.rename(columns={"time_median": "t_one_thread"})
    summary = summary.merge(one_thread, on=["size", "algo"], how="left")
    summary["self_speedup"] = summary["t_one_thread"] / summary["time_median"]
    return summary


def series(summary: pd.DataFrame, algorithm: str, n: int) -> pd.DataFrame:
    """Rows of one algorithm at size n, sorted by thread count."""
    rows = summary[(summary["algo"] == algorithm) & (summary["size"] == n)]
    return rows.sort_values("threads")


def thread_counts_at(summary: pd.DataFrame, n: int) -> list[int]:
    """Thread counts measured for the parallel kernels at size n."""
    rows = summary[(summary["size"] == n) & (summary["algo"].isin(PARALLEL_ALGORITHMS))]
    return sorted(int(p) for p in rows["threads"].unique())


# ---------------------------------------------------------------------------
# Plotting building blocks
# ---------------------------------------------------------------------------

def plain_number(value, _position) -> str:
    """Tick label as a plain number: 0.5, 1, 2, 8 (not 2^3 or 5x10^-1)."""
    return f"{value:g}"


def plot_median_with_iqr(ax, x, median, low, high, color: str, label: str) -> None:
    """
    One shaded band (low..high) under one thick median line.
    The band gets a lower zorder so every median line stays on top of it.
    """
    ax.fill_between(x, low, high, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, median, color=color, label=label, zorder=3)


def plot_parallel_kernels(ax, summary: pd.DataFrame, n: int, metric: str) -> None:
    """
    Median line + band of `metric` for omp_tiled and omp_simd_tiled.
    metric "time" uses the columns time_median/q1/q3; "speedup" and
    "efficiency" use <metric>, <metric>_low and <metric>_high.
    """
    for algorithm in PARALLEL_ALGORITHMS:
        data = series(summary, algorithm, n)
        if data.empty:
            continue
        if metric == "time":
            middle, low, high = data["time_median"], data["time_q1"], data["time_q3"]
        else:
            middle, low, high = data[metric], data[f"{metric}_low"], data[f"{metric}_high"]
        plot_median_with_iqr(ax, data["threads"], middle, low, high,
                             ALGORITHMS[algorithm]["color"], ALGORITHMS[algorithm]["legend"])


def format_threads_axis(ax, thread_counts: list[int]) -> None:
    """
    Log2 x-axis with a tick at every measured thread count.
    Doubling p is the natural step in a scaling study, and on a log2 axis
    1, 2, 4, 8 are evenly spaced.
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(thread_counts)
    ax.xaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Threads $p$")


def mark_physical_cores(ax, physical_cores: int | None) -> None:
    """
    Optional vertical line at p = number of physical cores. Beyond it, extra
    threads only get SMT siblings (two hardware threads sharing one core's
    execution units), so less gain per thread is expected there.
    """
    if physical_cores is None:
        return
    ax.axvline(physical_cores, color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH,
               linestyle=CORES_LINESTYLE, label=f"{physical_cores} cores",
               zorder=2.4)


def add_legend(fig: plt.Figure, ax, title: str | None = None) -> None:
    """
    One-row, frameless legend ABOVE the plot area, so it can never cover data.
    "outside upper center" lets the constrained layout reserve room for it.
    The optional figure title (slides only) is the legend's title, so it sits
    directly above the legend row without overlapping it.
    With four entries (physical-core marker on), shorter line samples and
    gaps keep the single row inside the canvas.
    """
    handles, labels = ax.get_legend_handles_labels()
    compact = len(handles) > 3
    fig.legend(handles, labels, loc="outside upper center", ncols=len(handles),
               title=title, title_fontproperties=TITLE_FONT,
               columnspacing=0.9 if compact else None,
               handlelength=1.6 if compact else None,
               handletextpad=0.5 if compact else None)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save the same figure as vector PDF (for LaTeX) and 300-dpi PNG."""
    for extension in ("pdf", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_time_vs_threads(summary: pd.DataFrame, n: int, physical_cores: int | None,
                         title: str | None) -> plt.Figure:
    """Kernel time vs. p for the two parallel kernels, with seq_tiled as a reference."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")

    # seq_tiled has one value per size (p = 1): a horizontal line with its
    # IQR as a horizontal band, so every parallel point can be read against it.
    sequential = series(summary, SEQUENTIAL, n)
    if not sequential.empty:
        row = sequential.iloc[0]
        color = ALGORITHMS[SEQUENTIAL]["color"]
        ax.axhspan(row["time_q1"], row["time_q3"], color=color, alpha=BAND_ALPHA,
                   linewidth=0, zorder=2)
        ax.axhline(row["time_median"], color=color, linewidth=SEQUENTIAL_LINEWIDTH,
                   linestyle=SEQUENTIAL_DASHES, label=ALGORITHMS[SEQUENTIAL]["legend"], zorder=3)

    plot_parallel_kernels(ax, summary, n, "time")
    mark_physical_cores(ax, physical_cores)

    format_threads_axis(ax, thread_counts_at(summary, n))
    # Log y: equal ratios look equal, so a constant SIMD gain is a constant
    # vertical gap. Ticks at 1, 2 and 5 of every decade, as plain numbers.
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Kernel time (s)")
    add_legend(fig, ax, title)
    return fig


def plot_speedup_vs_threads(summary: pd.DataFrame, n: int, physical_cores: int | None,
                            title: str | None) -> plt.Figure:
    """S(p) = T_seq / T_p vs. p, with the ideal linear speedup S(p) = p."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    plot_parallel_kernels(ax, summary, n, "speedup")

    thread_counts = thread_counts_at(summary, n)
    ax.plot(thread_counts, thread_counts, **REFERENCE_STYLE, label="Ideal $S(p) = p$")
    mark_physical_cores(ax, physical_cores)

    format_threads_axis(ax, thread_counts)
    # Log2 y as well: the ideal S = p becomes a straight diagonal, and a
    # constant efficiency is a line parallel to it.
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(LogLocator(base=2))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(r"Speedup $S(p) = T_{\mathrm{seq}}\,/\,T_p$")
    add_legend(fig, ax, title)
    return fig


def plot_efficiency_vs_threads(summary: pd.DataFrame, n: int, physical_cores: int | None,
                               title: str | None) -> plt.Figure:
    """E(p) = S(p) / p vs. p, with the ideal E(p) = 1 (100 %)."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    plot_parallel_kernels(ax, summary, n, "efficiency")
    ax.axhline(1.0, **REFERENCE_STYLE, label="Ideal $E(p) = 1$")
    mark_physical_cores(ax, physical_cores)

    format_threads_axis(ax, thread_counts_at(summary, n))
    # Linear y from 0, so the distance to 100 % is the share of the ideal
    # that was lost. The top leaves room for values above 100 % (possible
    # for omp_simd_tiled, whose S(p) also contains the SIMD gain).
    at_n = summary[(summary["size"] == n) & (summary["algo"].isin(PARALLEL_ALGORITHMS))]
    top = max(1.1, float(at_n["efficiency_high"].max()) * 1.08)
    ax.set_ylim(0.0, top)
    # About six "round" ticks (25 %, 50 % or 100 % apart), whatever the top.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Efficiency $E(p) = S(p)\\,/\\,p$")
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


def fmt(value, spec: str, suffix: str = "") -> str:
    """Format a number, or '-' for a metric that does not apply (NaN)."""
    return "-" if pd.isna(value) else format(value, spec) + suffix


def ordered(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort by size, then by the fixed algorithm order, then by threads."""
    order = list(ALGORITHMS)
    return (summary.assign(order=summary["algo"].map(order.index))
                   .sort_values(["size", "order", "threads"]))


def value_at(summary: pd.DataFrame, n: int, algorithm: str, threads: int, column: str) -> float:
    """One statistic of one configuration, or NaN if it was not measured."""
    rows = summary[(summary["size"] == n) & (summary["algo"] == algorithm)
                   & (summary["threads"] == threads)]
    return float(rows[column].iloc[0]) if not rows.empty else float("nan")


def key_results_markdown(summary: pd.DataFrame, n: int) -> list[str]:
    """Headline numbers for the plotted size (also the ones quoted in the README)."""
    lines = ["## Key results", ""]
    thread_counts = thread_counts_at(summary, n)
    if not thread_counts:
        return lines + [f"No parallel runs at n = {n}.", ""]
    p_max = thread_counts[-1]

    parts = []
    for algorithm in PARALLEL_ALGORITHMS:
        speedup = value_at(summary, n, algorithm, p_max, "speedup")
        efficiency = value_at(summary, n, algorithm, p_max, "efficiency")
        parts.append(f"{algorithm} S = {fmt(speedup, '.2f')} (E = {fmt(efficiency * 100, '.0f', ' %')})")
    lines.append(f"- n = {n}, p = {p_max} (largest measured thread count): " + "; ".join(parts) + ".")

    parts = []
    for algorithm in PARALLEL_ALGORITHMS:
        parts.append(f"{algorithm} {fmt(value_at(summary, n, algorithm, p_max, 'self_speedup'), '.2f')}x")
    lines.append(f"- Self-relative speedup T(algo, 1) / T(algo, {p_max}) at n = {n}: "
                 + ", ".join(parts) + ".")

    gain_1 = (value_at(summary, n, "omp_tiled", 1, "time_median")
              / value_at(summary, n, "omp_simd_tiled", 1, "time_median"))
    gain_max = (value_at(summary, n, "omp_tiled", p_max, "time_median")
                / value_at(summary, n, "omp_simd_tiled", p_max, "time_median"))
    lines.append(f"- SIMD gain T(omp_tiled) / T(omp_simd_tiled) at n = {n}: "
                 f"{fmt(gain_1, '.2f')}x at p = 1, {fmt(gain_max, '.2f')}x at p = {p_max}.")

    t_seq = value_at(summary, n, SEQUENTIAL, 1, "time_median")
    t_omp = value_at(summary, n, "omp_tiled", 1, "time_median")
    difference = t_omp - t_seq
    lines.append(f"- OpenMP at p = 1, n = {n}: omp_tiled {fmt(t_omp, '.4g')} s vs. seq_tiled "
                 f"{fmt(t_seq, '.4g')} s (difference {fmt(difference, '+.4g')} s, "
                 f"{fmt(difference / t_seq * 100, '+.1f')} %; negative = omp_tiled faster).")
    return lines + [""]


def scaling_markdown(summary: pd.DataFrame) -> list[str]:
    """Main table: every (size, algorithm, threads) with time, S(p), E(p) and memory."""
    lines = ["## Time, speedup and efficiency", "",
             "S(p) = median seq_tiled time / median time at the same n. E(p) = S(p) / p. "
             "Self-relative = median time of the same kernel at p = 1 / median time at p. "
             "Because seq_tiled is scalar, S(p) and E(p) of omp_simd_tiled include the SIMD gain; "
             "the self-relative speedup shows the threading part alone.", ""]
    rows = []
    for _, row in ordered(summary).iterrows():
        is_sequential = row["algo"] == SEQUENTIAL
        rows.append([str(row["size"]), ALGORITHMS[row["algo"]]["label"], str(row["threads"]),
                     str(row["runs"]),
                     f"{row['time_median']:.4g}", f"{row['time_iqr']:.2g}",
                     fmt(row["speedup"], ".2f"),
                     fmt(row["efficiency"] * 100, ".0f", " %"),
                     "-" if is_sequential else fmt(row["self_speedup"], ".2f"),
                     f"{row['memory_median']:.2f}"])
    lines += markdown_table(["Size n", "Algorithm", "Threads p", "Runs", "Median time (s)",
                             "IQR time (s)", "Speedup S(p)", "Efficiency E(p)",
                             "Self-relative speedup", "Median memory (MB)"],
                            rows, text_columns=2)
    return lines


def overhead_and_simd_markdown(summary: pd.DataFrame) -> list[str]:
    """OpenMP cost at p = 1 and SIMD gain per (size, threads)."""
    lines = ["", "### OpenMP at p = 1 vs. seq_tiled", "",
             "Same tiled loops and scalar inner loop; the only difference is the OpenMP "
             "parallel loop around them (here with a team of one thread). "
             "Negative difference = omp_tiled was faster.", ""]
    rows = []
    for n in sorted(summary["size"].unique()):
        t_seq = value_at(summary, n, SEQUENTIAL, 1, "time_median")
        t_omp = value_at(summary, n, "omp_tiled", 1, "time_median")
        difference = t_omp - t_seq
        rows.append([str(n), fmt(t_seq, ".4g"), fmt(t_omp, ".4g"),
                     fmt(difference, "+.4g"), fmt(difference / t_seq * 100, "+.1f", " %")])
    lines += markdown_table(["Size n", "seq_tiled median (s)", "omp_tiled p=1 median (s)",
                             "Difference (s)", "Difference (%)"], rows)

    lines += ["", "### SIMD gain", "",
              "SIMD gain = median omp_tiled time / median omp_simd_tiled time, same n and p "
              "(threads and scheduling are identical; only the inner loop differs).", ""]
    rows = []
    for n in sorted(summary["size"].unique()):
        for p in thread_counts_at(summary, n):
            t_scalar = value_at(summary, n, "omp_tiled", p, "time_median")
            t_vector = value_at(summary, n, "omp_simd_tiled", p, "time_median")
            rows.append([str(n), str(p), fmt(t_scalar, ".4g"), fmt(t_vector, ".4g"),
                         fmt(t_scalar / t_vector, ".2f", "x")])
    lines += markdown_table(["Size n", "Threads p", "omp_tiled median (s)",
                             "omp_simd_tiled median (s)", "SIMD gain"], rows)
    return lines


def excluded_markdown(failed_runs: pd.DataFrame) -> list[str]:
    """Runs without a valid measurement: reported, never silently dropped."""
    lines = ["## Excluded runs", ""]
    if failed_runs.empty:
        lines.append("None - every run finished successfully within the time budget.")
        return lines
    counts = (failed_runs.groupby(["size", "algo", "threads", "status"])
              .size().reset_index(name="runs"))
    rows = [[str(r["size"]), r["algo"], str(r["threads"]), r["status"], str(r["runs"])]
            for _, r in counts.iterrows()]
    lines += markdown_table(["Size n", "Algorithm", "Threads p", "Status", "Runs"],
                            rows, text_columns=4)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot the Assignment 3 results.")
    parser.add_argument("--input", type=Path, default=config.RESULTS_DIR / "measurements.csv")
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--size", type=int, default=None,
                        help="Matrix size n for the figures (default: the largest measured)")
    parser.add_argument("--physical-cores", type=int, default=None,
                        help="Mark p = number of physical cores in the figures (optional)")
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

    summary = add_parallel_metrics(summarize(ok_runs))

    n = args.size if args.size is not None else int(summary["size"].max())
    if n not in set(summary["size"]):
        raise SystemExit(f"No successful runs at n = {n}; measured sizes: "
                         f"{sorted(summary['size'].unique())}.")
    if pd.isna(value_at(summary, n, SEQUENTIAL, 1, "time_median")):
        print(f"Warning: no seq_tiled runs at n = {n}; speedup and efficiency are undefined.")
    print(f"Figures for n = {n}.")

    figures = [
        ("time_vs_threads", plot_time_vs_threads, f"Kernel time vs. threads ($n = {n}$)"),
        ("speedup_vs_threads", plot_speedup_vs_threads, f"Speedup vs. threads ($n = {n}$)"),
        ("efficiency_vs_threads", plot_efficiency_vs_threads, f"Parallel efficiency ($n = {n}$)"),
    ]
    for name, plot_function, slide_title in figures:
        title = slide_title if args.titles else None
        save_figure(plot_function(summary, n, args.physical_cores, title), args.output_dir, name)

    lines = ["# Assignment 3 - Results Summary", "",
             "Generated by `plot_results.py` from `results/measurements.csv`. "
             "Statistics use successful runs only. Time = kernel only (monotonic clock); "
             "memory = peak resident set size of the whole process (GNU time); IQR = Q3 - Q1. "
             f"Block size {config.BLOCK_SIZE}; speedup and efficiency use median times.", ""]
    lines += key_results_markdown(summary, n)
    lines += scaling_markdown(summary)
    lines += overhead_and_simd_markdown(summary)
    lines += [""] + excluded_markdown(failed_runs)

    path = args.output_dir / "results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
