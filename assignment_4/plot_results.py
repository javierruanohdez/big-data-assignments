"""
plot_results.py - Analysis and figures for Assignment 4.

Pipeline:
    1. Load results/measurements.csv (written by run_benchmark.py). Only runs
       with status "ok" enter the statistics. Timeouts, errors and checksum
       mismatches have no valid measurement, so they are excluded, but they
       are counted and listed in the summary so nothing is hidden.
    2. Per (experiment, mode, size, workers): median, Q1, Q3 and IQR of every
       time column and of the peak memory. The per-run sum of the barriered
       phases (scatter + compute + gather) is summarised the same way, so it
       can be compared with the pipelined end-to-end time.
    3. Metrics, all computed from MEDIAN end-to-end times (e2e_s):
         strong speedup       S(p)   = T_dist(1) / T_dist(p)          same n
         strong efficiency    E(p)   = S(p) / p
         vs. seq_python              T_seq_python / T_dist(p)         same n
         vs. C sequential            T_c_baseline / T(p)              same n
         vs. C OpenMP                T_c_omp_tiled(p) / T_dist(p)     same n and p
         weak efficiency      E_w(p) = T_dist(1, n_1) / T_dist(p, n_p)
       S(p) is SELF-relative: T_dist(1) already contains the Dask overhead
       that the single-process baseline does not pay, so S(p) shows how the
       distributed program scales against itself. "vs. seq_python" is the
       absolute gain over the same block algorithm without Dask (> 1 means
       distribution paid off). Both are reported.
    4. Four figures:
         time_breakdown_n<N>                (a) per-phase time vs. p, strong scaling
         strong_scaling_speedup_efficiency  (b) S(p) and E(p) vs. p, every strong size
         weak_scaling                       (c) time and data moved vs. p, weak scaling
         cross_paradigm_n<N>                (d) C sequential vs. C OpenMP vs. Dask
    5. results_summary.md - every statistic as Markdown tables.

Variability:
    Line figures show the median as a line and the IQR as a band (style
    guide). For S(p) and E(p) the band runs from T_dist(1) / Q3(T_p) to
    T_dist(1) / Q1(T_p), with T_dist(1) fixed at its median (as in
    Assignment 3). Bar figures show medians only: the style guide uses no
    error bars, and every IQR is listed in results_summary.md.

Style: every visual choice follows common/plotting_style.md and is applied
in ONE place, apply_plot_style(). Figures are drawn 5.6 in wide and meant to
be included at \\columnwidth (3.5 in) in LaTeX, i.e. scaled by 0.625, which
gives the printed text sizes of the style guide.

Extensions of the style guide for this assignment (documented in the README):
    - Three figures have two stacked panels that share the x axis. They keep
      the 5.6 in width (so all text sizes stay the same) and are taller.
      Two measures with different units go into two panels, never onto two
      y axes of one panel.
    - Series colors. The Dask program is Python, so it keeps crimson; the C
      OpenMP program keeps deep blue (its color in Assignment 3); the
      single-process NumPy baseline takes amber (the validated 4th color).
      The C sequential baseline is the unit of the cross-paradigm figure
      (speedup 1), so it is drawn as a charcoal reference line.
    - Shades of crimson. Everything that belongs to the Dask program keeps
      its hue. Where Dask is split into ordered parts, a one-hue ordinal ramp
      shows the order: the phases in pipeline order (setup, scatter,
      compute, gather; light to dark) and the matrix sizes (larger n =
      darker). The ramp was checked with the palette validator (monotone
      lightness, visible steps, light end >= 2:1 contrast on white).
    - The pipelined end-to-end bar of figure (a) is charcoal: it is the
      reference total the stacked phases are compared with (measured, so
      a solid bar, not a dashed line).
    - Stacked bar segments are separated by thin white gaps instead of
      outlines, so neighbouring shades never touch.

Run (from anywhere):
    python3 assignment_4/plot_results.py                  # paper figures (no titles)
    python3 assignment_4/plot_results.py --titles         # with titles, for slides
    python3 assignment_4/plot_results.py --breakdown-size 2000 --physical-cores 0
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # render to files only; no window/display is needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import (FuncFormatter, LogLocator, MaxNLocator,  # noqa: E402
                               NullLocator, PercentFormatter)

import benchmark_config as config  # noqa: E402

# ---------------------------------------------------------------------------
# Style constants (see common/plotting_style.md)
# ---------------------------------------------------------------------------

CRIMSON = "#dc143c"           # Python (style guide) -> the Dask program
DEEP_BLUE = "#1f43b2"         # C (style guide); omp_tiled's color in Assignment 3
AMBER = "#c98500"             # validated 4th series color -> NumPy in one process
REFERENCE_COLOR = "#3d3d3d"   # charcoal
TEXT_COLOR = "#1f1f1e"
GRID_COLOR = "#c8c8c5"

# One-hue ordinal ramp of the Dask crimson (OKLCH lightness 0.78 -> 0.44,
# validated; see the module docstring). Pipeline order = light to dark.
PHASES = {
    "setup_s":   {"label": "Setup",   "color": "#ff9595"},
    "scatter_s": {"label": "Scatter", "color": "#ff5260"},
    "compute_s": {"label": "Compute", "color": CRIMSON},
    "gather_s":  {"label": "Gather",  "color": "#9d0126"},
}
# Data moved, colored like the phase in which it moves.
TRAFFIC = {
    "scatter_bytes":       {"label": "Client to workers", "color": PHASES["scatter_s"]["color"]},
    "comm_bytes_measured": {"label": "Between workers",   "color": PHASES["compute_s"]["color"]},
    "gather_bytes":        {"label": "Workers to client", "color": PHASES["gather_s"]["color"]},
}
# Matrix sizes of the strong-scaling figure: the darkest steps of the same
# ramp, larger n = darker (n = 2000 keeps the plain Dask crimson, as in (d)).
SIZE_RAMP = [PHASES["scatter_s"]["color"], CRIMSON, PHASES["gather_s"]["color"]]

FIGURE_WIDTH = 5.6            # inches; printed at 3.5 in wide (scale 0.625)
FIGURE_SIZE = (FIGURE_WIDTH, 4.2)
TWO_PANEL_SIZE = (FIGURE_WIDTH, 6.6)
MEDIAN_LINEWIDTH = 2.5
BAND_ALPHA = 0.22             # transparency of the IQR band
REFERENCE_LINEWIDTH = 1.75    # thinner than the data lines
REFERENCE_DASHES = (0, (5, 6))  # "loosely dashed": 5 pt dash, 6 pt gap
MEASURED_REFERENCE_LINEWIDTH = 2.0
MEASURED_REFERENCE_DASHES = (0, (5, 3))   # a measured value drawn across all p (as in A3)
CORES_LINESTYLE = (0, (1, 3))             # dotted: cannot be mistaken for the dashed ideals
BAR_GAP_LINEWIDTH = 1.0                   # white gap between stacked segments
TITLE_FONT = {"size": 16, "weight": "bold"}   # slides only (--titles)
PNG_DPI = 300

REFERENCE_STYLE = {"color": REFERENCE_COLOR, "linewidth": REFERENCE_LINEWIDTH,
                   "linestyle": REFERENCE_DASHES, "zorder": 2.5}

MEGABYTE_DECIMAL = 1e6        # data volumes: MB = 10^6 bytes (a 500 x 500 tile = 2 MB)

# Columns summarised with median / Q1 / Q3.
TIME_COLUMNS = ["e2e_s", "setup_s", "cold_e2e_s", "scatter_s", "compute_s", "gather_s", "phase_sum_s"]
MEMORY_COLUMNS = ["worker_peak_mb_max", "worker_peak_mb_mean", "client_peak_mb"]
# Columns that are the same in every repetition of a configuration.
CONSTANT_COLUMNS = ["grid", "block_size", "tile_grid", "tasks", "scatter_bytes",
                    "comm_bytes_theoretical", "comm_bytes_measured", "gather_bytes"]

MODE_LABELS = {
    "distributed": "Dask (distributed)",
    "seq_python": "NumPy, 1 process",
    "c_baseline": "C sequential",
    "c_omp_tiled": "C OpenMP",
}
MODE_ORDER = ["c_baseline", "c_omp_tiled", "seq_python", "distributed"]
# Shorter names for the legends, so every legend fits in one row of the canvas.
LEGEND_LABELS = {
    "distributed": "Dask",
    "seq_python": "NumPy, 1 process",
    "c_baseline": "C sequential",
    "c_omp_tiled": "C OpenMP",
}
ANNOTATION_FONTSIZE = 12      # = tick size: 7.5 pt once printed at column width


def apply_plot_style() -> None:
    """Configure Matplotlib globally to match common/plotting_style.md (same as Assignment 3)."""
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
    Returns (ok_runs, failed_runs). The checksum is read as text, so it is
    compared exactly as the programs printed it.
    """
    df = pd.read_csv(path, dtype={"checksum": str})
    ok_runs = df[df["status"] == "ok"].copy()
    failed_runs = df[df["status"] != "ok"].copy()   # "timeout", "error" or "mismatch"
    # Per-run sum of the barriered phases (NaN for the C modes, which have no phases).
    ok_runs["phase_sum_s"] = ok_runs[["scatter_s", "compute_s", "gather_s"]].sum(axis=1, min_count=3)
    return ok_runs, failed_runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Median, Q1, Q3 and IQR of every time and memory column per
    (experiment, mode, size, workers), plus the per-configuration constants.

    Quartiles use pandas' default linear interpolation. With 5 repetitions,
    Q1 and Q3 are exactly the 2nd and 4th smallest values, so the IQR is
    the spread of the middle three runs - robust to one slow outlier run.
    """
    keys = ["experiment", "mode", "size", "workers"]
    grouped = runs.groupby(keys)
    summary = pd.DataFrame({"runs": grouped["e2e_s"].count()})
    for column in TIME_COLUMNS + MEMORY_COLUMNS:
        summary[f"{column}_median"] = grouped[column].median()
        summary[f"{column}_q1"] = grouped[column].quantile(0.25)
        summary[f"{column}_q3"] = grouped[column].quantile(0.75)
        summary[f"{column}_iqr"] = summary[f"{column}_q3"] - summary[f"{column}_q1"]
    for column in CONSTANT_COLUMNS:
        summary[column] = grouped[column].first()
    return summary.reset_index()


def lookup(summary: pd.DataFrame, experiment: str, mode: str, size: int,
           workers: int, column: str) -> float:
    """One statistic of one configuration, or NaN if it was not measured."""
    rows = summary[(summary["experiment"] == experiment) & (summary["mode"] == mode)
                   & (summary["size"] == size) & (summary["workers"] == workers)]
    return float(rows[column].iloc[0]) if not rows.empty else float("nan")


def add_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Add the metrics of the module docstring (step 3) to every row where they apply."""
    summary = summary.copy()
    for column in ["speedup", "speedup_low", "speedup_high", "efficiency", "efficiency_low",
                   "efficiency_high", "vs_seq_python", "vs_c_baseline", "vs_c_omp",
                   "weak_efficiency"]:
        summary[column] = float("nan")

    for index, row in summary.iterrows():
        experiment, mode, n, p = row["experiment"], row["mode"], row["size"], row["workers"]
        t_median = row["e2e_s_median"]

        # Every mode against the C baselines at the same n (strong experiment only).
        t_c_seq = lookup(summary, "strong", "c_baseline", n, 1, "e2e_s_median")
        summary.loc[index, "vs_c_baseline"] = t_c_seq / t_median

        if mode != "distributed":
            continue
        t_seq_python = lookup(summary, experiment, "seq_python", n, 1, "e2e_s_median")
        summary.loc[index, "vs_seq_python"] = t_seq_python / t_median
        summary.loc[index, "vs_c_omp"] = (lookup(summary, "strong", "c_omp_tiled", n, p, "e2e_s_median")
                                          / t_median)

        if experiment == "strong":
            t_one = lookup(summary, "strong", "distributed", n, 1, "e2e_s_median")
            summary.loc[index, "speedup"] = t_one / t_median
            summary.loc[index, "speedup_low"] = t_one / row["e2e_s_q3"]    # slower run -> lower S
            summary.loc[index, "speedup_high"] = t_one / row["e2e_s_q1"]
            for suffix in ["", "_low", "_high"]:
                summary.loc[index, f"efficiency{suffix}"] = summary.loc[index, f"speedup{suffix}"] / p
        else:
            weak_one = summary[(summary["experiment"] == "weak") & (summary["mode"] == "distributed")
                               & (summary["workers"] == 1)]
            if not weak_one.empty:
                summary.loc[index, "weak_efficiency"] = weak_one["e2e_s_median"].iloc[0] / t_median
    return summary


def select(summary: pd.DataFrame, experiment: str, mode: str, size: int | None = None) -> pd.DataFrame:
    """Rows of one experiment and mode (optionally one size), sorted by workers."""
    rows = summary[(summary["experiment"] == experiment) & (summary["mode"] == mode)]
    if size is not None:
        rows = rows[rows["size"] == size]
    return rows.sort_values(["size", "workers"])


# ---------------------------------------------------------------------------
# Plotting building blocks
# ---------------------------------------------------------------------------

def plain_number(value, _position) -> str:
    """Tick label as a plain number: 0.5, 1, 2, 8 (not 2^3 or 5x10^-1)."""
    return f"{value:g}"


def plot_median_with_iqr(ax, x, median, low, high, color: str, label: str) -> None:
    """One shaded band (low..high) under one thick median line."""
    ax.fill_between(x, low, high, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, median, color=color, label=label, zorder=3)


def format_workers_axis(ax, worker_counts: list[int], labels: list[str] | None = None,
                        axis_label: str = "Workers $p$") -> None:
    """
    Log2 x axis with a tick at every measured worker count: doubling p is the
    natural step of a scaling study, and 1, 2, 4, 8 are evenly spaced.
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(worker_counts)
    if labels is None:
        ax.xaxis.set_major_formatter(FuncFormatter(plain_number))
    else:
        ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(axis_label)


def mark_physical_cores(ax, physical_cores: int | None, label: bool = True) -> None:
    """
    Dotted vertical line at p = physical cores. Up to it, every worker has a
    core of its own; beyond it, two workers share one core (SMT), and the
    client and scheduler have no core left to themselves.
    """
    if not physical_cores:
        return
    ax.axvline(physical_cores, color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH,
               linestyle=CORES_LINESTYLE, zorder=2.4,
               label=f"{physical_cores} physical cores" if label else None)


def legend_fits(fig: plt.Figure, legend) -> bool:
    """True if the legend lies completely inside the figure canvas."""
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    canvas = fig.bbox
    tolerance = 0.5   # pixels
    return (box.x0 >= canvas.x0 - tolerance and box.x1 <= canvas.x1 + tolerance
            and box.y0 >= canvas.y0 - tolerance and box.y1 <= canvas.y1 + tolerance)


def add_figure_legend(fig: plt.Figure, handles: list, labels: list[str], title: str | None) -> None:
    """
    Frameless legend ABOVE the plot area ("outside upper center"), so it can
    never cover data; the optional slide title is the legend's title.
    One row, as the style guide asks. If one row would not fit inside the
    5.6 in canvas (it would be clipped at both sides), it is redrawn in two
    rows: a clipped legend is never saved. save_figure checks again.
    """
    for rows in (1, 2):
        columns = math.ceil(len(handles) / rows)
        compact = columns > 3
        legend = fig.legend(handles, labels, loc="outside upper center", ncols=columns,
                            title=title, title_fontproperties=TITLE_FONT,
                            columnspacing=0.9 if compact else None,
                            handlelength=1.6 if compact else None,
                            handletextpad=0.5 if compact else None)
        if legend_fits(fig, legend):
            return
        legend.remove()
    raise RuntimeError("The legend does not fit inside the figure even in two rows")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Check that no legend is clipped, then save as vector PDF (LaTeX) and 300-dpi PNG."""
    for legend in fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        if not legend_fits(fig, legend):
            raise RuntimeError(f"{name}: a legend extends beyond the figure canvas")
    for extension in ("pdf", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def stacked_bars(ax, x, values: dict, width: float) -> None:
    """
    Stacked bars, one segment per entry of `values` ({column: (heights, color, label)}),
    bottom to top in dict order. Thin white edges leave a gap between segments.
    """
    bottom = np.zeros(len(x))
    for heights, color, label in values.values():
        heights = np.asarray(heights, dtype=float)
        ax.bar(x, heights, width, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=BAR_GAP_LINEWIDTH, zorder=3)
        bottom += heights


# ---------------------------------------------------------------------------
# Figure (a): time breakdown vs. workers, strong scaling
# ---------------------------------------------------------------------------

def plot_time_breakdown(summary: pd.DataFrame, n: int, title: str | None) -> plt.Figure:
    """
    Two panels sharing the workers axis:
      top     one multiplication: the barriered phases stacked (scatter,
              compute, gather) next to the pipelined end-to-end time
      bottom  the one-time cluster setup
    Setup gets its own panel and scale: it is paid once per cluster (and
    could be amortised over many multiplications), and at n = 4000 it is
    still 1.5-4x one multiplication, so in one stack it would flatten the
    phases that change with p.
    """
    rows = select(summary, "strong", "distributed", n)
    worker_counts = [int(p) for p in rows["workers"]]
    x = np.arange(len(rows))
    width = 0.36

    fig, (ax_run, ax_setup) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 6.0), sharex=True,
                                           height_ratios=[2.2, 1.0], layout="constrained")

    phases = {column: (rows[f"{column}_median"], PHASES[column]["color"], PHASES[column]["label"])
              for column in ["scatter_s", "compute_s", "gather_s"]}
    stacked_bars(ax_run, x - width / 2 - 0.02, phases, width)
    ax_run.bar(x + width / 2 + 0.02, rows["e2e_s_median"], width, color=REFERENCE_COLOR,
               label="End-to-end (pipelined)", zorder=3)
    ax_run.set_ylabel("One multiplication (s)")

    ax_setup.bar(x, rows["setup_s_median"], width * 1.4, color=PHASES["setup_s"]["color"], zorder=3)
    ax_setup.set_ylabel("Cluster setup (s)")
    ax_setup.set_xticks(x)
    ax_setup.set_xticklabels([str(p) for p in worker_counts])
    ax_setup.set_xlabel("Workers $p$")

    for ax in (ax_run, ax_setup):
        ax.grid(axis="x", visible=False)          # vertical lines between bars are only noise
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    handles, labels = ax_run.get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, title)
    return fig


# ---------------------------------------------------------------------------
# Figure (b): strong-scaling speedup and efficiency
# ---------------------------------------------------------------------------

def size_colors(sizes: list[int]) -> dict[int, str]:
    """Darkest ramp steps for the sizes, larger n = darker."""
    if len(sizes) > len(SIZE_RAMP):
        raise ValueError(f"At most {len(SIZE_RAMP)} strong-scaling sizes can be colored")
    steps = SIZE_RAMP[len(SIZE_RAMP) - len(sizes):]
    return dict(zip(sorted(sizes), steps))


def plot_speedup_efficiency(summary: pd.DataFrame, physical_cores: int | None,
                            title: str | None) -> plt.Figure:
    """
    S(p) (top, log2 axes: the ideal S = p is the diagonal) and E(p) (bottom,
    linear from 0: the distance to 100 % is the share of the ideal lost),
    one line per strong-scaling size.
    """
    rows = select(summary, "strong", "distributed")
    sizes = sorted(int(n) for n in rows["size"].unique())
    colors = size_colors(sizes)
    worker_counts = sorted(int(p) for p in rows["workers"].unique())

    fig, (ax_s, ax_e) = plt.subplots(2, 1, figsize=TWO_PANEL_SIZE, sharex=True, layout="constrained")
    for n in sizes:
        data = rows[rows["size"] == n]
        label = f"$n = {n}$"
        plot_median_with_iqr(ax_s, data["workers"], data["speedup"], data["speedup_low"],
                             data["speedup_high"], colors[n], label)
        plot_median_with_iqr(ax_e, data["workers"], data["efficiency"], data["efficiency_low"],
                             data["efficiency_high"], colors[n], label)

    ax_s.plot(worker_counts, worker_counts, **REFERENCE_STYLE, label="Ideal ($S = p$, $E = 1$)")
    ax_e.axhline(1.0, **REFERENCE_STYLE)
    mark_physical_cores(ax_s, physical_cores)
    mark_physical_cores(ax_e, physical_cores, label=False)

    ax_s.set_yscale("log", base=2)
    ax_s.yaxis.set_major_locator(LogLocator(base=2))
    ax_s.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax_s.yaxis.set_minor_locator(NullLocator())
    ax_s.set_ylabel(r"Speedup $S(p) = T_1 / T_p$")

    top = max(1.1, float(rows["efficiency_high"].max()) * 1.08)
    ax_e.set_ylim(0.0, top)
    ax_e.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax_e.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_e.set_ylabel(r"Efficiency $E(p) = S(p) / p$")
    format_workers_axis(ax_e, worker_counts)

    handles, labels = ax_s.get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, title)
    return fig


# ---------------------------------------------------------------------------
# Figure (c): weak scaling
# ---------------------------------------------------------------------------

def plot_weak_scaling(summary: pd.DataFrame, physical_cores: int | None,
                      title: str | None) -> plt.Figure:
    """
    Top: end-to-end time when the work PER WORKER is constant
    (n_p = round(n_1 * p^(1/3))). Ideal weak scaling is a flat line at
    T_dist(1). The single-process NumPy baseline at the same n_p shows how
    much the total work grows (about 8x from p = 1 to 8).
    Bottom: the data moved in one multiplication, by direction. It grows
    with p although every worker multiplies the same number of FLOPs:
    that traffic, and the coordination around it, is the weak-scaling
    overhead.
    """
    dask = select(summary, "weak", "distributed")
    numpy_runs = select(summary, "weak", "seq_python")
    worker_counts = [int(p) for p in dask["workers"]]
    tick_labels = [f"{p}\n({int(n)})" for p, n in zip(dask["workers"], dask["size"])]

    fig, (ax_time, ax_data) = plt.subplots(2, 1, figsize=TWO_PANEL_SIZE, sharex=True,
                                           height_ratios=[1.6, 1.0], layout="constrained")

    plot_median_with_iqr(ax_time, dask["workers"], dask["e2e_s_median"], dask["e2e_s_q1"],
                         dask["e2e_s_q3"], CRIMSON, LEGEND_LABELS["distributed"])
    # NumPy runs once per n_p (p = 1); it is drawn at the p whose n_p it used.
    numpy_x = [int(dask[dask["size"] == n]["workers"].iloc[0]) for n in numpy_runs["size"]]
    plot_median_with_iqr(ax_time, numpy_x, numpy_runs["e2e_s_median"], numpy_runs["e2e_s_q1"],
                         numpy_runs["e2e_s_q3"], AMBER, LEGEND_LABELS["seq_python"])
    t_one = float(dask[dask["workers"] == 1]["e2e_s_median"].iloc[0]) if 1 in worker_counts else None
    if t_one is not None:
        ax_time.axhline(t_one, **REFERENCE_STYLE, label="Ideal")
    mark_physical_cores(ax_time, physical_cores)
    ax_time.set_ylim(bottom=0.0)
    ax_time.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_time.set_ylabel("End-to-end time (s)")

    # Bars on the shared log2 axis: from p / 1.3 to p * 1.3 gives every bar
    # the same visual width.
    left = np.array(worker_counts) / 1.3
    widths = np.array(worker_counts) * 1.3 - left
    bottom = np.zeros(len(dask))
    for column, style in TRAFFIC.items():
        heights = dask[column].to_numpy(dtype=float) / MEGABYTE_DECIMAL
        ax_data.bar(left, heights, widths, bottom=bottom, align="edge", color=style["color"],
                    label=style["label"], edgecolor="white", linewidth=BAR_GAP_LINEWIDTH, zorder=3)
        bottom += heights
    ax_data.grid(axis="x", visible=False)
    ax_data.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_data.set_ylabel("Data moved (MB)")
    # The traffic legend sits directly above its own panel, never on the bars.
    ax_data.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(TRAFFIC),
                   columnspacing=1.0, handlelength=1.4, handletextpad=0.5, borderaxespad=0.2)
    format_workers_axis(ax_data, worker_counts, tick_labels,
                        axis_label="Workers $p$ (matrix size $n_p$)")

    handles, labels = ax_time.get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, title)
    return fig


# ---------------------------------------------------------------------------
# Figure (d): cross-paradigm comparison
# ---------------------------------------------------------------------------

def plot_cross_paradigm(summary: pd.DataFrame, n: int, worker_counts: list[int],
                        title: str | None) -> plt.Figure:
    """
    Speed relative to the C sequential baseline (= 1) at one size, for the
    C OpenMP program and the Dask program with p = 1, 4, 8, plus the
    single-process NumPy baseline as a measured reference line.
    A ratio axis is used because the absolute times span 13.7 s to 0.65 s:
    on a linear time axis every bar except C sequential would be a sliver.
    Each bar is labelled with its median time, so the absolute value is
    never lost.
    """
    t_c_seq = lookup(summary, "strong", "c_baseline", n, 1, "e2e_s_median")
    x = np.arange(len(worker_counts))
    width = 0.36

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    ax.axhline(1.0, **REFERENCE_STYLE, label=LEGEND_LABELS["c_baseline"])

    tallest = 1.0
    for offset, mode, color in [(-width / 2 - 0.02, "c_omp_tiled", DEEP_BLUE),
                                (width / 2 + 0.02, "distributed", CRIMSON)]:
        times = [lookup(summary, "strong", mode, n, p, "e2e_s_median") for p in worker_counts]
        ratios = [t_c_seq / t for t in times]
        tallest = max(tallest, max(ratios))
        bars = ax.bar(x + offset, ratios, width, color=color, label=LEGEND_LABELS[mode], zorder=3)
        # Median time inside the top of each bar (white text on the dark bar).
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.97, f"{t:.2f} s",
                    ha="center", va="top", color="white", fontsize=ANNOTATION_FONTSIZE, zorder=4)

    numpy_median = lookup(summary, "strong", "seq_python", n, 1, "e2e_s_median")
    if not math.isnan(numpy_median):
        numpy_low = t_c_seq / lookup(summary, "strong", "seq_python", n, 1, "e2e_s_q3")
        numpy_high = t_c_seq / lookup(summary, "strong", "seq_python", n, 1, "e2e_s_q1")
        ax.axhspan(numpy_low, numpy_high, color=AMBER, alpha=BAND_ALPHA, linewidth=0, zorder=2)
        ax.axhline(t_c_seq / numpy_median, color=AMBER, linewidth=MEASURED_REFERENCE_LINEWIDTH,
                   linestyle=MEASURED_REFERENCE_DASHES, label=LEGEND_LABELS["seq_python"], zorder=2.6)
        tallest = max(tallest, numpy_high)
    # The two reference lines are labelled with their times in a free strip
    # to the right of the last group, where no bar can be underneath.
    # A 3-point offset keeps each label clear of its line (and band).
    label_x = x[-1] + 0.95
    label_style = {"xytext": (0, 3), "textcoords": "offset points", "ha": "right",
                   "va": "bottom", "fontsize": ANNOTATION_FONTSIZE}
    if not math.isnan(numpy_median):
        ax.annotate(f"{numpy_median:.2f} s", (label_x, numpy_high), **label_style)
    ax.annotate(f"{t_c_seq:.2f} s", (label_x, 1.0), **label_style)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$p = {p}$" for p in worker_counts])
    ax.set_xlim(-0.6, x[-1] + 1.0)
    ax.set_xlabel("Workers or threads $p$")
    ax.set_ylim(0.0, tallest * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylabel(r"Speedup over C sequential")
    ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    add_figure_legend(fig, handles, labels, title)
    return fig


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def markdown_table(header: list[str], rows: list[list[str]], text_columns: int = 1) -> list[str]:
    """Markdown table as a list of lines; the first `text_columns` columns left-aligned."""
    alignment = [":---" if i < text_columns else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(alignment) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def fmt(value, spec: str, suffix: str = "") -> str:
    """Format a number, or '-' for a metric that does not apply (NaN)."""
    return "-" if pd.isna(value) else format(value, spec) + suffix


def megabytes(value) -> str:
    """Bytes as decimal MB with one decimal, or '-'."""
    return "-" if pd.isna(value) else f"{value / MEGABYTE_DECIMAL:.1f}"


def ordered(rows: pd.DataFrame) -> pd.DataFrame:
    """Sort by experiment (strong first), size, the fixed mode order, then workers."""
    return (rows.assign(order=rows["mode"].map(MODE_ORDER.index),
                        experiment_order=rows["experiment"].map({"strong": 0, "weak": 1}))
                .sort_values(["experiment_order", "size", "order", "workers"]))


def key_results_markdown(summary: pd.DataFrame, breakdown_size: int, comparison_size: int) -> list[str]:
    """Headline numbers (also the ones quoted in the README)."""
    lines = ["## Key results", ""]
    strong = select(summary, "strong", "distributed")
    for n in sorted(strong["size"].unique()):
        data = strong[strong["size"] == n]
        best = data.loc[data["e2e_s_median"].idxmin()]
        p_max = int(data["workers"].max())
        largest = data[data["workers"] == p_max].iloc[0]
        lines.append(
            f"- Strong scaling, n = {n}: fastest at p = {int(best['workers'])} with "
            f"{best['e2e_s_median']:.3f} s, S = {best['speedup']:.2f} (E = {best['efficiency'] * 100:.0f} %); "
            f"at p = {p_max}: S = {largest['speedup']:.2f} (E = {largest['efficiency'] * 100:.0f} %). "
            f"Against NumPy in one process (same tiles and kernel, no Dask): "
            f"{best['vs_seq_python']:.2f}x at p = {int(best['workers'])}, "
            f"{data[data['workers'] == 1]['vs_seq_python'].iloc[0]:.2f}x at p = 1.")

    weak = select(summary, "weak", "distributed")
    if not weak.empty:
        last = weak.iloc[-1]
        lines.append(f"- Weak scaling: T = {weak.iloc[0]['e2e_s_median']:.3f} s at p = 1 "
                     f"(n = {int(weak.iloc[0]['size'])}) and {last['e2e_s_median']:.3f} s at "
                     f"p = {int(last['workers'])} (n = {int(last['size'])}): "
                     f"E_w = {last['weak_efficiency'] * 100:.0f} %. Data moved between workers: "
                     f"0 MB -> {megabytes(last['comm_bytes_measured'])} MB.")

    t_c = lookup(summary, "strong", "c_baseline", comparison_size, 1, "e2e_s_median")
    if not math.isnan(t_c):
        parts = []
        for mode in ["c_omp_tiled", "distributed"]:
            data = select(summary, "strong", mode, comparison_size)
            best = data.loc[data["e2e_s_median"].idxmin()]
            parts.append(f"{MODE_LABELS[mode]} best {best['e2e_s_median']:.3f} s at p = "
                         f"{int(best['workers'])} ({best['vs_c_baseline']:.1f}x)")
        t_numpy = lookup(summary, "strong", "seq_python", comparison_size, 1, "e2e_s_median")
        parts.append(f"NumPy, 1 process {t_numpy:.3f} s ({t_c / t_numpy:.1f}x)")
        lines.append(f"- Cross-paradigm, n = {comparison_size}: C sequential {t_c:.2f} s; "
                     + "; ".join(parts) + ".")

    setup = select(summary, "strong", "distributed", breakdown_size)
    if not setup.empty:
        lines.append(f"- Cluster setup (one-time, not in e2e): {setup['setup_s_median'].min():.1f} s "
                     f"to {setup['setup_s_median'].max():.1f} s at n = {breakdown_size}.")
    return lines + [""]


def strong_scaling_markdown(summary: pd.DataFrame) -> list[str]:
    """Time, speedup and the cross-paradigm ratios for every strong configuration."""
    lines = ["## Strong scaling: time and speedup", "",
             "Time = median end-to-end time (Dask: pipelined pass on warm workers, cluster setup "
             "excluded; NumPy: the whole block algorithm; C: the kernel time the program prints). "
             "S(p) = T_dist(1) / T_dist(p) (self-relative), E(p) = S(p) / p. "
             "vs. NumPy = T_seq_python / T_dist(p) (> 1: distribution paid off). "
             "vs. C seq. = T_c_baseline / T. vs. C OpenMP = T_c_omp_tiled(p) / T_dist(p).", ""]
    rows = []
    for _, row in ordered(summary[summary["experiment"] == "strong"]).iterrows():
        rows.append([str(row["size"]), MODE_LABELS[row["mode"]], str(row["workers"]),
                     "-" if pd.isna(row["grid"]) else str(row["grid"]),
                     str(row["runs"]), f"{row['e2e_s_median']:.4g}", f"{row['e2e_s_iqr']:.2g}",
                     fmt(row["speedup"], ".2f"), fmt(row["efficiency"] * 100, ".0f", " %"),
                     fmt(row["vs_seq_python"], ".2f", "x"), fmt(row["vs_c_baseline"], ".2f", "x"),
                     fmt(row["vs_c_omp"], ".2f", "x")])
    lines += markdown_table(["Size n", "Mode", "p", "Grid", "Runs", "Median time (s)", "IQR (s)",
                             "S(p)", "E(p)", "vs. NumPy", "vs. C seq.", "vs. C OpenMP"],
                            rows, text_columns=2)
    return lines + [""]


def phases_markdown(summary: pd.DataFrame) -> list[str]:
    """Setup, cold pass, barriered phases and their sum against the pipelined time."""
    lines = ["## Phase breakdown (Dask)", "",
             "Medians, with the IQR in parentheses. Scatter, compute and gather come from the "
             "barriered pass; phase sum = their per-run sum; e2e = the pipelined pass; "
             "difference = (phase sum - e2e) / e2e (negative: the pipelined pass was slower). "
             "Cold = the first full-size pass on fresh worker processes.", ""]
    rows = []
    for _, row in summary[summary["mode"] == "distributed"].sort_values(
            ["experiment", "size", "workers"]).iterrows():
        def cell(column, spec=".3f"):
            return f"{row[f'{column}_median']:{spec}} ({row[f'{column}_iqr']:.2g})"
        difference = (row["phase_sum_s_median"] - row["e2e_s_median"]) / row["e2e_s_median"]
        rows.append([row["experiment"], str(row["size"]), str(row["workers"]), cell("setup_s", ".2f"),
                     cell("cold_e2e_s"), cell("scatter_s"), cell("compute_s"), cell("gather_s"),
                     cell("phase_sum_s"), cell("e2e_s"), f"{difference * 100:+.1f} %"])
    lines += markdown_table(["Experiment", "Size n", "p", "Setup (s)", "Cold e2e (s)",
                             "Scatter (s)", "Compute (s)", "Gather (s)", "Phase sum (s)",
                             "e2e (s)", "Difference"], rows, text_columns=1)
    return lines + [""]


def weak_scaling_markdown(summary: pd.DataFrame) -> list[str]:
    """Weak-scaling time, efficiency and the single-process reference."""
    lines = ["## Weak scaling", "",
             f"n_p = round({config.WEAK_BASE_SIZE} * p^(1/3)), tile grid "
             f"{config.WEAK_TILE_GRID} x {config.WEAK_TILE_GRID} (b_p = ceil(n_p / "
             f"{config.WEAK_TILE_GRID})): the same work per worker at every p. "
             "E_w(p) = T_dist(1, n_1) / T_dist(p, n_p); ideal 100 %.", ""]
    rows = []
    for _, row in select(summary, "weak", "distributed").iterrows():
        t_numpy = lookup(summary, "weak", "seq_python", row["size"], 1, "e2e_s_median")
        rows.append([str(row["workers"]), str(row["size"]), fmt(row["block_size"], ".0f"),
                     str(row["grid"]), f"{row['e2e_s_median']:.4g}", f"{row['e2e_s_iqr']:.2g}",
                     fmt(row["weak_efficiency"] * 100, ".0f", " %"), f"{row['scatter_s_median']:.3f}",
                     f"{row['compute_s_median']:.3f}", fmt(t_numpy, ".4g"),
                     fmt(row["vs_seq_python"], ".2f", "x")])
    lines += markdown_table(["p", "n_p", "b_p", "Grid", "Dask median (s)", "IQR (s)", "E_w(p)",
                             "Scatter (s)", "Compute (s)", "NumPy median (s)", "vs. NumPy"],
                            rows, text_columns=0)
    return lines + [""]


def memory_markdown(summary: pd.DataFrame) -> list[str]:
    """Peak memory per process."""
    lines = ["## Memory", "",
             "Peak resident set size in MB (2^20 bytes), median over the runs, IQR in "
             "parentheses. Worker = a computing process: each Dask worker (largest and mean "
             "over the workers of a run), or the single NumPy or C process. Client = the Dask "
             "driver (client + scheduler). Total = all processes of one run: mean worker x p + "
             "client for Dask (separate processes, separate memory); the single process for "
             "NumPy and C (the OpenMP threads share one address space).", ""]
    rows = []
    for _, row in ordered(summary).iterrows():
        worker_max = f"{row['worker_peak_mb_max_median']:.1f} ({row['worker_peak_mb_max_iqr']:.2g})"
        worker_mean = f"{row['worker_peak_mb_mean_median']:.1f}"
        client = fmt(row["client_peak_mb_median"], ".1f")
        if row["mode"] == "distributed":
            total = row["worker_peak_mb_mean_median"] * row["workers"] + row["client_peak_mb_median"]
        else:
            total = row["worker_peak_mb_max_median"]
        rows.append([row["experiment"], str(row["size"]), MODE_LABELS[row["mode"]],
                     str(row["workers"]), worker_max, worker_mean, client, f"{total:.0f}"])
    lines += markdown_table(["Experiment", "Size n", "Mode", "p", "Worker peak, max (MB)",
                             "Worker peak, mean (MB)", "Client peak (MB)", "Total (MB)"],
                            rows, text_columns=3)
    return lines + [""]


def traffic_markdown(summary: pd.DataFrame, runs: pd.DataFrame) -> list[str]:
    """Data volumes per multiplication and the theoretical-vs-measured check."""
    dask_runs = runs[runs["mode"] == "distributed"]
    matching = int((dask_runs["comm_bytes_measured"] == dask_runs["comm_bytes_theoretical"]).sum())
    lines = ["## Data transfer", "",
             "Bytes moved in one multiplication, in MB (10^6 bytes). Scatter: client to workers "
             "(every A and B tile once). Between workers: theoretical = predicted by the 2D "
             "block-cyclic placement; measured = recorded by Dask's transfer logs in the "
             "barriered pass. Gather: workers to client (every C tile once). Per worker = "
             "between-workers volume / p.", "",
             f"Measured = theoretical in {matching} of {len(dask_runs)} Dask runs.", ""]
    rows = []
    for _, row in summary[summary["mode"] == "distributed"].sort_values(
            ["experiment", "size", "workers"]).iterrows():
        rows.append([row["experiment"], str(row["size"]), str(row["workers"]), str(row["grid"]),
                     fmt(row["block_size"], ".0f"), fmt(row["tasks"], ".0f"),
                     megabytes(row["scatter_bytes"]), megabytes(row["comm_bytes_theoretical"]),
                     megabytes(row["comm_bytes_measured"]),
                     megabytes(row["comm_bytes_measured"] / row["workers"]),
                     megabytes(row["gather_bytes"])])
    lines += markdown_table(["Experiment", "Size n", "p", "Grid", "b", "Tasks", "Scatter (MB)",
                             "Between workers, theoretical (MB)", "Between workers, measured (MB)",
                             "Per worker (MB)", "Gather (MB)"], rows, text_columns=1)
    return lines + [""]


def checksum_markdown(runs: pd.DataFrame) -> list[str]:
    """One checksum per input size, across every mode (C and BLAS round differently)."""
    lines = ["## Checksums", "",
             "Distinct checksums printed (%.10e) per input size, over all modes. The BLAS "
             "kernel rounds differently from the C loops in the last bits, which the 11 printed "
             "digits normally do not show.", ""]
    rows = []
    for n, group in runs.groupby("size"):
        values = sorted(group["checksum"].unique())
        rows.append([str(n), str(len(group)), str(len(values)), ", ".join(values)])
    lines += markdown_table(["Size n", "Runs", "Distinct", "Checksum"], rows, text_columns=0)
    return lines + [""]


def excluded_markdown(failed_runs: pd.DataFrame) -> list[str]:
    """Runs without a valid measurement: reported, never silently dropped."""
    lines = ["## Excluded runs", ""]
    if failed_runs.empty:
        lines.append("None - every run finished successfully within the time budget.")
        return lines
    counts = (failed_runs.groupby(["experiment", "mode", "size", "workers", "status"])
              .size().reset_index(name="runs"))
    rows = [[r["experiment"], r["mode"], str(r["size"]), str(r["workers"]), r["status"], str(r["runs"])]
            for _, r in counts.iterrows()]
    lines += markdown_table(["Experiment", "Mode", "Size n", "p", "Status", "Runs"], rows, text_columns=2)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot the Assignment 4 results.")
    parser.add_argument("--input", type=Path, default=config.RESULTS_DIR / "measurements.csv")
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--breakdown-size", type=int, default=None,
                        help="n for figure (a) (default: the largest strong-scaling size)")
    parser.add_argument("--comparison-size", type=int, default=config.C_COMPARISON_SIZES[0],
                        help="n for figure (d) (default: %(default)s)")
    parser.add_argument("--comparison-workers", type=int, nargs="+", default=[1, 4, 8],
                        help="p values in figure (d) (default: 1 4 8)")
    parser.add_argument("--physical-cores", type=int, default=4,
                        help="mark p = physical cores in (b) and (c); 4 on the reference "
                             "machine, 0 hides the line")
    parser.add_argument("--titles", action="store_true",
                        help="add titles to the figures (for slides; paper figures use captions)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    ok_runs, failed_runs = load_measurements(args.input)
    if ok_runs.empty:
        raise SystemExit(f"No successful runs in {args.input}; nothing to plot.")
    print(f"Loaded {len(ok_runs)} successful runs ({len(failed_runs)} excluded).")
    summary = add_metrics(summarize(ok_runs))

    strong_sizes = sorted(select(summary, "strong", "distributed")["size"].unique())
    if not strong_sizes:
        raise SystemExit("No successful strong-scaling Dask runs; nothing to plot.")
    breakdown_size = args.breakdown_size or int(strong_sizes[-1])
    if breakdown_size not in strong_sizes:
        raise SystemExit(f"No strong-scaling Dask runs at n = {breakdown_size}; "
                         f"measured: {[int(n) for n in strong_sizes]}.")

    def slide_title(text: str) -> str | None:
        return text if args.titles else None

    save_figure(plot_time_breakdown(summary, breakdown_size,
                                    slide_title(f"Time per phase ($n = {breakdown_size}$)")),
                args.output_dir, f"time_breakdown_n{breakdown_size}")
    save_figure(plot_speedup_efficiency(summary, args.physical_cores,
                                        slide_title("Strong scaling")),
                args.output_dir, "strong_scaling_speedup_efficiency")
    if not select(summary, "weak", "distributed").empty:
        save_figure(plot_weak_scaling(summary, args.physical_cores, slide_title("Weak scaling")),
                    args.output_dir, "weak_scaling")
    else:
        print("No weak-scaling runs: figure (c) skipped.")
    if not math.isnan(lookup(summary, "strong", "c_baseline", args.comparison_size, 1, "e2e_s_median")):
        save_figure(plot_cross_paradigm(summary, args.comparison_size, args.comparison_workers,
                                        slide_title(f"Paradigms compared ($n = {args.comparison_size}$)")),
                    args.output_dir, f"cross_paradigm_n{args.comparison_size}")
    else:
        print(f"No C baseline at n = {args.comparison_size}: figure (d) skipped.")

    lines = ["# Assignment 4 - Results Summary", "",
             "Generated by `plot_results.py` from `results/measurements.csv`. Statistics use "
             "successful runs only: median and IQR (Q3 - Q1) over the repetitions of each "
             "configuration. Every speedup and ratio uses median times. "
             f"Block kernel of the Python modes: {config.KERNEL} (NumPy matmul, one BLAS thread "
             "per process).", ""]
    lines += key_results_markdown(summary, breakdown_size, args.comparison_size)
    lines += strong_scaling_markdown(summary)
    lines += phases_markdown(summary)
    lines += weak_scaling_markdown(summary)
    lines += memory_markdown(summary)
    lines += traffic_markdown(summary, ok_runs)
    lines += checksum_markdown(ok_runs)
    lines += excluded_markdown(failed_runs)

    path = args.output_dir / "results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
