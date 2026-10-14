"""
plot_results.py - Analysis and figures for Assignment 1.

Pipeline:
    1. Load results/measurements.csv (written by run_benchmark.py) and keep
       only successful runs (status == "ok"). Timeouts and errors have no
       time/memory values, so they are excluded from the statistics, but they
       are counted and listed in the summary so nothing is hidden.
    2. For every (size, language): median and interquartile range (IQR) of
       the kernel time and of the peak memory.
    3. Two figures, each saved as PDF (LaTeX) and PNG (quick view, slides):
         time_vs_size   - log-log, with an O(n^3) reference line
         memory_vs_size - log-log, with the theoretical size of the matrices
    4. results_summary.md - the same statistics as a Markdown table, plus the
       empirical scaling exponent of each language.

Style: every visual choice follows common/plotting_style.md and is applied
in ONE place, apply_plot_style(). Figures are drawn on a 5.6 in canvas and
meant to be included at \\columnwidth (3.5 in) in LaTeX, i.e. scaled by 0.625.

Run (from anywhere):
    python3 assignment_1/plot_results.py               # paper figures (no titles)
    python3 assignment_1/plot_results.py --titles      # with titles, for slides
    python3 assignment_1/plot_results.py --input other.csv --output-dir somewhere/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # render to files only; no window/display is needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import benchmark_config as config  # noqa: E402

# ---------------------------------------------------------------------------
# Style constants (see common/plotting_style.md)
# ---------------------------------------------------------------------------

# Fixed order and color per language: color follows the language, never its rank.
# The three colors were checked with a palette validator (lightness, contrast,
# color-blindness separation) - see the style guide before changing them.
LANGUAGES = {
    "c":      {"label": "C",      "color": "#1f43b2"},   # deep blue
    "java":   {"label": "Java",   "color": "#9333ea"},   # vibrant purple
    "python": {"label": "Python", "color": "#dc143c"},   # crimson
}
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
    failed_runs = df[df["status"] != "ok"].copy()   # "timeout" or "error"
    return ok_runs, failed_runs


def summarize(ok_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Median, first/third quartile and IQR of time and memory per (size, language).

    Quartiles use pandas' default linear interpolation. With 5 repetitions,
    Q1 and Q3 are exactly the 2nd and 4th smallest values, so the IQR is
    the spread of the middle three runs - robust to one slow outlier run.
    """
    grouped = ok_runs.groupby(["size", "language"])
    summary = pd.DataFrame({
        "runs":          grouped["time_s"].count(),
        "time_median":   grouped["time_s"].median(),
        "time_q1":       grouped["time_s"].quantile(0.25),
        "time_q3":       grouped["time_s"].quantile(0.75),
        "memory_median": grouped["memory_mb"].median(),
        "memory_q1":     grouped["memory_mb"].quantile(0.25),
        "memory_q3":     grouped["memory_mb"].quantile(0.75),
    }).reset_index()

    summary["time_iqr"] = summary["time_q3"] - summary["time_q1"]
    summary["memory_iqr"] = summary["memory_q3"] - summary["memory_q1"]
    return summary


def scaling_exponent(sizes: np.ndarray, times: np.ndarray) -> float:
    """
    Slope of a straight-line fit of log(time) against log(n).

    If time = k * n^b, then log(time) = log(k) + b * log(n), so the slope is
    the empirical exponent b. The O(n^3) model predicts b = 3; a larger value
    means the program slows down faster than the operation count alone
    explains (typically cache misses once the matrices no longer fit in cache).
    """
    slope, _intercept = np.polyfit(np.log(sizes), np.log(times), deg=1)
    return slope


# ---------------------------------------------------------------------------
# Plotting building blocks
# ---------------------------------------------------------------------------

def plot_median_with_iqr(ax, summary: pd.DataFrame, metric: str) -> None:
    """
    Draw one shaded band (Q1..Q3) and one thick line (median) per language.
    `metric` is the column prefix: "time" or "memory".
    The bands get a lower zorder so every median line stays on top of them.
    """
    for language, style in LANGUAGES.items():
        data = summary[summary["language"] == language].sort_values("size")
        if data.empty:
            continue   # e.g. a language that timed out at every size

        ax.fill_between(data["size"], data[f"{metric}_q1"], data[f"{metric}_q3"],
                        color=style["color"], alpha=BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(data["size"], data[f"{metric}_median"],
                color=style["color"], label=style["label"], zorder=3)


def plot_reference(ax, sizes: np.ndarray, values: np.ndarray, label: str) -> None:
    """Draw a theoretical curve: charcoal, loosely dashed, thinner than the data."""
    ax.plot(sizes, values, color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH,
            linestyle=REFERENCE_DASHES, label=label, zorder=2.5)


def format_size_axis(ax, sizes: np.ndarray) -> None:
    """Log x-axis with a tick at every measured size, shown as plain numbers."""
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())   # hide the 200, 300, ... minor ticks
    ax.set_xlabel("Matrix size $n$")


def add_legend(fig: plt.Figure, ax, title: str | None = None) -> None:
    """
    One-row, frameless legend ABOVE the plot area, so it can never cover data.
    "outside upper center" lets the constrained layout reserve room for it.

    The optional figure title (slides only) is drawn as the legend's title:
    it then sits directly above the legend row, and the layout engine keeps
    the two from overlapping (a separate suptitle would collide with it).
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(handles),
               title=title, title_fontproperties=TITLE_FONT)


def reference_sizes(summary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Measured sizes, plus a dense log-spaced grid for drawing smooth reference curves."""
    sizes = np.sort(summary["size"].unique())
    dense = np.geomspace(sizes[0], sizes[-1], 100)
    return sizes, dense


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_time(summary: pd.DataFrame, title: str | None) -> plt.Figure:
    """Figure (a): kernel time vs. matrix size, log-log, with an O(n^3) reference."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    plot_median_with_iqr(ax, summary, "time")

    # O(n^3) reference: time = k * n^3. On a log-log plot this is a straight
    # line with slope 3, so comparing slopes compares growth rates directly.
    # Only the SLOPE carries meaning; k just shifts the line up or down.
    # We place it at the geometric mean of the fastest and slowest medians
    # at the smallest size, so it runs between the curves without hiding any.
    sizes, dense = reference_sizes(summary)
    smallest = summary[summary["size"] == sizes[0]]
    anchor_time = np.sqrt(smallest["time_median"].min() * smallest["time_median"].max())
    k = anchor_time / sizes[0] ** 3
    plot_reference(ax, dense, k * dense ** 3, r"$O(n^3)$ reference")

    format_size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("Execution time (s)")
    add_legend(fig, ax, title)
    return fig


def plot_memory(summary: pd.DataFrame, title: str | None) -> plt.Figure:
    """Figure (b): peak memory vs. matrix size, log-log, with the size of the data."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    plot_median_with_iqr(ax, summary, "memory")

    # Theoretical minimum: the three matrices A, B and C of n*n doubles.
    # Anything above this line is runtime overhead (interpreter, JVM,
    # boxed Python floats, input buffers). It grows as n^2 (slope 2).
    sizes, dense = reference_sizes(summary)
    matrix_mb = 3 * dense ** 2 * BYTES_PER_DOUBLE / BYTES_PER_MB
    plot_reference(ax, dense, matrix_mb, r"Matrix data ($24n^2$ B)")   # 3 matrices x 8n^2 bytes

    format_size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_ylabel("Peak memory, RSS (MB)")
    add_legend(fig, ax, title)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save the same figure as vector PDF (for LaTeX) and 300-dpi PNG."""
    for extension in ("pdf", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


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


def write_summary(summary: pd.DataFrame, failed_runs: pd.DataFrame, path: Path) -> None:
    lines = ["# Assignment 1 - Results Summary", "",
             "Generated by `plot_results.py` from `results/measurements.csv`.",
             "Time = kernel only (monotonic clock). Memory = peak resident set size "
             "of the whole process (GNU time). IQR = Q3 - Q1.", ""]

    # Main table, ordered by size and then by the fixed language order.
    language_order = list(LANGUAGES)
    ordered = summary.assign(order=summary["language"].map(language_order.index))
    ordered = ordered.sort_values(["size", "order"])

    rows = []
    for _, row in ordered.iterrows():
        rows.append([
            str(row["size"]),
            LANGUAGES[row["language"]]["label"],
            str(row["runs"]),
            f"{row['time_median']:.4g}",
            f"{row['time_iqr']:.2g}",
            f"{row['memory_median']:.2f}",
            f"{row['memory_iqr']:.2f}",
        ])
    lines += markdown_table(
        ["Size n", "Language", "Runs", "Median time (s)", "IQR time (s)",
         "Median memory (MB)", "IQR memory (MB)"], rows, text_columns=2)

    # Empirical exponent: how close each language is to the O(n^3) model.
    lines += ["", "## Empirical scaling exponent", "",
              "Slope of a least-squares fit of log(median time) vs. log(n). "
              "The O(n^3) model predicts 3.", ""]
    rows = []
    for language, style in LANGUAGES.items():
        data = summary[summary["language"] == language].sort_values("size")
        if len(data) >= 2:
            exponent = scaling_exponent(data["size"].to_numpy(), data["time_median"].to_numpy())
            sizes_used = f"{data['size'].min()}-{data['size'].max()}"
            rows.append([style["label"], f"{exponent:.2f}", sizes_used])
    lines += markdown_table(["Language", "Exponent", "Sizes used"], rows)

    # Excluded runs: reported, never silently dropped.
    lines += ["", "## Excluded runs", ""]
    if failed_runs.empty:
        lines.append("None - every run finished successfully within the time budget.")
    else:
        counts = failed_runs.groupby(["size", "language", "status"]).size().reset_index(name="runs")
        rows = [[str(r["size"]), LANGUAGES[r["language"]]["label"], r["status"], str(r["runs"])]
                for _, r in counts.iterrows()]
        lines += markdown_table(["Size n", "Language", "Status", "Runs"], rows, text_columns=3)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot benchmark results.")
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

    summary = summarize(ok_runs)
    time_title = "Execution time vs. matrix size" if args.titles else None
    memory_title = "Peak memory vs. matrix size" if args.titles else None

    save_figure(plot_time(summary, time_title), args.output_dir, "time_vs_size")
    save_figure(plot_memory(summary, memory_title), args.output_dir, "memory_vs_size")
    write_summary(summary, failed_runs, args.output_dir / "results_summary.md")


if __name__ == "__main__":
    main()
