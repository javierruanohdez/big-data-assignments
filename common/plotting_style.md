# Plotting Style Guide
Project-wide rules for every figure in this repository (Assignments 1–4).
The goal is striking, modern figures that drop straight into an IEEE two-column paper and also work on slides.
The rules are implemented once in code (`apply_plot_style()` in each `plot_results.py`), so following this guide means calling that function.

## 1. Figure geometry: design large, print at column width
- Design canvas: **5.6 in × 4.2 in**, layout `constrained` (no `bbox_inches="tight"`).
- In LaTeX, include every figure at the column width:
  `\includegraphics[width=\columnwidth]{time_vs_size.pdf}`
- This scales the figure by 3.5 / 5.6 = **0.625**. All sizes below are canvas sizes; the printed size is given alongside.
  The large canvas gives bold typography on screen and slides; in print, axis labels land at 8.75 pt (IEEE asks for 8–10 pt) and tick/legend text at 7.5 pt.

## 2. Typography
| Element | Canvas size | Printed size |
|---|---|---|
| Axis labels | **14 pt** | 8.75 pt |
| Tick labels | **12 pt** | 7.5 pt |
| Legend | **12 pt** | 7.5 pt |
| Titles (slides only, `--titles`) | **16 pt**, bold | — |

- Font family: Serif (Times New Roman → Liberation Serif → STIX → DejaVu Serif), matching IEEE body text.
- Math: `mathtext.fontset = "stix"`.
- No titles in paper figures: the LaTeX caption is the title. `plot_results.py --titles` adds them for slides.

## 3. Colors (fixed assignment)
| Series | Color | Hex |
|---|---|---|
| C | Deep blue | `#1f43b2` |
| Java | Vibrant purple | `#9333ea` |
| Python | Crimson | `#dc143c` |
| Theoretical / reference curves | Charcoal | `#3d3d3d` |
| Text and axes | Near-black | `#1f1f1e` |
| Grid lines | Light gray | `#c8c8c5` (at alpha 0.6) |

- **Validated, not picked by eye.** On a white background, the three series colors pass the palette validator's lightness, chroma, contrast (≥ 3:1) and color-blindness checks for every pair. The worst pair, blue ↔ purple under protanopia, has ΔE = 10.2 (target ≥ 8).
- The blue is the darkest blue that still passes the lightness check. A darker navy starts to read as the black of the text and axes.
- **Grayscale print:** purple (L\* 45) and crimson (L\* 47) become almost the same gray. In grayscale, Java and Python are identified by the legend and by their position, not by shade.
- **Color follows the language**, in every figure and every assignment. A 4th series (later assignments) uses amber `#c98500` (validated together with the three above; it also clears 3:1 contrast).

## 4. Marks
- Median: solid line, **2.5 pt** (≈ 1.6 pt printed), round caps and joins, no markers.
- Variability (IQR): `fill_between` band in the series color, **alpha 0.22**, no edge line, drawn underneath the medians.
- Theoretical model: charcoal, **loosely dashed** `(0, (5, 6))`, **1.75 pt** (thinner than the data), labeled in the legend. Placed so that it hides no data: on a log-log plot only its slope matters.
- Markers / error bars: **Not used.**

## 5. Axes and grid
- White background.
- Top and right spines removed (despine); left and bottom spines 1.0 pt, near-black.
- Ticks point outward, 1.0 pt wide, 5 pt long.
- Grid: **major ticks only**, dashed, light gray, alpha 0.6, 0.8 pt, drawn **behind** the data.
- Log scales use plain numbers (`100`, `1000`) on the size axis.

## 6. Legend
- No frame (`frameon=False`).
- **Above the plot area, in a single row** (`fig.legend(loc="outside upper center", ncols=4)`; with `--titles`, the title sits above it), so it never covers data, whatever the data looks like.
- Order: the languages in their fixed order (C, Java, Python), then the reference curve.

## 7. Export
- `.pdf`: vector, fonts embedded as TrueType (`pdf.fonttype = 42`). Use this in LaTeX.
- `.png`: 300 dpi on the canvas (≈ 480 dpi at printed size), white background. For quick viewing and slides.
