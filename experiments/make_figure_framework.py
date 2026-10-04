"""Figure: the retention framework (schematic).

Left: what is calibrated once per model on validation data (K_i, rho, tau).
Right: one decoding step -- the ring buffer (sink + tracked window), the
confidence check and the full-cache fallback.

Output: figures/fig_t11_framework.png|pdf
Usage:  python experiments/make_figure_framework.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from make_figures_track import BLUE, CM, FULL, INK, INK2, MUTED, ORANGE, save

PALE_B, PALE_O, PALE_G = "#e6f0fb", "#fdebe3", "#f1f0ec"


def box(ax, x, y, w, h, text, fc=PALE_G, ec=INK2, fs=7, bold=None, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
    if bold:
        ax.text(x + w / 2, y + h - 1.6, bold, ha="center", va="top", fontsize=fs, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h - 5.4, text, ha="center", va="top", fontsize=fs - 0.6, color=INK2, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.25)


def arrow(ax, p, q, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color, ls=ls,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(FULL, 8.6 * CM))
    ax.set_xlim(0, 172)
    ax.set_ylim(0, 86)
    ax.axis("off")

    # ---- calibration (left)
    ax.text(1, 82.5, "(a)  Calibration (per model)", fontsize=8, fontweight="bold", color=INK, va="center")
    box(ax, 1, 59, 52, 18, "split rule (f_r, f_p): smallest set\nwithin the gate δ = 0.2·WER", bold="Budget  K_i")
    box(ax, 1, 37, 52, 18, "smallest padding set holding ρ\nof the first-step padding mass", bold="Sink mass  ρ")
    box(ax, 1, 3, 52, 30, "cheapest arm of\n{no fallback, 0.6, … , 0.95} with\nupper CI bound of ΔWER < δ and\nno new looping utterance;\nelse the full cache is kept",
        fc=PALE_O, ec=ORANGE, bold="Fallback threshold  τ")
    arrow(ax, (27, 59), (27, 55))
    arrow(ax, (27, 37), (27, 33))

    # ---- decoding step (right)
    ax.text(62, 82.5, "(b)  Decoding step t", fontsize=8, fontweight="bold", color=INK, va="center")
    box(ax, 62, 64, 108, 12, "Full cross-attention cache K, V  (1500 positions per layer, stays in DRAM)", fc="white", ec=MUTED)
    box(ax, 62, 40, 40, 16, "sink S_ℓ + window W_t\n(k = 0.5·K_i positions)", fc=PALE_B, ec=BLUE, bold="Ring buffer")
    box(ax, 109, 40, 23, 16, "reads k\npositions", fc=PALE_B, ec=BLUE, bold="Decoder step")
    ax.add_patch(Polygon([[152, 57], [166, 48], [152, 39], [138, 48]], closed=True, fc=PALE_O, ec=ORANGE, lw=0.7))
    ax.text(152, 48, "top-1 prob.\n≥ τ ?", ha="center", va="center", fontsize=6.6, color=INK)
    box(ax, 110, 8, 60, 16, "same step on 1500 positions;\nits output replaces the step", fc=PALE_O, ec=ORANGE, bold="Redo on the full cache")
    box(ax, 62, 8, 40, 16, "window follows the\nalignment-head peak", fc=PALE_B, ec=BLUE, bold="Emit token, move window")

    arrow(ax, (82, 64), (82, 56), color=MUTED)                            # cache -> buffer
    ax.text(83.5, 60, "slots entering the window", fontsize=6.2, color=INK2, va="center")
    arrow(ax, (102, 48), (109, 48))
    arrow(ax, (132, 48), (138, 48))
    arrow(ax, (152, 39), (152, 24), color=ORANGE)                         # no -> redo
    ax.text(153.5, 31.5, "no", fontsize=6.6, color=INK2, va="center")
    arrow(ax, (168.5, 64), (168.5, 24.2), color=MUTED, ls=(0, (3, 2)))    # cache -> redo
    arrow(ax, (145, 43.5), (100, 24.2), color=BLUE)                       # yes -> emit
    ax.text(117, 35.8, "yes", fontsize=6.6, color=INK2, va="center")
    arrow(ax, (110, 16), (102, 16), color=ORANGE)                         # redo -> emit
    arrow(ax, (70, 24), (70, 40), color=BLUE)                             # emit -> buffer
    ax.text(71.5, 32, "t + 1", fontsize=6.6, color=INK2, va="center")

    # calibration feeds the step
    arrow(ax, (53, 68), (61.5, 54), color=MUTED, ls=(0, (3, 2)))
    arrow(ax, (53, 46), (61.5, 47), color=MUTED, ls=(0, (3, 2)))
    ax.text(152, 59.3, "τ from (a)", fontsize=6.2, color=INK2, ha="center", va="center")
    save(fig, "fig_t11_framework")


if __name__ == "__main__":
    main()
