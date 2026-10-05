"""Figure: the retention framework, detailed variant (schematic).

(a) calibration, once per model (Algorithm 3 for tau);
(b) one decoding step with the cache strip, the ring buffer, the confidence
    check and the full-cache fallback (Algorithm 2);
(c) what a step reads on the two paths, to scale for k = 196.

Output: figures/fig_t11_framework_v2.png|pdf
Usage:  python experiments/make_figure_framework2.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from make_figures_track import BLUE, CM, FULL, INK, INK2, MUTED, ORANGE, save

PALE_B, PALE_O, PALE_G = "#e6f0fb", "#fdebe3", "#f1f0ec"
SPEECH, PAD = "#dce8f7", "#eeeeea"
DASH = (0, (3, 2))


def badge(ax, x, y, num, ec):
    ax.add_patch(Circle((x, y), 1.75, fc="white", ec=ec, lw=0.7, zorder=5))
    ax.text(x, y - 0.1, str(num), ha="center", va="center", fontsize=6, color=INK, zorder=6)


def box(ax, x, y, w, h, head=None, text=None, fc=PALE_G, ec=INK2, num=None, fs=6.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc=fc, ec=ec, lw=0.7, ls=ls))
    if head:
        ax.text(x + w / 2, y + h - 1.5, head, ha="center", va="top", fontsize=7, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.3, text or "", ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.25)
    elif text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK2, linespacing=1.25)
    if num:
        badge(ax, x + 0.4, y + h - 0.4, num, ec)        # on the corner, clear of the heading


def arrow(ax, p, q, color=INK2, ls="-", head=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>" if head else "-", mutation_scale=7, lw=0.8, color=color,
                                 ls=ls, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(FULL, 12.8 * CM))
    ax.set_xlim(-1, 172)
    ax.set_ylim(-20, 106)
    ax.axis("off")
    t = dict(fontsize=6.0, color=INK2, va="center")
    small = {**t, "fontsize": 5.6}

    # ------------------------------------------------ (a) calibration
    ax.text(1, 103, "(a)  Calibration, per model", fontsize=8, fontweight="bold", color=INK, va="center")
    box(ax, 1, 88, 54, 10, "Validation set (300 utterances)", "reference: full cache;  gate δ = 0.2·WER$_{full}$",
        fc="white", ec=MUTED, fs=5.8)
    box(ax, 1, 71, 54, 13, "Budget  $K_i$", "split rule ($f_r$, $f_p$): smallest kept set\nwithin the gate", num=1)
    box(ax, 1, 55, 54, 13, "Sink mass  ρ", "smallest padding set holding ρ\nof the first-step padding mass", num=2)
    box(ax, 1, 21, 54, 31, "Fallback threshold  τ", "arms in order of cost:\nno fallback, 0.6, 0.7, 0.8, 0.9, 0.95",
        fc=PALE_O, ec=ORANGE, num=3)
    ax.add_patch(Polygon([[23, 38.5], [38, 33], [23, 27.5], [8, 33]], closed=True, fc="white", ec=ORANGE, lw=0.7))
    ax.text(23, 33, "U < δ and no new\nlooping utterance?", ha="center", va="center", fontsize=5.5, color=INK)
    box(ax, 43, 30, 10.5, 6, text="take τ*", fc="white", ec=ORANGE)
    arrow(ax, (38, 33), (43, 33), color=ORANGE)
    ax.text(40.4, 34.7, "yes", ha="center", **small)
    ax.text(28, 24.5, "no → next arm;\nafter the last arm → FULL CACHE", ha="center", linespacing=1.15, **small)
    box(ax, 1, 6, 54, 11, "Configuration", "$K_i$,  ρ,  τ*   (or the full cache)", fc="white", ec=INK2)
    for y0, y1 in ((88, 84), (71, 68), (55, 52), (21, 17)):
        arrow(ax, (28, y0), (28, y1))

    # ------------------------------------------------ (b) decoding step
    ax.text(62, 103, "(b)  Decoding step t", fontsize=8, fontweight="bold", color=INK, va="center")
    x0, x1, ys, hs = 62.0, 169.5, 87.0, 5.5
    xs = x0 + 0.2 * (x1 - x0)
    ax.add_patch(Rectangle((x0, ys), xs - x0, hs, fc=SPEECH, ec="none"))
    ax.add_patch(Rectangle((xs, ys), x1 - xs, hs, fc=PAD, ec="none"))
    ax.add_patch(Rectangle((66.5, ys), 10.5, hs, fc=BLUE, ec="none", alpha=0.45))
    for xb in (88.5, 91.5, 100, 119, 142):
        ax.add_patch(Rectangle((xb, ys), 0.9, hs, fc=ORANGE, ec="none"))
    ax.add_patch(Rectangle((x0, ys), x1 - x0, hs, fc="none", ec=INK2, lw=0.7))
    ax.plot([69.5], [ys + hs / 2], "o", ms=3.2, color=INK)
    ax.text(x0, 98.4, "Full cross-attention cache K, V — 1500 positions per layer (stays in DRAM)", fontsize=6.6, color=INK, va="center")
    ax.text(62, 94.4, "window $W_t$ around the peak $c_t$", ha="left", **t)
    ax.text(62, 84.6, "speech ($n_r$)", ha="left", **t)
    ax.text(112, 84.6, "padding (≈ 80 %)", ha="center", **t)
    ax.text(150, 84.6, "orange: padding sink $S_ℓ$", ha="center", **t)

    box(ax, 102.5, 67.5, 43, 13, text="First step (t = 1) on the full cache:\ncomputes K, V, selects $S_ℓ$\nand the start peak $c_1$",
        fc="white", ec=MUTED, ls=DASH, fs=5.8)
    box(ax, 62, 46, 38, 16, "Ring buffer  $B_ℓ$", "$S_ℓ ∪ W_t$\nk = 0.5·$K_i$ positions per layer", fc=PALE_B, ec=BLUE, num=4)
    box(ax, 106, 46, 26, 16, "Decoder step", "reads only $B_ℓ$ →\nlogits z, attention a", fc=PALE_B, ec=BLUE, num=5)
    ax.add_patch(Polygon([[150, 63.5], [165, 54], [150, 44.5], [135, 54]], closed=True, fc=PALE_O, ec=ORANGE, lw=0.7))
    ax.text(150, 54, "max softmax(z)\n≥ τ* ?", ha="center", va="center", fontsize=6.2, color=INK)
    badge(ax, 141.5, 60.4, 6, ORANGE)
    box(ax, 112, 6, 57.5, 18, "Redo on the full cache",
        "same step on 1500 positions;\nz, a and the decoder cache from this run;\nthe peak is then searched over all audio",
        fc=PALE_O, ec=ORANGE, num=7)
    box(ax, 62, 6, 42, 18, "Emit $y_{t+1}$, move window",
        "c ← max(c, alignment peak);\noverwrite only the slots\nthat left the window", fc=PALE_B, ec=BLUE, num=8)

    # ring-buffer cells
    cw, cy, ch = 4.1, 31.2, 4.6
    cells = [("108", "white", ORANGE), ("109", "white", ORANGE)] + [(str(n), PALE_B, BLUE) for n in range(103, 108)] \
        + [("s", PALE_O, ORANGE)] * 2
    for i, (lab, fc, ec) in enumerate(cells):
        ax.add_patch(Rectangle((62 + i * cw, cy), cw, ch, fc=fc, ec=ec, lw=0.6))
        ax.text(62 + (i + 0.5) * cw, cy + ch / 2, lab, ha="center", va="center", fontsize=4.8, color=INK)
    ax.text(62 + cw, 29.1, "entering", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62 + 4.5 * cw, 29.1, "window", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62 + 8 * cw, 29.1, "sink", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62, 38.2, "slots of $B_ℓ$ (one layer)", ha="left", **small)
    ax.plot([63.5, 63.5], [46, 39.8], color=BLUE, lw=0.6, ls=(0, (1, 1.5)))

    arrow(ax, (79, ys), (79, 62), color=MUTED)                          # cache -> buffer
    ax.text(80.5, 77.3, "slots entering\nthe window", linespacing=1.15, **t)
    arrow(ax, (102.5, 68.5), (97, 62), color=MUTED, ls=DASH)                # first step -> buffer
    arrow(ax, (100, 54), (106, 54))
    arrow(ax, (132, 54), (135, 54))
    arrow(ax, (150, 44.5), (150, 24), color=ORANGE)                     # no -> redo
    ax.text(151.5, 34, "no:\nfallback", linespacing=1.15, **t)
    arrow(ax, (167.6, ys), (167.6, 24.2), color=MUTED, ls=DASH)         # cache -> redo
    ax.text(166.4, 75, "all 1500\npositions", ha="right", linespacing=1.15, **t)
    arrow(ax, (142.6, 49.3), (104.3, 24.4), color=BLUE)                 # yes -> emit
    ax.text(127, 33.6, "yes: fast path", ha="left", **t)
    arrow(ax, (112, 15), (104, 15), color=ORANGE)                       # redo -> emit
    arrow(ax, (101.8, 24), (101.8, 43.5), color=BLUE, head=False)       # emit -> buffer
    arrow(ax, (101.8, 43.5), (99.6, 46.1), color=BLUE)
    ax.text(103, 33.5, "t + 1", **t)
    ax.text(150, 65.6, "τ* from (a)", ha="center", **t)

    # configuration feeds the step
    arrow(ax, (55, 11.5), (58.5, 11.5), color=MUTED, ls=DASH, head=False)
    arrow(ax, (58.5, 11.5), (58.5, 54), color=MUTED, ls=DASH, head=False)
    arrow(ax, (58.5, 54), (61.8, 54), color=MUTED, ls=DASH)

    # ------------------------------------------------ (c) what a step reads
    ax.plot([0, 171], [1.2, 1.2], color="#d9d8d3", lw=0.6)
    ax.text(62, -2.6, "(c)  What a step reads", fontsize=8, fontweight="bold", color=INK, va="center")
    full_len = 70.0
    ax.add_patch(Rectangle((62, -9.0), full_len * 196 / 1500, 2.8, fc=BLUE, ec="none"))
    ax.add_patch(Rectangle((62, -14.6), full_len, 2.8, fc=ORANGE, ec="none"))
    ax.text(62 + full_len * 196 / 1500 + 1.5, -7.6, "fast path: k positions per layer (here k = 196, to scale)", **t)
    ax.text(62 + full_len + 1.5, -13.2, "fallback: 1500 positions per layer", **t)
    # legend
    ax.text(1, -2.6, "Legend", fontsize=7, fontweight="bold", color=INK, va="center")
    ax.add_patch(Rectangle((1, -7.4), 3.4, 2.4, fc=PALE_B, ec=BLUE, lw=0.7))
    ax.text(6.5, -6.2, "PadSink-Track fast path", **t)
    ax.add_patch(Rectangle((1, -10.9), 3.4, 2.4, fc=PALE_O, ec=ORANGE, lw=0.7))
    ax.text(6.5, -9.7, "padding sink / fallback", **t)
    arrow(ax, (0.8, -13.4), (4.8, -13.4))
    ax.text(6.5, -13.4, "every step", **t)
    arrow(ax, (0.8, -16.9), (4.8, -16.9), color=MUTED, ls=DASH)
    ax.text(6.5, -16.9, "calibration, first step or fallback only", **t)
    save(fig, "fig_t11_framework_v2")


if __name__ == "__main__":
    main()
