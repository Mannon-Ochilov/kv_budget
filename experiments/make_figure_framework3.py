"""Figure: the retention framework (schematic), final variant.

(a) calibration, once per model, with the two acceptance conditions of the
    tau rule drawn as separate checks (Algorithm 3);
(b) one decoding step with the cache strip, the ring buffer, the confidence
    check and the full-cache fallback (Algorithm 2);
(c) what a step reads on the two paths, to scale for k = 196.

Output: figures/fig_t11_framework_v3.png|pdf
Usage:  python experiments/make_figure_framework3.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from make_figures_track import BLUE, CM, FULL, INK, INK2, MUTED, ORANGE, save

PALE_B, PALE_O, PALE_G = "#e6f0fb", "#fdebe3", "#f1f0ec"
PAD = "#eeeeea"
DASH = (0, (3, 2))
X0, X1, Y0, Y1 = -1, 172, -28, 108


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


def arrow(ax, *pts, color=INK2, ls="-", head=True):
    """Polyline through pts; the arrowhead sits on the last segment."""
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.8, ls=ls, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=7, lw=0.8,
                                 color=color, ls=ls, shrinkA=0, shrinkB=0))


def diamond(ax, cx, cy, hw, hh, text, fc="white", fs=5.5):
    ax.add_patch(Polygon([[cx, cy + hh], [cx + hw, cy], [cx, cy - hh], [cx - hw, cy]], closed=True, fc=fc, ec=ORANGE, lw=0.7))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)


def main():
    fig, ax = plt.subplots(figsize=(FULL, FULL * (Y1 - Y0) / (X1 - X0)))   # equal units: badges stay round
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    t = dict(fontsize=6.0, color=INK2, va="center")
    small = {**t, "fontsize": 5.4}

    # ------------------------------------------------ (a) calibration
    ax.text(1, 105.5, "(a)  Calibration, once per model", fontsize=8, fontweight="bold", color=INK, va="center")
    box(ax, 1, 91.5, 54, 9, "Validation set (300 utterances)", "full cache → WER$_{full}$;  gate δ = 0.2·WER$_{full}$",
        fc="white", ec=MUTED, fs=5.8)
    box(ax, 1, 77, 54, 12, "Budget  $K_i$", "split rule ($f_r$, $f_p$): smallest kept set\nwithin the gate δ", num=1)
    box(ax, 1, 62.5, 54, 12, "Sink mass  ρ", "smallest padding set holding ρ of the\nfirst-step padding mass;  |$S_ℓ$| ≤ 0.25·k", num=2)
    box(ax, 1, 8.5, 54, 51.5, "Fallback threshold  τ",
        "arms in order of cost: no fallback, 0.6, 0.7,\n0.8, 0.9, 0.95.  U = upper 95 % CI bound\n"
        "of WER$_{arm}$ − WER$_{full}$ (paired bootstrap)", fc=PALE_O, ec=ORANGE, num=3)
    for ya, yb in ((91.5, 89), (77, 74.5), (62.5, 60)):
        arrow(ax, (28, ya), (28, yb))

    diamond(ax, 19, 36, 15, 3.8, "(i)  U < δ ?")
    diamond(ax, 19, 23.5, 15, 5, "(ii)  new utterance\nwith WER > 1 ?")
    box(ax, 40, 28, 13, 5, text="next arm", fc="white", ec=INK2, fs=5.8)
    box(ax, 8, 10.5, 22, 4.5, text="take this arm:  τ*", fc="white", ec=ORANGE, fs=5.8)
    box(ax, 37, 10.5, 16.5, 4.5, text="FULL CACHE", fc="white", ec=INK2, fs=5.8)
    arrow(ax, (19, 32.2), (19, 28.5), color=ORANGE)                         # (i) yes -> (ii)
    ax.text(20.3, 30.4, "yes", **small)
    arrow(ax, (19, 18.5), (19, 15), color=ORANGE)                           # (ii) no -> take
    ax.text(20.3, 16.9, "no", **small)
    arrow(ax, (34, 36), (46.5, 36), (46.5, 33))                             # (i) no -> next arm
    ax.text(36.5, 37.4, "no", ha="center", **small)
    arrow(ax, (34, 23.5), (38.3, 23.5), (38.3, 30.5), (40, 30.5))               # (ii) yes -> next arm
    ax.text(36.1, 24.9, "yes", ha="center", **small)
    arrow(ax, (53, 30.5), (54.1, 30.5), (54.1, 43.8), (19, 43.8), (19, 39.8))   # next arm -> (i)
    arrow(ax, (46.5, 28), (46.5, 15), color=MUTED, ls=DASH)                 # last arm -> full cache
    ax.text(45.7, 20.6, "after the\nlast arm", ha="right", linespacing=1.15, **small)

    box(ax, 1, -4, 54, 9, "Configuration", "$K_i$,  ρ,  τ*   (or the full cache)", fc="white", ec=INK2)
    arrow(ax, (19, 10.5), (19, 5))
    arrow(ax, (46.5, 10.5), (46.5, 5), color=MUTED, ls=DASH)

    # ------------------------------------------------ (b) decoding step
    ax.text(62, 105.5, "(b)  Decoding step t", fontsize=8, fontweight="bold", color=INK, va="center")
    x0, x1, ys, hs = 62.0, 169.5, 89.0, 5.5
    xs = x0 + 0.2 * (x1 - x0)                        # end of speech
    wx, ww = 67.0, 10.5                              # window: the peak sits 0.1 w from its left edge
    ax.add_patch(Rectangle((x0, ys), xs - x0, hs, fc="white", ec="none"))
    ax.add_patch(Rectangle((xs, ys), x1 - xs, hs, fc=PAD, ec="none"))
    ax.add_patch(Rectangle((wx, ys), ww, hs, fc=BLUE, ec="none", alpha=0.45))
    for xb in (88.5, 91.5, 100, 119, 142):
        ax.add_patch(Rectangle((xb, ys), 0.9, hs, fc=ORANGE, ec="none"))
    ax.plot([xs, xs], [ys, ys + hs], color=MUTED, lw=0.5)
    ax.add_patch(Rectangle((x0, ys), x1 - x0, hs, fc="none", ec=INK2, lw=0.7))
    ax.plot([wx + 0.1 * ww], [ys + hs / 2], "o", ms=3.0, color=INK)
    ax.text(x0, 100.6, "Full cross-attention cache K, V — 1500 positions per layer (stays in DRAM)", fontsize=6.6, color=INK, va="center")
    ax.text(x0, 96.7, "blue: window $W_t$ = [$c_t$ − 0.1·$w_ℓ$, $c_t$ + 0.9·$w_ℓ$), moves with decoding;  ● = alignment peak $c_t$", ha="left", **t)
    ax.text(x0, 86.6, "speech ($n_r$ positions)", ha="left", **t)
    ax.text(112, 86.6, "padding (≈ 80 %)", ha="center", **t)
    ax.text(165.6, 86.6, "orange: padding sink $S_ℓ$ (fixed)", ha="right", **t)

    box(ax, 102.5, 69.5, 43, 13, text="First step (t = 1) on the full cache:\ncomputes K, V, selects $S_ℓ$\nand the start peak $c_1$",
        fc="white", ec=MUTED, ls=DASH, fs=5.8)
    box(ax, 62, 48, 38, 16, "Ring buffer  $B_ℓ$", "$S_ℓ ∪ W_t$\nk = 0.5·$K_i$ positions per layer", fc=PALE_B, ec=BLUE, num=4)
    box(ax, 106, 48, 26, 16, "Decoder step", "reads only $B_ℓ$ →\nlogits z, attention a", fc=PALE_B, ec=BLUE, num=5)
    diamond(ax, 150, 56, 15, 9.5, "max softmax(z)\n≥ τ* ?", fc=PALE_O, fs=6.2)
    badge(ax, 141.5, 62.4, 6, ORANGE)
    box(ax, 112, 6, 57.5, 18, "Redo on the full cache",
        "same step on 1500 positions;\nz, a and the self-attention cache from this run;\nthe peak is then searched over all audio",
        fc=PALE_O, ec=ORANGE, num=7)
    box(ax, 62, 6, 42, 18, "Emit token $y_t$, move window",
        "c ← max(c, alignment-head peak);\noverwrite only the slots\nthat left the window", fc=PALE_B, ec=BLUE, num=8)

    # ring-buffer cells of one layer
    cw, cy, ch = 3.4, 33.0, 4.6
    cells = [("109", "white", ORANGE), ("110", "white", ORANGE)] + [(str(n), PALE_B, BLUE) for n in range(103, 109)] \
        + [("s", PALE_O, ORANGE)] * 3
    for i, (lab, fc, ec) in enumerate(cells):
        ax.add_patch(Rectangle((62 + i * cw, cy), cw, ch, fc=fc, ec=ec, lw=0.6))
        ax.text(62 + (i + 0.5) * cw, cy + ch / 2, lab, ha="center", va="center", fontsize=4.6, color=INK)
    ax.text(62 + cw, 30.9, "entering", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62 + 5 * cw, 30.9, "kept from step t − 1", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62 + 9.5 * cw, 30.9, "sink", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62, 28.1, "109, 110 overwrite the slots of 101, 102", ha="left", **{**t, "fontsize": 5.3})
    ax.text(62, 40.1, "slots of $B_ℓ$ (one layer)", ha="left", **small)
    ax.plot([63.5, 63.5], [48, 41.7], color=BLUE, lw=0.6, ls=(0, (1, 1.5)))

    arrow(ax, (86, ys - 4.6), (86, 64), color=MUTED)                          # cache -> buffer
    ax.text(87.5, 75, "slots entering\nthe window", linespacing=1.15, **t)
    arrow(ax, (102.5, 70.5), (97, 64), color=MUTED, ls=DASH)            # first step -> buffer
    arrow(ax, (100, 56), (106, 56))
    arrow(ax, (132, 56), (135, 56))
    arrow(ax, (150, 46.5), (150, 24), color=ORANGE)                     # no -> redo
    ax.text(151.5, 35, "no:\nfallback", linespacing=1.15, **t)
    arrow(ax, (167.6, ys), (167.6, 24), color=MUTED, ls=DASH)           # cache -> redo
    ax.text(166.4, 77, "all 1500\npositions", ha="right", linespacing=1.15, **t)
    arrow(ax, (142.6, 51.3), (104.3, 24.4), color=BLUE)                 # yes -> emit
    ax.text(127, 34.6, "yes: fast path", ha="left", **t)
    arrow(ax, (112, 15), (104, 15), color=ORANGE)                       # redo -> emit
    arrow(ax, (101.8, 24), (101.8, 45.5), (99.6, 48.1), color=BLUE)     # emit -> buffer
    ax.text(103, 37, "t + 1", **t)
    ax.text(150, 67.6, "τ* from (a)", ha="center", **t)

    # configuration feeds the ring buffer
    arrow(ax, (55, 0.5), (58.5, 0.5), (58.5, 56), (62, 56))
    ax.text(60.3, 28, "$K_i$, ρ", rotation=90, ha="center", **t)

    # ------------------------------------------------ (c) what a step reads
    ax.plot([0, 171], [-7.5, -7.5], color="#d9d8d3", lw=0.6)
    ax.text(62, -11.5, "(c)  What a step reads", fontsize=8, fontweight="bold", color=INK, va="center")
    full_len = 70.0
    ax.add_patch(Rectangle((62, -18.4), full_len * 196 / 1500, 2.8, fc=BLUE, ec="none"))
    ax.add_patch(Rectangle((62, -24.4), full_len, 2.8, fc=ORANGE, ec="none"))
    ax.text(62 + full_len * 196 / 1500 + 1.5, -17, "fast path: k positions per layer (here k = 196, to scale)", **t)
    ax.text(62 + full_len + 1.5, -23, "fallback: 1500 positions per layer", **t)
    # legend
    ax.text(1, -11.5, "Legend", fontsize=7, fontweight="bold", color=INK, va="center")
    ax.add_patch(Rectangle((1, -16.2), 3.4, 2.4, fc=PALE_B, ec=BLUE, lw=0.7))
    ax.text(6.5, -15, "PadSink-Track fast path", **t)
    ax.add_patch(Rectangle((1, -19.7), 3.4, 2.4, fc=PALE_O, ec=ORANGE, lw=0.7))
    ax.text(6.5, -18.5, "padding sink / fallback", **t)
    arrow(ax, (0.8, -22.2), (4.8, -22.2))
    ax.text(6.5, -22.2, "every step / every arm", **t)
    arrow(ax, (0.8, -25.7), (4.8, -25.7), color=MUTED, ls=DASH)
    ax.text(6.5, -25.7, "conditional: first step, fallback, last arm", **t)
    save(fig, "fig_t11_framework_v3")


if __name__ == "__main__":
    main()
