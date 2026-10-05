"""Figure: the retention framework (schematic), compact two-panel layout.

Same content as make_figure_framework3.py --no-c, laid out to take less height:
(a) calibration, once per model (Algorithm 3 for tau);
(b) one decoding step with the confidence check and the fallback (Algorithm 2).

Output: figures/fig_t11_framework_v5.png|pdf
Usage:  python experiments/make_figure_framework4.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from make_figure_framework3 import DASH, PAD, PALE_B, PALE_G, PALE_O, arrow, badge, box, diamond
from make_figures_track import BLUE, FULL, INK, INK2, MUTED, ORANGE, save

X0, X1, Y0, Y1 = -1, 172, 4, 108


def main():
    fig, ax = plt.subplots(figsize=(FULL, FULL * (Y1 - Y0) / (X1 - X0)))   # equal units: badges stay round
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    t = dict(fontsize=6.0, color=INK, va="center")
    small = {**t, "fontsize": 5.4}

    # ------------------------------------------------ (a) calibration
    ax.text(1, 105.5, "(a)  Calibration, once per model", fontsize=8, fontweight="bold", color=INK, va="center")
    box(ax, 1, 92, 54, 8.5, "Validation set (300 utterances)", "full cache $\\rightarrow$ WER$_{full}$;  gate δ = 0.2·WER$_{full}$",
        fc="white", ec=MUTED, fs=5.8)
    box(ax, 1, 81.5, 54, 8.5, "Budget  $K_i$", "split rule ($f_r$, $f_p$): smallest set within the gate δ", num=1, fs=5.8)
    box(ax, 1, 68.5, 54, 11, "Sink mass  ρ", "smallest padding set holding ρ of the\nfirst-step padding mass;  |$S_ℓ$| ≤ 0.25·k", num=2)
    box(ax, 1, 23.7, 54, 42.8, "Fallback threshold  τ",
        "arms by cost: no fallback, 0.6, 0.7, 0.8, 0.9, 0.95\nU = upper 95 % CI bound of WER$_{arm}$ − WER$_{full}$",
        fc=PALE_O, ec=ORANGE, num=3, fs=5.8)
    for ya, yb in ((92, 90), (81.5, 79.5), (68.5, 66.5)):
        arrow(ax, (28, ya), (28, yb))

    diamond(ax, 19, 48.4, 15, 3.4, "(i)  U < δ ?")
    diamond(ax, 19, 37.2, 15, 4.6, "(ii)  new utterance\nwith WER > 1 ?")
    box(ax, 40, 40.5, 13, 5, text="next arm", fc="white", ec=INK2, fs=5.8)
    box(ax, 8, 25.2, 22, 4.4, text="take this arm:  τ*", fc="white", ec=ORANGE, fs=5.8)
    box(ax, 37, 25.2, 16.5, 4.4, text="FULL CACHE", fc="white", ec=INK2, fs=5.8)
    arrow(ax, (19, 45), (19, 41.8), color=ORANGE)                           # (i) yes -> (ii)
    ax.text(20.3, 43.5, "yes", **small)
    arrow(ax, (19, 32.6), (19, 29.6), color=ORANGE)                         # (ii) no -> take
    ax.text(20.3, 31.2, "no", **small)
    arrow(ax, (34, 48.4), (46.5, 48.4), (46.5, 45.5))                       # (i) no -> next arm
    ax.text(36.5, 49.8, "no", ha="center", **small)
    arrow(ax, (34, 37.2), (38.3, 37.2), (38.3, 43), (40, 43))               # (ii) yes -> next arm
    ax.text(36.1, 38.6, "yes", ha="center", **small)
    arrow(ax, (53, 43), (54.1, 43), (54.1, 54.5), (19, 54.5), (19, 51.8))   # next arm -> (i)
    arrow(ax, (46.5, 40.5), (46.5, 29.6), color=MUTED, ls=DASH)             # last arm -> full cache
    ax.text(45.7, 34.6, "after the\nlast arm", ha="right", linespacing=1.15, **small)

    box(ax, 1, 15.2, 54, 5.5, fc="white", ec=INK2)
    ax.text(28, 17.95, "Configuration:  $K_i$,  ρ,  τ*   (or the full cache)", ha="center", va="center", fontsize=6.4, color=INK)
    arrow(ax, (19, 25.2), (19, 20.7))
    arrow(ax, (46.5, 25.2), (46.5, 20.7), color=MUTED, ls=DASH)

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

    box(ax, 101.5, 73.5, 40, 10, text="First step (t = 1) on the full cache:\ncomputes K, V, selects $S_ℓ$\nand the start peak $c_1$",
        fc="white", ec=MUTED, ls=DASH, fs=5.8)
    box(ax, 62, 57, 38, 13, "Ring buffer  $B_ℓ$", "$S_ℓ ∪ W_t$\nk = 0.5·$K_i$ positions per layer", fc=PALE_B, ec=BLUE, num=4)
    box(ax, 106, 57, 26, 13, "Decoder step", "reads only $B_ℓ$ $\\rightarrow$\nlogits z, attention a", fc=PALE_B, ec=BLUE, num=5)
    diamond(ax, 150, 63.5, 15, 9, "max softmax(z)\n≥ τ* ?", fc=PALE_O, fs=6.2)
    badge(ax, 139, 66.6, 6, ORANGE)
    box(ax, 112, 15.2, 57.5, 15, "Redo on the full cache",
        "same step on 1500 positions;\nz, a and the self-attention cache from this run;\nthe peak is then searched over all audio",
        fc=PALE_O, ec=ORANGE, num=7)
    box(ax, 62, 15.2, 42, 15, "Emit token $y_t$, move window",
        "c $\\leftarrow$ max(c, alignment-head peak);\noverwrite only the slots\nthat left the window", fc=PALE_B, ec=BLUE, num=8)

    # ring-buffer cells of one layer
    cw, cy, ch = 3.4, 43.5, 4.6
    cells = [("109", "white", ORANGE), ("110", "white", ORANGE)] + [(str(n), PALE_B, BLUE) for n in range(103, 109)] \
        + [("s", PALE_O, ORANGE)] * 3
    for i, (lab, fc, ec) in enumerate(cells):
        ax.add_patch(Rectangle((62 + i * cw, cy), cw, ch, fc=fc, ec=ec, lw=0.6))
        ax.text(62 + (i + 0.5) * cw, cy + ch / 2, lab, ha="center", va="center", fontsize=4.6, color=INK)
    ax.text(62 + cw, 41.4, "entering", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62 + 5 * cw, 41.4, "kept from step t − 1", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62 + 9.5 * cw, 41.4, "sink", ha="center", **{**t, "fontsize": 5.3})
    ax.text(62, 38.6, "109, 110 overwrite the slots of 101, 102", ha="left", **{**t, "fontsize": 5.3})
    ax.text(62, 50.6, "slots of $B_ℓ$ (one layer)", ha="left", **small)
    ax.plot([63.5, 63.5], [57, 52.2], color=BLUE, lw=0.6, ls=(0, (1, 1.5)))

    arrow(ax, (86, ys - 4.6), (86, 70), color=MUTED)                    # cache -> buffer
    ax.text(87.5, 78.5, "slots entering\nthe window", linespacing=1.15, **t)
    arrow(ax, (101.5, 74.5), (97, 70), color=MUTED, ls=DASH)            # first step -> buffer
    arrow(ax, (100, 63.5), (106, 63.5))
    arrow(ax, (132, 63.5), (135, 63.5))
    arrow(ax, (150, 54.5), (150, 30.2), color=ORANGE)                   # no -> redo
    ax.text(151.5, 42.5, "no:\nfallback", linespacing=1.15, **t)
    arrow(ax, (167.6, ys), (167.6, 30.2), color=MUTED, ls=DASH)         # cache -> redo
    ax.text(166.4, 79, "all 1500\npositions", ha="right", linespacing=1.15, **t)
    arrow(ax, (142.6, 58.9), (104.3, 30.6), color=BLUE)                 # yes -> emit
    ax.text(127, 43.5, "yes: fast path", ha="left", **t)
    arrow(ax, (112, 22.7), (104, 22.7), color=ORANGE)                   # redo -> emit
    arrow(ax, (101.8, 30.2), (101.8, 54.5), (99.6, 57.1), color=BLUE)   # emit -> buffer
    ax.text(103, 45, "t + 1", **t)
    ax.text(150, 74.4, "τ* from (a)", ha="center", **t)

    # configuration feeds the ring buffer
    arrow(ax, (55, 17.95), (58.5, 17.95), (58.5, 63.5), (62, 63.5))
    ax.text(60.3, 23.5, "$K_i$, ρ", rotation=90, ha="center", **t)

    # ------------------------------------------------ legend, one row
    ax.plot([0, 171], [11.6, 11.6], color="#d9d8d3", lw=0.6)
    ax.add_patch(Rectangle((1, 6.3), 3.4, 2.4, fc=PALE_B, ec=BLUE, lw=0.7))
    ax.text(6, 7.5, "PadSink-Track fast path", **t)
    ax.add_patch(Rectangle((36, 6.3), 3.4, 2.4, fc=PALE_O, ec=ORANGE, lw=0.7))
    ax.text(41, 7.5, "padding sink / fallback", **t)
    arrow(ax, (70, 7.5), (74.5, 7.5))
    ax.text(76, 7.5, "every step / every arm", **t)
    arrow(ax, (105, 7.5), (109.5, 7.5), color=MUTED, ls=DASH)
    ax.text(111, 7.5, "conditional: first step, fallback, last arm", **t)
    save(fig, "fig_t11_framework_v5")


if __name__ == "__main__":
    main()
