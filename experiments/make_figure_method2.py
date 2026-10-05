"""Figure: the PadSink-Track retention rule (schematic, method only).

(a) one-shot retention: the set is chosen at t = 1 and goes stale;
(b) PadSink-Track: the window follows the alignment peak, the sink is fixed;
(c) one decoding step and the ring buffer of one layer.

Calibration, the confidence check and the full-cache fallback are left to the
framework figure (make_figure_framework3.py).

Output: figures/fig_t1_method_v2.png|pdf
Usage:  python experiments/make_figure_method2.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from make_figures_track import BLUE, FULL, GREYS, INK, INK2, MUTED, ORANGE, save

PALE_B, PALE_O, PALE_G = "#e6f0fb", "#fdebe3", "#f1f0ec"
RED = "#d23b3b"
DASH = (0, (3, 2))
X0, X1, Y0, Y1 = 0, 173, 24, 120


def arrow(ax, *pts, color=INK2):
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.8, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(FULL, FULL * (Y1 - Y0) / (X1 - X0)))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    t = dict(fontsize=6.0, color=INK, va="center")

    T, P, RN = 14, 40, 20                        # schematic: steps, positions, speech positions
    rng = np.random.default_rng(7)
    need = np.clip(np.round(np.linspace(1, RN - 2, T) + rng.normal(0, 0.5, T)), 0, RN - 1).astype(int)
    sinks = np.array([RN + 1, RN + 2, RN + 5, RN + 9, RN + 14])
    gw, y_top, y_bot = 47.0, 106.0, 84.0
    cw, rh = gw / P, (y_top - y_bot) / T

    def grid(x0, xt, title, sub):
        ax.add_patch(Rectangle((x0, y_bot), RN * cw, y_top - y_bot, fc="white", ec="none"))
        ax.add_patch(Rectangle((x0 + RN * cw, y_bot), (P - RN) * cw, y_top - y_bot, fc="#f2f1ec", ec="none"))
        ax.plot([x0 + RN * cw] * 2, [y_bot, y_top], color=MUTED, lw=0.5)
        ax.add_patch(Rectangle((x0, y_bot), gw, y_top - y_bot, fc="none", ec=INK2, lw=0.6, zorder=4))
        ax.text(xt, 116.5, title, fontsize=8, fontweight="bold", color=INK, va="center")
        ax.text(xt, 112.3, sub, **t)
        ax.text(x0 + gw / 2, y_top + 2.2, "encoder positions →", ha="center", **{**t, "color": INK2})
        ax.text(x0 + RN * cw / 2, y_bot - 2.3, "speech ($n_r$ positions)", ha="center", **t)
        ax.text(x0 + RN * cw + (P - RN) * cw / 2, y_bot - 2.3, "padding (≈ 80 %)", ha="center", **t)
        ax.text(x0 - 1, y_top - rh / 2, "t = 1", ha="right", **t)
        ax.text(x0 - 1, y_bot + rh / 2, f"t = {T}", ha="right", **t)
        arrow(ax, (x0 - 3.5, y_top - 3.5), (x0 - 3.5, y_bot + 3.5), color=MUTED)
        return (lambda p_: x0 + (p_ + 0.5) * cw), (lambda t_: y_top - (t_ + 0.5) * rh)

    # (a) one-shot
    cx, cy = grid(9, 2, "(a)  One-shot retention", "H2O, SnapKV, PyramidKV, sink + one-shot audio")
    kept = np.unique(np.concatenate([[0, 1, 2, 3, 5, 6], sinks]))
    for p_ in kept:
        ax.add_patch(Rectangle((cx(p_) - cw * 0.42, y_bot), cw * 0.84, y_top - y_bot, fc=GREYS["h2o_layer"], alpha=0.8, ec="none"))
    for t_, p_ in enumerate(need):
        hit = p_ in kept
        ax.plot(cx(p_), cy(t_), marker="o" if hit else "x", ms=3.2 if hit else 3.6, color=INK if hit else RED, mew=1.1, zorder=3)

    # (b) PadSink-Track: the peak sits 0.1 w from the left edge of the window
    cx, cy = grid(73, 66.5, "(b)  PadSink-Track", "the selection is renewed at every step")
    w = 7
    for t_, p_ in enumerate(need):
        left = max(cx(p_) - 0.1 * w * cw, 73)
        ax.add_patch(Rectangle((left, cy(t_) - rh * 0.45), w * cw, rh * 0.9, fc=BLUE, alpha=0.4, ec="none"))
    for p_ in sinks:
        ax.add_patch(Rectangle((cx(p_) - cw * 0.42, y_bot), cw * 0.84, y_top - y_bot, fc=ORANGE, alpha=0.9, ec="none"))
    for t_, p_ in enumerate(need):
        ax.plot(cx(p_), cy(t_), marker="o", ms=3.2, color=INK, zorder=3)

    # legend
    items = [(INK, "o", "position the decoder\nneeds at step t"),
             (RED, "x", "needed but evicted\n→ errors, loops"),
             (GREYS["h2o_layer"], "s", "one-shot set: chosen\nat t = 1, fixed"),
             (BLUE, "s", "window $W_t$: follows\nthe alignment peak $c_t$"),
             (ORANGE, "s", "padding sink $S_ℓ$: exact\nK/V, chosen at t = 1")]
    for n, (col, mk, txt) in enumerate(items):
        y = 111.6 - n * 6.2                    # centred in the legend column
        ax.plot(131.5, y, marker=mk, ms=5.5 if mk == "s" else 3.6, color=col, mew=1.1, alpha=0.6 if col == BLUE else 1)
        ax.text(135.5, y, txt, linespacing=1.15, **t)

    # separators between the panels and the legend
    for xa in (62.0, 126.0):
        ax.plot([xa, xa], [79.6, 119.6], color=MUTED, lw=0.6, ls=DASH)
    ax.plot([1, 172], [78, 78], color=MUTED, lw=0.6, ls=DASH)

    # (c) the step loop
    ax.text(2, 73.5, "(c)  One decoding step of PadSink-Track", fontsize=8, fontweight="bold", color=INK, va="center")
    bw, bh, gap, by = 30.6, 17.5, 4.5, 52.0
    boxes = [("1. Initialize (t = 1)", "use the full cache; select\nsink $S_ℓ$: smallest set with ρ\nof padding mass, at most\n0.25·k;  start peak $c_1$",
              PALE_O, ORANGE),
             ("2. Ring buffer", "$B_ℓ = W_t ∪ S_ℓ$\nk = |$S_ℓ$| + $w_ℓ$\npositions per layer", PALE_B, BLUE),
             ("3. Decoder step t", "reads only $B_ℓ$\n→ token $y_t$", PALE_G, INK2),
             ("4. Alignment heads", "their attention over $W_t$\n→ peak;\n$c_{t+1}$ = max($c_t$, peak)", PALE_G, INK2),
             ("5. Shift the window", "move the window to\n[$c_{t+1}$ − 0.1·$w_ℓ$, $c_{t+1}$ + 0.9·$w_ℓ$);\nat the end of speech\nit runs into padding",
              PALE_G, INK2)]
    xs = [1 + i * (bw + gap) for i in range(5)]
    for x, (head, txt, fc, ec) in zip(xs, boxes):
        ax.add_patch(FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
        ax.text(x + bw / 2, by + bh - 1.8, head, ha="center", va="top", fontsize=6.8, fontweight="bold", color=INK)
        ax.text(x + bw / 2, by + bh - 6.4, txt, ha="center", va="top", fontsize=5.9, color=INK, linespacing=1.3)
    for x in xs[:-1]:
        arrow(ax, (x + bw, by + bh / 2), (x + bw + gap, by + bh / 2))
    x5, x2 = xs[4] + bw / 2, xs[1] + bw / 2
    arrow(ax, (x5, by), (x5, 48.5), (x2, 48.5), (x2, by))
    ax.text((x5 + x2) / 2, 46.7, "next step", ha="center", **t)

    # ring buffer of one layer (zoom of box 2)
    zx0, zx1, zy0, zy1 = 1.0, 86.0, 25.5, 43.5
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fc="none", ec=BLUE, lw=0.6, ls=DASH))
    ax.plot([xs[1], zx0], [by, zy1], color=BLUE, lw=0.6, ls=DASH)
    ax.plot([xs[1] + bw, zx1], [by, zy1], color=BLUE, lw=0.6, ls=DASH)
    ax.text((zx0 + zx1) / 2, 41, "Ring buffer of one layer (size k)", ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
    sw, sx, sy, sh = 6.6, 7.2, 33.2, 5.0
    labels = ["109", "110", "103", "104", "105", "106", "107", "108", "s", "s", "s"]
    for n, lab in enumerate(labels):
        fc, ec = ("white", ORANGE) if n < 2 else ((PALE_O, ORANGE) if lab == "s" else (PALE_B, BLUE))
        ax.add_patch(Rectangle((sx + n * sw, sy), sw, sh, fc=fc, ec=ec, lw=0.6))
        ax.text(sx + (n + 0.5) * sw, sy + sh / 2, lab, ha="center", va="center", fontsize=5.6, color=INK)
    for c, txt in ((1, "entering positions\n(overwrite 101, 102)"), (5, "positions kept\nfrom step t − 1"), (9.5, "sink positions\n($S_ℓ$)")):
        ax.text(sx + c * sw, 29.4, txt, ha="center", linespacing=1.15, **{**t, "fontsize": 5.6})
    ax.text(91, 34.5,
            "Slots of positions that left the window (101, 102) are overwritten\n"
            "with the entering ones (109, 110), with no per-step gather.\n"
            "s = sink slots. Attention does not depend on slot order.", linespacing=1.3, **t)
    save(fig, "fig_t1_method_v2")


if __name__ == "__main__":
    main()
