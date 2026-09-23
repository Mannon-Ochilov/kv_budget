"""System figure: static encoder compression and dynamic decoder KV
compression as two working sets of one model (schematic, MDPI style)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
BLUE, ORANGE, GREEN, DARK = "#0072B2", "#D55E00", "#009E73", "#1a1a1a"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
                     "font.size": 8.5, "figure.facecolor": "white", "savefig.facecolor": "white"})


def box(ax, xy, w, h, text, fc="white", ec=DARK, fs=7.2, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=DARK,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color=DARK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(6.69, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    box(ax, (5, 9.4), 4.8, 0.7, "Whisper encoder–decoder ASR on a CPU\n(audio in, text out)", fc="#f2f2f2")
    arrow(ax, (3.8, 9.05), (2.5, 8.35))
    arrow(ax, (6.2, 9.05), (7.5, 8.35))
    # left column: encoder / static
    box(ax, (2.5, 7.95), 4.4, 0.7, "ENCODER\nstatic model state: weights W", fc="#e6f0fa", ec=BLUE, bold=True)
    box(ax, (2.5, 6.55), 4.4, 1.1, "τ-based functional redundancy\ncompensated removal\nINT8 + low-rank", fc="white", ec=BLUE)
    box(ax, (2.5, 5.0), 4.4, 0.7, "cache-anchored weight working set\n$M_w^{(\\ell)} \\leq \\alpha\\,M_{L3}$", fc="#e6f0fa", ec=BLUE)
    ax.text(2.5, 4.35, "the weight paper", ha="center", fontsize=7, color=BLUE, style="italic")
    # right column: decoder / runtime
    box(ax, (7.5, 7.95), 4.4, 0.7, "DECODER\nruntime state: cross-attention K, V", fc="#fbeee6", ec=ORANGE, bold=True)
    box(ax, (7.5, 6.55), 4.4, 1.1, "precision q (KIVI int8/int4)\nretention r: audio / padding split\nWER gate, least-compressed fit", fc="white", ec=ORANGE)
    box(ax, (7.5, 5.0), 4.4, 0.7, "cache-budgeted KV working set\n$M_w^{(\\ell)} + M_{KV}(q,r)/L \\leq \\alpha\\,M_{L3}$", fc="#fbeee6", ec=ORANGE)
    ax.text(7.5, 4.35, "this paper", ha="center", fontsize=7, color=ORANGE, style="italic")
    for x in (2.5, 7.5):
        arrow(ax, (x, 7.6), (x, 7.1))
        arrow(ax, (x, 6.0), (x, 5.35))
    # merge
    arrow(ax, (2.5, 4.2), (3.9, 3.15))
    arrow(ax, (7.5, 4.2), (6.1, 3.15))
    box(ax, (5, 2.65), 7.2, 0.9, "one model, two working sets, one budget rule\n$T_{\\mathrm{ASR}} = T_{\\mathrm{enc}}(W_\\tau) + \\sum_t T_{\\mathrm{dec}}(KV_{q,r}),\\;\\; \\Delta\\mathrm{WER} \\leq \\delta$", fc="#e8f5ef", ec=GREEN, bold=True)
    arrow(ax, (5, 2.2), (5, 1.75))
    box(ax, (5, 1.35), 7.2, 0.75, "system-level composition (system table): A FP32/FP32, B τ-encoder,\nC KV-only, D both — WER, $T_{\\mathrm{enc}}$, $T_{\\mathrm{dec}}$, RTF, $M_w$, $M_{KV}$", fc="white", ec=GREEN)
    ax.text(5, 0.5, "cache-aware optimization of both static and runtime working sets",
            ha="center", fontsize=8, color=DARK, fontweight="bold")
    out = os.path.join(FIG, "fig_kv_c5_system.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


if __name__ == "__main__":
    main()
