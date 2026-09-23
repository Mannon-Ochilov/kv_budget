"""Figure: quality against cache bytes for each cache precision (E1).

One point per scheme: cross-attention cache size per step on a log axis,
WER on the ordinate, with the paired 95% interval of the difference from the
FP32 cache drawn as the FP32 WER plus the bounds of that difference -- the
same construction as the paper's Figures 6 and 7. The FP32 reference is a
dashed line; the accuracy gate of the paper, FP32 + delta with delta = 0.2 x
WER_FP32, is drawn so each scheme's acceptance can be read directly.

Everything is read from results_cache_precision_wer.json at draw time.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = next((os.path.join(HERE, f"results_cache_precision_wer_{t}.json")
            for t in ("cascade", "fp32")
            if os.path.exists(os.path.join(HERE, f"results_cache_precision_wer_{t}.json"))),
           os.path.join(HERE, "results_cache_precision_wer_fp32.json"))
OUT = os.path.join(HERE, "..", "figures", "fig_cache_precision_wer.png")
EPS = 0.20

BLUE, ORANGE, GREEN, PINK, DARK, GREY = ("#0072B2", "#D55E00", "#009E73",
                                         "#CC79A7", "#1a1a1a", "#888888")
STYLE = {
    "fp32": (DARK, "D", "FP32 (reference)"),
    "fp16": (GREY, "o", "FP16"),
    "int8_tensor": (BLUE, "s", "int8, per tensor"),
    "int8_head": (BLUE, "o", "int8, per head"),
    "int8_kivi": (GREEN, "^", "int8, KIVI grouping"),
    "int4_kivi": (ORANGE, "v", "int4, KIVI grouping"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua",
                   "DejaVu Serif"],
    "font.size": 9, "font.weight": "normal",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def main():
    d = json.load(open(SRC))
    sch = d["schemes"]
    fp32 = sch["fp32"]["wer"]
    gate = round(fp32, 4) + round(fp32 * EPS, 4)

    fig, ax = plt.subplots(figsize=(4.72, 3.5))
    ax.axhline(fp32, color=GREY, lw=0.9, ls=(0, (5, 4)))
    ax.axhline(gate, color=ORANGE, lw=1.1, ls=(0, (6, 4)))
    ax.text(0.98, gate, f"gate FP32 + δ = {gate:.4f}", fontsize=7.5,
            color=ORANGE, ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.text(0.98, fp32, f"FP32 cache: {fp32:.4f}", fontsize=7.5, color=GREY,
            ha="right", va="top", transform=ax.get_yaxis_transform())

    for name, r in sch.items():
        col, mk, label = STYLE.get(name, (DARK, "o", name))
        x, y = r["cross_cache_mib"], r["wer"]
        if "delta_vs_fp32" in r and name != "fp32":
            _, lo, hi = r["delta_vs_fp32"]
            ax.errorbar([x], [y], yerr=[[y - (fp32 + lo)], [(fp32 + hi) - y]],
                        fmt="none", ecolor=col, elinewidth=0.9, capsize=3)
        ax.plot([x], [y], marker=mk, markersize=6.5, linestyle="none",
                markerfacecolor="white" if name != "fp32" else col,
                markeredgecolor=col, markeredgewidth=1.2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Cross-attention cache per step, MiB (log scale)", fontsize=9)
    ax.set_ylabel("WER, 300 test utterances", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", handlelength=1.2)
    ax.tick_params(labelsize=8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight")
    print(f"  {OUT}")


if __name__ == "__main__":
    main()
