"""Figure: alignment-head attention and the tracked window (one utterance),
in the colours of the schematics: attention in grey/black (the positions the
decoder needs), the window W_t in blue, the padding sink in orange.

Same data as fig_trace() in make_figures_track.py (trace_track_medium_uz_22.npz).

Output: figures/fig_t4_trace_v2.png|pdf
Usage:  python experiments/make_figure_trace2.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from make_figures_track import BLUE, CM, FULL, HERE, INK, ORANGE, save


def main():
    z = np.load(os.path.join(HERE, "trace_track_medium_uz_22.npz"), allow_pickle=True)
    att, wins, rn, sinks = z["att"], z["wins"], int(z["rn"]), z["sinks"]
    T = len(att)
    fig, ax = plt.subplots(figsize=(FULL, 5.8 * CM))
    xmax = min(1500, rn + 260)
    im = ax.imshow(np.sqrt(att[:, :xmax]), aspect="auto", origin="lower", cmap="Greys",
                   extent=[0, xmax, -0.5, T - 0.5], interpolation="nearest")
    for t, (lo, hi) in enumerate(wins):
        ax.add_patch(Rectangle((lo, t - 0.5), hi - lo, 1.0, fc=BLUE, alpha=0.16, ec="none"))
        ax.plot([lo, lo], [t - 0.5, t + 0.5], color=BLUE, lw=0.8)
        ax.plot([hi, hi], [t - 0.5, t + 0.5], color=BLUE, lw=0.8)
    ax.axvline(rn, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(rn - 4, 1.0, "end of speech", color=INK, fontsize=7, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", pad=1.0))
    ss = sinks[sinks < xmax]
    ax.plot(ss, np.full(len(ss), -1.6), "|", color=ORANGE, ms=5, mew=1.4, clip_on=False)
    ax.set_ylim(-2.2, T - 0.5)
    ax.set_xlabel("encoder position (20 ms each)")
    ax.set_ylabel("decoding step t")
    ax.tick_params(colors=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("√ attention", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=6, colors=INK)
    handles = [Patch(fc="#555555", ec="none", label="alignment-head attention (full cache)"),
               Patch(fc=BLUE, alpha=0.35, ec=BLUE, label="window $W_t$"),
               Line2D([], [], color=ORANGE, marker="|", ls="none", ms=6, mew=1.4, label="padding sink $S_ℓ$ (one layer)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5,
              handlelength=1.4, columnspacing=1.8, labelcolor=INK)
    save(fig, "fig_t4_trace_v2")


if __name__ == "__main__":
    main()
