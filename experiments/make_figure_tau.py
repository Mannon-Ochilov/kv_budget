"""Figure: the fallback threshold tau -- quality, redone steps and saving.

  (a) dWER against the full cache on the 300 test utterances, one panel per
      model, for PadSink-Track without the fallback and with tau = 0.5 ... 0.95
      (95 % CI; dashed line = the gate delta). The ringed point is the arm the
      calibration (framework.py, 300 validation utterances) chose.
  (b) share of the decoder steps redone on the full cache.
  (c) whole-utterance saving in ms/token against the full cache (measured at
      no fallback, tau = 0.5, 0.7, 0.9; e2e_latency.py, 100 utterances).

Data: results_align_track_*, results_adaptive_fb_*, results_e2e_latency_*_fb*,
results_framework_*.  Output: figures/fig_t10_tau.png|pdf

Usage:  python experiments/make_figure_tau.py
"""

import matplotlib.pyplot as plt
import numpy as np

from make_figures_track import FULL, CM, GRID, INK, INK2, MUTED, J, delta_of, label, save, symlog_axis

MODELS = [("medium_uz", "Whisper-medium, Uzbek", "#2a78d6", "o"), ("small_uz", "Whisper-small, Uzbek", "#eb6834", "s"),
          ("medium_en", "Whisper-medium, English", "#1baf7a", "^"), ("small_en", "Whisper-small, English", "#eda100", "D")]
TAUS = ["none", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95"]
XT = ["no\nfallback", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95"]


def quality(m):
    rows = [J(f"results_align_track_{m}.json")["arms"]["track/s0.5"]["delta_vs_full"] + [0.0]]
    t = J(f"results_adaptive_fb_{m}.json")["test"]
    rows += [t[f"fb/{x}"]["delta_vs_full"] + [t[f"fb/{x}"]["redone_share"]] for x in TAUS[1:]]
    return np.array(rows)                                   # d, lo, hi, redone


def saving(m):
    a, b = J(f"results_e2e_latency_{m}_fb.json")["arms"], J(f"results_e2e_latency_{m}_fb_tau.json")["arms"]
    c = J(f"results_e2e_latency_{m}_fw.json")["arms"]          # the final session: full / no fallback / chosen tau
    s = lambda arms, k: 100 * (1 - arms[k]["ms_per_token_median"] / arms["full"]["ms_per_token_median"])   # noqa: E731
    return [0, 1, 3, 5], [s(c, "track_ring"), s(b, "track_fb05"), s(b, "track_fb07"), s(a, "track_fb")]


def main():
    fig = plt.figure(figsize=(FULL, 11.6 * CM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.42)
    x = np.arange(len(TAUS))
    for j, (m, name, col, mk) in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, j])
        q, dl = quality(m), delta_of(m)
        chosen = J(f"results_framework_{m}.json")["choice"]
        ax.axhline(dl, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=GRID, lw=0.6, zorder=1)
        ax.errorbar(x, q[:, 0], yerr=[q[:, 0] - q[:, 1], q[:, 2] - q[:, 0]], fmt="none", ecolor=col, elinewidth=0.8,
                    capsize=1.8, capthick=0.8, zorder=2)
        ax.plot(x, q[:, 0], color=col, lw=1.2, marker=mk, ms=4, mfc="white", mec=col, mew=1.1, zorder=3)
        if chosen in TAUS:
            i = TAUS.index(chosen)
            ax.plot([x[i]], [q[i, 0]], marker=mk, ms=4.6, mfc=col, mec=col, zorder=4)
            ax.plot([x[i]], [q[i, 0]], marker="o", ms=10, mfc="none", mec=INK, mew=0.8, zorder=4)
        symlog_axis(ax, lin=0.01)
        ax.set_ylim(-0.012, 0.2)
        ax.set_yticks([-0.01, 0, 0.01, 0.1])
        ax.set_yticklabels(["−0.01", "0", "0.01", "0.1"] if j == 0 else [])
        ax.set_xticks(x)
        ax.set_xticklabels(["none"] + TAUS[1:], rotation=90)
        ax.set_title(name, fontsize=7.5, color=INK, pad=3)
        ax.text(-0.3 if m == "medium_en" else len(TAUS) - 0.6, dl * (0.72 if m == "medium_en" else 1.0),
                f"δ = {dl:.4f}".rstrip("0"), fontsize=6.3, color=INK2,
                va="top" if m == "medium_en" else "bottom", ha="left" if m == "medium_en" else "right")
        if j == 0:
            ax.set_ylabel("ΔWER vs full cache")
            label(ax, "(a)", x=-0.42, y=1.08)
    fig.text(0.5, 0.493, "fallback threshold τ   (ring = arm chosen on the validation set)", ha="center", fontsize=7.5, color=INK)

    ax_b, ax_c = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])
    for m, name, col, mk in MODELS:
        q = quality(m)
        ax_b.plot(x, 100 * q[:, 3], color=col, lw=1.3, marker=mk, ms=4, mfc="white", mec=col, mew=1.1, label=name)
        xs, sv = saving(m)
        ax_c.plot(xs, sv, color=col, lw=1.3, marker=mk, ms=4, mfc="white", mec=col, mew=1.1)
        ax_c.text(xs[-1] + 0.18, sv[-1], f"{sv[-1]:.0f} %", fontsize=6.5, color=INK2, va="center")
    for ax, yl, lab, ttl in ((ax_b, "steps redone, %", "(b)", "Decoder steps redone on the full cache"),
                             (ax_c, "saving, %", "(c)", "Whole-utterance saving in ms/token vs full cache")):
        ax.set_title(ttl, fontsize=7.5, color=INK, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(XT)
        ax.set_xlabel("fallback threshold τ")
        ax.set_ylabel(yl)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        label(ax, lab, x=-0.15, y=1.03)
    ax_b.legend(loc="upper left", handlelength=1.8, labelspacing=0.3, borderaxespad=0.2)
    ax_c.set_xlim(-0.4, 6.4)
    ax_c.set_ylim(-1.5, 20)
    ax_c.axhline(0, color=INK2, lw=0.6)
    ax_c.text(0.98, 0.93, "measured at no fallback, τ = 0.5, 0.7, 0.9", transform=ax_c.transAxes, fontsize=6.5, color=INK2, ha="right")
    save(fig, "fig_t10_tau")


if __name__ == "__main__":
    main()
