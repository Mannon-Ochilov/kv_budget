"""Result figures in the colours of the schematics.

Colour is kept for the method only: blue = PadSink-Track, orange = the padding
sink (and the sink-preserving one-shot variant), greys = the full cache and the
one-shot baselines. Where four models share an axis they are told apart without
colour: Uzbek dark grey, English light grey; medium solid, small dashed; one marker each.
All text is black.

The data and the drawing code of figures 3 and 5-9 are those of
make_figures_track.py; only the styling differs. Figures 2 and 10 are redrawn
here because they used colour for the models.

Output: figures/<name>_v2.png|pdf
Usage:  python experiments/make_figures_v2.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

import make_figures_track as M
from make_figures_track import BLUE, CM, FULL, GRID, INK, INK2, MUTED, J, delta_of, label, symlog_axis

plt.rcParams.update({"xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.labelcolor": INK, "legend.labelcolor": INK})
_save = M.save


def save(fig, name):
    for txt in fig.findobj(Text):
        txt.set_color(INK)
    _save(fig, name + "_v2")


M.save = save                                    # the functions of make_figures_track pick it up

DARK, GREY = "#3b3a37", "#a3a29c"
STYLE = {"medium_uz": dict(color=DARK, marker="o", ls="-"), "small_uz": dict(color=DARK, marker="s", ls=(0, (4, 2))),
         "medium_en": dict(color=GREY, marker="^", ls="-"), "small_en": dict(color=GREY, marker="D", ls=(0, (4, 2)))}
NAMES = dict(M.MODELS)


def line(ax, x, y, m, **kw):
    s = STYLE[m]
    return ax.plot(x, y, color=s["color"], ls=s["ls"], marker=s["marker"], ms=4, lw=1.2, mfc="white", mec=s["color"],
                   mew=1.1, **kw)


# ------------------------------------------------------------------ padding (figure 2 of the old numbering)
def fig_padding():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 6.0 * CM), gridspec_kw={"width_ratios": [1, 1.25]})
    for m, name in M.MODELS:
        r = J(f"results_eviction_budget_{m}.json")
        calib, f_r = r["calib"], r["choice"]["rule"][1]
        pts = sorted((float(k.split("/")[2]), v["wer"]) for k, v in calib.items() if k.startswith(f"split/{f_r}/"))
        full = calib["full"]["wer"]
        xp = [max(p[0], 0.012) for p in pts]     # f_p = 0 drawn at the left edge of the log axis
        line(a, xp, [p[1] - full for p in pts], m, label=name)
    a.set_xscale("log")
    a.set_xticks([0.012, 0.05, 0.1, 0.25, 0.5, 1.0])
    a.set_xticklabels(["0", "0.05", "0.1", "0.25", "0.5", "1"])
    symlog_axis(a, 0.02)
    a.set_xlabel("kept padding fraction f_p (validation, calibrated f_r)")
    a.set_ylabel("ΔWER vs full cache")
    a.axvspan(0.009, 0.03, color="#eeeeea", lw=0, zorder=0)
    a.text(0.0195, -0.06, "cliff", fontsize=7, ha="center")
    a.legend(loc="upper right", handlelength=2.6)
    label(a, "(a)")
    # causal test
    arms = [("sink_mean", "sink K/V\n→ mean"), ("audio_mean", "same number of\naudio K/V → mean"),
            ("pad_rest_mean", "rest of padding\n→ mean")]
    width, handles = 0.2, []
    for j, (m, name) in enumerate(M.MODELS):
        sc = J(f"results_sink_causal_{m}.json")
        sc = sc.get("arms", sc)
        vals = [sc[k]["delta_vs_full"][0] for k, _ in arms]
        lo = [sc[k]["delta_vs_full"][1] for k, _ in arms]
        hi = [sc[k]["delta_vs_full"][2] for k, _ in arms]
        xs = np.arange(3) + (j - 1.5) * width
        col, small = STYLE[m]["color"], m.startswith("small")
        kw = dict(fc="white", ec=col, hatch="/////", lw=0.7) if small else dict(fc=col, ec=col, lw=0.7)
        b.bar(xs, vals, width * 0.86, zorder=2, **kw)
        b.errorbar(xs, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, lw=0.7,
                   capsize=1.6, zorder=3)
        handles.append(Patch(label=name, **kw))
    symlog_axis(b, 0.01)
    b.axhline(0, color=INK2, lw=0.6)
    b.set_xticks(range(3))
    b.set_xticklabels([n for _, n in arms])
    b.set_ylabel("ΔWER vs full cache [95 % CI]")
    b.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=6.3, handlelength=1.4,
             labelspacing=0.3, columnspacing=1.2, borderaxespad=0.2)
    label(b, "(b)", x=-0.12)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_t2_padding")


# ------------------------------------------------------------------ fallback threshold
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


def fig_tau():
    fig = plt.figure(figsize=(FULL, 11.6 * CM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.42)
    x = np.arange(len(TAUS))
    for j, (m, name) in enumerate(M.MODELS):
        ax = fig.add_subplot(gs[0, j])
        q, dl, mk = quality(m), delta_of(m), STYLE[m]["marker"]
        chosen = J(f"results_framework_{m}.json")["choice"]
        ax.axhline(dl, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=GRID, lw=0.6, zorder=1)
        ax.errorbar(x, q[:, 0], yerr=[q[:, 0] - q[:, 1], q[:, 2] - q[:, 0]], fmt="none", ecolor=BLUE, elinewidth=0.8,
                    capsize=1.8, capthick=0.8, zorder=2)
        ax.plot(x, q[:, 0], color=BLUE, lw=1.2, marker=mk, ms=4, mfc="white", mec=BLUE, mew=1.1, zorder=3)
        if chosen in TAUS:
            i = TAUS.index(chosen)
            ax.plot([x[i]], [q[i, 0]], marker=mk, ms=4.6, mfc=BLUE, mec=BLUE, zorder=4)
            ax.plot([x[i]], [q[i, 0]], marker="o", ms=10, mfc="none", mec=INK, mew=0.8, zorder=4)
        symlog_axis(ax, lin=0.01)
        ax.set_ylim(-0.012, 0.2)
        ax.set_yticks([-0.01, 0, 0.01, 0.1])
        ax.set_yticklabels(["−0.01", "0", "0.01", "0.1"] if j == 0 else [])
        ax.set_xticks(x)
        ax.set_xticklabels(["none"] + TAUS[1:], rotation=90)
        ax.set_title(name, fontsize=7.5, pad=3)
        ax.text(-0.3 if m == "medium_en" else len(TAUS) - 0.6, dl * (0.72 if m == "medium_en" else 1.0),
                f"δ = {dl:.4f}".rstrip("0"), fontsize=6.3,
                va="top" if m == "medium_en" else "bottom", ha="left" if m == "medium_en" else "right")
        if j == 0:
            ax.set_ylabel("ΔWER vs full cache")
            label(ax, "(a)", x=-0.42, y=1.08)
    fig.text(0.5, 0.493, "fallback threshold τ   (ring = arm chosen on the validation set)", ha="center", fontsize=7.5)

    ax_b, ax_c = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])
    for m, name in M.MODELS:
        q = quality(m)
        line(ax_b, x, 100 * q[:, 3], m, label=name)
        xs, sv = saving(m)
        line(ax_c, xs, sv, m)
        ax_c.text(xs[-1] + 0.18, sv[-1], f"{sv[-1]:.0f} %", fontsize=6.5, va="center")
    for ax, yl, lab, ttl in ((ax_b, "steps redone, %", "(b)", "Decoder steps redone on the full cache"),
                             (ax_c, "saving, %", "(c)", "Whole-utterance saving in ms/token vs full cache")):
        ax.set_title(ttl, fontsize=7.5, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(XT)
        ax.set_xlabel("fallback threshold τ")
        ax.set_ylabel(yl)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        label(ax, lab, x=-0.15, y=1.03)
    ax_b.legend(loc="upper left", handlelength=2.6, labelspacing=0.3, borderaxespad=0.2)
    ax_c.set_xlim(-0.4, 6.4)
    ax_c.set_ylim(-1.5, 20)
    ax_c.axhline(0, color=INK2, lw=0.6)
    ax_c.text(0.98, 0.93, "measured at no fallback, τ = 0.5, 0.7, 0.9", transform=ax_c.transAxes, fontsize=6.5, ha="right")
    save(fig, "fig_t10_tau")


if __name__ == "__main__":
    fig_padding()
    M.fig_budget()
    M.fig_ablation()
    M.fig_efficiency()
    M.fig_long()
    M.fig_roofline()
    M.fig_system()
    fig_tau()
