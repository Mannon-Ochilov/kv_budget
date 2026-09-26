"""Journal figures for the PadSink-Track manuscript (MDPI style).

Arial 8 pt, full width 17 cm (single column 8.5 cm), 600 dpi PNG + PDF,
panel labels (a), (b), ... Colours by role: PadSink-Track blue, the earlier
one-shot PadSink-KV orange, the other one-shot rules in greys with distinct
markers, the oracle black dashed, the gate delta grey dashed.

  fig_t1_method.png       method schematic
  fig_t2_padding.png      padding cliff (validation) + sink causal test 2x2
  fig_t3_budget.png       dWER vs budget, 4 models, one-shot vs track vs oracle
  fig_t4_trace.png        alignment-head attention and the tracked window (one utterance)
  fig_t5_ablation.png     component ablation at 0.5 K_i, 4 models
  fig_t6_efficiency.png   step time, L3 misses, whole-utterance ms/token

Usage:  python experiments/make_figures_track.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
CM = 1 / 2.54
FULL, HALF = 17.0 * CM, 8.5 * CM

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#1a1a1a", "#52514e", "#8a8984", "#e4e3df"
GREYS = {"split": "#6f6e69", "h2o_layer": "#9a9994", "pyramidkv": "#b9b8b2"}
MODELS = [("medium_uz", "Whisper-medium, Uzbek"), ("small_uz", "Whisper-small, Uzbek"),
          ("medium_en", "Whisper-medium, English"), ("small_en", "Whisper-small, English")]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 600, "pdf.fonttype": 42, "lines.linewidth": 1.4,
})


def J(n):
    return json.load(open(os.path.join(HERE, n)))


def delta_of(m):
    return round(J(f"results_eviction_budget_{m}.json")["test"]["full"]["wer"] * 0.2, 4)


def label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left", color=INK)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", name)


def symlog_axis(ax, lin=0.01):
    ax.set_yscale("symlog", linthresh=lin, linscale=0.6)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ figure 1
def fig_method():
    fig, ax = plt.subplots(figsize=(FULL, 6.4 * CM))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 40)
    ax.axis("off")
    x0, x1, y = 4, 96, 25
    xa = x0 + (x1 - x0) * 0.3
    ax.text(x0, 37.5, "Cross-attention K/V cache: 1500 encoder positions per layer (the full cache stays in DRAM)",
            color=INK, ha="left", va="center")
    ax.add_patch(Rectangle((x0, y), xa - x0, 4, fc="#dbe8f7", ec=INK2, lw=0.6))
    ax.add_patch(Rectangle((xa, y), x1 - xa, 4, fc="#f1f0ec", ec=INK2, lw=0.6))
    ax.text(x0 + 1, y + 2, "speech", ha="left", va="center", color=INK)
    ax.text(x1 - 1, y + 2, "contextualized padding (≈ 80 %)", ha="right", va="center", color=INK2)
    for p_ in (35.5, 38.0, 41.5, 44.0, 48.0):
        ax.add_patch(Rectangle((p_, y), 0.8, 4, fc=ORANGE, ec="none"))
    ax.annotate(r"padding sink $S_\ell$: kept exactly, fixed", xy=(41.9, y), xytext=(41.9, y - 4.2),
                ha="center", va="center", color=ORANGE, fontsize=7,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
    for cx, a, ls in ((6, 0.35, "--"), (13, 0.6, "--"), (20, 1.0, "-")):
        ax.add_patch(Rectangle((cx, y - 0.4), 11, 4.8, fc="none", ec=BLUE, lw=1.2, alpha=a, ls=ls))
    ax.annotate("", xy=(31, y + 6.4), xytext=(7, y + 6.4), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0))
    ax.text(19, y + 7.4, "window $W_t$ follows the alignment peak $c_t$", color=BLUE, ha="center", va="bottom",
            fontsize=7)

    def box(x, yb, w, h, txt, ec=INK2):
        ax.add_patch(FancyBboxPatch((x, yb), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc="white", ec=ec, lw=0.7))
        ax.text(x + w / 2, yb + h / 2, txt, ha="center", va="center", color=INK, fontsize=7, linespacing=1.3)
    yb, h = 3.5, 11
    box(3, yb, 22, h, "working set of step $t$:\n"
        r"$k = |W_t| + |S_\ell|$ positions" "\nin a ring buffer\n(0–3 slots rewritten)", ec=BLUE)
    box(30, yb, 16, h, "decoder step $t$\nreads only the\n$k$ positions")
    box(51, yb, 24, h, "alignment heads' attention\nover $W_t$ gives the peak;\n"
        r"$c_{t+1} = \max(c_t, \mathrm{peak})$")
    box(80, yb, 17, h, "next token; at the\nend of speech the\nwindow runs into\nthe padding")
    for xa_, xb_ in ((25.4, 29.6), (46.4, 50.6), (75.4, 79.6)):
        ax.annotate("", xy=(xb_, yb + h / 2), xytext=(xa_, yb + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.annotate("", xy=(14, yb - 0.6), xytext=(88.5, yb - 0.6),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, connectionstyle="arc3,rad=-0.06"))
    ax.text(51, -3.4, "next step", color=INK2, ha="center", va="bottom", fontsize=7)
    ax.annotate("", xy=(13.5, yb + h + 0.4), xytext=(25.5, y - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8))
    save(fig, "fig_t1_method")


# ------------------------------------------------------------------ figure 2
def fig_padding():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 6.0 * CM), gridspec_kw={"width_ratios": [1, 1.25]})
    markers = {"medium_uz": "o", "small_uz": "s", "medium_en": "^", "small_en": "D"}
    cols = {"medium_uz": BLUE, "small_uz": AQUA, "medium_en": ORANGE, "small_en": INK2}
    for m, name in MODELS:
        calib = J(f"results_eviction_budget_{m}.json")["calib"]
        f_r = J(f"results_eviction_budget_{m}.json")["choice"]["rule"][1]
        pts = sorted((float(k.split("/")[2]), v["wer"]) for k, v in calib.items()
                     if k.startswith(f"split/{f_r}/"))
        full = calib["full"]["wer"]
        x = [p[0] for p in pts]
        yv = [p[1] - full for p in pts]
        xp = [max(v, 0.012) for v in x]      # f_p = 0 drawn at the left edge of the log axis
        a.plot(xp, yv, marker=markers[m], ms=4, color=cols[m], lw=1.2, label=name, mec="white", mew=0.5)
    a.set_xscale("log")
    a.set_xticks([0.012, 0.05, 0.1, 0.25, 0.5, 1.0])
    a.set_xticklabels(["0", "0.05", "0.1", "0.25", "0.5", "1"])
    symlog_axis(a, 0.02)
    a.set_xlabel("kept padding fraction f_p (validation, calibrated f_r)")
    a.set_ylabel("ΔWER vs full cache")
    a.axvspan(0.009, 0.03, color="#f6e3dc", lw=0)
    a.text(0.0195, -0.06, "cliff", color=ORANGE, fontsize=7, ha="center")
    a.legend(loc="upper right", handlelength=1.8)
    label(a, "(a)")
    # causal test
    arms = [("sink_mean", "sink K/V\n→ mean"), ("audio_mean", "same number of\naudio K/V → mean"),
            ("pad_rest_mean", "rest of padding\n→ mean")]
    width = 0.2
    for j, (m, name) in enumerate(MODELS):
        sc = J(f"results_sink_causal_{m}.json")
        sc = sc.get("arms", sc)
        vals = [sc[k]["delta_vs_full"][0] for k, _ in arms]
        lo = [sc[k]["delta_vs_full"][1] for k, _ in arms]
        hi = [sc[k]["delta_vs_full"][2] for k, _ in arms]
        xs = np.arange(3) + (j - 1.5) * width
        b.bar(xs, vals, width * 0.9, color=cols[m], label=name, zorder=2)
        b.errorbar(xs, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, lw=0.6,
                   capsize=1.5, zorder=3)
    symlog_axis(b, 0.01)
    b.axhline(0, color=INK2, lw=0.6)
    b.set_xticks(range(3))
    b.set_xticklabels([n for _, n in arms])
    b.set_ylabel("ΔWER vs full cache [95 % CI]")
    b.set_title("medium: the sink carries content; small: it is a placeholder (colours as in (a))",
                fontsize=7, color=INK2, pad=10)
    label(b, "(b)", x=-0.12)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_t2_padding")


# ------------------------------------------------------------------ figure 3
def budget_series(m):
    e6 = J(f"results_eviction_budget_{m}.json")["test"]
    sk = [k for k in e6 if k.startswith("split")][0]
    sb = J(f"results_sota_baselines_{m}.json")["arms"]
    sp = J(f"results_spar_{m}.json")["test"]
    spk = [k for k in sp if k.startswith("spar/")]
    rho = {"medium_uz": "0.9", "small_uz": "0.95", "medium_en": "0.8", "small_en": "0.9"}[m]
    dg = J(f"results_diag_budget_{m}.json")["arms"]
    tr = J(f"results_align_track_{m}.json")["arms"]
    ab = J(f"results_track_ablation_{m}.json")["arms"]
    S = {}
    S["split"] = [e6[sk]["delta_vs_full"], dg["split/s0.5"]["delta_vs_full"], dg["split/s0.25"]["delta_vs_full"]]
    for r in ("h2o_layer", "pyramidkv"):
        S[r] = [sb[r]["delta_vs_full"], dg[f"{r}/s0.5"]["delta_vs_full"], dg[f"{r}/s0.25"]["delta_vs_full"]]
    S["padsink"] = [sp[f"spar/rho{rho}"]["delta_vs_full"], dg["padsink/s0.5"]["delta_vs_full"],
                    dg["padsink/s0.25"]["delta_vs_full"]]
    S["track"] = [ab["track/s1.0"]["delta_vs_full"], tr["track/s0.5"]["delta_vs_full"],
                  tr["track/s0.25"]["delta_vs_full"]]
    S["oracle"] = [dg["oracle/s1"]["delta_vs_full"], dg["oracle/s0.5"]["delta_vs_full"],
                   dg["oracle/s0.25"]["delta_vs_full"]]
    return S


def fig_budget():
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 5.6 * CM), sharey=True)
    xs = [1.0, 0.5, 0.25]
    style = {
        "split": dict(color=GREYS["split"], marker="s", label="calibrated split (one-shot)"),
        "h2o_layer": dict(color=GREYS["h2o_layer"], marker="v", label="H2O per layer (one-shot)"),
        "pyramidkv": dict(color=GREYS["pyramidkv"], marker="^", label="PyramidKV (one-shot)"),
        "padsink": dict(color=ORANGE, marker="o", label="PadSink-KV (one-shot)"),
        "track": dict(color=BLUE, marker="o", label="PadSink-Track", lw=2.0),
        "oracle": dict(color=INK, marker="x", ls="--", label="per-step oracle (not realizable)", lw=1.0),
    }
    for ax, (m, name), tag in zip(axs, MODELS, "abcd"):
        S = budget_series(m)
        d = delta_of(m)
        ax.axhline(d, color=MUTED, ls=(0, (4, 3)), lw=0.8)
        ax.axhline(0, color=INK2, lw=0.5)
        for key, st in style.items():
            y = [v[0] for v in S[key]]
            kw = dict(st)
            kw.setdefault("lw", 1.2)
            ax.plot(xs, y, ms=4, mec=kw["color"] if key == "oracle" else "white", mew=0.8 if key == "oracle" else 0.5, zorder=3 if key in ("track", "oracle") else 2, **kw)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["$K_i$", "$K_i/2$", "$K_i/4$"])
        ax.invert_xaxis()
        symlog_axis(ax, 0.01)
        ax.set_title(f"({tag}) {name}", color=INK, pad=8)
        ax.text(0.97, d * 1.25, "δ", color=MUTED, fontsize=7, ha="right", transform=ax.get_yaxis_transform())
    axs[0].set_ylabel("ΔWER vs full cache (symlog)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_t3_budget")


# ------------------------------------------------------------------ figure 4
def fig_trace():
    z = np.load(os.path.join(HERE, "trace_track_medium_uz_22.npz"), allow_pickle=True)
    att, wins, rn = z["att"], z["wins"], int(z["rn"])
    sinks = z["sinks"]
    T = len(att)
    fig, ax = plt.subplots(figsize=(FULL, 5.4 * CM))
    xmax = min(1500, rn + 260)
    im = ax.imshow(np.sqrt(att[:, :xmax]), aspect="auto", origin="lower", cmap="Blues",
                   extent=[0, xmax, -0.5, T - 0.5], interpolation="nearest")
    for t, (lo, hi) in enumerate(wins):
        ax.add_patch(Rectangle((lo, t - 0.5), hi - lo, 1.0, fc=ORANGE, alpha=0.13, ec="none"))
        ax.plot([lo, lo], [t - 0.5, t + 0.5], color=ORANGE, lw=0.8)
        ax.plot([hi, hi], [t - 0.5, t + 0.5], color=ORANGE, lw=0.8)
    ax.axvline(rn, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(rn - 4, 1.0, "end of speech", color=INK, fontsize=7, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", pad=1.0))
    ss = sinks[sinks < xmax]
    ax.plot(ss, np.full(len(ss), -1.6), "|", color=ORANGE, ms=5, clip_on=False)
    ax.set_ylim(-2.2, T - 0.5)
    ax.set_xlabel("encoder position (20 ms each)")
    ax.set_ylabel("decoding step t")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("√ attention", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig_t4_trace")


# ------------------------------------------------------------------ figure 5
def fig_ablation():
    rows = [("track", "PadSink-Track"), ("track_int8", "+ int8 cache"), ("track_sum", "+ padding summary slot"),
            ("v1", "window kept inside speech (v1)"), ("allheads", "peak from all heads"),
            ("fixed_rate", "fixed-rate window (MURMUR-style)"), ("fixed_rate_utt", "fixed rate, per-utterance best"),
            ("static", "window does not move"), ("nosink", "no padding sink")]
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 6.4 * CM), sharey=True)
    for ax, (m, name), tag in zip(axs, MODELS, "abcd"):
        tr = J(f"results_align_track_{m}.json")["arms"]
        ab = J(f"results_track_ablation_{m}.json")["arms"]
        d = delta_of(m)
        src = {"track": tr["track/s0.5"], "track_sum": tr["track_sum/s0.5"], "track_int8": ab["track_int8/s0.5"]}
        for k in ("v1", "allheads", "fixed_rate", "fixed_rate_utt", "static", "nosink"):
            src[k] = ab[f"{k}/s0.5"]
        for i, (k, _) in enumerate(rows):
            v = src[k]["delta_vs_full"]
            col = BLUE if k in ("track", "track_int8", "track_sum") else GREYS["split"]
            ax.errorbar(v[0], i, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=3.5, color=col, ecolor=col,
                        lw=0.8, capsize=0, mec="white", mew=0.4)
        ax.axvline(d, color=MUTED, ls=(0, (4, 3)), lw=0.8)
        ax.axvline(0, color=INK2, lw=0.5)
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
        ax.set_xlim(-0.01, 8)
        ax.set_ylim(len(rows) - 0.5, -1.0)
        ax.set_xticks([0, 0.01, 0.1, 1])
        ax.set_xticklabels(["0", "0.01", "0.1", "1"])
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"({tag}) " + name.replace(", ", "\n"), color=INK)
        ax.text(d, -0.75, " δ", color=MUTED, fontsize=7, va="bottom")
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([r[1] for r in rows])
    fig.supxlabel("ΔWER vs full cache at $K_i/2$ [95 % CI] (symlog)", fontsize=8, y=0.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_t5_ablation")


# ------------------------------------------------------------------ figure 6
def fig_efficiency():
    lat = J("results_track_latency.json")["models"]
    llc = J("results_llc_miss_step.json")
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 5.4 * CM))
    # (a) step time
    keys = [("full", "full cache (1500)"), ("oneshot_Ki", "one-shot, $K_i$"), ("oneshot_k", "one-shot, $K_i/2$"),
            ("track_k", "Track, per-step gather"), ("track_ring", "Track, ring buffer")]
    cols = [MUTED, GREYS["split"], GREYS["h2o_layer"], "#9fc3ee", BLUE]
    for j, (m, title) in enumerate((("medium_uz", "medium"), ("small_en", "small"))):
        d = lat[m]
        vals = []
        for k, _ in keys:
            kk = [x for x in d if x.startswith(k)][0]
            vals.append(d[kk]["median_ms"])
        xs = np.arange(len(keys)) * 0.17 + j
        axs[0].bar(xs, vals, 0.15, color=cols, zorder=2)
        for x, v in zip(xs, vals):
            axs[0].text(x, v + 0.6, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", fontsize=6, color=INK2)
    axs[0].set_xticks([0.34, 1.34])
    axs[0].set_xticklabels(["Whisper-medium", "Whisper-small"])
    axs[0].set_ylabel("decoder step, ms (t = 30, 1 thread)")
    axs[0].grid(axis="y", color=GRID, lw=0.5)
    axs[0].set_axisbelow(True)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in cols]
    axs[0].legend(handles, [n for _, n in keys], loc="upper right", fontsize=6)
    axs[0].set_ylim(0, 88)
    label(axs[0], "(a)")
    # (b) L3 misses
    rowsb = [("medium  fp32 cache, 1500", "full"), ("medium  fp32 cache, 393", "one-shot $K_i$"),
             ("medium  oneshot_k (0.5 K_i)", "one-shot $K_i/2$"), ("medium  track_ring (0.5 K_i)", "Track (ring)")]
    rowss = [("small   fp32 cache, 1500", "full"), ("small   fp32 cache, 348", "one-shot K_i"),
             ("small   oneshot_k (0.5 K_i)", "one-shot K_i/2"), ("small   track_ring (0.5 K_i)", "Track (ring)")]
    cb = [MUTED, GREYS["split"], GREYS["h2o_layer"], BLUE]
    for j, rows in enumerate((rowsb, rowss)):
        vals = [llc[k]["llc_load_misses_per_step"] / 1e6 for k, _ in rows]
        xs = np.arange(4) * 0.2 + j
        axs[1].bar(xs, vals, 0.18, color=cb, zorder=2)
        for x, v in zip(xs, vals):
            axs[1].text(x, v + 0.1, f"{v:.2f}", ha="center", fontsize=6, color=INK2)
    axs[1].set_xticks([0.3, 1.3])
    axs[1].set_xticklabels(["Whisper-medium", "Whisper-small"])
    axs[1].set_ylabel("L3 load misses per step, millions")
    axs[1].grid(axis="y", color=GRID, lw=0.5)
    axs[1].set_axisbelow(True)
    axs[1].legend([Rectangle((0, 0), 1, 1, color=c) for c in cb], [n for _, n in rowsb], fontsize=6)
    axs[1].set_ylim(0, 10)
    label(axs[1], "(b)")
    # (c) end-to-end
    for j, m in enumerate(("medium_uz", "small_en")):
        e = J(f"results_e2e_latency_{m}.json")["arms"]
        vals = [e[k]["ms_per_token_median"] for k in ("full", "oneshot_Ki", "track_ring")]
        xs = np.arange(3) * 0.22 + j
        axs[2].bar(xs, vals, 0.2, color=[MUTED, GREYS["split"], BLUE], zorder=2)
        for x, v in zip(xs, vals):
            axs[2].text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=6, color=INK2)
    axs[2].set_xticks([0.22, 1.22])
    axs[2].set_xticklabels(["Whisper-medium", "Whisper-small"])
    axs[2].set_ylabel("whole-utterance decoding, ms per token")
    axs[2].grid(axis="y", color=GRID, lw=0.5)
    axs[2].set_axisbelow(True)
    axs[2].legend([Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, GREYS["split"], BLUE)],
                  ["full cache", "one-shot, $K_i$", "Track, $K_i/2$ (ring)"], fontsize=6)
    axs[2].set_ylim(0, 85)
    label(axs[2], "(c)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_t6_efficiency")


# ------------------------------------------------------------------ figure 1 (full method figure)
def fig_method_full():
    fig, ax = plt.subplots(figsize=(FULL, 11.0 * CM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    T, P, RN = 14, 40, 20                        # schematic: steps, positions, speech positions
    rng = np.random.default_rng(7)
    need = np.clip(np.round(np.linspace(1, RN - 2, T) + rng.normal(0, 0.5, T)), 0, RN - 1).astype(int)
    sinks = np.array([RN + 1, RN + 2, RN + 5, RN + 9, RN + 14])

    def grid(x0, title):
        gw, y_top, y_bot = 27.0, 55.0, 34.0
        cw, rh = gw / P, (y_top - y_bot) / T
        ax.add_patch(Rectangle((x0, y_bot), RN * cw, y_top - y_bot, fc="#e6eff9", ec="none"))
        ax.add_patch(Rectangle((x0 + RN * cw, y_bot), (P - RN) * cw, y_top - y_bot, fc="#f2f1ec", ec="none"))
        ax.add_patch(Rectangle((x0, y_bot), gw, y_top - y_bot, fc="none", ec=INK2, lw=0.5))
        ax.text(x0, 62.5, title[0], fontsize=7.5, color=INK, va="top")
        ax.text(x0, 59.9, title[1], fontsize=6.5, color=INK2, va="top")
        ax.text(x0 + RN * cw / 2, y_bot - 1.2, "speech", fontsize=6.5, color=INK2, ha="center", va="top")
        ax.text(x0 + RN * cw + (P - RN) * cw / 2, y_bot - 1.2, "padding (≈ 80 %)", fontsize=6.5, color=INK2,
                ha="center", va="top")
        ax.text(x0 - 0.6, y_top - rh / 2, "t = 1", fontsize=6, color=INK2, ha="right", va="center")
        ax.text(x0 - 0.6, y_bot + rh / 2, f"t = {T}", fontsize=6, color=INK2, ha="right", va="center")
        ax.annotate("", xy=(x0 - 3.2, y_bot + 1), xytext=(x0 - 3.2, y_top - 1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6))
        ax.text(x0 + gw / 2, y_top + 0.8, "encoder positions →", fontsize=6, color=MUTED, ha="center", va="bottom")
        cx = lambda p_: x0 + (p_ + 0.5) * cw                       # noqa: E731
        cy = lambda t_: y_top - (t_ + 0.5) * rh                    # noqa: E731
        return cw, rh, cx, cy, y_top, y_bot

    # (a) one-shot
    cw, rh, cx, cy, yt, yb = grid(3, ("(a) One-shot retention", "H2O, SnapKV, PyramidKV, PadSink-KV"))
    kept = np.unique(np.concatenate([[0, 1, 2, 3, 5, 6], sinks]))
    for p_ in kept:
        ax.add_patch(Rectangle((cx(p_) - cw * 0.42, yb), cw * 0.84, yt - yb, fc=GREYS["h2o_layer"], alpha=0.8, ec="none"))
    for t_, p_ in enumerate(need):
        hit = p_ in kept
        ax.plot(cx(p_), cy(t_), marker="o" if hit else "x", ms=3.4 if hit else 3.8,
                color=INK if hit else "#d23b3b", mew=1.2, zorder=3)
    # (b) track
    cw, rh, cx, cy, yt, yb = grid(36, ("(b) PadSink-Track", "the selection is renewed at every step"))
    w = 7
    for t_, p_ in enumerate(need):
        lo = int(np.clip(p_ - 1, 0, P - w))
        ax.add_patch(Rectangle((cx(lo) - cw / 2, cy(t_) - rh * 0.45), w * cw, rh * 0.9, fc=BLUE, alpha=0.3, ec="none"))
    for p_ in sinks:
        ax.add_patch(Rectangle((cx(p_) - cw * 0.42, yb), cw * 0.84, yt - yb, fc=ORANGE, alpha=0.85, ec="none"))
    for t_, p_ in enumerate(need):
        ax.plot(cx(p_), cy(t_), marker="o", ms=3.4, color=INK, zorder=3)
    # legend
    items = [(INK, "o", "position the decoder needs at step t"),
             ("#d23b3b", "x", "needed but evicted → errors, loops"),
             (GREYS["h2o_layer"], "s", "one-shot set: chosen at t = 1, fixed"),
             (BLUE, "s", "window $W_t$: follows the alignment peak"),
             (ORANGE, "s", r"padding sink $S_\ell$: exact K/V, fixed")]
    for n, (col, mk, txt) in enumerate(items):
        y = 54 - n * 4.4
        ax.plot(71.5, y, marker=mk, ms=6 if mk == "s" else 4.2, color=col, mew=1.2, alpha=0.85 if mk == "s" else 1)
        ax.text(73.5, y, txt, fontsize=6.5, color=INK, va="center")

    # (c) the step loop
    ax.text(3, 28.5, "(c) One decoding step of PadSink-Track", fontsize=7.5, color=INK, va="bottom")

    def box(x, y, wdt, h, txt, ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), wdt, h, boxstyle="round,pad=0.35,rounding_size=0.8", fc="white", ec=ec, lw=0.7))
        ax.text(x + wdt / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.0, color=INK, linespacing=1.35)

    yb2, h2 = 14.5, 11.5
    boxes = [(3, 17.6, "once, at t = 1\n(full cache):\n" r"sink $S_\ell$ = smallest set" "\n"
              "with ρ of padding mass;\n" r"start peak $c_1$", ORANGE),
             (23.4, 17.4, r"ring buffer $B_\ell = W_t \cup S_\ell$" "\nk positions per layer;\n"
              "the full cache\nstays in DRAM", BLUE),
             (43.6, 12.6, "decoder step t\n" r"reads only $B_\ell$" "\n" r"→ token $y_t$", INK2),
             (58.9, 19.8, "alignment heads'\n" r"attention over $W_t$ → peak;" "\n"
              r"$c_{t+1} = \max(c_t, \mathrm{peak})$", INK2),
             (81.4, 16.2, "shift the window to\n" r"$[c-0.1w,\ c+0.9w)$;" "\n"
              "at the end of speech\nit runs into padding", INK2)]
    for x, wdt, txt, ec in boxes:
        box(x, yb2, wdt, h2, txt, ec)
    for (x1, w1, *_), (x2, *_r) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x2 - 0.6, yb2 + h2 / 2), xytext=(x1 + w1 + 0.6, yb2 + h2 / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.annotate("", xy=(32.1, yb2 - 0.8), xytext=(89.5, yb2 - 0.8),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, connectionstyle="arc3,rad=-0.09"))
    ax.text(59.8, 9.2, "next step", fontsize=6.3, color=INK2, ha="center", va="top")
    # ring buffer slots
    x0, y0, sw = 3, 0.8, 3.0
    labels = ["109", "110", "103", "104", "105", "106", "107", "108", "s", "s", "s"]
    for n, lab in enumerate(labels):
        fc = "#f7d9cc" if lab == "s" else ("#fde0d2" if n < 2 else "#dbe8f7")
        ax.add_patch(Rectangle((x0 + n * sw, y0), sw * 0.94, 3.2, fc=fc, ec=INK2, lw=0.4))
        ax.text(x0 + n * sw + sw * 0.47, y0 + 1.6, lab, fontsize=5.0, ha="center", va="center",
                color=ORANGE if n < 2 else INK)
    ax.text(x0 + 11 * sw + 1.2, y0 + 1.6,
            "ring buffer of one layer: the slots of positions 101, 102 that left the window are overwritten with the entering "
            "109, 110 (0–3 slots per step,\nno per-step gather); s = sink slots. Attention ignores slot order, so the result equals "
            "reading the positions in order.", fontsize=6, color=INK2, va="center")
    save(fig, "fig_t1_method")


# ------------------------------------------------------------------ figure 7
def fig_long():
    sets = [("small_en_long", "(a) Whisper-small, English, ≥ 10 s (n = 80)"),
            ("medium_uz_long", "(b) Whisper-medium, Uzbek, 15–30 s composites (n = 89)")]
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 6.2 * CM), sharey=True)
    style = {"h2o_layer": dict(color=GREYS["h2o_layer"], marker="v", label="H2O per layer (one-shot)"),
             "padsink": dict(color=ORANGE, marker="o", label="PadSink-KV (one-shot)"),
             "track": dict(color=BLUE, marker="o", label="PadSink-Track", lw=2.0)}
    for ax, (name, title) in zip(axs, sets):
        r = J(f"results_long_fixed_k_{name}.json")
        d = r["delta"]
        ax.axhline(d, color=MUTED, ls=(0, (4, 3)), lw=0.8)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.text(1480, d * 1.25, "δ", color=MUTED, fontsize=7, ha="right")
        for key, st in style.items():
            ys = [r["arms"][f"{key}/k{k}"]["delta_vs_full"] for k in (200, 400)]
            kw = dict(st)
            kw.setdefault("lw", 1.2)
            ax.plot([200, 400], [v[0] for v in ys], ms=4.5, mec="white", mew=0.5,
                    zorder=3 if key == "track" else 2, **kw)
            for k, v in zip((200, 400), ys):
                ax.plot([k, k], [v[1], v[2]], color=st["color"], lw=0.8, zorder=1)
        sp = r["arms"]["split/Ki"]["delta_vs_full"]
        ki = r["split_Ki_mean"]
        ax.plot([ki], [sp[0]], marker="s", ms=5, color=GREYS["split"], mec="white", mew=0.5, ls="none",
                label="calibrated split at its own $K_i$ (one-shot)")
        ax.annotate(f"mean $K_i$ = {ki:.0f}", xy=(ki, sp[0]), xytext=(ki, 0.35), ha="center", fontsize=7,
                    color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.axvline(r["speech_positions_mean"], color=INK2, lw=0.6, ls=(0, (1, 2)))
        ax.text(r["speech_positions_mean"] + 15, 2.2, "mean speech\npositions", fontsize=6.5, color=INK2, va="top")
        ax.set_xscale("log")
        ax.set_xlim(150, 1500)
        ax.set_xticks([200, 400, 800, 1500])
        ax.set_xticklabels(["200", "400", "800", "1500"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        symlog_axis(ax, 0.01)
        ax.set_title(title, color=INK)
        ax.set_xlabel("positions read per step and layer")
    axs[0].set_ylabel("ΔWER vs full cache (symlog)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_t7_long")


if __name__ == "__main__":
    fig_long()
    fig_method_full()
    fig_padding()
    fig_budget()
    fig_trace()
    fig_ablation()
    fig_efficiency()
