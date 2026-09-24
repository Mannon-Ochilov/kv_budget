"""The three summary figures for the cache paper, drawn from the result files.

  fig_kv_cache_wer.png     WER against cross-attention cache bytes, one panel
                           per model: the precision family, the eviction
                           family, and their composition; gate and budget.
  fig_kv_calib_grid.png    the E6 calibration grids, WER over
                           (audio fraction x padding fraction), chosen cell.
  fig_kv_step_time.png     decoder step at t = 30, full vs evicted cache.

MDPI conventions as for the paper's figures: Palatino, bold only for panel
and axis titles, 17 cm wide, 600 dpi, Okabe-Ito colours, annotations inside
the drawing. Nothing is computed here; every number is read from the JSON
the experiments wrote.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
EPS, M_L3, ALPHA = 0.20, 24.0, 0.7
W_LAYER = {"medium_uz": 14.00, "small_en": 7.88}
LAYERS = {"medium_uz": 24, "small_en": 12}

BLUE, ORANGE, GREEN, PINK, DARK, GREY = ("#0072B2", "#D55E00", "#009E73",
                                         "#CC79A7", "#1a1a1a", "#888888")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "font.size": 8.5, "font.weight": "normal",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def J(name):
    return json.load(open(os.path.join(HERE, name)))


def ci_bar(ax, x, y, ref, lo, hi, col):
    ax.errorbar([x], [y], yerr=[[y - (ref + lo)], [(ref + hi) - y]], fmt="none",
                ecolor=col, elinewidth=0.8, capsize=2.5)


# ------------------------------------------------------------- figure 1
def cache_wer():
    panels = []
    # medium: precision from E1 (cascade), H2O from E5, split from E6/E6b
    e1 = J("results_cache_precision_wer_cascade.json")["schemes"]
    e5 = J("results_cache_eviction_wer_cascade_n300.json")["arms"]
    e6 = J("results_eviction_budget_medium_uz.json")["test"]
    cb = J("results_eviction_combo_medium_uz.json")["arms"]
    bd = J("results_boundary_int8_medium_uz.json")
    ref = e6["full"]["wer"]
    pts = [("prec", e1["fp16"]["cross_cache_mib"], e1["fp16"]["wer"], e1["fp16"]["delta_vs_fp32"], "FP16"),
           ("prec", e1["int8_kivi"]["cross_cache_mib"], e1["int8_kivi"]["wer"], e1["int8_kivi"]["delta_vs_fp32"], "int8"),
           ("prec", e1["int4_kivi"]["cross_cache_mib"], e1["int4_kivi"]["wer"], e1["int4_kivi"]["delta_vs_fp32"], "int4"),
           ("evict", e5["mass50/fp32"]["cross_cache_mib_mean"], e5["mass50/fp32"]["wer"], e5["mass50/fp32"]["delta_vs_full"], "H2O 50%"),
           ("evict", e5["mass25/fp32"]["cross_cache_mib_mean"], e5["mass25/fp32"]["wer"], e5["mass25/fp32"]["delta_vs_full"], "H2O 25%"),
           ("ours", e6["split/1.000/0.100"]["mib"], e6["split/1.000/0.100"]["wer"], e6["split/1.000/0.100"]["delta_vs_full"], "calibrated"),
           ("both", cb["split/int4_kivi"]["mib"], cb["split/int4_kivi"]["wer"], cb["split/int4_kivi"]["delta_vs_full"], "calibrated + int4"),
           ("lat", cb["split/int8_head"]["mib"], cb["split/int8_head"]["wer"], cb["split/int8_head"]["delta_vs_full"], "calibrated + int8 (latency)"),
           ("exec", bd["mib"], bd["wer"], bd["delta_vs_full"], f"int8, K = {bd['K']} (selected)")]
    panels.append(("(a) Whisper-medium, Uzbek, 300 test utterances", e6["full"]["mib"], ref, pts, "medium_uz"))

    s7 = J("results_cache_sweep_small_en_n300.json")["arms"]
    e6s = J("results_eviction_budget_small_en.json")["test"]
    cbs = J("results_eviction_combo_small_en.json")["arms"]
    ref = e6s["full"]["wer"]
    sk = [k for k in e6s if k.startswith("split")][0]
    pts = [("prec", s7["full/fp16"]["mib"], s7["full/fp16"]["wer"], s7["full/fp16"]["delta_vs_full"], "FP16"),
           ("prec", s7["full/int8_kivi"]["mib"], s7["full/int8_kivi"]["wer"], s7["full/int8_kivi"]["delta_vs_full"], "int8"),
           ("prec", s7["full/int4_kivi"]["mib"], s7["full/int4_kivi"]["wer"], s7["full/int4_kivi"]["delta_vs_full"], "int4"),
           ("evict", s7["mass50/fp32"]["mib"], s7["mass50/fp32"]["wer"], s7["mass50/fp32"]["delta_vs_full"], "H2O 50%"),
           ("evict", s7["mass25/fp32"]["mib"], s7["mass25/fp32"]["wer"], s7["mass25/fp32"]["delta_vs_full"], "H2O 25%"),
           ("ours", e6s[sk]["mib"], e6s[sk]["wer"], e6s[sk]["delta_vs_full"], "calibrated"),
           ("both", cbs["split/int4_kivi"]["mib"], cbs["split/int4_kivi"]["wer"], cbs["split/int4_kivi"]["delta_vs_full"], "calibrated + int4")]
    panels.append(("(b) whisper-small, English, LibriSpeech test-clean", e6s["full"]["mib"], ref, pts, "small_en"))

    style = {"prec": (BLUE, "o", "precision only (full cache)"),
             "evict": (ORANGE, "s", "eviction only (H2O, FP32)"),
             "ours": (GREEN, "^", "calibrated split rule (FP32)"),
             "both": (PINK, "D", "calibrated + int4 (sim.)"),
             "lat": (GREEN, "P", "calibrated + int8, executable (latency option)"),
             "exec": (GREEN, "*", "int8, budget boundary: hardware-selected")}
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.3))
    for ax, (title, full_mib, ref, pts, name) in zip(axes, panels):
        gate = round(ref, 4) + round(ref * EPS, 4)
        BUDGET = LAYERS[name] * (ALPHA * M_L3 - W_LAYER[name])
        ax.axhline(ref, color=GREY, lw=0.8, ls=(0, (5, 4)))
        ax.axhline(gate, color=DARK, lw=0.9, ls=(0, (6, 3)))
        ax.axvline(BUDGET, color=GREY, lw=0.8, ls=":")
        ax.plot([full_mib], [ref], marker="D", ms=6, color=DARK, ls="none")
        ax.annotate("full FP32", (full_mib, ref), xytext=(-4, -11), textcoords="offset points",
                    ha="right", fontsize=7, color=DARK)
        for fam, x, y, (_, lo, hi), lab in pts:
            col, mk, _ = style[fam]
            ci_bar(ax, x, y, ref, lo, hi, col)
            ax.plot([x], [y], marker=mk, ms=11 if fam == "exec" else (7 if fam == "lat" else 6),
                    mfc=col if fam in ("exec", "lat") else "white",
                    mec=col, mew=1.2, ls="none")
            off = {"prec": ((-6, -9), "right", "top"),
                   "evict": ((7, 0), "left", "center"),
                   "ours": ((0, 9), "center", "bottom"),
                   "both": ((0, 9), "center", "bottom"),
                   "lat": ((0, -11), "center", "top"),
                   "exec": ((-6, 24), "right", "bottom")}[fam]
            if fam == "exec":
                ax.annotate(lab, (x, y), xytext=(28, 62), textcoords="offset points", ha="left", va="bottom",
                            fontsize=6.6, color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.7))
                continue
            ax.annotate(lab, (x, y), xytext=off[0], textcoords="offset points",
                        ha=off[1], va=off[2], fontsize=6.6, color=col)
        ax.set_xscale("log")
        ax.set_xlim(min(q[1] for q in pts) * 0.45, max(full_mib, BUDGET) * 1.6)
        ax.text(0.99, gate, f"gate: FP32 + {EPS:.1f}·WER = {gate:.4f}", fontsize=6.8,
                ha="right", va="bottom", transform=ax.get_yaxis_transform(), color=DARK)
        ax.text(BUDGET * 0.96, 0.985, f"KV budget {BUDGET:.0f} MiB\n= L·(α·M_L3 − M_w)", fontsize=6.4, color=GREY,
                transform=ax.get_xaxis_transform(), va="top", ha="right")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("Cross-attention cache per step, MiB (log)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("WER")
    axes[0].set_ylim(0.17, 0.225)
    axes[1].set_ylim(0.043, 0.075)
    h = [plt.Line2D([], [], marker=m, mfc="white", mec=c, mew=1.2, ls="none", ms=6, label=l)
         for c, m, l in style.values()]
    axes[1].legend(handles=h, frameon=False, fontsize=6.8, loc="upper left",
                   bbox_to_anchor=(0.0, 0.97))
    fig.tight_layout(w_pad=1.5)
    out = os.path.join(FIG, "fig_kv_cache_wer.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


# ------------------------------------------------------------- figure 2
def calib_grid():
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.0))
    for ax, (name, title) in zip(axes, [("medium_uz", "(a) Whisper-medium, Uzbek validation"),
                                        ("small_en", "(b) whisper-small, LibriSpeech dev-clean")]):
        r = J(f"results_eviction_budget_{name}.json")
        g, ch = r["calib"], r["choice"]
        fr = sorted({v["rule"][1] for v in g.values() if v["rule"][0] == "split"}, reverse=True)
        fp = sorted({v["rule"][2] for v in g.values() if v["rule"][0] == "split"})
        W = np.array([[g[f"split/{a}/{b}"]["wer"] for b in fp] for a in fr])
        ref, delta = g["full"]["wer"], ch["delta"]
        im = ax.imshow(W, cmap="Blues", norm=LogNorm(vmin=ref * 0.9, vmax=W.max()),
                       aspect="auto")
        for i in range(len(fr)):
            for j in range(len(fp)):
                w = W[i, j]
                inside = round(w, 4) <= round(ref, 4) + delta
                ax.text(j, i, f"{w:.3f}" if w < 1 else f"{w:.1f}", ha="center",
                        va="center", fontsize=6.8,
                        color="white" if w > 0.5 else DARK,
                        fontweight="bold" if inside else "normal")
        ci, cj = fr.index(ch["rule"][1]), fp.index(ch["rule"][2])
        ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False,
                                   ec=ORANGE, lw=1.8))
        ax.set_xticks(range(len(fp)), [f"{x:g}" for x in fp])
        ax.set_yticks(range(len(fr)), [f"{x:g}" for x in fr])
        ax.set_xlabel("padding fraction kept, $f_p$")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.text(0.5, -0.30, f"full cache {ref:.4f}, gate {ref + delta:.4f};  "
                            f"bold: within gate;  box: chosen", transform=ax.transAxes,
                fontsize=6.8, ha="center", va="top", color=DARK)
        ax.tick_params(labelsize=7.5, length=0)
    axes[0].set_ylabel("audio fraction kept, $f_r$")
    fig.tight_layout(w_pad=1.5)
    out = os.path.join(FIG, "fig_kv_calib_grid.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


# ------------------------------------------------------------- figure 3
def step_time():
    r = J("results_step_latency_evicted.json")["arms"]
    llc = J("results_llc_miss_step.json")
    mibs = {"medium  fp32 cache, 1500": 281.2, "medium  fp32 cache, 393": 73.7,
            "medium  int8 cache, 1500": 70.3, "medium  int8 cache, 393": 18.4, "medium  int8 cache, 1433": 67.2,
            "small   fp32 cache, 1500": 105.5, "small   fp32 cache, 348": 24.5}
    order = ["medium  fp32 cache, 1500", "medium  fp32 cache, 393", "medium  int8 cache, 1500",
             "medium  int8 cache, 1433", "medium  int8 cache, 393", "small   fp32 cache, 1500",
             "small   fp32 cache, 348"]
    labels = [k for k in order if k in r]
    med = [r[k]["median_ms"] for k in labels]
    cols = [DARK if r[k]["positions"] == 1500 else (BLUE if r[k]["positions"] == 1433 else GREEN) for k in labels]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.69, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.array([0, 1, 2.4, 3.4, 4.4, 5.8, 6.8])
    ax.bar(x, med, width=0.8, color=cols, alpha=0.85)
    rng = np.random.default_rng(0)
    for xi, k, m in zip(x, labels, med):
        runs = r[k].get("runs_ms") or [r[k]["min_ms"], r[k]["max_ms"]]
        ax.plot(xi + rng.uniform(-0.22, 0.22, len(runs)), runs, marker="o", ms=2.6, ls="none",
                mfc="white", mec=DARK, mew=0.6)
        ax.text(xi, max(runs) + 1.0, f"{m:.1f}", ha="center", fontsize=7)
        ax.text(xi, 0.6, f"{r[k]['positions']}\npos.", ha="center", va="bottom", fontsize=6.4, color="white")
    ax.set_xticks([0.5, 3.4, 6.3], ["medium\nFP32 cache", "medium\nint8 cache", "small\nFP32 cache"])
    ax.set_ylabel("Decoder step at t = 30, ms")
    ax.set_ylim(0, max(med) * 1.45)
    ax.set_title("(a) step time: median bar; min and max of 21 rounds as points", fontsize=8, loc="left")
    ax.text(0.98, 0.97, "black: full cache (1500)\ngreen: calibrated retention", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.6, color=DARK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    # (b) LLC misses against cache bytes
    for k, v in llc.items():
        col = BLUE if k.startswith("medium") else ORANGE
        mk = "s" if "int8" in k else "o"
        ax2.plot([mibs[k]], [v["llc_all_misses_per_step"] / 1e6], marker=mk, ms=6, ls="none",
                 mfc="white" if v["positions"] == 1500 else col, mec=col, mew=1.2)
    # predicted slope: one 64-byte line per 64 bytes, anchored at the full FP32 point
    for k0, col in [("medium  fp32 cache, 1500", BLUE), ("small   fp32 cache, 1500", ORANGE)]:
        m0, y0 = mibs[k0], llc[k0]["llc_all_misses_per_step"] / 1e6
        xs = np.array([0, m0])
        ax2.plot(xs, y0 - (m0 - xs) * 2 ** 20 / 64 / 1e6, color=col, lw=0.9, ls=(0, (5, 3)))
    ax2.set_xlabel("Cross-attention cache per step, MiB")
    ax2.set_ylabel("L3 misses per step, millions")
    ax2.set_title("(b) LLC-miss events against cache bytes", fontsize=8, loc="left")
    h = [plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=BLUE, mew=1.2, label="medium, FP32 cache"),
         plt.Line2D([], [], marker="s", ls="none", mfc="white", mec=BLUE, mew=1.2, label="medium, int8 cache"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=ORANGE, mew=1.2, label="small, FP32 cache"),
         plt.Line2D([], [], color=GREY, lw=0.9, ls=(0, (5, 3)), label="one line per 64 B (predicted)")]
    ax2.legend(handles=h, frameon=False, fontsize=6.4, loc="upper left")
    ax2.text(0.98, 0.04, "hollow: 1500 positions\nfilled: calibrated retention", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=6.4, color=DARK)
    ax2.set_ylim(0, None)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=7.5)
    fig.tight_layout(w_pad=1.5)
    out = os.path.join(FIG, "fig_kv_step_time.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    cache_wer()
    calib_grid()
    step_time()
