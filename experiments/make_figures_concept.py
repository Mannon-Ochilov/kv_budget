"""Four conceptual figures for the cache paper (MDPI style, 600 dpi).

  fig_kv_c1_budget.png     why a weights-only budget is not enough: the
                           layer-local hot set with and without the cache
  fig_kv_c2_method.png     the method as a flow: split ranking -> retention
                           r -> precision q -> layer-local budget test ->
                           WER gate -> select / stop
  fig_kv_c3_qr_space.png   the (r, q) plane: every measured candidate, the
                           budget iso-lines for alpha, gate pass/fail, and
                           the selected point per model
  fig_kv_c4_padding.png    the 30 s timeline: audio vs padding, attention
                           mass on each, and the three retention outcomes

C1, C3 and C4 carry measured numbers read from the result files; C2 is a
schematic. Palatino, bold only for titles, Okabe-Ito colours.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
BLUE, ORANGE, GREEN, PINK, DARK, GREY, SKY = ("#0072B2", "#D55E00", "#009E73",
                                              "#CC79A7", "#1a1a1a", "#888888", "#56B4E9")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
})
M_L3, ALPHA = 24.0, 0.7


def J(n):
    return json.load(open(os.path.join(HERE, n)))


# ------------------------------------------------------------------ C1
def c1_budget():
    hw = J("results_hardware_select.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 2.3), gridspec_kw={"width_ratios": [1, 1]})
    for ax, (name, title, kv_layer) in zip(axes, [
            ("medium_uz", "(a) Whisper-medium: 24 layers, FP32 cache 281.2 MiB", 281.2 / 24),
            ("small_en", "(b) whisper-small: 12 layers, FP32 cache 105.5 MiB", 105.5 / 12)]):
        w = hw[name]["w_layer"]
        B = ALPHA * M_L3
        ax.barh([1], [w], color=DARK, height=0.55)
        ax.barh([0], [w], color=DARK, height=0.55)
        ax.barh([0], [kv_layer], left=[w], color=ORANGE, height=0.55)
        ax.axvline(B, color=GREEN, lw=1.4, ls=(0, (5, 3)))
        ax.text(B, 1.62, f"budget α·M_L3 = {B:.1f} MiB", color=GREEN, fontsize=7.5,
                ha="center", va="bottom")
        ax.text(w / 2, 1, f"weights {w:.1f}", color="white", ha="center", va="center", fontsize=7.5)
        ax.text(w / 2, 0, f"weights {w:.1f}", color="white", ha="center", va="center", fontsize=7.5)
        ax.text(w + kv_layer / 2, 0, f"cache {kv_layer:.1f}", color="white", ha="center",
                va="center", fontsize=7.5)
        tot = w + kv_layer
        ax.text(max(tot, w) + 0.4, 1, "fits" if w <= B else "does not fit",
                va="center", fontsize=7.5, color=GREEN if w <= B else ORANGE)
        ax.text(tot + 0.4, 0, f"{'fits' if tot <= B else 'does not fit'} ({tot:.1f})",
                va="center", fontsize=7.5, color=GREEN if tot <= B else ORANGE)
        ax.set_yticks([1, 0], ["weights only\n(rule of the weight paper)",
                               "weights + own\ncache slice"])
        ax.set_xlim(0, max(tot, B) * 1.35)
        ax.set_ylim(-0.6, 2.0)
        ax.set_xlabel("per-layer hot working set, MiB")
        ax.set_title(title, fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout(w_pad=2)
    out = os.path.join(FIG, "fig_kv_c1_budget.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


# ------------------------------------------------------------------ C2
def box(ax, xy, w, h, text, fc="white", ec=DARK, fs=7.5, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=DARK)


def arrow(ax, p, q, text=None, color=DARK, side=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=color,
                                 shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx + side, my, text, fontsize=6.8, color=color, ha="left" if side > 0 else "right",
                va="center")


def c2_method():
    fig, ax = plt.subplots(figsize=(6.69, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    box(ax, (5, 9.35), 5.6, 0.85, "Whisper encoder: 1500 encoder states per utterance\n(one pass; the decoder re-reads their K, V every step)", fc="#f2f2f2")
    # two candidate families
    box(ax, (2.45, 7.6), 4.7, 1.55, "retention candidates R\naudio states: rank by attention mass, keep f_r\ncontextualized padding states: rank, keep f_p\n(calibrated on validation under the WER gate)", fc="#e6f0fa", ec=BLUE, fs=7)
    box(ax, (7.55, 7.6), 4.7, 1.55, "precision candidates Q\nFP32, FP16, int8, int4\n(KIVI grouping: K per channel, V per token)", fc="#fbeee6", ec=ORANGE, fs=7)
    arrow(ax, (3.6, 8.93), (2.7, 8.4))
    arrow(ax, (6.4, 8.93), (7.3, 8.4))
    box(ax, (5, 5.95), 4.6, 0.7, "candidate set C = Q × R\nevery (q, r) is decoded and measured", fc="white", ec=DARK, bold=True)
    arrow(ax, (2.8, 6.82), (4.2, 6.32))
    arrow(ax, (7.2, 6.82), (5.8, 6.32))
    box(ax, (4.6, 4.6), 5.4, 0.7, "hardware filter:  M_w(ℓ) + M_KV(q, r) / L  ≤  α·M_L3", fc="white", ec=DARK, fs=7)
    arrow(ax, (5, 5.6), (4.6, 4.97))
    box(ax, (4.6, 3.25), 5.4, 0.7, "quality filter:  U_ΔWER(q, r)  ≤  δ = ε·WER_full  (paired bootstrap)", fc="white", ec=DARK, fs=7)
    arrow(ax, (4.6, 4.25), (4.6, 3.62))
    box(ax, (5, 1.9), 5.4, 0.7, "(q*, r*) = argmax M_KV over the survivors\nthe least-compressed configuration that fits — then STOP", fc="#e8f5ef", ec=GREEN, bold=True)
    arrow(ax, (4.6, 2.9), (5, 2.27))
    ax.text(5, 0.95, "no ordering of q before r: precision and retention are selected jointly;\n"
                     "if nothing survives the hardware filter, the weight budget must be lowered first (Section 3)",
            ha="center", va="center", fontsize=6.8, color=DARK)
    ax.text(7.45, 4.6, "drops (q, r) that\ndo not fit", fontsize=6.6, color=ORANGE, va="center")
    ax.text(7.45, 3.25, "drops (q, r) that\nfail the gate", fontsize=6.6, color=ORANGE, va="center")
    out = os.path.join(FIG, "fig_kv_c2_method.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


# ------------------------------------------------------------------ C3
def c3_qr_space():
    import hardware_select as hs
    hw = J("results_hardware_select.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.2))
    for ax, (name, title, d, L) in zip(axes, [("medium_uz", "(a) Whisper-medium", 1024, 24),
                                              ("small_en", "(b) whisper-small", 768, 12)]):
        C, ref = hs.candidates(name)
        delta = round(ref * 0.2, 4)
        w = hw[name]["w_layer"]
        full_mib = L * 2 * 1500 * d * 32 / 8 / 2 ** 20
        rs = np.linspace(0.05, 1.0, 200)
        for a, col, ls in [(0.6, GREY, ":"), (0.7, GREEN, (0, (5, 3))), (0.8, GREY, "-.")]:
            b_kv = a * M_L3 - w
            if b_kv <= 0:
                continue
            # r * q/32 * full_mib / L <= b_kv  ->  q = 32 * b_kv * L / (r * full_mib)
            q = 32 * b_kv * L / (rs * full_mib)
            ax.plot(rs, q, color=col, lw=1.1, ls=ls)
            inside = np.where((q > 3.3) & (q < 44))[0]
            if len(inside):
                j = inside[len(inside) // 3]
                ax.annotate(f"α = {a}", (rs[j], q[j]), xytext=(4, 4), textcoords="offset points",
                            color=col, fontsize=6.8, ha="left", va="bottom",
                            bbox=dict(fc="white", ec="none", pad=0.5))
        sel = hw[name]["alpha"]["0.7"]["hardware_point"]
        for lab, mib, dw, _ in C:
            bits = 32 if "FP32" in lab else 16 if "FP16" in lab else 8 if "int8" in lab else 4
            r = mib / (full_mib * bits / 32)
            ok = round(dw[2], 4) < delta
            is_sel = sel and lab == sel[1]
            execu = hs.executable(name, lab)
            # circle: executable in ONNX Runtime; diamond: NumPy simulation (capacity headroom)
            ax.plot([r], [bits], marker="o" if execu else "D", ms=9 if is_sel else (6 if execu else 5),
                    mfc=(GREEN if is_sel else (BLUE if ok else "white")),
                    mec=GREEN if is_sel else (BLUE if ok else ORANGE), mew=1.3, ls="none")
            if is_sel:
                ax.annotate(f"selected (α = 0.7):\n{lab}, {mib:.0f} MiB", (r, bits), xytext=((-16, -30) if name == "medium_uz" else (-14, -28)),
                            textcoords="offset points", ha="right", va="top", fontsize=6.8, color=GREEN)
        ax.set_yscale("log", base=2)
        ax.set_yticks([4, 8, 16, 32], ["int4", "int8", "FP16", "FP32"])
        ax.set_ylim(3, 48)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("retained fraction of the cache, r")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("cache precision, q")
    h = [plt.Line2D([], [], marker="o", ls="none", mfc=BLUE, mec=BLUE, ms=6, label="executable in ORT, passes the gate"),
         plt.Line2D([], [], marker="D", ls="none", mfc=BLUE, mec=BLUE, ms=5, label="simulated (int4 / KIVI), passes the gate"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=ORANGE, mew=1.2, ms=6, label="fails the gate"),
         plt.Line2D([], [], marker="o", ls="none", mfc=GREEN, mec=GREEN, ms=8, label="selected at α = 0.7 (executable candidates)"),
         plt.Line2D([], [], color=GREEN, lw=1.1, ls=(0, (5, 3)), label="budget iso-line: below it fits")]
    fig.legend(handles=h, frameon=False, fontsize=6.4, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    axes[0].text(0.03, 0.60, "α = 0.5: B_KV < 0 —\nweights alone exceed the budget", transform=axes[0].transAxes,
                 fontsize=6.6, color=GREY, va="bottom")
    fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
    out = os.path.join(FIG, "fig_kv_c3_qr_space.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


# ------------------------------------------------------------------ C4
def c4_padding():
    ma = J("results_mass_analysis_medium_uz.json")["mean"]
    e5 = J("results_cache_eviction_wer_cascade_n300.json")["arms"]
    e6 = J("results_eviction_budget_medium_uz.json")["test"]
    sk = [k for k in e6 if k.startswith("split")][0]
    real, pad_share = ma["real"], ma["pad_share"]
    fig, ax = plt.subplots(figsize=(6.69, 2.9))
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    # timeline
    ax.add_patch(FancyBboxPatch((0, 5.2), real, 0.6, boxstyle="square,pad=0", fc=BLUE, ec="none"))
    ax.add_patch(FancyBboxPatch((real, 5.2), 1500 - real, 0.6, boxstyle="square,pad=0", fc="#f3c9b3", ec="none"))
    ax.text(real / 2, 5.5, f"{real / 1500:.0%}", ha="center", va="center", color="white", fontsize=7)
    ax.text(real + (1500 - real) / 2, 5.5, f"padding of the 30 s window ({1 - real / 1500:.0%})",
            ha="center", va="center", color=DARK, fontsize=7)
    ax.text(0, 5.9, f"encoder timeline, Whisper-medium test set (mean): audio {real:.0f} positions, padding {1500 - real:.0f}", fontsize=7.5, va="bottom", color=DARK)
    # mass bar
    m_audio = (1 - pad_share) * 1500
    ax.add_patch(FancyBboxPatch((0, 3.9), m_audio, 0.6, boxstyle="square,pad=0", fc=BLUE, ec="none"))
    ax.add_patch(FancyBboxPatch((m_audio, 3.9), 1500 - m_audio, 0.6, boxstyle="square,pad=0", fc=ORANGE, ec="none"))
    ax.text(m_audio / 2, 4.2, f"{1 - pad_share:.1%}", ha="center", va="center", color="white", fontsize=7.5)
    ax.text(m_audio + (1500 - m_audio) / 2, 4.2, f"{pad_share:.1%} of the first-step cross-attention mass",
            ha="center", va="center", color="white", fontsize=7.5)
    ax.text(0, 4.65, "where the decoder looks", fontsize=7.5, va="bottom", color=DARK)
    # outcomes
    rows = [("FULL cache (1500)", e6["full"]["wer"], f"{e6['full']['mib']:.0f} MiB", GREEN),
            ("TRIM: audio only (270)", e5["trim/fp32"]["wer"], f"{e5['trim/fp32']['cross_cache_mib_mean']:.0f} MiB", ORANGE),
            ("SPLIT: all audio + top 10% of padding positions by attention mass (393)", e6[sk]["wer"], f"{e6[sk]['mib']:.0f} MiB", GREEN)]
    for i, (lab, wer, mib, col) in enumerate(rows):
        y = 2.9 - i * 0.95
        ax.text(0, y, lab, fontsize=7.5, va="center", color=DARK)
        ax.text(760, y, f"WER {wer:.3f}", fontsize=7.5, va="center", color=col, fontweight="bold")
        ax.text(1000, y, mib, fontsize=7.5, va="center", color=DARK)
        ax.text(1180, y, "catastrophic — runaway repetition" if wer > 1 else
                ("reference" if i == 0 else f"ΔWER {e6[sk]['delta_vs_full'][0]:+.4f}, Accepted; 3.8× smaller"), fontsize=7, va="center", color=col)
    ax.text(0, 3.45, "three retention outcomes (300 test utterances)", fontsize=7.5, va="bottom", color=DARK)
    out = os.path.join(FIG, "fig_kv_c4_padding.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(" ", out)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    c1_budget()
    c2_method()
    c3_qr_space()
    c4_padding()
