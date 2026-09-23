"""Decoder step cost before and after moving the activation term.

Three decoders, one machine, one thread:

  no-cache   -- the deployed decoder: recomputes cross-attention K/V from
                1500 encoder positions and the whole prefix every step
  with-past  -- the same weights with a KV cache: reads the cache instead
  untied     -- with-past plus the vocabulary projection given its own int8
                weight, so the tied table is no longer dequantized per step

(a) step latency against decode position, with the linear fit a + b t for
    each arm; the intercept a is the cost that has nothing to do with the
    token being generated.
(b) kernel time by operator role at t = 30, so the two removed terms can be
    seen leaving.

Every number is read from the result files; arms whose files are absent are
skipped rather than invented. Styling matches the paper's figures.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "fig_decoder_step_cost.png")

BLUE, ORANGE, GREEN, GREY, DARK = ("#0072B2", "#D55E00", "#009E73",
                                   "#888888", "#1a1a1a")
ARMS = [
    ("no-cache (deployed)", "results_step_latency.json",
     "results_node_profile.json", ORANGE, "s", "--"),
    ("with KV cache", "results_step_latency_past.json",
     "results_node_profile_past.json", BLUE, "o", "-"),
    ("with KV cache + untied lm_head", "results_step_latency_untied.json",
     "results_node_profile_untied.json", GREEN, "^", "-."),
    ("+ int8 cache, dequantize", "results_step_latency_dq.json",
     "results_node_profile_dq.json", GREY, "v", ":"),
    ("+ int8 cache, integer attention", "results_step_latency_int.json",
     "results_node_profile_int.json", "#CC79A7", "D", "-"),
]
ROLE_ORDER = ["cross-attn K/V proj", "lm_head + dequantize",
              "cache dequantize", "integer attention",
              "attention products", "FFN fc1/fc2", "self-attn proj",
              "cross-attn Q/out proj", "layer norm", "other"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua",
                   "DejaVu Serif"],
    "font.size": 9, "font.weight": "normal",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "custom", "mathtext.rm": "Palatino Linotype",
    "mathtext.it": "Palatino Linotype:italic",
    "axes.linewidth": 0.8, "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.69, 3.7), gridspec_kw={"width_ratios": [1.15, 1.0],
                                                "wspace": 0.38})
    arms = []
    for label, lat_f, prof_f, col, mk, ls in ARMS:
        lat, prof = load(lat_f), load(prof_f)
        if lat is None:
            continue
        arms.append((label, lat, prof, col))
        t = np.array([r["t"] for r in lat["rows"]], float)
        ms = np.array([r["ms"] for r in lat["rows"]])
        ax1.plot(t, ms, ls, color=col, lw=1.0, marker=mk, markersize=4.2,
                 markerfacecolor="white", markeredgewidth=1.1,
                 label=f"{label}:  {lat['fit_a_ms']:.0f} + "
                       f"{lat['fit_b_ms_per_token']:.2f}·t ms")

    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, None)
    ax1.set_xlabel("Decode position $t$", fontsize=9)
    ax1.set_ylabel("Decoder step, ms (1 thread)", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=7, loc="upper left", handlelength=2.4)
    ax1.set_title("(a) Step latency against position", fontsize=9,
                  loc="left", pad=5)

    # (b) stacked kernel time by role, one bar per arm
    x = np.arange(len(arms))
    bottoms = np.zeros(len(arms))
    shades = plt.cm.Greys(np.linspace(0.85, 0.25, len(ROLE_ORDER)))
    for i, role in enumerate(ROLE_ORDER):
        vals = np.array([(a[2] or {}).get("ms_by_role", {}).get(role, 0.0)
                         for a in arms])
        col = ORANGE if role == "cross-attn K/V proj" else \
            BLUE if role == "lm_head + dequantize" else shades[i]
        ax2.bar(x, vals, 0.6, bottom=bottoms, color=col, edgecolor="white",
                linewidth=0.5, label=role)
        bottoms += vals
    for xi, a in zip(x, arms):
        if a[2]:
            ax2.text(xi, bottoms[xi] + 12, f"{bottoms[xi]:.0f} ms",
                     ha="center", fontsize=7.5, color=DARK)
    ax2.set_xticks(x)
    short = {"no-cache (deployed)": "no-cache\n(deployed)",
             "with KV cache": "KV cache\nFP32",
             "with KV cache + untied lm_head": "+ untied\nlm_head",
             "+ int8 cache, dequantize": "+ int8 cache\ndequantize",
             "+ int8 cache, integer attention": "+ int8 cache\ninteger attn"}
    ax2.set_xticklabels([short.get(a[0], a[0]).replace("\n", " ") for a in arms],
                        fontsize=6.8, rotation=28, ha="right",
                        rotation_mode="anchor")
    ax2.set_ylabel("Kernel time at $t$ = 30, ms", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=6.4, loc="upper right",
               ncol=1, handlelength=1.2, labelspacing=0.25)
    ax2.set_title("(b) Where the step goes", fontsize=9, loc="left", pad=5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight")
    print(f"  {OUT}")
    print("\nARMS")
    for label, lat, prof, _ in arms:
        s30 = next(r["ms"] for r in lat["rows"] if r["t"] == 30)
        print(f"  {label:<32} floor {lat['fit_a_ms']:6.1f} ms   "
              f"slope {lat['fit_b_ms_per_token']:5.2f} ms/tok   "
              f"t=30: {s30:6.1f} ms")


if __name__ == "__main__":
    main()
