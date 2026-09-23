"""The selection rule itself: least-aggressive (q, r) that fits the budget.

    (q*, r*) = argmax M_KV(q, r)   s.t.   M_w(l) + M_KV(q, r)/L <= alpha * M_L3
                                          U_dWER(q, r) <= delta

Every measured configuration of both models is a candidate: the precision
sweep (full retention), the eviction arms (FP32), and the composed arms.
Per-layer INT8 weights are read from the with-past graphs (14.00 MiB medium,
7.88 MiB small -- the with-past decoder has no cross-attention K/V
projections, so this is smaller than the no-past graph's 16.0 MiB). The rule
is evaluated at alpha in {0.5, 0.6, 0.7, 0.8} and the maximum quality-gated
point (argmin M_KV under the gate alone) is reported beside it.

Also: WER per audio-length stratum for the main arms, from the per-sample
results already on disk, since the padding share -- and hence what
retention can buy -- depends on utterance length.

Usage:  python experiments/hardware_select.py
"""

import json
import os

import numpy as np

from kvlib import SETUPS, load_audio, real_positions

HERE = os.path.dirname(os.path.abspath(__file__))
M_L3 = 24.0
EPS = 0.20
W_LAYER = {"medium_uz": 14.00, "small_en": 7.88}
LAYERS = {"medium_uz": 24, "small_en": 12}


def J(n):
    return json.load(open(os.path.join(HERE, n)))


def candidates(name):
    """(label, M_KV MiB, dWER [d, lo, hi], per_sample) for every measured arm."""
    C = []
    if name == "medium_uz":
        e1 = J("results_cache_precision_wer_cascade.json")["schemes"]
        for k, lab in [("fp32", "FP32, full"), ("fp16", "FP16, full"), ("int8_head", "int8 head, full"),
                       ("int8_kivi", "int8 KIVI, full"), ("int4_kivi", "int4 KIVI, full")]:
            C.append((lab, e1[k]["cross_cache_mib"], e1[k]["delta_vs_fp32"], e1[k]["per_sample_wer"]))
        e5 = J("results_cache_eviction_wer_cascade_n300.json")["arms"]
        for k, lab in [("mass50/fp32", "FP32, H2O 50%"), ("mass25/fp32", "FP32, H2O 25%"),
                       ("mass25/int4_kivi", "int4 KIVI, H2O 25%")]:
            C.append((lab, e5[k]["cross_cache_mib_mean"], e5[k]["delta_vs_full"], e5[k]["per_sample_wer"]))
        e6 = J("results_eviction_budget_medium_uz.json")["test"]
        sk = [k for k in e6 if k.startswith("split")][0]
        C.append(("FP32, split (1.0, 0.1)", e6[sk]["mib"], e6[sk]["delta_vs_full"], e6[sk]["per_sample_wer"]))
        cb = J("results_eviction_combo_medium_uz.json")["arms"]
        for k, lab in [("split/int8_kivi", "int8 KIVI, split"), ("split/int4_kivi", "int4 KIVI, split")]:
            C.append((lab, cb[k]["mib"], cb[k]["delta_vs_full"], cb[k]["per_sample_wer"]))
        ref = e1["fp32"]["wer"]
    else:
        s7 = J("results_cache_sweep_small_en_n300.json")["arms"]
        for k, lab in [("full/fp32", "FP32, full"), ("full/fp16", "FP16, full"), ("full/int8_head", "int8 head, full"),
                       ("full/int8_kivi", "int8 KIVI, full"), ("full/int4_kivi", "int4 KIVI, full"),
                       ("mass50/fp32", "FP32, H2O 50%"), ("mass25/fp32", "FP32, H2O 25%"),
                       ("mass25/int4_kivi", "int4 KIVI, H2O 25%")]:
            C.append((lab, s7[k]["mib"], s7[k]["delta_vs_full"], s7[k]["per_sample_wer"]))
        e6 = J("results_eviction_budget_small_en.json")["test"]
        sk = [k for k in e6 if k.startswith("split")][0]
        C.append(("FP32, split (0.75, 0.05)", e6[sk]["mib"], e6[sk]["delta_vs_full"], e6[sk]["per_sample_wer"]))
        cb = J("results_eviction_combo_small_en.json")["arms"]
        for k, lab in [("split/int8_kivi", "int8 KIVI, split"), ("split/int4_kivi", "int4 KIVI, split")]:
            C.append((lab, cb[k]["mib"], cb[k]["delta_vs_full"], cb[k]["per_sample_wer"]))
        ref = s7["full/fp32"]["wer"]
    return C, ref


def select(name, C, ref, alpha, m_l3=M_L3):
    L, w = LAYERS[name], W_LAYER[name]
    delta = round(ref * EPS, 4)
    b_kv = alpha * m_l3 - w
    fits = [(mib, lab) for lab, mib, d, _ in C if mib / L <= b_kv and round(d[2], 4) < delta]
    gated = [(mib, lab) for lab, mib, d, _ in C if round(d[2], 4) < delta]
    hw = max(fits) if fits else None
    mx = min(gated)
    return b_kv, hw, mx


def strata(name, C, ref):
    setup = SETUPS[name]
    waves, _ = load_audio(setup.audio["test"], 300)
    secs = np.array([len(w) / 16000 for w in waves])
    bins = [(0, 5), (5, 10), (10, 20), (20, 30.01)]
    out = {}
    full = [c for c in C if c[0] == "FP32, full"][0][3]
    for lo, hi in bins:
        idx = np.where((secs >= lo) & (secs < hi))[0]
        if len(idx) == 0:
            continue
        row = {"n": int(len(idx)), "full": float(np.mean(np.array(full)[idx]))}
        for lab, _, _, ps in C:
            row[lab] = float(np.mean(np.array(ps)[idx]))
        out[f"{lo:.0f}-{min(hi, 30):.0f} s"] = row
    return out


def main():
    global M_L3
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3", type=float, default=24.0, help="L3 size in MiB of the machine to select for")
    args = ap.parse_args()
    M_L3 = args.l3
    res = {"m_l3": M_L3}
    for name in ("medium_uz", "small_en"):
        C, ref = candidates(name)
        L, w = LAYERS[name], W_LAYER[name]
        print(f"\n{name}: per-layer INT8 weights {w:.2f} MiB, {L} layers, full-cache WER {ref:.4f}, "
              f"delta {round(ref * EPS, 4)}")
        print(f"  {'candidate':<26}{'M_KV MiB':>10}{'per layer':>11}{'U_dWER':>9}")
        for lab, mib, d, _ in sorted(C, key=lambda c: -c[1]):
            print(f"  {lab:<26}{mib:>10.1f}{mib / L:>11.2f}{d[2]:>+9.4f}")
        res[name] = {"w_layer": w, "layers": L, "ref": ref, "alpha": {}}
        for a in (0.5, 0.6, 0.7, 0.8):
            b_kv, hw, mx = select(name, C, ref, a)
            res[name]["alpha"][str(a)] = {"budget_kv_layer": b_kv, "hardware_point": hw, "max_gated": mx}
            print(f"  alpha {a}: B_KV/layer {b_kv:5.2f} MiB -> hardware-selected: "
                  f"{hw[1] + f' ({hw[0]:.1f} MiB)' if hw else 'NONE fits (weights alone exceed budget)'}"
                  f";  max gated: {mx[1]} ({mx[0]:.1f} MiB)")
        # the weight paper's L3 sweep, alpha = 0.7: which (q, r) the rule picks per cache size
        res[name]["l3"] = {}
        for m in sorted({8, 12, 16, 24, 32, 48, M_L3}):
            b_kv, hw, mx = select(name, C, ref, 0.7, m)
            res[name]["l3"][str(m)] = {"budget_kv_layer": b_kv, "hardware_point": hw, "max_gated": mx}
            print(f"  L3 {m:4.1f} MiB (alpha 0.7): B_KV/layer {b_kv:6.2f} -> "
                  f"{hw[1] + f' ({hw[0]:.1f} MiB)' if hw else 'NONE fits (weights alone exceed budget)'}")
        st = strata(name, C, ref)
        res[name]["strata"] = st
        keys = ["FP32, full", "int4 KIVI, full", "FP32, H2O 25%",
                [c[0] for c in C if "split (" in c[0]][0], "int4 KIVI, split"]
        print(f"  {'stratum':<10}{'n':>4}" + "".join(f"{k:>24}" for k in keys))
        for s, row in st.items():
            print(f"  {s:<10}{row['n']:>4}" + "".join(f"{row[k]:>24.4f}" for k in keys))
    json.dump(res, open(os.path.join(HERE, "results_hardware_select.json"), "w"), indent=1)
    print("\nsaqlandi: results_hardware_select.json")


if __name__ == "__main__":
    main()
