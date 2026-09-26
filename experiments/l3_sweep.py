"""L1 -- what the selection rule picks as the L3 size changes (analytic).

The WER of a configuration does not depend on the L3; which configurations
the rule (2) admits does. For M_L3 in L3_SIZES and alpha = 0.7, per model:

  per-layer working set  M_w + 2 * k * d * 4 B  <= alpha * M_L3     (FP32 cache)

k is the mean number of cross-attention positions a decoder step reads per
layer: the kept set of a one-shot rule, the tracked set of PadSink-Track
(whose full cache stays in DRAM -- the budget is on the per-step working
set, not on the stored cache). Candidates are only MEASURED arms (300 test
utterances): full cache, the one-shot rules at K_i / 0.5 K_i / 0.25 K_i,
PadSink-Track (v2 and with the padding summary slot) at 0.5 / 0.25 K_i.
The oracle is listed apart (not realizable).

For each (model, M_L3) the table gives k_max and, per family, the
least-aggressive admitted candidate that passes the gate (U < delta).
INT8 per-head caches count a quarter of the bytes per position: one-shot
int8 arms (medium_uz: split + int8 head, the K = 1433 boundary point) and
PadSink-Track with the int8 cache (track_ablation.py) are candidates too.

Usage:  python experiments/l3_sweep.py
"""

import json
import os

from kvlib import SETUPS

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, EPS = 0.7, 0.20
L3_SIZES = (12, 16, 20, 24, 32, 48)
W_LAYER = {"medium_uz": 14.00, "medium_en": 14.00, "small_en": 7.88, "small_uz": 7.88}


def J(n):
    p = os.path.join(HERE, n)
    return json.load(open(p)) if os.path.exists(p) else {}


def candidates(m):
    """(family, label, k, [d, lo, hi], bits)"""
    C = []
    e6 = J(f"results_eviction_budget_{m}.json")["test"]
    C.append(("full", "full cache", 1500, [0, 0, 0]))
    sk = [k for k in e6 if k.startswith("split")][0]
    C.append(("one-shot", "split K_i", e6[sk]["kept"], e6[sk]["delta_vs_full"]))
    for k, v in J(f"results_sota_baselines_{m}.json").get("arms", {}).items():
        C.append(("one-shot", f"{k} K_i", v["kept"], v["delta_vs_full"]))
    for k, v in J(f"results_spar_{m}.json").get("test", {}).items():
        if k.startswith("spar/"):
            C.append(("one-shot", f"padsink K_i ({k.split('/')[1]})", v["kept"], v["delta_vs_full"]))
    for k, v in J(f"results_diag_budget_{m}.json").get("arms", {}).items():
        fam = "oracle" if k.startswith("oracle") else "one-shot"
        C.append((fam, k.replace("/s", " "), v["kept"], v["delta_vs_full"]))
    for k, v in J(f"results_align_track_{m}.json").get("arms", {}).items():
        C.append(("track", k.replace("/s", " "), v["kept"], v["delta_vs_full"]))
    k_i = e6[sk]["kept"]
    for k, v in J(f"results_track_ablation_{m}.json").get("arms", {}).items():
        if k.startswith("track"):
            sc = float(k.split("/s")[1])
            C.append(("track", k.replace("/s", " "), sc * k_i, v["delta_vs_full"], 8 if "int8" in k else 32))
    cb = J(f"results_eviction_combo_{m}.json").get("arms", {}).get("split/int8_head")
    if cb:
        C.append(("one-shot", "split int8 K_i", cb["kept"], cb["delta_vs_full"], 8))
    bd = J(f"results_boundary_int8_{m}.json")
    if bd:
        C.append(("one-shot", "int8 head K=1433", bd["K"], bd["delta_vs_full"], 8))
    return [c if len(c) == 5 else c + (32,) for c in C]


def main():
    out = {}
    for m, setup in SETUPS.items():
        full_wer = J(f"results_eviction_budget_{m}.json")["test"]["full"]["wer"]
        delta = round(full_wer * EPS, 4)
        C = candidates(m)
        slot = 2 * setup.d_model * 4 / 1024 ** 2          # MiB per position per layer, FP32
        print(f"\n== {m}  (M_w {W_LAYER[m]:.2f} MiB/layer, {slot * 1024:.1f} KiB/position/layer, delta {delta})")
        print(f"{'M_L3':>5} {'k_max':>6}   {'one-shot (best admitted & passing)':<38} {'PadSink-Track':<30} oracle")
        rows = []
        for l3 in L3_SIZES:
            head = ALPHA * l3 - W_LAYER[m]
            k_max = int(head / slot) if head > 0 else 0
            pick = {}
            for fam in ("full", "one-shot", "track", "oracle"):
                # admitted if its bytes fit: k * bits/32 FP32-equivalent positions
                ok = [c for c in C if c[0] == fam and c[2] * c[4] / 32 <= k_max and round(c[3][2], 4) < delta]
                pick[fam] = max(ok, key=lambda c: c[2] * c[4] / 32) if ok else None
            best_os = pick["full"] or pick["one-shot"]
            fmt = lambda c: f"{c[1]} (k={c[2]:.0f}, {c[3][0]:+.4f})" if c else "-- none --"
            print(f"{l3:>5} {k_max:>6}   {fmt(best_os):<38} {fmt(pick['track']):<30} {fmt(pick['oracle'])}")
            rows.append({"l3": l3, "k_max": k_max,
                         **{f: (list(pick[f][1:3]) + [pick[f][3][0]] if pick[f] else None) for f in pick}})
        # the smallest L3 at which each family has an admitted, passing configuration
        need = {}
        for fam in ("one-shot", "track", "oracle"):
            ok = [c for c in C if c[0] in (fam, "full" if fam == "one-shot" else fam) and round(c[3][2], 4) < delta]
            if ok:
                c = min(ok, key=lambda c: c[2] * c[4] / 32)
                need[fam] = [(W_LAYER[m] + c[2] * c[4] / 32 * slot) / ALPHA, c[1], c[2], c[4]]
                print(f"  minimal L3 {fam:<9} {need[fam][0]:5.1f} MiB  via {c[1]} (k={c[2]:.0f}, {c[4]}-bit, {c[3][0]:+.4f})")
        out[m] = {"delta": delta, "rows": rows, "min_l3": need}
    # long audio (results_long_fixed_k_*): the one-shot set grows with the speech, the tracked set does
    # not; minimal L3 for the passing configurations, FP32 (int8 was not measured on long audio)
    print("\n== long audio, FP32 (minimal L3 of the passing configuration per family)")
    out["long"] = {}
    for name, m in (("small_en_long", "small_en"), ("medium_uz_long", "medium_uz")):
        r = J(f"results_long_fixed_k_{name}.json")
        slot = 2 * SETUPS[m].d_model * 4 / 1024 ** 2
        fam = {"one-shot": [], "track": []}
        for arm, v in r["arms"].items():
            if arm == "full" or round(v["delta_vs_full"][2], 4) >= r["delta"]:
                continue
            k = r["split_Ki_mean"] if arm == "split/Ki" else float(arm.split("/k")[1])
            fam["track" if arm.startswith("track") else "one-shot"].append((k, arm))
        row = {}
        for f, c in fam.items():
            if c:
                k, arm = min(c)
                row[f] = [(W_LAYER[m] + k * slot) / ALPHA, arm, k]
                print(f"  {name:<15} {f:<9} {row[f][0]:5.1f} MiB  via {arm} (k={k:.0f})")
        out["long"][name] = row
    json.dump(out, open(os.path.join(HERE, "results_l3_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
