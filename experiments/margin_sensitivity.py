"""R2 -- how much do the verdicts depend on the non-inferiority margin?

delta = eps * WER_full with eps = 0.2 throughout the paper. Every verdict of
the main tables is recomputed from the saved per-sample WERs (same paired
bootstrap, seed and CI) at eps in {0.10, 0.15, 0.20, 0.30}:
  Table 4 (sink causal), Table 5 (K_i), Table 6 (0.5 K_i), Table 7 (long audio).
Printed: each arm's verdict per eps, and the count of verdicts that differ
from eps = 0.20.

Usage:  python experiments/margin_sensitivity.py
"""

import json
import os

import numpy as np

from kvlib import SEED, paired_ci

HERE = os.path.dirname(os.path.abspath(__file__))
EPSS = (0.10, 0.15, 0.20, 0.30)
MODELS = ("medium_uz", "small_uz", "medium_en", "small_en")
RHO = {"medium_uz": "0.9", "small_uz": "0.95", "medium_en": "0.8", "small_en": "0.9"}


def J(n):
    return json.load(open(os.path.join(HERE, n)))


def verdict(ci, delta):
    return "Q" if round(ci[2], 4) < delta else "R" if round(ci[1], 4) > delta else "N"


def arms_of(m):
    full = J(f"results_eviction_budget_{m}.json")["test"]["full"]["per_sample_wer"]
    e6 = J(f"results_eviction_budget_{m}.json")["test"]
    sk = [k for k in e6 if k.startswith("split")][0]
    sb = J(f"results_sota_baselines_{m}.json")["arms"]
    sp = J(f"results_spar_{m}.json")["test"]
    dg = J(f"results_diag_budget_{m}.json")["arms"]
    tr = J(f"results_align_track_{m}.json")["arms"]
    ab = J(f"results_track_ablation_{m}.json")["arms"]
    sc = J(f"results_sink_causal_{m}.json")
    sc = sc.get("arms", sc)
    A = {
        "T4 sink->mean": sc["sink_mean"], "T4 audio->mean": sc["audio_mean"], "T4 rest pad->mean": sc["pad_rest_mean"],
        "T5 split": e6[sk], "T5 H2O layer": sb["h2o_layer"], "T5 SnapKV": sb["snapkv"], "T5 PyramidKV": sb["pyramidkv"],
        "T5 PadSink-KV": sp[f"spar/rho{RHO[m]}"], "T5 Track": ab["track/s1.0"],
        "T6 split": dg["split/s0.5"], "T6 H2O layer": dg["h2o_layer/s0.5"], "T6 PyramidKV": dg["pyramidkv/s0.5"],
        "T6 PadSink-KV": dg["padsink/s0.5"], "T6 fixed rate": ab["fixed_rate/s0.5"], "T6 Track": tr["track/s0.5"],
        "T6 Track int8": ab["track_int8/s0.5"], "T6 oracle": dg["oracle/s0.5"],
    }
    return full, {k: v["per_sample_wer"] for k, v in A.items()}


def long_arms(name):
    r = J(f"results_long_fixed_k_{name}.json")
    full = (r["arms"]["full"]["per_sample_wer"] if "full" in r["arms"]
            else J("results_long_audio_medium_uz.json")["arms"]["full/fp32"]["per_sample_wer"])
    return full, {f"T7 {k}": v["per_sample_wer"] for k, v in r["arms"].items() if k != "full"}


def main():
    blocks = [(m, *arms_of(m)) for m in MODELS] + [(n, *long_arms(n)) for n in ("small_en_long", "medium_uz_long")]
    out, changed, total = {}, 0, 0
    for name, full, arms in blocks:
        wf = float(np.mean(full))
        print(f"\n== {name}  (full WER {wf:.4f}; delta at eps 0.10/0.15/0.20/0.30 = "
              + "/".join(f"{wf * e:.4f}" for e in EPSS) + ")")
        out[name] = {}
        for arm, ps in arms.items():
            ci = paired_ci(ps, full, np.random.default_rng(SEED))
            v = [verdict(ci, round(wf * e, 4)) for e in EPSS]
            out[name][arm] = {"ci": ci, "verdicts": dict(zip(map(str, EPSS), v))}
            ref = v[EPSS.index(0.20)]
            changed += sum(x != ref for x in v)
            total += len(v) - 1
            flag = "" if len(set(v)) == 1 else "   <- depends on eps"
            print(f"  {arm:<22} {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]   " + "  ".join(v) + flag)
    print(f"\nverdicts differing from eps = 0.20: {changed} of {total}")
    json.dump(out, open(os.path.join(HERE, "results_margin_sensitivity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
