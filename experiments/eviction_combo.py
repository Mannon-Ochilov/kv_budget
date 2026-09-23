"""E6b -- the calibrated split rule composed with a low-precision cache.

E6 chose (f_r, f_p) per model on the validation split. E1/E7 showed the
cache can be held at int4 (KIVI grouping) without measurable cost. This
measures the two together on the test split: the calibrated split rule with
the kept entries at int8 and at int4, paired against the full FP32 cache
from the E6 test run so the whole ladder shares one reference.

Usage:  python experiments/eviction_combo.py --setup medium_uz [--n 300]
"""

import argparse
import json
import os
import time

import numpy as np

from cache_precision_wer import SCHEMES
from eviction_budget import EPS, keep_split
from kvlib import (SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
BITS = {"int8_kivi": 8, "int4_kivi": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    assert full["n"] == args.n
    out_json = os.path.join(HERE, f"results_eviction_combo_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    res["rule"] = [f_r, f_p]

    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)

    for scheme in ("int8_kivi", "int4_kivi"):
        name = f"split/{scheme}"
        if name in res["arms"]:
            continue
        wers, kept = [], []
        t0 = time.time()
        for i in range(args.n):
            ids, k = greedy(setup, first, step, states[i], prompt,
                            keep_split(f_r, f_p, real_positions(waves[i])),
                            SCHEMES[scheme])
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            kept.append(k)
        res["arms"][name] = {"n": args.n, "scheme": scheme, "bits": BITS[scheme],
                             "wer": float(np.mean(wers)), "per_sample_wer": wers,
                             "kept": float(np.mean(kept)),
                             "mib": cache_mib(setup, float(np.mean(kept)), BITS[scheme])}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {name:<16} WER {np.mean(wers):.4f}  kept {np.mean(kept):5.0f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)

    rng = np.random.default_rng(SEED)
    delta = round(full["wer"] * EPS, 4)
    print(f"\n{setup.name}: split f_r={f_r} f_p={f_p}; full FP32 WER {full['wer']:.4f}, "
          f"{full['mib']:.1f} MiB, delta {delta}")
    print(f"{'arm':<18}{'kept':>6}{'MiB':>8}{'WER':>9}{'dWER vs full [95% CI]':>30}   gate")
    for name, r in res["arms"].items():
        d, lo, hi = paired_ci(r["per_sample_wer"], full["per_sample_wer"], rng)
        r["delta_vs_full"] = [d, lo, hi]
        verdict = ("Accepted" if round(hi, 4) < delta else
                   "Rejected" if round(lo, 4) > delta else "Inconclusive")
        print(f"{name:<18}{r['kept']:>6.0f}{r['mib']:>8.1f}{r['wer']:>9.4f}"
              f"   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   {verdict}")
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
