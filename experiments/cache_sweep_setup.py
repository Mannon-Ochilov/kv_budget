"""E7 -- the E1 precision sweep and the E5 eviction check on another model.

The compact replication: does the second model show the same three things
as Whisper-medium/uz -- precision headroom down to int4 (E1), catastrophe
when the padding positions are dropped (E5 trim), recovery under global
attention-mass ranking (E5 H2O)? Same decoding loop, same paired bootstrap,
same three-way gate. Runs for either setup; the medium_uz rows reproduce E1
and E5 from the shared code path (a check on kvlib itself).

Usage:  python experiments/cache_sweep_setup.py --setup small_en [--n 300]
"""

import argparse
import json
import os
import time

import numpy as np

from cache_precision_wer import SCHEMES  # numpy quantizers, unchanged
from kvlib import (ENC_POS, SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.20
BITS = {"fp32": 32, "fp16": 16, "int8_head": 8, "int8_kivi": 8, "int4_kivi": 4}
N_SINK = 32

# (name, keep rule, precision scheme)
ARMS = [
    ("full/fp32", "full", "fp32"),
    ("full/fp16", "full", "fp16"),
    ("full/int8_head", "full", "int8_head"),
    ("full/int8_kivi", "full", "int8_kivi"),
    ("full/int4_kivi", "full", "int4_kivi"),
    ("trim/fp32", "trim", "fp32"),
    ("sink32/fp32", "sink", "fp32"),
    ("mass50/fp32", "mass50", "fp32"),
    ("mass25/fp32", "mass25", "fp32"),
    ("mass25/int4_kivi", "mass25", "int4_kivi"),
]


def keeper(rule, real_n):
    def fn(mass):
        if rule == "full":
            return None
        if rule == "trim":
            return np.arange(real_n)
        if rule == "sink":
            pad = real_n + np.argsort(-mass[real_n:])[:N_SINK]
            return np.sort(np.concatenate([np.arange(real_n), pad]))
        k = int(round({"mass50": 0.5, "mass25": 0.25}[rule] * ENC_POS))
        return np.sort(np.argsort(-mass)[:k])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="small_en", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    out_json = os.path.join(HERE, f"results_cache_sweep_{setup.name}_n{args.n}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}

    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)
    print(f"{setup.name}: {args.n} utterances, real positions mean "
          f"{np.mean([real_positions(w) for w in waves]):.0f} of {ENC_POS}")

    for name, rule, scheme in ARMS:
        if name in res["arms"]:
            continue
        wers, kept = [], []
        t0 = time.time()
        for i in range(args.n):
            ids, k = greedy(setup, first, step, states[i], prompt,
                            keeper(rule, real_positions(waves[i])), SCHEMES[scheme])
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            kept.append(k)
        res["arms"][name] = {
            "n": args.n, "rule": rule, "scheme": scheme, "bits": BITS[scheme],
            "wer": float(np.mean(wers)), "per_sample_wer": wers,
            "kept": float(np.mean(kept)),
            "mib": cache_mib(setup, float(np.mean(kept)), BITS[scheme]),
        }
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {name:<18} WER {np.mean(wers):.4f}  kept {np.mean(kept):5.0f}"
              f"  {res['arms'][name]['mib']:6.1f} MiB  [{time.time() - t0:.0f}s]",
              flush=True)

    base = res["arms"]["full/fp32"]["per_sample_wer"]
    ref = res["arms"]["full/fp32"]["wer"]
    delta = round(ref * EPS, 4)
    rng = np.random.default_rng(SEED)
    print(f"\n{'arm':<20}{'kept':>6}{'MiB':>8}{'WER':>9}"
          f"{'dWER vs full/fp32 [95% CI]':>32}   gate (delta {delta})")
    for name, r in res["arms"].items():
        d, lo, hi = paired_ci(r["per_sample_wer"], base, rng)
        r["delta_vs_full"] = [d, lo, hi]
        verdict = ("Accepted" if round(hi, 4) < delta else
                   "Rejected" if round(lo, 4) > delta else "Inconclusive")
        print(f"{name:<20}{r['kept']:>6.0f}{r['mib']:>8.1f}{r['wer']:>9.4f}"
              f"   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   {verdict}")
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
