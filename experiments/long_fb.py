"""Long audio with the fallback (framework.py): PadSink-Track at a fixed k and
a step redone on the full cache when the top-1 probability is below tau.

Same two sets and the same fixed k = 200 / 400 as long_fixed_k.py (Table 7).
Arms: tau = 0.9 and, if different, the tau framework.py chose for the model
on short audio. Reference: the full cache on the same utterances.

Prediction (before the run): at k = 400 both sets stay accepted; at k = 200
the fallback brings medium_uz_long (plain Track: +0.099, rejected) below
+0.03 but the verdict there is not predicted.

Usage:  python experiments/long_fb.py --set medium_uz_long|small_en_long
"""

import argparse
import json
import os
import time

import numpy as np

import long_fixed_k as LK          # sets MAX_NEW = 224 in kvlib and align_track
import adaptive_track as T
import align_track as A
from eviction_budget import EPS
from kvlib import SEED, error_rate, paired_ci, prompt_ids, real_positions, session, text_norm, with_attention

T.MAX_NEW = 224
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["medium_uz_long", "small_en_long"])
    args = ap.parse_args()
    setup, states, waves, texts, idx, full_ps = LK.data(args.set)
    if full_ps is None:
        full_ps = json.load(open(os.path.join(HERE, f"results_long_fixed_k_{args.set}.json")))["arms"]["full"]["per_sample_wer"]
    fw = os.path.join(HERE, f"results_framework_{setup.name}.json")
    chosen = json.load(open(fw)).get("choice") if os.path.exists(fw) else None
    taus = ["0.9"] + ([chosen] if chosen not in (None, "none", "0.9") else [])
    heads, norm = A.align_heads(setup), text_norm(setup)
    tok, prompt = prompt_ids(setup)
    first, sa = session(with_attention(setup.first)), session(with_attention(setup.step))
    out_json = os.path.join(HERE, f"results_long_fb_{args.set}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    delta = round(float(np.mean(full_ps)) * EPS, 4)
    res.update({"delta": delta, "chosen_tau_short_audio": chosen})
    for k in LK.KS:
        for tau in taus:
            name = f"fb{tau}/k{k}"
            if name in res["arms"]:
                continue
            wers, red, st, t0 = [], 0, 0, time.time()
            for i in idx:
                ids, _, r, t = T.decode(setup, first, sa, states[i], prompt, k, real_positions(waves[i]), heads, tau=float(tau))
                wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
                red += r
                st += t
            d = paired_ci(wers, full_ps, np.random.default_rng(SEED))
            v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
            res["arms"][name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d), "gate": v,
                                 "redone_share": red / max(st, 1), "catastrophic": int(sum(w > 1 for w in wers))}
            json.dump(res, open(out_json, "w"), indent=1)
            print(f"  {args.set} {name:<12} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v} (delta {delta})  redone {red / max(st, 1):.0%}"
                  f"  WER>1: {res['arms'][name]['catastrophic']}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
