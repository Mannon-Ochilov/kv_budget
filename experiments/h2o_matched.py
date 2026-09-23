"""E6d -- H2O at exactly the per-utterance budget of the calibrated rule.

The E6 test compared the calibrated split rule against H2O at a fraction
chosen from the calibration split, which on test kept a different number of
positions (393 vs 352 on medium, 348 vs 375 on small). That is not the same
budget. Here H2O's global ranking keeps, for every utterance i, exactly the
K_i positions the split rule keeps for that utterance: same utterance, same
number of KV slots, different selection rule. Paired against the full cache
and directly against the split rule.

Usage:  python experiments/h2o_matched.py --setup medium_uz [--n 300]
"""

import argparse
import json
import os
import time

import numpy as np

from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))


def keep_global_k(k):
    return lambda mass: np.sort(np.argsort(-mass)[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    split_key = [k for k in e6["test"] if k.startswith("split")][0]
    split = e6["test"][split_key]
    assert full["n"] == args.n and split["n"] == args.n

    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)

    wers, kept = [], []
    t0 = time.time()
    for i in range(args.n):
        rn = real_positions(waves[i])
        k_i = len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS)))
        ids, k = greedy(setup, first, step, states[i], prompt, keep_global_k(k_i),
                        lambda a, b: (a, b))
        assert k == k_i
        hyp = norm(tok.decode(ids, skip_special_tokens=True))
        wers.append(error_rate(refs[i], hyp.split()))
        kept.append(k)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n}  WER {np.mean(wers):.4f}  [{time.time() - t0:.0f}s]",
                  flush=True)

    rng = np.random.default_rng(SEED)
    d_full = paired_ci(wers, full["per_sample_wer"], rng)
    d_split = paired_ci(wers, split["per_sample_wer"], rng)
    delta = round(full["wer"] * EPS, 4)
    res = {"n": args.n, "rule": ["global_matched", f_r, f_p], "wer": float(np.mean(wers)),
           "per_sample_wer": wers, "kept": float(np.mean(kept)),
           "mib": cache_mib(setup, float(np.mean(kept)), 32),
           "delta_vs_full": list(d_full), "delta_vs_split": list(d_split),
           "split_wer": split["wer"], "split_kept": split["kept"], "delta_gate": delta}
    out = os.path.join(HERE, f"results_h2o_matched_{setup.name}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n{setup.name}: H2O at the split rule's per-utterance K_i "
          f"(mean {np.mean(kept):.0f} positions, same as split {split['kept']:.0f})")
    print(f"  full      WER {full['wer']:.4f}")
    print(f"  split     WER {split['wer']:.4f}")
    print(f"  H2O(K_i)  WER {np.mean(wers):.4f}   vs full {d_full[0]:+.4f} "
          f"[{d_full[1]:+.4f}, {d_full[2]:+.4f}]   H2O - split {d_split[0]:+.4f} "
          f"[{d_split[1]:+.4f}, {d_split[2]:+.4f}]   (delta {delta})")
    print(f"\nsaqlandi: {out}")


if __name__ == "__main__":
    main()
