"""L8 -- WER of the ring-buffer implementation that was timed.

The WER tables were computed with align_track.track_greedy (a fresh gather
every step); the timing used e2e_latency.track_ring (persistent ring
buffer). Both hold the same positions; only the slot order -- and so the
floating-point summation order inside attention -- differs, and the tokens
agree on 94-99 % of utterances. This measures the ring implementation's own
WER on the 300 test utterances at 0.5 K_i and compares it with the gather
implementation (paired).

Prediction (before the run): the ring-vs-gather difference is within
+-delta/4 on every model and the gate verdicts of Table 6 do not change.

Usage:  python experiments/ring_wer.py --setup medium_uz
"""

import argparse
import json
import os
import time

import numpy as np

import align_track as A
from e2e_latency import track_ring
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, encoder_states, error_rate, paired_ci,
                   prompt_ids, real_positions, session, text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    heads = A.align_heads(setup)
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    delta = round(full["wer"] * EPS, 4)
    gather = json.load(open(os.path.join(HERE, f"results_align_track_{setup.name}.json")))["arms"]["track/s0.5"]
    states, waves, texts = encoder_states(setup, "test", 300)
    norm = text_norm(setup)
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    sa = session(with_attention(setup.step))
    wers, t0 = [], time.time()
    for i in range(300):
        rn = real_positions(waves[i])
        k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
        ids = track_ring(setup, first, sa, states[i], prompt, k, rn, heads)
        wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
    rng = np.random.default_rng(SEED)
    d = paired_ci(wers, full["per_sample_wer"], rng)
    dg = paired_ci(wers, gather["per_sample_wer"], rng)
    v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
    out = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d),
           "delta_vs_gather": list(dg), "gate": v, "gather_gate": gather["gate"]}
    json.dump(out, open(os.path.join(HERE, f"results_ring_wer_{setup.name}.json"), "w"), indent=1)
    print(f"  {setup.name} ring  WER {np.mean(wers):.4f}  vs full {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] {v}"
          f"  (gather: {gather['gate']})  ring - gather {dg[0]:+.4f} [{dg[1]:+.4f}, {dg[2]:+.4f}]"
          f"  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
