"""R5 -- why PadSink-Track fails on whisper-medium/English (validation only).

Total k = round(0.5 K_i) is kept fixed; two factors are varied:
  sink cap      0.25 (the frozen setting) / 0.40 / 0.50 of k
  window end    runs into the padding (v2, frozen) / stops at the end of speech (v1)
= 6 configurations, 100 validation utterances (dev-clean), paired against the
full cache on the same utterances.

Reading (stated before the run):
  * better with a larger sink cap        -> padding coverage was short;
  * better when the window stops at speech -> the run into the padding hurts;
  * neither helps                        -> a local, monotone window steered by
                                            the alignment heads is not enough
                                            for this checkpoint.

Usage:  python experiments/medium_en_diag.py
"""

import json
import os
import time

import numpy as np

import align_track as A
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, encoder_states, error_rate, paired_ci, prompt_ids,
                   real_positions, session, text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    s = SETUPS["medium_en"]
    e6 = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_en.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["calib"]["full"]["per_sample_wer"][:100]
    wf = float(np.mean(full))
    states, waves, texts = encoder_states(s, "validation", 100)
    heads = A.align_heads(s)
    norm = text_norm(s)
    tok, prompt = prompt_ids(s)
    first, sa = session(with_attention(s.first)), session(with_attention(s.step))
    out_json = os.path.join(HERE, "results_medium_en_diag.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    res["full_wer_valid"] = wf
    for cap in (0.25, 0.40, 0.50):
        for clip in (False, True):
            name = f"cap{cap:.2f}/{'speech_end' if clip else 'into_padding'}"
            if name in res["arms"]:
                continue
            wers, t0 = [], time.time()
            for i in range(100):
                rn = real_positions(waves[i])
                k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
                ids, _ = A.track_greedy(s, first, sa, states[i], prompt, k, rn, heads, False, cap=cap, clip_rn=clip)
                wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
            d = paired_ci(wers, full, np.random.default_rng(SEED))
            res["arms"][name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d)}
            json.dump(res, open(out_json, "w"), indent=1)
            print(f"  {name:<24} WER {np.mean(wers):.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
                  f"  WER>1: {sum(w > 1 for w in wers)}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nvalidation full WER {wf:.4f}, delta (eps 0.2) {round(wf * EPS, 4)}")


if __name__ == "__main__":
    main()
