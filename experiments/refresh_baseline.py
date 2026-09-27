"""R3 -- periodic re-selection: a dynamic rival to PadSink-Track.

Every R decoder steps (R in {4, 8, 16}) the step is run on the FULL cross
cache (so that step is exact) and, from that step's attention, each layer
keeps its k heaviest positions for the next R - 1 steps, which read only
those k positions. k = round(0.5 K_i), the same active budget as
PadSink-Track in Table 6. The cost of the refresh (a full-cache step every R
steps) is part of the method and is timed in e2e_latency.py.

Pre-registered predictions (before the run):
  P1  R = 4 passes the gate on at least three models (it refreshes often);
  P2  R = 16 fails on at least two models (the set goes stale, as the
      one-shot rules do);
  P3  at the R that passes on as many models as PadSink-Track, its
      whole-utterance ms/token is higher than PadSink-Track's (every R-th
      step reads the full cache).

Results (300 test utterances, 0.5 K_i): R = 4 passes only on medium_uz
(+0.016; small_uz inconclusive +0.034; small_en +0.020 and medium_en +0.267
rejected) with a full-cache step on 23 % of the steps; R = 8 and R = 16 fail on
all four. P1 refuted (1 of 4), P2 confirmed (4 of 4). P3 confirmed: whole
utterance, R = 4 costs 67.5 ms/token (medium) and 22.7 (small), no faster than
the full cache (64.5 / 23.6) and slower than PadSink-Track (55.0 / 20.1).

Usage:  python experiments/refresh_baseline.py --setup medium_uz
"""

import argparse
import json
import os
import time

import numpy as np

from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, SEED, SETUPS, SOT, encoder_states, error_rate,
                   paired_ci, prompt_ids, real_positions, session, text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
PERIODS = (4, 8, 16)


def refresh_greedy(setup, first, sa, enc, prompt, k, period):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    cross = {n: v for n, v in present.items() if ".encoder." in n}
    L = setup.n_layers

    def select(att_by_layer):
        return {l: np.sort(np.argsort(-att_by_layer[l])[:k]) for l in range(L)}

    att0 = {int(n.split("layers.")[1].split("/")[0]): v[0, :, -1, :].sum(0) for n, v in zip(names, out)
            if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    keep = select(att0)
    s_in = [i.name for i in sa.get_inputs()]
    s_out = [o.name for o in sa.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    nxt = int(np.argmax(logits[0, -1]))
    t, refreshes = 0, 0
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        t += 1
        full = t % period == 0
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in s_in:
            if n.startswith("past_key_values"):
                pn = "present" + n[len("past_key_values"):]
                if ".encoder." in pn:
                    feed[n] = cross[pn] if full else np.ascontiguousarray(cross[pn][:, :, keep[int(pn.split(".")[1])], :])
                else:
                    feed[n] = present[pn]
        res = sa.run(None, feed)
        logits = res[s_out.index("logits")]
        for n, v in zip(s_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        if full:
            keep = select({l: res[att_idx[l]][0, :, -1, :].sum(0) for l in range(L)})
            refreshes += 1
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):], refreshes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    args = ap.parse_args()
    s = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{s.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    delta = round(full["wer"] * EPS, 4)
    states, waves, texts = encoder_states(s, "test", 300)
    norm = text_norm(s)
    tok, prompt = prompt_ids(s)
    first, sa = session(with_attention(s.first)), session(with_attention(s.step))
    out_json = os.path.join(HERE, f"results_refresh_{s.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    for R in PERIODS:
        name = f"refresh_R{R}/s0.5"
        if name in res["arms"]:
            continue
        wers, frac, t0 = [], [], time.time()
        for i in range(300):
            k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, real_positions(waves[i]))(np.zeros(ENC_POS))))))
            ids, nref = refresh_greedy(s, first, sa, states[i], prompt, k, R)
            wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
            frac.append(nref / max(1, len(ids)))
        d = paired_ci(wers, full["per_sample_wer"], np.random.default_rng(SEED))
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        res["arms"][name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d),
                             "gate": v, "full_step_share": float(np.mean(frac)),
                             "catastrophic": int(sum(w > 1 for w in wers))}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {s.name} {name:<16} WER {np.mean(wers):.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}"
              f"  full-step share {np.mean(frac):.0%}  WER>1: {res['arms'][name]['catastrophic']}  [{time.time() - t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
