"""D4 -- what does the tracked set miss? (why medium_en fails at 0.5*K_i)

Decodes with PadSink-Track v2 exactly as align_track.py, but at every step
also runs the step on the FULL cross cache (analysis only) and splits each
layer's attention mass of that step's query into:
  kept        positions the track set holds
  audio_out   audio positions outside the window
  pad_out     padding positions outside the sink and the window
Reported per layer, averaged over steps and utterances, for the given
models, at 0.5*K_i. Output tokens follow the track decode (the full-cache
run is only measured, never used).

Prediction (before the run): on medium_en the missed mass sits in
audio_out (non-local reading of the audio), concentrated in a few layers;
on medium_uz, which passes, the missed mass is smaller.

Usage:  python experiments/track_coverage.py --setup medium_en --n 40
"""

import argparse
import json
import os

import numpy as np

import align_track as A
from diag_budget import RHO
from eviction_budget import keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, SETUPS, SOT, encoder_states, prompt_ids,
                   real_positions, session, with_attention)
from spar import sink_set

HERE = os.path.dirname(os.path.abspath(__file__))


def run(setup, first, sa, enc, prompt, k, rn, heads):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    cross = {n: v for n, v in present.items() if ".encoder." in n}
    att = {int(n.split("layers.")[1].split("/")[0]): v[0, :, -1, :] for n, v in zip(names, out)
           if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    L = setup.n_layers
    sinks = []
    for l in range(L):
        s = rn + sink_set(att[l].sum(0)[rn:], RHO[setup.name])
        sinks.append(np.sort(s[:max(1, int(A.TRACK_CAP * k))]))
    c = int(np.argmax(sum(att[l][h, :rn] for l, h in heads)))
    s_in = [i.name for i in sa.get_inputs()]
    s_out = [o.name for o in sa.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    rows = []
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        keep = {}
        for l in range(L):
            w = max(1, k - len(sinks[l]))
            lo = int(np.clip(c - int(A.BACK * w), 0, max(0, ENC_POS - w)))
            keep[l] = np.unique(np.concatenate([np.arange(lo, min(ENC_POS, lo + w)), sinks[l]]).astype(int))
        base = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        full_feed, feed = dict(base), dict(base)
        for n in s_in:
            if n.startswith("past_key_values"):
                pn = "present" + n[len("past_key_values"):]
                if ".encoder." in pn:
                    full_feed[n] = cross[pn]
                    feed[n] = np.ascontiguousarray(cross[pn][:, :, keep[int(pn.split(".")[1])], :])
                else:
                    full_feed[n] = feed[n] = present[pn]
        rf = sa.run(None, full_feed)
        row = np.zeros((L, 3))
        for l in range(L):
            m = rf[att_idx[l]][0, :, -1, :].mean(0)          # head-mean attention, sums to 1
            inset = np.zeros(ENC_POS, bool)
            inset[keep[l]] = True
            row[l] = [m[inset].sum(), m[~inset & (np.arange(ENC_POS) < rn)].sum(),
                      m[~inset & (np.arange(ENC_POS) >= rn)].sum()]
        rows.append(row)
        res = sa.run(None, feed)
        logits = res[s_out.index("logits")]
        for n, v in zip(s_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        score = np.zeros(ENC_POS)
        for l, h in heads:
            np.add.at(score, keep[l], res[att_idx[l]][0, h, -1, :])
        score[rn:] = 0
        c = max(c, int(np.argmax(score)))
        nxt = int(np.argmax(logits[0, -1]))
    return np.mean(rows, axis=0) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_en", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    heads = A.align_heads(setup)
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    states, waves, _ = encoder_states(setup, "test", 300)
    _, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    sa = session(with_attention(setup.step))
    per = []
    for i in range(args.n):
        rn = real_positions(waves[i])
        k = max(1, int(round(args.scale * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
        r = run(setup, first, sa, states[i], prompt, k, rn, heads)
        if r is not None:
            per.append(r)
    per = np.mean(per, axis=0)
    print(f"{setup.name} s={args.scale}  mean over layers: kept {per[:, 0].mean():.3f}  "
          f"audio_out {per[:, 1].mean():.3f}  pad_out {per[:, 2].mean():.3f}")
    for l in range(setup.n_layers):
        print(f"  L{l:02d}  kept {per[l, 0]:.3f}  audio_out {per[l, 1]:.3f}  pad_out {per[l, 2]:.3f}")
    out = os.path.join(HERE, f"results_track_coverage_{setup.name}.json")
    json.dump({"n": args.n, "scale": args.scale, "per_layer": per.tolist()}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
