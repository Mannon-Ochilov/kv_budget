"""Data for the window-trajectory figure: one utterance, every decoding step.

For each step of the PadSink-Track decode (medium_uz, 0.5 K_i) the step is
also run on the full cache (analysis only) to record the alignment heads'
attention over all 1500 positions; the tracked window [lo, lo + w), the
sink positions and the peak c are saved next to it.

Usage:  python experiments/trace_track.py --setup medium_uz --utt 12
"""

import argparse
import os

import numpy as np

import align_track as A
from diag_budget import RHO
from eviction_budget import keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, SETUPS, SOT, encoder_states, prompt_ids,
                   real_positions, session, with_attention)
from spar import sink_set

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz")
    ap.add_argument("--utt", type=int, default=12)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    heads = A.align_heads(setup)
    import json
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    states, waves, texts = encoder_states(setup, "test", 300)
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    sa = session(with_attention(setup.step))
    i = args.utt
    rn = real_positions(waves[i])
    k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
    enc = states[i]
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
    s_in = [x.name for x in sa.get_inputs()]
    s_out = [o.name for o in sa.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    rows, wins, peaks, toks = [], [], [], []
    first_row = sum(att[l][h] for l, h in heads) / len(heads)
    rows.append(first_row)
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        toks.append(tok.decode([nxt]))
        keep = {}
        w = max(1, k - len(sinks[0]))
        lo = int(np.clip(c - int(A.BACK * w), 0, max(0, ENC_POS - w)))
        for l in range(L):
            wl = max(1, k - len(sinks[l]))
            lol = int(np.clip(c - int(A.BACK * wl), 0, max(0, ENC_POS - wl)))
            keep[l] = np.unique(np.concatenate([np.arange(lol, min(ENC_POS, lol + wl)), sinks[l]]).astype(int))
        wins.append((lo, lo + w))
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
        rows.append(sum(rf[att_idx[l]][0, h, -1, :] for l, h in heads) / len(heads))
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
        peaks.append(c)
        nxt = int(np.argmax(logits[0, -1]))
    out = os.path.join(HERE, f"trace_track_{setup.name}_{i}.npz")
    np.savez(out, att=np.array(rows[1:]), wins=np.array(wins), peaks=np.array(peaks), rn=rn, k=k,
             sinks=np.array(sinks[heads[0][0]]), toks=np.array(toks), ref=texts[i],
             hyp=tok.decode(ids[1 + len(prompt):], skip_special_tokens=True))
    print(f"utt {i}: rn {rn}, k {k}, steps {len(wins)}\n REF {texts[i]}\n HYP {tok.decode(ids[1 + len(prompt):], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
