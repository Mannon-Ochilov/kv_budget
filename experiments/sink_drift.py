"""R7 -- does the padding sink move during decoding? (analysis only)

PadSink-Track fixes the sink from the first step's attention. On medium_en
35 % of a step's attention falls on padding outside the sink and the window.
Full-cache greedy decode; at every step and layer the padding attention mass
(head mean) is split by how much of it a sink of the Track size would hold if
it were chosen from
  first    the first step (what PadSink-Track does)
  prev     the previous step's full attention
  r8       the most recent of every 8th step (a periodic sink refresh)
  oracle   the current step
and n90 = padding positions needed for 90 % of the step's padding mass.

Prediction (before the run): on medium_en `first` holds a clearly smaller
share than on medium_uz, and `r8` recovers most of the gap to `oracle` --
i.e. the sink drifts and a periodic refresh of the sink alone would fix it.
Refutation: `r8` and `oracle` are also low / n90 is large -> the padding
attention is diffuse and no small sink can hold it.

Usage:  python experiments/sink_drift.py --setup medium_en --n 30
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


def run(setup, first, sa, enc, prompt, k, rn):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    att = {int(n.split("layers.")[1].split("/")[0]): v[0, :, -1, :] for n, v in zip(names, out)
           if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    L = setup.n_layers
    n_s = max(1, int(A.TRACK_CAP * k))
    top = lambda p, n: np.argsort(-p)[:n]                                    # noqa: E731
    sets = {"first": [], "prev": [], "r8": []}
    for l in range(L):
        s = np.asarray(sink_set(att[l].sum(0)[rn:], RHO[setup.name]))[:n_s]
        for key in sets:
            sets[key].append(s)
    s_in = [i.name for i in sa.get_inputs()]
    s_out = [o.name for o in sa.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    rows, t = [], 0
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        t += 1
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in s_in:
            if n.startswith("past_key_values"):
                feed[n] = present["present" + n[len("past_key_values"):]]
        res = sa.run(None, feed)
        for n, v in zip(s_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        row = np.zeros((L, 6))
        for l in range(L):
            p = res[att_idx[l]][0, :, -1, :].mean(0)[rn:]
            tot = p.sum()
            cum = np.cumsum(np.sort(p)[::-1]) / max(tot, 1e-12)
            row[l] = [tot, p[sets["first"][l]].sum(), p[sets["prev"][l]].sum(), p[sets["r8"][l]].sum(),
                      p[top(p, len(sets["first"][l]))].sum(), int(np.searchsorted(cum, 0.9)) + 1]
            sets["prev"][l] = top(p, len(sets["first"][l]))
            if t % 8 == 0:
                sets["r8"][l] = sets["prev"][l]
        rows.append(row)
        nxt = int(np.argmax(res[s_out.index("logits")][0, -1]))
    return np.mean(rows, axis=0) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_en", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    s = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{s.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    states, waves, _ = encoder_states(s, "test", 300)
    _, prompt = prompt_ids(s)
    first, sa = session(with_attention(s.first), 4), session(with_attention(s.step), 4)
    per = []
    for i in range(args.n):
        rn = real_positions(waves[i])
        k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
        r = run(s, first, sa, states[i], prompt, k, rn)
        if r is not None:
            per.append(r)
    m = np.mean(per, axis=0)                                   # layers x 6
    tot = m[:, 0].sum()
    share = {key: float(m[:, j].sum() / tot) for key, j in (("first", 1), ("prev", 2), ("r8", 3), ("oracle", 4))}
    print(f"{s.name}: padding mass/step {m[:, 0].mean():.3f} | share of it held by a sink chosen from: "
          + "  ".join(f"{k} {v:.3f}" for k, v in share.items()) + f" | n90 {m[:, 5].mean():.0f} positions", flush=True)
    json.dump({"n": args.n, "per_layer": m.tolist(), "share": share},
              open(os.path.join(HERE, f"results_sink_drift_{s.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
