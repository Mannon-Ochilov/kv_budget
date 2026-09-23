"""Where the cross-attention mass goes over decoding steps and over layers.

The retention rule ranks positions by first-step mass. Two questions that
leaves open: (1) is the padding share stable over the decoding steps, or is
the first step special; (2) is it uniform over the decoder layers. Both
first-step and step graphs are run with their cross-attention Softmax
outputs exposed, and for every utterance the mass on padding is recorded per
step and per layer. Also recorded: how well the first-step ranking predicts
the all-step ranking (overlap of the top-K_i sets).

Usage:  python experiments/mass_profile.py --setup medium_uz [--n 300]
"""

import argparse
import json
import os
import time

import numpy as np

from eviction_budget import keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, SETUPS, SOT, encoder_states,
                   prompt_ids, real_positions, session, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))


def attn_mass(names, out, n_layers):
    """(layers, positions) mass from one run, summed over heads and query rows."""
    m = np.zeros((n_layers, ENC_POS))
    for n, v in zip(names, out):
        if "encoder_attn" in n and n.endswith("Softmax_output_0"):
            layer = int(n.split("layers.")[1].split("/")[0])
            m[layer] = v.sum(axis=(0, 1, 2))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    L = setup.n_layers
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]

    states, waves, _ = encoder_states(setup, "test", args.n)
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    step = session(with_attention(setup.step))
    fn, sn = [o.name for o in first.get_outputs()], [o.name for o in step.get_outputs()]
    step_in = [i.name for i in step.get_inputs()]

    pad_by_step = []      # per utterance: list over steps of padding share
    pad_by_layer_first, pad_by_layer_all = [], []
    overlap = []
    t0 = time.time()
    for i in range(args.n):
        rn = real_positions(waves[i])
        ids = [SOT, *prompt]
        out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                               "encoder_hidden_states": np.ascontiguousarray(states[i][None])})
        logits = out[fn.index("logits")]
        present = {n: v for n, v in zip(fn, out) if n.startswith("present")}
        m1 = attn_mass(fn, out, L)
        acc = m1.copy()
        shares = [m1.sum(axis=0)[rn:].sum() / m1.sum()]
        nxt = int(np.argmax(logits[0, -1]))
        for _ in range(MAX_NEW):
            if nxt == EOT:
                break
            ids.append(nxt)
            feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
            for n in step_in:
                if n.startswith("past_key_values"):
                    feed[n] = present["present" + n[len("past_key_values"):]]
            res = step.run(None, feed)
            logits = res[sn.index("logits")]
            for n, v in zip(sn, res):
                if n.startswith("present") and ".decoder." in n:
                    present[n] = v
            ms = attn_mass(sn, res, L)
            acc += ms
            shares.append(ms.sum(axis=0)[rn:].sum() / ms.sum())
            nxt = int(np.argmax(logits[0, -1]))
        pad_by_step.append(shares)
        pad_by_layer_first.append(m1[:, rn:].sum(axis=1) / m1.sum(axis=1))
        pad_by_layer_all.append(acc[:, rn:].sum(axis=1) / acc.sum(axis=1))
        # does the first-step ranking pick the same positions as the all-step one?
        k1 = keep_split(f_r, f_p, rn)(m1.sum(axis=0))
        ka = keep_split(f_r, f_p, rn)(acc.sum(axis=0))
        overlap.append(len(np.intersect1d(k1, ka)) / len(k1))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n}  [{time.time() - t0:.0f}s]", flush=True)

    def at(step_idx):
        v = [s[step_idx] for s in pad_by_step if len(s) > step_idx]
        return float(np.mean(v)), len(v)

    steps = {}
    for j in (0, 1, 2, 4, 9, 19):
        mean, cnt = at(j)
        steps[str(j + 1)] = {"pad_share": mean, "n": cnt}
    first5 = float(np.mean([np.mean(s[:5]) for s in pad_by_step]))
    allsteps = float(np.mean([np.mean(s) for s in pad_by_step]))
    lay_first = np.mean(pad_by_layer_first, axis=0)
    lay_all = np.mean(pad_by_layer_all, axis=0)
    res = {"setup": setup.name, "n": args.n, "pad_share_by_step": steps,
           "pad_share_first5": first5, "pad_share_all_steps": allsteps,
           "pad_share_by_layer_first": lay_first.tolist(),
           "pad_share_by_layer_all": lay_all.tolist(),
           "topk_overlap_first_vs_all": float(np.mean(overlap))}
    json.dump(res, open(os.path.join(HERE, f"results_mass_profile_{setup.name}.json"), "w"),
              indent=1)
    print(f"\n{setup.name}: padding share of cross-attention mass")
    print("  by step:  " + "  ".join(f"t={k}: {v['pad_share']:.3f}" for k, v in steps.items()))
    print(f"  first 5 steps {first5:.3f}   all steps {allsteps:.3f}")
    print("  by layer (first step): " + " ".join(f"{x:.2f}" for x in lay_first))
    print("  by layer (all steps):  " + " ".join(f"{x:.2f}" for x in lay_all))
    print(f"  top-K_i overlap, first-step vs all-step ranking: {np.mean(overlap):.3f}")


if __name__ == "__main__":
    main()
