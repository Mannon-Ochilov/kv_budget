"""D1/D2 -- where do the retention rules separate, and how much is left?

At the calibrated budget K_i every rule is within the CI of the full cache
(Table 5a), so a ranking between them is not measurable there. Two
diagnostics:

  D1  budget sweep: the same rules at s*K_i, s in {0.5, 0.25} (s = 1 is
      already measured: results_eviction_budget / sota_baselines / spar).
        split_s     the calibrated rule with both counts scaled by s
        h2o_layer   per-layer first-step mass, top s*K_i
        pyramidkv   2:1 pyramid, total L*s*K_i
        padsink     PadSink-KV static, validation-selected rho
      (SnapKV already fails at s = 1 on three models; not repeated.)

  D2  oracle headroom: at EVERY decoding step, each layer keeps the s*K_i
      positions with the highest attention mass of THAT step's query
      (computed on the full cache, then the step is re-run on the kept
      positions). Not realizable -- it reads the full cache every step --
      it bounds what any query-aware (per-step) selection could gain over
      the one-shot first-step selection all rules above use.

Pre-registered predictions (written before the run):
  P1  the spread between rules grows as s shrinks: at s = 0.25 the best and
      worst of {split, h2o_layer, pyramidkv, padsink} differ by more than
      delta on at least two of the four models.
  P2  padsink passes the gate in at least as many (model, s) cells as any
      other rule.
  P3  oracle at s = 0.25 is better than the best one-shot rule by more than
      delta/2 on at least two models (=> per-step selection is worth
      building); if not, the one-shot selection is not the bottleneck.

300 test utterances, FP32 cache, paired bootstrap vs full, gate as E6.

Usage:  python experiments/diag_budget.py --setup medium_uz
"""

import argparse
import json
import os
import time

import numpy as np

from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, SEED, SETUPS, SOT, encoder_states,
                   error_rate, greedy, paired_ci, prompt_ids, real_positions,
                   session, text_norm, with_attention)
from sota_baselines import make_rule
from spar import spar_rule

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES = (0.5, 0.25)
RHO = {"medium_uz": 0.9, "small_uz": 0.95, "medium_en": 0.8, "small_en": 0.9}


def split_scaled(f_r, f_p, rn, s):
    n_r = max(1, int(round(s * max(1, int(round(f_r * rn))))))
    n_p = max(1, int(round(s * int(round(f_p * (ENC_POS - rn))))))

    def fn(mass):
        r = np.argsort(-mass[:rn])[:n_r]
        p = rn + np.argsort(-mass[rn:])[:n_p]
        return np.sort(np.concatenate([r, p]))
    return fn


def oracle_greedy(setup, first, step_attn, enc, prompt, k):
    """Per-step top-k per layer by the current query's attention mass."""
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    cross = {n: v for n, v in present.items() if ".encoder." in n}
    s_in = [i.name for i in step_attn.get_inputs()]
    s_out = [o.name for o in step_attn.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        base = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        feed = dict(base)
        for n in s_in:
            if n.startswith("past_key_values"):
                feed[n] = present["present" + n[len("past_key_values"):]]
        # 1) the query's attention on the full cross cache
        res = step_attn.run(None, feed)
        feed2 = dict(base)
        for n in s_in:
            if n.startswith("past_key_values"):
                pn = "present" + n[len("past_key_values"):]
                if ".encoder." in pn:
                    layer = int(pn.split(".")[1])
                    m = res[att_idx[layer]].sum(axis=(0, 1, 2))
                    keep = np.sort(np.argsort(-m)[:k])
                    feed2[n] = np.ascontiguousarray(cross[pn][:, :, keep, :])
                else:
                    feed2[n] = present[pn]
        # 2) the step on the kept positions only
        res = step_attn.run(None, feed2)
        logits = res[s_out.index("logits")]
        for n, v in zip(s_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    delta = round(full["wer"] * EPS, 4)
    out_json = os.path.join(HERE, f"results_diag_budget_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    res["rho"] = RHO[setup.name]

    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    step = session(setup.step)
    step_attn = session(with_attention(setup.step))

    def k_of(i, s):
        return max(1, int(round(s * len(keep_split(f_r, f_p, real_positions(waves[i]))(np.zeros(ENC_POS))))))

    arms = [(f"oracle/s1", "oracle", 1.0)]
    for s in SCALES:
        arms += [(f"{r}/s{s}", r, s) for r in ("split", "h2o_layer", "pyramidkv", "padsink")]
        arms.append((f"oracle/s{s}", "oracle", s))
    for name, rule, s in arms:
        if name in res["arms"]:
            continue
        wers, kept, t0 = [], [], time.time()
        for i in range(args.n):
            rn, k = real_positions(waves[i]), k_of(i, s)
            if rule == "oracle":
                ids, kk = oracle_greedy(setup, first, step_attn, states[i], prompt, k), k
            else:
                if rule == "split":
                    fn = split_scaled(f_r, f_p, rn, s)
                elif rule == "padsink":
                    fn = spar_rule(k, rn, RHO[setup.name], setup.n_layers)
                else:
                    fn = make_rule(rule, k, setup.n_layers)
                ids, kk = greedy(setup, first, step, states[i], prompt, fn, lambda a, b: (a, b))
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            kept.append(kk)
        d = paired_ci(wers, full["per_sample_wer"][:args.n], np.random.default_rng(SEED))
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        res["arms"][name] = {"n": args.n, "wer": float(np.mean(wers)), "per_sample_wer": wers,
                             "kept": float(np.mean(kept)), "delta_vs_full": list(d), "gate": v}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {setup.name} {name:<18} kept {np.mean(kept):6.0f}  WER {np.mean(wers):.4f}  "
              f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
