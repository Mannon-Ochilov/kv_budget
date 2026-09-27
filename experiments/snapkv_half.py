"""SnapKV-style at 0.5 K_i (missing from Table 6), same protocol as diag_budget.py.

Prediction (before the run): SnapKV-style fails the gate at 0.5 K_i on all four
models (it already fails at K_i on three of them).

Usage:  python experiments/snapkv_half.py --setup medium_uz
"""
import argparse, json, os, time
import numpy as np
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, encoder_states, error_rate, greedy, paired_ci,
                   prompt_ids, real_positions, session, text_norm, with_attention)
from sota_baselines import make_rule

HERE = os.path.dirname(os.path.abspath(__file__))


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
    first, step = session(with_attention(s.first)), session(s.step)
    wers, kept, t0 = [], [], time.time()
    for i in range(300):
        k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, real_positions(waves[i]))(np.zeros(ENC_POS))))))
        ids, kk = greedy(s, first, step, states[i], prompt, make_rule("snapkv", k, s.n_layers), lambda a, b: (a, b))
        wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
        kept.append(kk)
    d = paired_ci(wers, full["per_sample_wer"], np.random.default_rng(SEED))
    v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
    p = os.path.join(HERE, f"results_diag_budget_{s.name}.json")
    r = json.load(open(p))
    r["arms"]["snapkv/s0.5"] = {"n": 300, "wer": float(np.mean(wers)), "per_sample_wer": wers,
                               "kept": float(np.mean(kept)), "delta_vs_full": list(d), "gate": v}
    json.dump(r, open(p, "w"), indent=1)
    print(f"  {s.name} snapkv/s0.5 kept {np.mean(kept):.0f} WER {np.mean(wers):.4f} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] {v}"
          f" WER>1: {sum(w > 1 for w in wers)} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
