"""E13 -- more retention baselines at the same per-utterance budget K_i.

H2O with aggregated mass is one point of the eviction literature. Three
others are re-implemented for the cross-attention cache, each given for
every utterance exactly the slot count K_i the calibrated split rule keeps,
so the comparison is: same utterance, same total KV slots, different
selection rule.

  h2o_layer   H2O per layer: each decoder layer ranks the 1500 positions by
              its OWN first-step attention mass and keeps K_i (the original
              H2O is per layer; our main arm aggregates mass over layers).
  snapkv      SnapKV: the attention scores of the observation window (the
              prompt tokens, i.e. the first step) are pooled along the
              position axis with a 1-D average kernel of 7 before ranking,
              per layer, K_i kept -- the clustering step SnapKV adds to H2O.
  pyramidkv   PyramidKV: the same total budget L*K_i is allocated over the
              layers as a pyramid (arithmetic decrease from the first to the
              last layer, ratio 2:1 between the ends, as in the paper's
              default), each layer ranking by its own mass.

All FP32 cache, 300 test utterances, both models, paired against the full
cache and against the calibrated split rule. Methods that cannot be
compared this way (Whisper-MLA: retraining + architecture change; SpeechKV:
learned pooling for speech LLMs; VoxZip: multimodal LLM) are listed with
the reason in the paper, not run.

Usage:  python experiments/sota_baselines.py --setup medium_uz [--n 300]
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


def pooled(m, k=7):
    pad = k // 2
    mp = np.pad(m, (pad, pad), mode="edge")
    return np.convolve(mp, np.ones(k) / k, mode="valid")


def make_rule(rule, k_i, n_layers):
    if rule == "h2o_layer":
        return lambda mass, mass_l: {l: np.sort(np.argsort(-mass_l[l])[:k_i]) for l in range(n_layers)}
    if rule == "snapkv":
        return lambda mass, mass_l: {l: np.sort(np.argsort(-pooled(mass_l[l]))[:k_i]) for l in range(n_layers)}
    if rule == "pyramidkv":
        # budgets decreasing linearly from 2x to 1x of the mean, sum = L*k_i
        w = np.linspace(2.0, 1.0, n_layers)
        b = np.round(w / w.sum() * n_layers * k_i).astype(int)
        b = np.clip(b, 1, ENC_POS)
        return lambda mass, mass_l: {l: np.sort(np.argsort(-mass_l[l])[:int(b[l])]) for l in range(n_layers)}
    raise ValueError(rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    split = e6["test"][[k for k in e6["test"] if k.startswith("split")][0]]
    out_json = os.path.join(HERE, f"results_sota_baselines_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}

    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)

    for rule in ("h2o_layer", "snapkv", "pyramidkv"):
        if rule in res["arms"]:
            continue
        wers, kept = [], []
        t0 = time.time()
        for i in range(args.n):
            k_i = len(keep_split(f_r, f_p, real_positions(waves[i]))(np.zeros(ENC_POS)))
            ids, k = greedy(setup, first, step, states[i], prompt,
                            make_rule(rule, k_i, setup.n_layers), lambda a, b: (a, b))
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            kept.append(k)
            if (i + 1) % 100 == 0:
                print(f"  {rule} {i + 1}/{args.n}  WER {np.mean(wers):.4f}  [{time.time() - t0:.0f}s]", flush=True)
        res["arms"][rule] = {"n": args.n, "wer": float(np.mean(wers)), "per_sample_wer": wers,
                             "kept": float(np.mean(kept)), "mib": cache_mib(setup, float(np.mean(kept)), 32)}
        json.dump(res, open(out_json, "w"), indent=1)

    rng = np.random.default_rng(SEED)
    delta = round(full["wer"] * EPS, 4)
    print(f"\n{setup.name}: full {full['wer']:.4f}, split {split['wer']:.4f} ({split['kept']:.0f} slots), delta {delta}")
    print(f"{'rule':<12}{'kept':>6}{'WER':>9}{'vs full [95% CI]':>28}{'vs split [95% CI]':>28}   gate")
    for rule, r in res["arms"].items():
        d = paired_ci(r["per_sample_wer"], full["per_sample_wer"], rng)
        ds = paired_ci(r["per_sample_wer"], split["per_sample_wer"], rng)
        r["delta_vs_full"], r["delta_vs_split"] = list(d), list(ds)
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        print(f"{rule:<12}{r['kept']:>6.0f}{r['wer']:>9.4f}   {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
              f"   {ds[0]:+.4f} [{ds[1]:+.4f}, {ds[2]:+.4f}]   {v}")
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
