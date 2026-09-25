"""A1-A3 -- is the alignment steering what makes PadSink-Track work?

Same budget, same sink, same window width as track v2 (align_track.py);
only one component changes per arm. 300 test utterances, 0.5 * K_i (and
track itself at K_i).

  fixed_rate      MURMUR-style (arXiv 2606.01483): the window advances a
                  fixed number of encoder positions per output token; the
                  rate is the validation-split mean of (speech positions /
                  reference tokens) -- realizable.
  fixed_rate_utt  the same with the rate of THIS utterance from its
                  reference length -- not realizable, the best any fixed
                  rate could do.
  static          the window never moves (a one-shot window at the start).
  nosink          no padding sink; the window gets the whole budget.
  allheads        the peak from every head of every layer instead of the
                  alignment heads.
  v1              the window is confined to the speech (no run into the
                  padding at the end).
  track/s1.0      track v2 at the full calibrated budget K_i.

Pre-registered predictions (before the run):
  A1  track beats fixed_rate at 0.5 K_i on at least three of the four
      models (paired CI below zero); fixed_rate_utt closes part of the gap
      but not all of it (speaking rate varies within an utterance).
  A2  static and nosink fail the gate on every model; allheads and v1 are
      worse than track on at least two models.
  A3  track at K_i passes the gate on all four models, medium_en included.
  Results A1-A3: A1 confirmed (fixed_rate rejected on all four, +0.15..+0.56;
  fixed_rate_utt is no better); A2 confirmed (static, nosink fail everywhere;
  allheads worse on all four; v1 worse on small_en/small_uz, but BETTER on
  medium_en, -0.0064); A3 refuted (K_i: small_en, small_uz accepted;
  medium_uz +0.017 and medium_en +0.0076 inconclusive).
  A4 (added before the int8 arms) track_int8 differs from track (FP32) by
      less than delta/2 at both budgets on every model.

Usage:  python experiments/track_ablation.py --setup medium_uz
"""

import argparse
import json
import os
import time

import numpy as np

import align_track as A
from cache_precision_wer import SCHEMES
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, encoder_states, error_rate, paired_ci,
                   prompt_ids, real_positions, session, text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))


def speech_rate(setup, split, n, tok):
    waves, texts = __import__("kvlib").load_audio(setup.audio[split], n)
    rn = np.array([real_positions(w) for w in waves])
    nt = np.array([max(1, len(tok(t, add_special_tokens=False).input_ids)) for t in texts])
    return rn / nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    heads = A.align_heads(setup)
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    delta = round(full["wer"] * EPS, 4)
    track = json.load(open(os.path.join(HERE, f"results_align_track_{setup.name}.json")))["arms"]
    out_json = os.path.join(HERE, f"results_track_ablation_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}

    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    rate_val = float(np.mean(speech_rate(setup, "validation", 100, tok)))
    rate_utt = speech_rate(setup, "test", args.n, tok)
    res["rate_validation"] = rate_val
    first = session(with_attention(setup.first))
    sa = session(with_attention(setup.step))

    arms = {
        "fixed_rate/s0.5": dict(peak="fixed", rate="val"),
        "fixed_rate_utt/s0.5": dict(peak="fixed", rate="utt"),
        "static/s0.5": dict(peak="static"),
        "nosink/s0.5": dict(cap=0.0),
        "allheads/s0.5": dict(peak="all"),
        "v1/s0.5": dict(clip_rn=True),
        "track/s1.0": dict(),
        # item 4: the int8 per-head cache (the graph-executable scheme) under the tracked set
        "track_int8/s0.5": dict(cache_fn="int8_head"),
        "track_int8/s1.0": dict(cache_fn="int8_head"),
    }
    for name, kw in arms.items():
        if name in res["arms"]:
            continue
        scale = float(name.split("/s")[1])
        wers, t0 = [], time.time()
        for i in range(args.n):
            rn = real_positions(waves[i])
            k = max(1, int(round(scale * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
            kw_i = dict(kw)
            if kw_i.get("rate") == "val":
                kw_i["rate"] = rate_val
            elif kw_i.get("rate") == "utt":
                kw_i["rate"] = float(rate_utt[i])
            if "cache_fn" in kw_i:
                kw_i["cache_fn"] = SCHEMES[kw_i["cache_fn"]]
            ids, _ = A.track_greedy(setup, first, sa, states[i], prompt, k, rn, heads, False, **kw_i)
            wers.append(error_rate(refs[i], norm(tok.decode(ids, skip_special_tokens=True)).split()))
        rng = np.random.default_rng(SEED)
        d = paired_ci(wers, full["per_sample_wer"][:args.n], rng)
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        entry = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d), "gate": v,
                 "catastrophic": int(sum(w > 1 for w in wers))}
        if name.endswith("s0.5"):
            dt = paired_ci(wers, track["track/s0.5"]["per_sample_wer"][:args.n], rng)
            entry["delta_vs_track"] = list(dt)
            extra = f"  vs track {dt[0]:+.4f} [{dt[1]:+.4f}, {dt[2]:+.4f}]"
        else:
            extra = ""
        res["arms"][name] = entry
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {setup.name} {name:<20} WER {np.mean(wers):.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}"
              f"  WER>1: {entry['catastrophic']}{extra}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nsaqlandi: {out_json}  (validation rate {rate_val:.2f} positions/token)")


if __name__ == "__main__":
    main()
