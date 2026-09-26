"""L7 -- a fixed per-step working set on long audio.

A one-shot rule keeps a set whose size must grow with the utterance (the
calibrated K_i is ~ f_r * speech + f_p * padding; at 30 s it approaches the
full 1500 positions). PadSink-Track reads a local window, so its k need not
grow with the audio. Test: long utterances, the SAME fixed k for every
utterance, one-shot rules against PadSink-Track.

  medium_uz_long   89 Uzbek composites of 15-30 s (long_audio_uz.py; unused
                   test clips 300-599 joined with 0.3 s silence)
  small_en_long    the 80 LibriSpeech test-clean utterances >= 10 s among
                   the 300 used everywhere else

Arms, k in {200, 400} positions per layer for every utterance:
  h2o_layer   one-shot, per-layer first-step mass, top k
  padsink     one-shot PadSink-KV (sink + top audio), total k
  track       PadSink-Track v2 (same settings as everywhere; k fixed)
Reference: the full cache (per-sample WER from the earlier runs) and the
calibrated split at its own K_i (reported with its mean K_i).

Pre-registered predictions (before the run):
  L7a  track at k = 200 passes the gate on both long sets.
  L7b  both one-shot rules fail at k = 200 on both sets, and at k = 400 at
       least on medium_uz_long (about 950 speech positions).
  L7c  the calibrated split needs a mean K_i above 700 on both sets.

Usage:  python experiments/long_fixed_k.py [--set medium_uz_long|small_en_long]
"""

import argparse
import json
import os
import time

import numpy as np

import align_track as A
import kvlib
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, Setup, error_rate, greedy, load_audio,
                   paired_ci, prompt_ids, real_positions, session, text_norm,
                   with_attention)
from sota_baselines import make_rule
from spar import spar_rule

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
KS = (200, 400)
RHO = {"medium_uz": 0.9, "small_en": 0.9}
kvlib.MAX_NEW = 224
A.MAX_NEW = 224


def data(name):
    if name == "medium_uz_long":
        base = SETUPS["medium_uz"]
        npz = os.path.join(ROOT, "models", "_calib_cache", "cv_uz_long100.npz")
        ls = Setup("medium_uz_long", base.hf_dir, base.encoder, base.first, base.step,
                   {"test": npz, "validation": base.audio["validation"]},
                   base.language, base.n_layers, base.d_model, base.normalizer)
        states, waves, texts = kvlib.encoder_states(ls, "test", 89)
        full = json.load(open(os.path.join(HERE, "results_long_audio_medium_uz.json")))["arms"]["full/fp32"]
        return base, states, waves, texts, list(range(89)), full["per_sample_wer"]
    base = SETUPS["small_en"]
    states, waves, texts = kvlib.encoder_states(base, "test", 300)
    idx = [i for i, w in enumerate(waves) if len(w) >= 10 * 16000]
    return base, states, waves, texts, idx, None     # full recomputed here with MAX_NEW = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="medium_uz_long", choices=["medium_uz_long", "small_en_long"])
    args = ap.parse_args()
    setup, states, waves, texts, idx, full_ps = data(args.set)
    heads = A.align_heads(setup)
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    norm = text_norm(setup)
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    step = session(setup.step)
    sa = session(with_attention(setup.step))
    out_json = os.path.join(HERE, f"results_long_fixed_k_{args.set}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    rn = [real_positions(waves[i]) for i in idx]
    if full_ps is None:
        if "full" not in res["arms"]:
            fw = []
            for i in idx:
                ids, _ = greedy(setup, first, step, states[i], prompt, lambda m: None, lambda a, b: (a, b))
                fw.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
            res["arms"]["full"] = {"wer": float(np.mean(fw)), "per_sample_wer": fw}
            json.dump(res, open(out_json, "w"), indent=1)
        full_ps = res["arms"]["full"]["per_sample_wer"]
    res.update({"n": len(idx), "audio_s_mean": float(np.mean([len(waves[i]) / 16000 for i in idx])),
                "speech_positions_mean": float(np.mean(rn)),
                "split_Ki_mean": float(np.mean([len(keep_split(f_r, f_p, r)(np.zeros(ENC_POS))) for r in rn]))})
    print(f"{args.set}: n {len(idx)}, {res['audio_s_mean']:.1f} s, speech positions {res['speech_positions_mean']:.0f}, "
          f"split K_i mean {res['split_Ki_mean']:.0f}", flush=True)
    delta = round(float(np.mean(full_ps)) * EPS, 4)
    arms = [("split/Ki", None)] + [(f"{r}/k{k}", k) for k in KS for r in ("h2o_layer", "padsink", "track")]
    for name, k in arms:
        if name in res["arms"]:
            continue
        rule = name.split("/")[0]
        wers, t0 = [], time.time()
        for j, i in enumerate(idx):
            if rule == "split":
                ids, _ = greedy(setup, first, step, states[i], prompt, keep_split(f_r, f_p, rn[j]), lambda a, b: (a, b))
            elif rule == "h2o_layer":
                ids, _ = greedy(setup, first, step, states[i], prompt, make_rule("h2o_layer", k, setup.n_layers),
                                lambda a, b: (a, b))
            elif rule == "padsink":
                ids, _ = greedy(setup, first, step, states[i], prompt,
                                spar_rule(k, rn[j], RHO[setup.name], setup.n_layers, fill=True), lambda a, b: (a, b))
            else:
                ids, _ = A.track_greedy(setup, first, sa, states[i], prompt, k, rn[j], heads, False)
            wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
        d = paired_ci(wers, full_ps, np.random.default_rng(SEED))
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        res["arms"][name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d),
                             "gate": v, "catastrophic": int(sum(w > 1 for w in wers))}
        res["delta"], res["full_wer"] = delta, float(np.mean(full_ps))
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {args.set} {name:<16} WER {np.mean(wers):.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}"
              f"  WER>1: {res['arms'][name]['catastrophic']}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nsaqlandi: {out_json}  (full WER {np.mean(full_ps):.4f}, delta {delta})")


if __name__ == "__main__":
    main()
