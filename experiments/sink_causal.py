"""E15 -- is the padding sink a place or content?

Dropping the sink positions is catastrophic (Table 3). Two readings: the
decoder needs WHAT those states contain, or it only needs somewhere to park
attention mass (the attention-sink reading of StreamingLLM). The test keeps
every position but replaces the K and V of the sink set (top rho = 0.9 of
each layer's padding mass, per layer, per head) with their mean vector:
the slots remain, the content is flattened.

  sink_mean     sink K/V -> mean over the sink positions
  audio_mean    control: the same NUMBER of audio positions (highest-mass
                ones) replaced by their mean -- content that must matter
  pad_rest_mean the non-sink padding replaced by its mean -- content that
                should not matter

If sink_mean stays within the gate while audio_mean does not, the sink is a
place, not content, and "attention sink" is the right word.

Usage:  python experiments/sink_causal.py --setup medium_uz [--n 300]
"""

import argparse
import json
import os
import time

import numpy as np

from eviction_budget import EPS
from kvlib import (ENC_POS, SEED, SETUPS, encoder_states, greedy, error_rate,
                   paired_ci, prompt_ids, real_positions, session, text_norm,
                   with_attention)
from spar import sink_set

HERE = os.path.dirname(os.path.abspath(__file__))
RHO = 0.9


def make_fns(arm, real_n, n_layers):
    """keep_fn records per-layer target sets; cache_fn flattens them."""
    state = {"targets": None, "layer": 0}

    def keep(mass, mass_l):
        tg = {}
        for l in range(n_layers):
            sink = real_n + sink_set(mass_l[l, real_n:], RHO)
            if arm == "sink_mean":
                tg[l] = sink
            elif arm == "audio_mean":
                tg[l] = np.argsort(-mass_l[l, :real_n])[:min(len(sink), real_n)]
            else:  # pad_rest_mean
                allpad = np.arange(real_n, ENC_POS)
                tg[l] = np.setdiff1d(allpad, sink)
        state["targets"] = tg
        return None  # keep every position

    def cache(k, v):
        if k.shape[2] != ENC_POS:      # self-attention cache: untouched
            return k, v
        l = state["layer"]
        state["layer"] += 1
        t = state["targets"][l]
        if len(t) == 0:
            return k, v
        k, v = k.copy(), v.copy()
        k[:, :, t, :] = k[:, :, t, :].mean(axis=2, keepdims=True)
        v[:, :, t, :] = v[:, :, t, :].mean(axis=2, keepdims=True)
        return k, v
    return keep, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fp32-encoder", action="store_true",
                    help="medium only: FP32 encoder instead of the cascade one (confound check)")
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    full = e6["test"]["full"]
    tag = setup.name
    if args.fp32_encoder:
        import dataclasses
        from kvlib import NNOPT, ROOT
        setup = dataclasses.replace(setup, name="medium_uz_fp32enc",
                                    encoder=os.path.join(NNOPT, "models", "uzbek_stt_v1_onnx", "encoder_model.onnx"))
        # reuse E1's FP32-encoder states; reference = E1's FP32-encoder full cache
        src = os.path.join(ROOT, "models", "enc_states_test300_fp32.npy")
        dst = os.path.join(ROOT, "models", f"enc_states_{setup.name}_test{args.n}.npy")
        if not os.path.exists(dst) and os.path.exists(src):
            os.link(src, dst)
        full = json.load(open(os.path.join(HERE, "results_cache_precision_wer_fp32.json")))["schemes"]["fp32"]
        tag = setup.name
    out_json = os.path.join(HERE, f"results_sink_causal_{tag}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)
    delta = round(full["wer"] * EPS, 4)
    for arm in ("sink_mean", "audio_mean", "pad_rest_mean"):
        if arm in res["arms"]:
            continue
        wers, t0 = [], time.time()
        for i in range(args.n):
            keep, cache = make_fns(arm, real_positions(waves[i]), setup.n_layers)
            ids, _ = greedy(setup, first, step, states[i], prompt, keep, cache)
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
        d = paired_ci(wers, full["per_sample_wer"], np.random.default_rng(SEED))
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        res["arms"][arm] = {"n": args.n, "wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d), "gate": v}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {arm:<14} WER {np.mean(wers):.4f}  vs full {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
