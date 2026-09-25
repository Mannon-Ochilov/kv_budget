"""D3 -- a realizable per-step selection: padding sink + alignment-tracked window.

D2 showed that per-step (query-aware) selection holds the full-cache WER at
0.25*K_i on every model while every one-shot rule collapses. The oracle
reads the whole cache each step; this rule does not.

PadSink-Track, per utterance:
  * first step on the full cache (as every rule); from it, per layer, the
    padding sink S_l (rho fraction of the padding mass, capped at SINK_CAP
    of the budget) and the start of the alignment centre c;
  * every later step each layer holds S_l plus a window of the remaining
    k - |S_l| audio positions, placed a quarter behind / three quarters
    ahead of c;
  * the step runs on that set only; the attention of Whisper's own
    alignment heads (generation_config.alignment_heads, the heads used for
    word timestamps) over the window gives the new peak, and c advances
    monotonically: c = max(c, peak).
  track_heavy additionally spends HEAVY of the window on the audio
  positions with the highest first-step mass (fixed, H2O-style), for
  layers that also read global context.

Only k positions per layer are read per step; the full cache stays in DRAM.

Selection: back / sink cap / peak source were tuned ONLY on small_en
validation (100 utt.): peak = alignment heads, back = 0.1, sink cap = 0.25
(val: +0.020 at 0.25*K_i, -0.0014 at 0.5*K_i). The same setting is applied
unchanged to all four models' test sets.

Pre-registered predictions (before the run):
  P4  track at 0.25*K_i passes the gate on small_en (the oracle does; the
      best one-shot rule is +2.36 there).
  P5  track is better than the best one-shot rule at 0.25*K_i on every
      model where it is run, by more than delta.
  (added before the test run, after the validation sweep)
  P6  track at 0.5*K_i passes the gate on at least three of the four models;
      at 0.25*K_i it passes on at most two (val: +0.020 > delta on small_en).
  v1 result (window clipped at the end of speech): small_en +0.037 N / +0.093 R,
  small_uz +0.014 Q / +0.045 N at 0.5 / 0.25*K_i; the outliers were decoding
  loops that start when the window reaches the end of speech (utt. 43:
  c = rn-1, then the peak jumps back to already-read audio). v2 lets the
  window run into the padding. v2 is re-checked on small_en validation
  before the test; v1 numbers are kept in the json as arms_v1.
  P7  v2 removes most loops: at 0.5*K_i the number of WER>1 utterances on
      the small_en / small_uz test is below v1's (1 / 5).
  v2 result: 0.5*K_i accepted on small_en, small_uz, medium_uz; medium_en
  +0.022 rejected. D4 (track_coverage.py): medium_en puts 35% of its
  attention on padding outside sink + window (medium_uz 17%).
  track_sum adds one summary slot for that padding (mean K/V, bias log n,
  exactly n identical copies; the slot is paid from the window).
  P8  track_sum at 0.5*K_i passes on medium_en and on all four models;
      at 0.25*K_i it improves on track on at least three models.

Usage:  python experiments/align_track.py --setup small_en [--n 300]
"""

import argparse
import json
import os
import time

import numpy as np

from diag_budget import RHO
from eviction_budget import EPS, keep_split
from kvlib import (with_cross_bias, ENC_POS, EOT, MAX_NEW, SEED, SETUPS, SOT, encoder_states,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)
from spar import SINK_CAP, sink_set

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES = (0.25, 0.5)
BACK = 0.1        # chosen on small_en validation (results_align_track_small_en.json, 'valid')
TRACK_CAP = 0.25  # sink share of k, idem; applied unchanged to the other three models
HEAVY = 0.25


def align_heads(setup):
    g = json.load(open(os.path.join(setup.hf_dir, "generation_config.json")))
    return [tuple(x) for x in g["alignment_heads"]]


def track_greedy(setup, first, step_attn, enc, prompt, k, rn, heads, heavy, back=BACK, peak="align", cap=TRACK_CAP, summary=False,
                 rate=None, clip_rn=False, cache_fn=None):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    cross = {n: v for n, v in present.items() if ".encoder." in n}
    if cache_fn is not None:
        # the whole cross cache is stored quantized (scales over all 1500
        # positions, as it would sit in DRAM); the decoder's own cache is
        # quantized after every step, as in the one-shot combo arms
        for l in range(setup.n_layers):
            kn, vn = f"present.{l}.encoder.key", f"present.{l}.encoder.value"
            cross[kn], cross[vn] = cache_fn(cross[kn], cross[vn])
    att = {}
    for n, v in zip(names, out):
        if "encoder_attn" in n and n.endswith("Softmax_output_0"):
            att[int(n.split("layers.")[1].split("/")[0])] = v[0, :, -1, :]   # heads x 1500
    L = setup.n_layers
    sinks = []
    for l in range(L):
        m = att[l].sum(0)
        s = rn + sink_set(m[rn:], RHO[setup.name])
        sinks.append(np.sort(s[:max(1, int(cap * k)) if cap > 0 else 0]).astype(int))
    if summary:
        # D4: the padding outside the sink is replaceable by its mean (sink
        # causal test, pad_rest_mean accepted on all models) -> one slot with
        # the mean K/V and bias log(n) == n identical copies. Sums over the
        # padding are formed once; window/padding overlap is subtracted per step.
        rest0, sumk, sumv = [], [], []
        for l in range(L):
            r = np.setdiff1d(np.arange(rn, ENC_POS), sinks[l])
            rest0.append(r)
            sumk.append(cross[f"present.{l}.encoder.key"][:, :, r, :].sum(axis=2, keepdims=True))
            sumv.append(cross[f"present.{l}.encoder.value"][:, :, r, :].sum(axis=2, keepdims=True))
    a0 = sum(att[l][h, :rn] for l, h in heads)
    c = c0 = int(np.argmax(a0))
    t_step = 0
    mass_audio = sum(att[l].sum(0)[:rn] for l in range(L))
    s_in = [i.name for i in step_attn.get_inputs()]
    s_out = [o.name for o in step_attn.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    nxt = int(np.argmax(logits[0, -1]))
    kept = []
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        keep = {}
        for l in range(L):
            w = max(1, k - len(sinks[l]) - (1 if summary else 0))   # the summary slot counts
            fixed = np.array([], dtype=int)
            if heavy:
                nh = int(HEAVY * w)
                fixed = np.argsort(-mass_audio)[:nh]
                w -= nh
            # v2: the window may run past the end of speech into the padding --
            # the frames right after speech are what tells the decoder to stop
            # (v1 clipped it at rn: at the end the peak jumped back and looped)
            if clip_rn:   # v1: window confined to the speech
                lo = int(np.clip(c - int(back * w), 0, max(0, rn - w)))
                win = np.arange(lo, min(rn, lo + w))
            else:
                lo = int(np.clip(c - int(back * w), 0, max(0, ENC_POS - w)))
                win = np.arange(lo, min(ENC_POS, lo + w))
            keep[l] = np.unique(np.concatenate([win, fixed, sinks[l]]).astype(int))
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in s_in:
            if n.startswith("past_key_values"):
                pn = "present" + n[len("past_key_values"):]
                if ".encoder." in pn:
                    l = int(pn.split(".")[1])
                    x = cross[pn][:, :, keep[l], :]
                    if summary:
                        ov = keep[l][np.isin(keep[l], rest0[l])]
                        tot = (sumk if pn.endswith("key") else sumv)[l]
                        nr = len(rest0[l]) - len(ov)
                        mean = (tot - cross[pn][:, :, ov, :].sum(axis=2, keepdims=True)) / max(nr, 1)
                        x = np.concatenate([x, mean.astype(x.dtype)], axis=2)
                        bias = np.zeros((1, 1, 1, len(keep[l]) + 1), np.float32)
                        bias[..., -1] = np.log(nr) if nr > 0 else -1e4
                        feed[f"cross_bias.{l}"] = bias
                    feed[n] = np.ascontiguousarray(x)
                else:
                    feed[n] = present[pn]
        res = step_attn.run(None, feed)
        logits = res[s_out.index("logits")]
        for n, v in zip(s_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        if cache_fn is not None:
            for l in range(L):
                kn, vn = f"present.{l}.decoder.key", f"present.{l}.decoder.value"
                present[kn], present[vn] = cache_fn(present[kn], present[vn])
        # alignment peak over the audio positions actually held
        score = np.zeros(ENC_POS)
        if peak == "align":
            for l, h in heads:
                np.add.at(score, keep[l], res[att_idx[l]][0, h, -1, :len(keep[l])])
        else:  # all heads of all layers
            for l in range(L):
                np.add.at(score, keep[l], res[att_idx[l]][0, :, -1, :len(keep[l])].sum(0))
        score[rn:] = 0
        t_step += 1
        if peak == "fixed":       # MURMUR-style: the window advances at a fixed rate
            c = int(c0 + rate * t_step)
        elif peak == "static":    # the window never moves (a one-shot window)
            c = c0
        else:
            c = max(c, int(np.argmax(score)))
        kept.append(np.mean([len(x) for x in keep.values()]) + (1 if summary else 0))
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):], float(np.mean(kept)) if kept else float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="small_en", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--valid", action="store_true", help="tune BACK / peak on the validation split")
    args = ap.parse_args()
    if args.valid:
        return tune(args)
    setup = SETUPS[args.setup]
    heads = align_heads(setup)
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    delta = round(full["wer"] * EPS, 4)
    out_json = os.path.join(HERE, f"results_align_track_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}

    states, waves, texts = encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    step_attn = session(with_attention(setup.step))

    res["config"] = {"version": 2, "back": BACK, "sink_cap": TRACK_CAP, "peak": "alignment_heads", "rho": RHO[setup.name]}
    step_bias = session(with_cross_bias(setup.step))
    for s in SCALES:
        for heavy in (False, "sum"):
            name = f"track{'_sum' if heavy else ''}/s{s}"
            if name in res["arms"] and res["arms"][name]["n"] == args.n:
                continue
            wers, kept, t0 = [], [], time.time()
            for i in range(args.n):
                rn = real_positions(waves[i])
                k = max(1, int(round(s * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
                ids, kk = (track_greedy(setup, first, step_bias, states[i], prompt, k, rn, heads, False, summary=True) if heavy
                            else track_greedy(setup, first, step_attn, states[i], prompt, k, rn, heads, False))
                wers.append(error_rate(refs[i], norm(tok.decode(ids, skip_special_tokens=True)).split()))
                kept.append(kk)
            d = paired_ci(wers, full["per_sample_wer"][:args.n], np.random.default_rng(SEED))
            v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
            res["arms"][name] = {"n": args.n, "wer": float(np.mean(wers)), "per_sample_wer": wers,
                                 "kept": float(np.mean(kept)), "delta_vs_full": list(d), "gate": v,
                                 "catastrophic": int(sum(w > 1 for w in wers))}
            json.dump(res, open(out_json, "w"), indent=1)
            print(f"  {setup.name} {name:<18} kept {np.mean(kept):6.0f}  WER {np.mean(wers):.4f}  "
                  f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}  WER>1: {sum(w > 1 for w in wers)}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nsaqlandi: {out_json}")


def tune(args):
    """Validation sweep (100 utterances): back in {0.1, 0.25, 0.5} x peak in {align, all}."""
    setup = SETUPS[args.setup]
    heads = align_heads(setup)
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    calib = e6["calib"]["full"]["per_sample_wer"]
    out_json = os.path.join(HERE, f"results_align_track_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    v = res.setdefault("valid", {})
    states, waves, texts = encoder_states(setup, "validation", 100)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    step_attn = session(with_attention(setup.step))
    grid = [(s, peak, back, SINK_CAP) for s in SCALES for peak in ("align", "all") for back in (0.1, 0.25, 0.5)]
    # second round: the sink reservation (60% of k) leaves ~35 audio frames at 0.25*K_i
    grid += [(s, "align", back, cap) for s in SCALES for back in (0.0, 0.1) for cap in (0.1, 0.25)]
    grid += [(s, "align", BACK, TRACK_CAP, "v2") for s in SCALES]
    grid += [(s, "align", BACK, TRACK_CAP, "v2", "sum") for s in SCALES]
    step_bias = session(with_cross_bias(setup.step))
    for s, peak, back, cap, *ver in grid:
                name = f"track/s{s}/{peak}/b{back}" + (f"/cap{cap}" if cap != SINK_CAP else "") + ("/v2" if ver else "") + ("/sum" if len(ver) > 1 else "")
                if name in v:
                    continue
                wers, t0 = [], time.time()
                for i in range(100):
                    rn = real_positions(waves[i])
                    k = max(1, int(round(s * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
                    ids, _ = track_greedy(setup, first, step_bias if len(ver) > 1 else step_attn, states[i], prompt, k, rn, heads,
                                          False, back, peak, cap, summary=len(ver) > 1)
                    wers.append(error_rate(refs[i], norm(tok.decode(ids, skip_special_tokens=True)).split()))
                d = paired_ci(wers, calib[:100], np.random.default_rng(SEED))
                v[name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "delta_vs_full": list(d)}
                json.dump(res, open(out_json, "w"), indent=1)
                print(f"  valid {name:<24} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  WER>1: {sum(w > 1 for w in wers)}"
                      f"  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
