"""R1 -- an untouched test set for whisper-small / English.

PadSink-Track's settings were tuned on small_en validation, and the v2 change
(the window may run into the padding) came from a decoding loop found in the
small_en TEST set, so the small_en test result is not independent. This
builds a fresh LibriSpeech test-clean set that nothing so far has seen: 300
utterances from speakers that do not occur in the 300 used everywhere else,
and runs the frozen method (no change of any setting) with the one-shot
rules at 0.5 K_i and the full cache.

Pre-registered predictions (before the run):
  R1a  PadSink-Track at 0.5 K_i passes the gate (delta = 0.2 * full WER).
  R1b  every one-shot rule (split, H2O per layer, PyramidKV, PadSink-KV)
       fails at 0.5 K_i, as on the original test set.

Usage:  python experiments/fresh_small_en.py
"""

import io
import json
import os
import time

import numpy as np

import align_track as A
import kvlib
from diag_budget import split_scaled
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, Setup, error_rate, greedy, paired_ci, prompt_ids,
                   real_positions, session, text_norm, with_attention)
from sota_baselines import make_rule
from spar import spar_rule

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NPZ = os.path.join(ROOT, "models", "_calib_cache", "ls_test_clean_fresh300.npz")
N = 300


def build():
    import soundfile as sf
    from datasets import Audio, load_dataset
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test",
                      streaming=True).cast_column("audio", Audio(decode=False))
    used, waves, texts, spk = set(), [], [], []
    for j, ex in enumerate(ds):
        if j < 300:
            used.add(ex["speaker_id"])          # the speakers of the original test set
            continue
        if ex["speaker_id"] in used:
            continue
        wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        waves.append(np.asarray(wav, np.float32))
        texts.append(ex["text"].lower())
        spk.append(ex["speaker_id"])
        if len(waves) == N:
            break
    np.savez(NPZ, audio=np.concatenate(waves), lengths=np.array([len(w) for w in waves]),
             texts=np.array(texts, dtype=object), speakers=np.array(spk))
    print(f"fresh set: {len(waves)} utterances, {len(set(spk))} new speakers "
          f"(original set: {len(used)} speakers), mean {np.mean([len(w) for w in waves]) / 16000:.1f} s", flush=True)


def main():
    if not os.path.exists(NPZ):
        build()
    base = SETUPS["small_en"]
    fresh = Setup("small_en_fresh", base.hf_dir, base.encoder, base.first, base.step,
                  {"test": NPZ, "validation": base.audio["validation"]},
                  base.language, base.n_layers, base.d_model, base.normalizer)
    states, waves, texts = kvlib.encoder_states(fresh, "test", N)
    heads = A.align_heads(base)
    e6 = json.load(open(os.path.join(HERE, "results_eviction_budget_small_en.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    norm = text_norm(base)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(base)
    first, step = session(with_attention(base.first)), session(base.step)
    sa = session(with_attention(base.step))
    out = os.path.join(HERE, "results_fresh_small_en.json")
    res = json.load(open(out)) if os.path.exists(out) else {"arms": {}}
    rn = [real_positions(w) for w in waves]
    ki = [len(keep_split(f_r, f_p, r)(np.zeros(ENC_POS))) for r in rn]
    ident = lambda a, b: (a, b)                                                        # noqa: E731
    arms = {
        "full": lambda i: greedy(base, first, step, states[i], prompt, lambda m: None, ident)[0],
        "split/Ki": lambda i: greedy(base, first, step, states[i], prompt, keep_split(f_r, f_p, rn[i]), ident)[0],
        "split/s0.5": lambda i: greedy(base, first, step, states[i], prompt, split_scaled(f_r, f_p, rn[i], 0.5), ident)[0],
        "h2o_layer/s0.5": lambda i: greedy(base, first, step, states[i], prompt,
                                           make_rule("h2o_layer", round(0.5 * ki[i]), base.n_layers), ident)[0],
        "pyramidkv/s0.5": lambda i: greedy(base, first, step, states[i], prompt,
                                           make_rule("pyramidkv", round(0.5 * ki[i]), base.n_layers), ident)[0],
        "padsink/s0.5": lambda i: greedy(base, first, step, states[i], prompt,
                                         spar_rule(round(0.5 * ki[i]), rn[i], 0.9, base.n_layers), ident)[0],
        "track/s0.5": lambda i: A.track_greedy(base, first, sa, states[i], prompt, round(0.5 * ki[i]), rn[i],
                                               heads, False)[0],
    }
    for name, fn in arms.items():
        if name in res["arms"]:
            continue
        wers, t0 = [], time.time()
        for i in range(N):
            wers.append(error_rate(refs[i], norm(tok.decode(fn(i), skip_special_tokens=True)).split()))
        res["arms"][name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers}
        json.dump(res, open(out, "w"), indent=1)
        print(f"  {name:<16} WER {np.mean(wers):.4f}  [{time.time() - t0:.0f}s]", flush=True)
    full = res["arms"]["full"]["per_sample_wer"]
    delta = round(float(np.mean(full)) * EPS, 4)
    res["delta"], res["Ki_mean"] = delta, float(np.mean(ki))
    print(f"\nfresh small_en: full WER {np.mean(full):.4f}, delta {delta}, mean K_i {np.mean(ki):.0f}")
    for name, r in res["arms"].items():
        if name == "full":
            continue
        d = paired_ci(r["per_sample_wer"], full, np.random.default_rng(SEED))
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        r["delta_vs_full"], r["gate"] = list(d), v
        print(f"  {name:<16} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}  WER>1: {sum(w > 1 for w in r['per_sample_wer'])}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
