"""E5 -- shrinking the cross-attention cache by dropping positions.

E1 lowers the precision of every cache entry. This lowers the number of
entries, the other family of KV-cache compression (H2O, Scissorhands,
StreamingLLM), applied to the one cache that dominates here: the 1500
encoder positions every decode step attends to.

Two levers, both measured on the same 300 test utterances with the same
paired bootstrap as E1:

  trim    Whisper's encoder always processes a 30 s window; an utterance of
          s seconds occupies only ceil(s * 16000 / 320) positions and the
          rest is padding. Keeping only the real positions is exact in the
          sense that nothing the audio produced is dropped -- but the
          decoder was trained attending to the padding too, so its effect
          on WER is an empirical question, not zero by construction.
  mass    keep the k of all 1500 positions with the highest attention mass
          accumulated over all heads and layers at the first decode step
          (the prompt tokens), the criterion of H2O / Scissorhands. The
          attention probabilities are read from the first-step graph by
          exposing its cross-attention Softmax outputs. Padding positions
          compete on equal terms: the first run showed that dropping them
          all (trim) is catastrophic (WER 2.7 -- runaway repetition), i.e.
          the decoder uses them as attention sinks, exactly the effect
          StreamingLLM describes for the first tokens of an LLM.
  sink    trim plus the 32 padding positions with the highest mass: the
          StreamingLLM recipe transplanted to a cross-attention cache.

Arms combine these with the best E1 scheme so the two families can be seen
to compose. Cache bytes per step are averaged over utterances, since trimmed
lengths differ.

Usage:  python experiments/cache_eviction_wer.py [--n 300] [--encoder cascade]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cache_precision_wer import (BITS, FIRST, MAX_NEW, N_LAYERS,  # noqa: E402
                                 SCHEMES, SEED, STEP, encoder_states,
                                 paired_ci, session)
from calib_utils import MODEL_DIR, SPLIT_CACHE, load_audio  # noqa: E402
from wer_cer_whole_network import EOT, SOT, error_rate, normalize  # noqa: E402
from transformers import WhisperTokenizer  # noqa: E402

FIRST_ATTN = FIRST.replace(".onnx", "_attn.onnx")
OUT_FMT = os.path.join(HERE, "results_cache_eviction_wer_{tag}_n{n}.json")
N_SINK = 32
SAMPLES_PER_POSITION = 320
ENC_POS = 1500

# (name, keep rule, precision scheme)
ARMS = [
    ("full/fp32", "full", "fp32"),
    ("trim/fp32", "trim", "fp32"),
    ("sink/fp32", "sink", "fp32"),
    ("sink/int8_kivi", "sink", "int8_kivi"),
    ("sink/int4_kivi", "sink", "int4_kivi"),
    ("mass50/fp32", "mass50", "fp32"),
    ("mass25/fp32", "mass25", "fp32"),
    ("mass25/int4_kivi", "mass25", "int4_kivi"),
]


def expose_attention(src, dst):
    """Add every cross-attention Softmax output to the first-step graph."""
    if os.path.exists(dst):
        return
    m = onnx.load(src, load_external_data=False)
    names = [n.output[0] for n in m.graph.node
             if n.op_type == "Softmax" and "encoder_attn" in n.name]
    assert len(names) == N_LAYERS, f"found {len(names)} cross-attn softmax"
    for nm in names:
        m.graph.output.append(onnx.helper.make_tensor_value_info(
            nm, onnx.TensorProto.FLOAT, None))
    onnx.save(m, dst)
    print(f"  wrote {os.path.basename(dst)} with {len(names)} attention outputs")


def keep_indices(rule, real_n, mass):
    """Encoder positions to retain, as a sorted index array."""
    if rule == "full":
        return None
    if rule == "trim":
        return np.arange(real_n)
    if rule == "sink":
        pad = real_n + np.argsort(-mass[real_n:])[:N_SINK]
        return np.sort(np.concatenate([np.arange(real_n), pad]))
    frac = {"mass50": 0.5, "mass25": 0.25}[rule]
    k = max(1, int(round(frac * ENC_POS)))
    return np.sort(np.argsort(-mass)[:k])


def greedy(first, step, enc, prompt, rule, scheme, real_n):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    mass = None
    if rule in ("sink", "mass50", "mass25"):
        # sum of attention probability over heads, prompt tokens and layers
        mass = np.zeros(ENC_POS, np.float64)
        for n, v in zip(names, out):
            if "encoder_attn" in n and n.endswith("Softmax_output_0"):
                mass += v.sum(axis=(0, 1, 2))
    idx = keep_indices(rule, real_n, mass)
    fn = SCHEMES[scheme]
    for i in range(N_LAYERS):
        kn, vn = f"present.{i}.encoder.key", f"present.{i}.encoder.value"
        k, v = present[kn], present[vn]
        if idx is not None:
            k, v = k[:, :, idx, :], v[:, :, idx, :]
        present[kn], present[vn] = fn(np.ascontiguousarray(k),
                                      np.ascontiguousarray(v))
    step_in = [i.name for i in step.get_inputs()]
    step_out = [o.name for o in step.get_outputs()]
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
        logits = res[step_out.index("logits")]
        for n, v in zip(step_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        for i in range(N_LAYERS):
            kn, vn = f"present.{i}.decoder.key", f"present.{i}.decoder.value"
            present[kn], present[vn] = fn(present[kn], present[vn])
        nxt = int(np.argmax(logits[0, -1]))
    kept = ENC_POS if idx is None else len(idx)
    return ids[1 + len(prompt):], kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--encoder", default="cascade", choices=["fp32", "cascade"])
    ap.add_argument("--arms", default=",".join(a[0] for a in ARMS))
    args = ap.parse_args()
    out_json = OUT_FMT.format(tag=args.encoder, n=args.n)
    wanted = [a for a in ARMS if a[0] in args.arms.split(",")]

    expose_attention(FIRST, FIRST_ATTN)
    states, texts = encoder_states(args.n, args.encoder)
    waves, _ = load_audio(0, args.n, source=SPLIT_CACHE["test"])
    real = [min(ENC_POS, math.ceil(len(w) / SAMPLES_PER_POSITION)) for w in waves]
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt = [t for _, t in tok.get_decoder_prompt_ids(language="uz",
                                                       task="transcribe")]
    first, step = session(FIRST_ATTN), session(STEP)
    refs = [normalize(t).split() for t in texts[:args.n]]

    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    results.setdefault("n", args.n)
    results.setdefault("arms", {})
    print(f"real positions: mean {np.mean(real):.0f} of {ENC_POS} "
          f"(audio {np.mean([len(w) for w in waves]) / 16000:.1f} s)")

    for name, rule, scheme in wanted:
        if name in results["arms"] and results["arms"][name].get("n") == args.n:
            print(f"{name}: cached, skipping")
            continue
        wers, kept = [], []
        t0 = time.time()
        for i in range(args.n):
            ids, k = greedy(first, step, states[i], prompt, rule, scheme, real[i])
            hyp = normalize(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            kept.append(k)
            if (i + 1) % 100 == 0:
                print(f"  {name} {i + 1}/{args.n}  WER {np.mean(wers):.4f}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        mib = N_LAYERS * 2 * np.mean(kept) * 1024 * BITS[scheme] / 8 / 1024 ** 2
        results["arms"][name] = {
            "n": args.n, "rule": rule, "scheme": scheme,
            "wer": float(np.mean(wers)), "per_sample_wer": wers,
            "positions_kept_mean": float(np.mean(kept)),
            "cross_cache_mib_mean": float(mib),
        }
        json.dump(results, open(out_json, "w"), indent=1)
        print(f"{name}: WER {np.mean(wers):.4f}  kept {np.mean(kept):.0f} pos  "
              f"cache {mib:.1f} MiB", flush=True)

    if "full/fp32" in results["arms"]:
        base = results["arms"]["full/fp32"]["per_sample_wer"]
        rng = np.random.default_rng(SEED)
        print(f"\n{'arm':<18}{'kept':>6}{'cache MiB':>11}{'WER':>9}"
              f"{'dWER vs full/fp32 [95% CI]':>34}")
        for name, r in results["arms"].items():
            d, lo, hi = paired_ci(r["per_sample_wer"], base, rng)
            r["delta_vs_full"] = [d, lo, hi]
            print(f"{name:<18}{r['positions_kept_mean']:>6.0f}"
                  f"{r['cross_cache_mib_mean']:>11.1f}{r['wer']:>9.4f}"
                  f"   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        json.dump(results, open(out_json, "w"), indent=1)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
