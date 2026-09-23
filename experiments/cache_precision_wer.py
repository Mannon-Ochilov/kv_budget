"""E1 -- what quantizing the KV cache costs in WER, at the paper's rigor.

The untied with-past decoder reads a 281.5 MiB FP32 cross-attention cache on
every step, the largest single read of the step. This measures what happens
to task quality when that cache -- and the small self-attention cache -- is
held at lower precision, on the same 300-utterance test split, with the same
paired percentile bootstrap (seed 20260916, 10 000 resamples) the paper uses
for every interval.

Schemes, all applied between steps to the `present.*` tensors:

  fp32         reference (verified identical to the deployed decoder)
  fp16         round-trip through float16
  int8_tensor  symmetric int8, one scale per tensor
  int8_head    symmetric int8, one scale per attention head
  int8_kivi    asymmetric int8, K per (head, channel) over tokens,
               V per (head, token) over channels  -- the KIVI grouping
  int4_kivi    the same grouping at 4 bits

The cross-attention cache is quantized once, when the first step emits it,
and dequantized before every later step; that is exactly the storage the
scheme implies. Quantization runs in NumPy here, so this file answers the
accuracy question only; the latency of an integrated kernel is E2.

Encoder states are computed once and memory-mapped, so each scheme costs a
decoder pass and nothing else. Decoding is single-threaded so the fp32 arm
can be checked against the deployed decoder's per-sample WER exactly.

Usage:  python experiments/cache_precision_wer.py [--n 300] [--schemes a,b]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir("D:/DSc/ISH/nnopt")
sys.path.insert(0, "D:/DSc/ISH/nnopt/experiments")
from calib_utils import MODEL_DIR, SPLIT_CACHE, TARGET_SR, load_audio  # noqa: E402
from wer_cer_whole_network import EOT, SOT, error_rate, normalize  # noqa: E402
from transformers import WhisperFeatureExtractor, WhisperTokenizer  # noqa: E402

EXPORT = os.path.join(HERE, "..", "models", "whisper_with_past")
ENCODERS = {
    "fp32": "models/uzbek_stt_v1_onnx/encoder_model.onnx",
    "cascade": "models/_gptq/enc_gptq_pruned.onnx",     # the paper's encoder
}
FIRST = os.path.join(EXPORT, "decoder_model_int8.onnx")
STEP = os.path.join(EXPORT, "decoder_with_past_untied_int8.onnx")
STATES_FMT = os.path.join(HERE, "..", "models", "enc_states_test300_{tag}.npy")
OUT_FMT = os.path.join(HERE, "results_cache_precision_wer_{tag}.json")
SEED, RESAMPLES = 20260916, 10000
MAX_NEW = 96
ENC_THREADS = 8
N_LAYERS = 24


# ----------------------------------------------------------------- schemes
def q_sym(x, bits, axes):
    """Symmetric quantization with one scale per slice over `axes`."""
    qmax = 2 ** (bits - 1) - 1
    s = np.max(np.abs(x), axis=axes, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return (np.clip(np.round(x / s), -qmax, qmax) * s).astype(np.float32)


def q_asym(x, bits, axes):
    """Asymmetric (min/max) quantization, KIVI style, per slice over `axes`."""
    qmax = 2 ** bits - 1
    lo = np.min(x, axis=axes, keepdims=True)
    hi = np.max(x, axis=axes, keepdims=True)
    s = (hi - lo) / qmax
    s = np.where(s == 0, 1.0, s)
    return (np.clip(np.round((x - lo) / s), 0, qmax) * s + lo).astype(np.float32)


# cache tensors are (batch, heads, seq, head_dim); axis 2 = tokens, 3 = channels
SCHEMES = {
    "fp32": lambda k, v: (k, v),
    "fp16": lambda k, v: (k.astype(np.float16).astype(np.float32),
                          v.astype(np.float16).astype(np.float32)),
    "int8_tensor": lambda k, v: (q_sym(k, 8, (1, 2, 3)), q_sym(v, 8, (1, 2, 3))),
    "int8_head": lambda k, v: (q_sym(k, 8, (2, 3)), q_sym(v, 8, (2, 3))),
    "int8_kivi": lambda k, v: (q_asym(k, 8, (2,)), q_asym(v, 8, (3,))),
    "int4_kivi": lambda k, v: (q_asym(k, 4, (2,)), q_asym(v, 4, (3,))),
}
BITS = {"fp32": 32, "fp16": 16, "int8_tensor": 8, "int8_head": 8,
        "int8_kivi": 8, "int4_kivi": 4}


# ----------------------------------------------------------------- pipeline
def session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def encoder_states(n, tag):
    """Compute and memory-map the encoder output for the first n test items."""
    states = STATES_FMT.format(tag=tag)
    waves, texts = load_audio(0, n, source=SPLIT_CACHE["test"])
    if os.path.exists(states):
        st = np.load(states, mmap_mode="r")
        if st.shape[0] >= n:
            return st, texts
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    enc = session(ENCODERS[tag], ENC_THREADS)
    st = np.lib.format.open_memmap(states, mode="w+", dtype=np.float32,
                                   shape=(n, 1500, 1024))
    t0 = time.time()
    for i, wav in enumerate(waves):
        f = fe(wav, sampling_rate=TARGET_SR,
               return_tensors="np").input_features.astype(np.float32)
        st[i] = enc.run(None, {"input_features": f})[0][0]
        if (i + 1) % 25 == 0:
            print(f"  encoder {i + 1}/{n}  [{time.time() - t0:.0f}s]",
                  flush=True)
    st.flush()
    del enc
    return np.load(states, mmap_mode="r"), texts


def greedy(first, step, enc, prompt, cache_fn):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    # Cross-attention cache: quantize once, it never changes afterwards.
    for i in range(N_LAYERS):
        kn, vn = f"present.{i}.encoder.key", f"present.{i}.encoder.value"
        present[kn], present[vn] = cache_fn(present[kn], present[vn])
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
        # Self-attention cache: re-quantized as it grows, same scheme.
        for i in range(N_LAYERS):
            kn, vn = f"present.{i}.decoder.key", f"present.{i}.decoder.value"
            present[kn], present[vn] = cache_fn(present[kn], present[vn])
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):]


def paired_ci(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), (RESAMPLES, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--encoder", default="fp32", choices=list(ENCODERS))
    args = ap.parse_args()
    schemes = [s for s in args.schemes.split(",") if s]
    OUT_JSON = OUT_FMT.format(tag=args.encoder)

    states, texts = encoder_states(args.n, args.encoder)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt = [t for _, t in tok.get_decoder_prompt_ids(language="uz",
                                                       task="transcribe")]
    first, step = session(FIRST), session(STEP)
    refs = [normalize(t).split() for t in texts[:args.n]]

    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    results.setdefault("n", args.n)
    results.setdefault("schemes", {})

    for name in schemes:
        if name in results["schemes"] and results["schemes"][name].get("n") == args.n:
            print(f"{name}: cached, skipping")
            continue
        fn = SCHEMES[name]
        wers, cers, lens = [], [], []
        t0 = time.time()
        for i in range(args.n):
            ids = greedy(first, step, states[i], prompt, fn)
            hyp = normalize(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            cers.append(error_rate(list(" ".join(refs[i])), list(hyp)))
            lens.append(len(ids))
            if (i + 1) % 50 == 0:
                print(f"  {name} {i + 1}/{args.n}  WER so far {np.mean(wers):.4f}"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
        results["schemes"][name] = {
            "n": args.n, "bits": BITS[name],
            "wer": float(np.mean(wers)), "cer": float(np.mean(cers)),
            "mean_tokens": float(np.mean(lens)),
            "per_sample_wer": wers,
            # cross-attention cache bytes per step, 24 layers x K,V x 1500 x 1024
            "cross_cache_mib": N_LAYERS * 2 * 1500 * 1024 * BITS[name] / 8 / 1024 ** 2,
        }
        json.dump(results, open(OUT_JSON, "w"), indent=1)
        print(f"{name}: WER {np.mean(wers):.4f}  CER {np.mean(cers):.4f}  "
              f"cache {results['schemes'][name]['cross_cache_mib']:.1f} MiB",
              flush=True)

    # Paired intervals against the fp32 cache, one seed for all rows.
    if "fp32" in results["schemes"]:
        base = results["schemes"]["fp32"]["per_sample_wer"]
        rng = np.random.default_rng(SEED)
        print(f"\n{'scheme':<13}{'bits':>5}{'cache MiB':>11}{'WER':>9}"
              f"{'dWER vs fp32 [95% CI]':>32}")
        for name, r in results["schemes"].items():
            d, lo, hi = paired_ci(r["per_sample_wer"], base, rng)
            r["delta_vs_fp32"] = [d, lo, hi]
            print(f"{name:<13}{r['bits']:>5}{r['cross_cache_mib']:>11.1f}"
                  f"{r['wer']:>9.4f}   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        json.dump(results, open(OUT_JSON, "w"), indent=1)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
