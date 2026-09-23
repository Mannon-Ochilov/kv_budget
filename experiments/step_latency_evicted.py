"""E6c -- what an evicted cache is worth in decoder-step time.

Eviction needs no new kernel: the with-past graphs take the encoder cache
length as a free dimension, so a kept cache is just a shorter tensor. This
times one decoder step at t = 30 with the cross-attention cache at its full
1500 positions and at the length the calibrated split rule keeps on the
test split (E6), for the FP32 cache and -- on Whisper-medium, where the
integer-attention graph exists (E2) -- the int8 cache. Interleaved rounds,
single thread, median with [min-max], the paper's protocol.

Usage:  python experiments/step_latency_evicted.py [--rounds 7] [--t 30]
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ENC_POS = 1500

# (label, model path, heads, kept positions)
ARMS = []


def kept_positions(name):
    r = json.load(open(os.path.join(HERE, f"results_eviction_budget_{name}.json")))
    key = [k for k in r["test"] if k.startswith("split")][0]
    return int(round(r["test"][key]["kept"]))


def build_arms():
    med = os.path.join(ROOT, "models", "whisper_with_past")
    sml = os.path.join(ROOT, "models", "whisper_small_onnx")
    k_med, k_sml = kept_positions("medium_uz"), kept_positions("small_en")
    return [
        ("medium  fp32 cache, 1500", f"{med}/decoder_with_past_untied_int8.onnx", 16, ENC_POS),
        (f"medium  fp32 cache, {k_med}", f"{med}/decoder_with_past_untied_int8.onnx", 16, k_med),
        ("medium  int8 cache, 1500", f"{med}/decoder_with_past_cache_int.onnx", 16, ENC_POS),
        (f"medium  int8 cache, {k_med}", f"{med}/decoder_with_past_cache_int.onnx", 16, k_med),
        ("small   fp32 cache, 1500", f"{sml}/decoder_with_past_untied_int8.onnx", 12, ENC_POS),
        (f"small   fp32 cache, {k_sml}", f"{sml}/decoder_with_past_untied_int8.onnx", 12, k_sml),
    ]


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def feed_for(sess, t, heads, enc_len, rng):
    feed = {}
    for inp in sess.get_inputs():
        name, shape = inp.name, list(inp.shape)
        if name == "input_ids":
            feed[name] = np.array([[50258]], dtype=np.int64)
            continue
        if name.endswith("_scale"):
            dims = [d if isinstance(d, int) else heads for d in shape]
            feed[name] = np.full(tuple(dims), 0.05, dtype=np.float32)
            continue
        seq = enc_len if ".encoder." in name else max(t - 1, 0)
        dims = [1 if isinstance(d, str) and "batch" in d else d for d in shape]
        dims = [seq if isinstance(d, str) else d for d in dims]
        if "int8" in inp.type:
            feed[name] = rng.integers(-127, 128, size=tuple(dims), dtype=np.int8)
        else:
            feed[name] = rng.standard_normal(tuple(dims), dtype=np.float32)
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    arms = build_arms()
    sess = {}
    for label, path, heads, k in arms:
        s = sess.setdefault(path, session(path))
        feed = feed_for(s, args.t, heads, k, rng)
        for _ in range(3):
            s.run(None, feed)
    times = {a[0]: [] for a in arms}
    for r in range(args.rounds):
        for label, path, heads, k in arms:
            s = sess[path]
            feed = feed_for(s, args.t, heads, k, rng)
            t0 = time.perf_counter()
            s.run(None, feed)
            times[label].append((time.perf_counter() - t0) * 1e3)
        print(f"  round {r + 1}/{args.rounds}", flush=True)

    out = {"t": args.t, "rounds": args.rounds, "arms": {}}
    print(f"\ndecoder step at t = {args.t}, {args.rounds} interleaved rounds, 1 thread")
    print(f"{'arm':<28}{'median ms':>11}{'[min-max]':>20}")
    for label, path, heads, k in arms:
        v = np.array(times[label])
        out["arms"][label] = {"model": os.path.basename(path), "positions": k,
                              "median_ms": float(np.median(v)),
                              "min_ms": float(v.min()), "max_ms": float(v.max())}
        print(f"{label:<28}{np.median(v):>11.1f}   [{v.min():.1f}-{v.max():.1f}]")
    json.dump(out, open(os.path.join(HERE, "results_step_latency_evicted.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
