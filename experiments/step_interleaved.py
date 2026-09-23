"""Decoder step at fixed t, all arms interleaved -- the paper's latency design.

step_latency_past.py measures one arm at a time. For a comparison that will
be quoted with an interval, the arms have to share the same minutes on the
machine: each round runs every arm once, in order, so drift inside a round
cancels and only within-round ratios are read. Seven rounds after three
warm-ups, single thread, median with the observed [min-max], exactly as
Table 11 of the paper was measured.

Arms are the decoder variants that decode identically (verify_with_past.py,
verify_int8_cache.py); the deployed no-cache decoder is included so the whole
chain from the paper's baseline can be read off one table.

Usage:  python experiments/step_interleaved.py [--t 30] [--rounds 7]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = os.getcwd()
os.chdir("D:/DSc/ISH/nnopt")
sys.path.insert(0, "D:/DSc/ISH/nnopt/experiments")
sys.path.insert(0, HERE)
from calib_utils import decoder_feeds  # noqa: E402
from step_latency_past import config, feed_for  # noqa: E402

EXPORT = os.path.join(HERE, "..", "models", "whisper_with_past")
ARMS = [
    ("no-cache (deployed)", "D:/DSc/ISH/nnopt/models/_whole_net/dec_int8.onnx"),
    ("KV cache, FP32", os.path.join(EXPORT, "decoder_with_past_model_int8.onnx")),
    ("+ untied lm_head", os.path.join(EXPORT, "decoder_with_past_untied_int8.onnx")),
    ("+ int8 cache, dequantize", os.path.join(EXPORT, "decoder_with_past_cache_dq.onnx")),
    ("+ int8 cache, integer attention", os.path.join(EXPORT, "decoder_with_past_cache_int.onnx")),
]
OUT_JSON = os.path.join(HERE, "results_step_interleaved.json")
WARMUP = 3


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()

    heads, head_dim = config()
    rng = np.random.default_rng(0)
    enc = decoder_feeds(12, 1)[0]["encoder_hidden_states"]

    sessions, feeds = [], []
    for label, path in ARMS:
        if not os.path.exists(path):
            print(f"  skip {label}: missing")
            continue
        s = session(path)
        if any(i.name.startswith("past_key_values") for i in s.get_inputs()):
            f = feed_for(s, args.t, heads, head_dim, rng)
        else:
            f = {"input_ids": np.full((1, args.t), 50258, dtype=np.int64),
                 "encoder_hidden_states": enc}
        sessions.append((label, s))
        feeds.append(f)

    for _ in range(WARMUP):
        for (_, s), f in zip(sessions, feeds):
            s.run(None, f)

    times = {label: [] for label, _ in sessions}
    for r in range(args.rounds):
        for (label, s), f in zip(sessions, feeds):
            t0 = time.perf_counter()
            s.run(None, f)
            times[label].append((time.perf_counter() - t0) * 1000)
        print(f"  round {r + 1}/{args.rounds}", flush=True)

    base = np.median(times[sessions[0][0]])
    print(f"\ndecoder step at t = {args.t}, {args.rounds} interleaved rounds, "
          f"1 thread\n{'arm':<34}{'median':>9}{'[min-max]':>18}{'vs deployed':>13}")
    out = {"t": args.t, "rounds": args.rounds, "arms": {}}
    for label, _ in sessions:
        v = np.array(times[label])
        med = float(np.median(v))
        print(f"{label:<34}{med:>9.1f}   [{v.min():6.1f}-{v.max():6.1f}]"
              f"{base / med:>12.1f}x")
        out["arms"][label] = {"median_ms": med, "min_ms": float(v.min()),
                              "max_ms": float(v.max()), "runs_ms": v.tolist(),
                              "speedup_vs_deployed": base / med}
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
