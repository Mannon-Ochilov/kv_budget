"""Decoder step latency against decode position, for the with-past decoder.

The counterpart of step_latency.py for a decoder that carries a KV cache.
Here a step reads one new token plus the cached keys and values: the
self-attention cache grows with t, the cross-attention cache is a fixed
1500-position block, and nothing is recomputed. If the static model is right,
the step is a small fraction of the no-cache decoder's and its t-independent
floor is the weight read alone.

Feeds are built from the model's own input signature: every `past_key_values`
input gets a tensor of the declared shape, with the sequence dimension set to
t - 1 for the decoder-side cache and to the encoder length for the
encoder-side cache. Values are random; with dynamic INT8 quantization the
arithmetic does not depend on them, so timing does not either.

Usage:  python experiments/step_latency_past.py --model path/to/decoder.onnx
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = "D:/DSc/ISH/nnopt/models/uzbek_stt_v1_onnx/config.json"
DEFAULT = os.path.join(HERE, "..", "models", "whisper_with_past",
                       "decoder_with_past_model_int8.onnx")
POSITIONS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 64, 80, 96]
REPEATS = 3
ENCODER_POSITIONS = 1500


def config():
    with open(CONFIG, encoding="utf-8") as f:
        c = json.load(f)
    return c["decoder_attention_heads"], c["d_model"] // c["decoder_attention_heads"]


def feed_for(sess, t, heads, head_dim, rng):
    """One feed at decode position t, shaped from the session's inputs."""
    feed = {}
    for inp in sess.get_inputs():
        name, shape = inp.name, list(inp.shape)
        if name == "input_ids":
            feed[name] = np.array([[50258]], dtype=np.int64)
        elif name == "encoder_hidden_states":
            feed[name] = rng.standard_normal(
                (1, ENCODER_POSITIONS, heads * head_dim), dtype=np.float32)
        elif name.startswith("past_key_values"):
            if name.endswith("_scale"):
                # per-head scale of an int8 cache tensor; magnitude is
                # irrelevant to timing, so a constant is enough
                dims = [d if isinstance(d, int) else heads for d in shape]
                feed[name] = np.full(tuple(dims), 0.05, dtype=np.float32)
                continue
            seq = ENCODER_POSITIONS if ".encoder." in name else max(t - 1, 0)
            dims = [1 if isinstance(d, str) and "batch" in d else d
                    for d in shape]
            dims = [seq if isinstance(d, str) else d for d in dims]
            # Symbolic dims other than batch and sequence are not expected;
            # fall back to the config for anything still unresolved.
            dims = [heads if d is None else d for d in dims]
            if "int8" in inp.type:
                feed[name] = rng.integers(-127, 128, size=tuple(dims),
                                          dtype=np.int8)
            else:
                feed[name] = rng.standard_normal(tuple(dims), dtype=np.float32)
        elif name == "use_cache_branch":
            feed[name] = np.array([True])
        else:
            raise SystemExit(f"unexpected input {name} {shape}")
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--out", default=os.path.join(HERE,
                    "results_step_latency_past.json"))
    args = ap.parse_args()

    heads, head_dim = config()
    rng = np.random.default_rng(0)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()][:4], "...")

    for _ in range(2):
        sess.run(None, feed_for(sess, 10, heads, head_dim, rng))

    rows = []
    for t in POSITIONS:
        feed = feed_for(sess, t, heads, head_dim, rng)
        best = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            sess.run(None, feed)
            dt = (time.perf_counter() - t0) * 1000
            best = dt if best is None else min(best, dt)
        rows.append({"t": t, "ms": best})
        print(f"  t={t:>3}  {best:8.1f} ms", flush=True)

    T = np.array([r["t"] for r in rows], float)
    Y = np.array([r["ms"] for r in rows])
    b, a = np.polyfit(T, Y, 1)
    print(f"\nfit: ms = {a:.1f} + {b:.3f} * t")
    print(f"  floor a {a:.1f} ms   growth b {b:.3f} ms/token   "
          f"t=1/t=96: {Y[0]:.0f}/{Y[-1]:.0f} ms")

    json.dump({"decoder": args.model, "rows": rows, "fit_a_ms": a,
               "fit_b_ms_per_token": b}, open(args.out, "w"), indent=2)
    print(f"\nsaqlandi: {args.out}")


if __name__ == "__main__":
    main()
