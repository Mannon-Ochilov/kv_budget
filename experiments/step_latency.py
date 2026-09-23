"""Decoder step latency as a function of decode position t.

The static traffic model says the step is dominated by a term that does not
depend on t: recomputing cross-attention K/V from 1500 encoder positions,
75.5 GMAC every step, plus a full read of the weight working set. If that is
right, step time is nearly flat in t and its intercept is most of the cost.
If instead the step grows steeply with t, the prefix recompute dominates and
the KV-cache story is the ordinary one.

A linear fit  ms = a + b*t  separates the two: a is the t-independent floor,
b the per-token growth. Content of the tokens does not matter for timing, so
the prompt is followed by a repeated filler token.

Single thread, min of REPEATS runs per t, one utterance's encoder states.
"""

import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir("D:/DSc/ISH/nnopt")          # calib_utils uses relative paths
sys.path.insert(0, "D:/DSc/ISH/nnopt/experiments")
from calib_utils import MODEL_DIR, decoder_feeds  # noqa: E402
from transformers import WhisperTokenizer  # noqa: E402

DECODER = "D:/DSc/ISH/nnopt/models/_whole_net/dec_int8.onnx"
POSITIONS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 64, 80, 96]
REPEATS = 3
OUT_JSON = os.path.join(HERE, "results_step_latency.json")


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(DECODER, sess_options=so,
                                providers=["CPUExecutionProvider"])

    enc = decoder_feeds(12, 1)[0]["encoder_hidden_states"]
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt = [50258] + [t for _, t in tok.get_decoder_prompt_ids(
        language="uz", task="transcribe")]
    filler = tok("a", add_special_tokens=False).input_ids[0]

    def ids(t):
        seq = (prompt + [filler] * t)[:max(t, len(prompt))]
        return np.array([seq[:t]], dtype=np.int64) if t >= len(prompt) \
            else np.array([prompt[:t]], dtype=np.int64)

    for _ in range(2):                                  # warm-up
        sess.run(None, {"input_ids": ids(10), "encoder_hidden_states": enc})

    rows = []
    for t in POSITIONS:
        feed = {"input_ids": ids(t), "encoder_hidden_states": enc}
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
    print(f"\nfit: ms = {a:.1f} + {b:.2f} * t")
    print(f"  t-independent floor a   : {a:.1f} ms  "
          f"({100 * a / Y[T == 30][0]:.0f}% of the step at t = 30)")
    print(f"  growth per token b      : {b:.2f} ms")
    print(f"  step at t=1 / t=96      : {Y[0]:.0f} / {Y[-1]:.0f} ms  "
          f"= {Y[-1] / Y[0]:.2f}x for 96x the prefix")

    json.dump({"decoder": DECODER, "rows": rows, "fit_a_ms": a,
               "fit_b_ms_per_token": b}, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
