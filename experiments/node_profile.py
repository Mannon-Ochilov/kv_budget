"""Where one decoder step spends its time, by operator role.

ONNX Runtime's built-in profiler records the kernel time of every node. The
events are aggregated by the same roles the static model uses, so the
measured split can be laid next to the predicted MAC split. The vocabulary
projection is tracked together with the DequantizeLinear that feeds it, since
the static model's claim is that the dequantize runs on every step.

One thread, t = 30, the first run discarded as warm-up.
"""

import json
import re
import os
import sys
from collections import defaultdict

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH_DIR = os.getcwd()
os.chdir("D:/DSc/ISH/nnopt")          # calib_utils uses relative paths
sys.path.insert(0, "D:/DSc/ISH/nnopt/experiments")
from calib_utils import decoder_feeds  # noqa: E402

DECODER = "D:/DSc/ISH/nnopt/models/_whole_net/dec_int8.onnx"
T = 30
RUNS = 3
OUT_JSON = os.path.join(HERE, "results_node_profile.json")

# A with-past decoder takes cached keys and values instead of the prefix;
# its feed is built from the input signature the same way step_latency_past
# does, so both scripts describe the identical step.
from step_latency_past import config as _cfg, feed_for as _feed_for  # noqa: E402

ROLES = [
    ("cross-attn K/V proj", r"encoder_attn/(k|v)_proj"),
    ("cross-attn Q/out proj", r"encoder_attn/(q|out)_proj"),
    ("self-attn proj", r"self_attn/(q|k|v|out)_proj"),
    ("FFN fc1/fc2", r"/fc[12]/"),
    ("lm_head + dequantize", r"proj_out|embed_tokens\.weight_transposed"),
    ("integer attention", r"encoder_attn/MatMul(_\d+)?_(int|cast|scale|rescale)$"
                          r"|encoder_attn/(Mul_1|Softmax)_output_0_dyq$"),
    ("cache dequantize", r"past_key_values\.\d+\.encoder\.(key|value)_dq$"),
    ("attention products", r"/(self_attn|encoder_attn)/MatMul(_\d+)?$|Softmax$"),
    ("layer norm", r"layer_norm|LayerNorm"),
    ("quantize activations", r"DynamicQuantizeLinear|QuantizeLinear"),
]


def role_of(name):
    for label, pat in ROLES:
        if re.search(pat, name):
            return label
    return "other"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DECODER)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    if not os.path.isabs(args.model):
        args.model = os.path.normpath(os.path.join(LAUNCH_DIR, args.model))
    if not os.path.isabs(args.out):
        args.out = os.path.normpath(os.path.join(LAUNCH_DIR, args.out))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_profiling = True
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    if any(i.name.startswith("past_key_values") for i in sess.get_inputs()):
        heads, head_dim = _cfg()
        feed = _feed_for(sess, T, heads, head_dim, np.random.default_rng(0))
    else:
        enc = decoder_feeds(12, 1)[0]["encoder_hidden_states"]
        ids = np.full((1, T), 50258, dtype=np.int64)
        feed = {"input_ids": ids, "encoder_hidden_states": enc}
    for _ in range(RUNS + 1):
        sess.run(None, feed)
    path = sess.end_profiling()

    events = json.load(open(path, encoding="utf-8"))
    runs = defaultdict(lambda: defaultdict(float))
    seen = defaultdict(int)
    for e in events:
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
            continue
        node = e["name"][:-len("_kernel_time")]
        seen[node] += 1
        run_idx = seen[node] - 1
        runs[run_idx][role_of(node)] += e["dur"] / 1000.0      # us -> ms

    # Drop the first run (warm-up), average the rest.
    keep = [runs[i] for i in sorted(runs) if i > 0]
    agg = defaultdict(float)
    for r in keep:
        for k, v in r.items():
            agg[k] += v / len(keep)
    total = sum(agg.values())

    print(f"{os.path.basename(args.model)}: step at t = {T}, "
          f"mean of {len(keep)} runs after warm-up")
    print(f"{'role':<26}{'ms':>9}{'share':>8}")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {k:<24}{v:>9.1f}{100 * v / total:>7.1f}%")
    print(f"  {'total (kernel time)':<24}{total:>9.1f}")

    json.dump({"t": T, "runs_kept": len(keep), "ms_by_role": dict(agg),
               "total_ms": total, "model": args.model},
              open(args.out, "w"), indent=2)
    print(f"\nsaqlandi: {args.out}")


if __name__ == "__main__":
    main()
