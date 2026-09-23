"""Per-token traffic of the decoder graph, split into what the weight-only
target sees and what it does not.

The deployed decoder has no KV cache: `decoder_model.onnx` takes `input_ids`
and `encoder_hidden_states` and nothing else, so at decode position t it
recomputes every projection over the whole prefix of t tokens, and recomputes
cross-attention K and V from all 1500 encoder positions. The weight-only
target M_eff counts the bytes of the weight matrices and nothing else. This
script counts everything the step actually moves and multiplies, by role, so
the size of the omission can be stated before any measurement is taken.

Three kinds of term appear per decode step:

  weights      -- read once per step regardless of t; this is M_eff
  activations  -- input rows read by each projection: t rows for most,
                  1500 rows for the cross-attention K/V projections
  arithmetic   -- MACs, of which the cross-attention K/V recompute is
                  48 projections x 1500 x 1024 x 1024 every single step

Bytes are counted at the deployed INT8 precision for weights and at INT8 for
the once-quantized encoder states that feed the cross-attention projections
(`encoder_hidden_states_quantized` is a single tensor reused by all 48). The
reuse-distance argument of Figure 5 applies to that tensor too: 16 MiB of
weights stream between consecutive layers' reads of it, so it is not assumed
to stay resident.

Usage:  python experiments/graph_traffic.py
        python experiments/graph_traffic.py --model path/to/decoder.onnx
"""

import argparse
import json
import os
import re

import onnx

NNOPT = "D:/DSc/ISH/nnopt/"
DEFAULT_MODEL = NNOPT + "models/_whole_net/dec_int8.onnx"
CONFIG = NNOPT + "models/uzbek_stt_v1_onnx/config.json"
OUT_JSON = "experiments/results_graph_traffic.json"
MIB = 1024.0 ** 2

ENCODER_POSITIONS = 1500
POSITIONS = [1, 10, 30, 96]            # 96 = MAX_NEW_TOKENS in the evaluator
BUDGET_MIB = 0.7 * 24.0

ROLE = re.compile(r"layers\.(\d+)/(self_attn|encoder_attn)/(q|k|v|out)_proj"
                  r"|layers\.(\d+)/(fc1|fc2)"
                  r"|(proj_out|lm_head)")


def classify(name):
    m = ROLE.search(name)
    if not m:
        return None
    if m.group(1) is not None:
        return f"{m.group(2)}.{m.group(3)}_proj"
    if m.group(4) is not None:
        return m.group(5)
    return "lm_head"


def rows_for(role, t):
    """Activation rows a projection reads at decode position t."""
    if role in ("encoder_attn.k_proj", "encoder_attn.v_proj"):
        return ENCODER_POSITIONS
    return t


def load(path):
    model = onnx.load(path, load_external_data=False)
    dims = {}
    for init in model.graph.initializer:
        dims[init.name] = (list(init.dims), init.data_type)
    ops = []
    for nd in model.graph.node:
        if nd.op_type not in ("MatMul", "MatMulInteger"):
            continue
        role = classify(nd.name)
        if role is None:
            continue
        w = next((x for x in nd.input if x in dims and len(dims[x][0]) == 2),
                 None)
        if w is None and role == "lm_head":
            # The vocabulary projection reuses the tied embedding table. In
            # the INT8 export it is stored as int8 but dequantized and
            # transposed before a plain float MatMul, so the working set the
            # step actually reads is the FP32 matrix, four times the on-disk
            # size. Counted at 4 bytes per element for that reason.
            emb = next((x for x in dims if x.endswith("embed_tokens.weight")
                        or x.endswith("embed_tokens.weight_quantized")), None)
            if emb is None:
                continue
            n, k = dims[emb][0]                    # (vocab, d_model)
            ops.append({"name": nd.name, "role": role, "k": k, "n": n,
                        "wbytes": k * n * 4, "dequantized": True})
            continue
        if w is None:
            continue
        shape, dtype = dims[w]
        k, n = shape
        ops.append({"name": nd.name, "role": role, "k": k, "n": n,
                    "wbytes": k * n * (1 if dtype in (2, 3) else 4)})
    return ops, model


def hidden_size():
    with open(CONFIG, encoding="utf-8") as f:
        c = json.load(f)
    return c["d_model"], c["decoder_layers"], c["decoder_attention_heads"]


def kv_cache_bytes(model, t, d_model, n_layers):
    """Bytes of cached K/V a with-past step reads, at the cache's own dtype.

    The decoder-side cache holds t-1 positions per layer, the encoder-side
    cache all 1500. Both are float32 in the optimum export: quantize_dynamic
    touches weights only, so the cache is read at four bytes per element.
    """
    past = [i for i in model.graph.input if i.name.startswith("past_key_values")]
    if not past:
        return 0, 0
    elem = 4
    dec = sum(1 for i in past if ".decoder." in i.name)
    enc = sum(1 for i in past if ".encoder." in i.name)
    return (dec * max(t - 1, 0) * d_model * elem,
            enc * ENCODER_POSITIONS * d_model * elem)


def per_step(ops, t, d_model, n_layers, model=None):
    out = {"t": t, "weight_bytes": 0, "act_bytes": 0, "macs": 0,
           "macs_cross_kv": 0, "by_role": {}}
    if model is not None:
        d, e = kv_cache_bytes(model, t, d_model, n_layers)
        out["kv_dec_bytes"], out["kv_enc_bytes"] = d, e
        out["act_bytes"] += d + e
    for op in ops:
        r = rows_for(op["role"], t)
        macs = r * op["k"] * op["n"]
        act = r * op["k"]                      # INT8 input rows
        out["weight_bytes"] += op["wbytes"]
        out["act_bytes"] += act
        out["macs"] += macs
        if op["role"] in ("encoder_attn.k_proj", "encoder_attn.v_proj"):
            out["macs_cross_kv"] += macs
        b = out["by_role"].setdefault(op["role"], {"w": 0, "act": 0, "macs": 0})
        b["w"] += op["wbytes"]
        b["act"] += act
        b["macs"] += macs
    # Attention products carry no weights: QK^T and attn x V.
    out["macs_attn_self"] = n_layers * 2 * t * t * d_model
    out["macs_attn_cross"] = n_layers * 2 * t * ENCODER_POSITIONS * d_model
    out["macs"] += out["macs_attn_self"] + out["macs_attn_cross"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    d_model, n_layers, _ = hidden_size()
    ops, model = load(args.model)
    inputs = [i.name for i in model.graph.input]
    has_past = any("past" in n for n in inputs)

    print("=" * 78)
    print(f"model   : {os.path.basename(args.model)}")
    print(f"inputs  : {inputs}")
    print(f"KV cache: {'yes' if has_past else 'NO -- every step recomputes'}")
    print(f"weighted matmuls classified: {len(ops)}")
    print("=" * 78)

    rows = [per_step(ops, t, d_model, n_layers, model) for t in POSITIONS]
    w = rows[0]["weight_bytes"] / MIB
    print(f"\nweight working set per step (INT8): {w:.1f} MiB  "
          f"= {w / BUDGET_MIB:.1f}x the {BUDGET_MIB:.1f} MiB budget")

    print(f"\n{'t':>4}{'weights MiB':>13}{'act MiB':>10}{'act/weights':>13}"
          f"{'GMAC':>9}{'cross-KV GMAC':>15}{'share':>8}")
    for r in rows:
        print(f"{r['t']:>4}{r['weight_bytes'] / MIB:>13.1f}"
              f"{r['act_bytes'] / MIB:>10.1f}"
              f"{r['act_bytes'] / r['weight_bytes']:>12.2f}x"
              f"{r['macs'] / 1e9:>9.1f}{r['macs_cross_kv'] / 1e9:>15.1f}"
              f"{100 * r['macs_cross_kv'] / r['macs']:>7.1f}%")

    print("\nper role at t = 30:")
    r30 = next(r for r in rows if r["t"] == 30)
    print(f"  {'role':<24}{'weights MiB':>13}{'act MiB':>10}{'GMAC':>9}")
    for role, b in sorted(r30["by_role"].items(),
                          key=lambda x: -x[1]["macs"]):
        print(f"  {role:<24}{b['w'] / MIB:>13.1f}{b['act'] / MIB:>10.1f}"
              f"{b['macs'] / 1e9:>9.1f}")
    print(f"  {'attention (self)':<24}{'':>13}{'':>10}"
          f"{r30['macs_attn_self'] / 1e9:>9.1f}")
    print(f"  {'attention (cross)':<24}{'':>13}{'':>10}"
          f"{r30['macs_attn_cross'] / 1e9:>9.1f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "has_past": has_past,
                   "budget_mib": BUDGET_MIB, "rows": rows}, f, indent=2)
    print(f"\nsaqlandi: {args.out}")


if __name__ == "__main__":
    main()
