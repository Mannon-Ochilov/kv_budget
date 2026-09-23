"""E2 -- hold the cross-attention KV cache as int8 inside the graph.

E1 measures what int8 cache costs in WER with the quantization done outside
the graph. That says nothing about speed: the deployed decoder's lm_head
showed that a DequantizeLinear executed on every step can cost more than the
bytes it saves. So two in-graph variants are built and timed against each
other:

  --mode dq   cache inputs are int8 + per-head scale; a DequantizeLinear in
              front of each attention MatMul restores FP32. The naive
              implementation: fewer bytes read from memory, but a full
              dequantized copy written and read again every step.
  --mode int  the attention products consume int8 directly. Q (and later the
              probabilities P) are dynamically quantized to uint8 per step,
              multiplied by the int8 K (or V) with MatMulInteger, and the
              result rescaled. No FP32 copy of the cache is ever materialised.

Both take the untied with-past int8 decoder and replace, per layer, the two
cross-attention products QK^T and P V. The self-attention cache is left in
FP32: at t <= 96 it is under 6 MiB and not the term under study.

The rewrite discovers the nodes by following the `past_key_values.i.encoder.*`
inputs to their consumers rather than assuming names, and refuses to run if
the pattern it finds is not the one it expects. `--inspect` prints that
chain for layer 0 and exits.
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "models", "whisper_with_past",
                   "decoder_with_past_untied_int8.onnx")
N_LAYERS, HEADS = 24, 16


def consumers_of(graph, name):
    return [n for n in graph.node if name in n.input]


def chain(graph, tensor, depth=6):
    """Follow a tensor through single-consumer ops until a MatMul."""
    path = []
    cur = tensor
    for _ in range(depth):
        cs = consumers_of(graph, cur)
        if not cs:
            break
        n = cs[0]
        path.append(n)
        if n.op_type in ("MatMul", "MatMulInteger"):
            break
        cur = n.output[0]
    return path


def inspect(model):
    g = model.graph
    for kind in ("key", "value"):
        name = f"past_key_values.0.encoder.{kind}"
        print(f"\n{name}:")
        for n in chain(g, name):
            print(f"  -> {n.op_type:<12} {n.name}")
            if n.op_type == "Transpose":
                print("       perm =", [a.ints for a in n.attribute if a.name == "perm"][0])
            if n.op_type == "MatMul":
                print("       inputs =", list(n.input))


def find_products(g, i):
    """(matmul_qk, transpose_k, matmul_pv) for layer i, or raise."""
    k_chain = chain(g, f"past_key_values.{i}.encoder.key")
    v_chain = chain(g, f"past_key_values.{i}.encoder.value")
    tr = next((n for n in k_chain if n.op_type == "Transpose"), None)
    kmul = next((n for n in k_chain if n.op_type == "Mul"), None)
    qk = next((n for n in k_chain if n.op_type == "MatMul"), None)
    pv = next((n for n in v_chain if n.op_type == "MatMul"), None)
    if qk is None or pv is None:
        raise SystemExit(f"layer {i}: could not find attention products "
                         f"({[n.op_type for n in k_chain]} / "
                         f"{[n.op_type for n in v_chain]})")
    return qk, tr, kmul, pv


def add_input(g, name, dtype, shape):
    g.input.append(helper.make_tensor_value_info(name, dtype, shape))


def rewrite(model, mode):
    g = model.graph
    old_inputs = {i.name: i for i in g.input}
    for i in range(N_LAYERS):
        qk, tr, kmul, pv = find_products(g, i)
        for kind, mm, transpose in (("key", qk, tr), ("value", pv, None)):
            fp = f"past_key_values.{i}.encoder.{kind}"
            q_name, s_name = fp + "_int8", fp + "_scale"
            shape = [d.dim_param or d.dim_value
                     for d in old_inputs[fp].type.tensor_type.shape.dim]
            # replace the FP32 input with an int8 tensor and a per-head scale
            g.input.remove(old_inputs[fp])
            add_input(g, q_name, TensorProto.INT8, shape)
            # Per-axis DequantizeLinear in ORT insists on a 1-D scale of the
            # axis length; the int path multiplies by the scale and needs it
            # broadcastable, so each mode declares the shape it needs.
            add_input(g, s_name, TensorProto.FLOAT,
                      [HEADS] if mode == "dq" else [1, HEADS, 1, 1])

            if mode == "dq":
                # int8 -> FP32 right at the input; the rest of the graph is
                # untouched, so this isolates the cost of the dequantize.
                zp = numpy_helper.from_array(np.zeros((HEADS,), np.int8),
                                             fp + "_zp")
                g.initializer.append(zp)
                g.node.insert(0, helper.make_node(
                    "DequantizeLinear", [q_name, s_name, fp + "_zp"], [fp],
                    name=f"{fp}_dq", axis=1))
                continue

            # mode == "int": the product itself consumes int8. Every new node
            # is inserted at the position of the MatMul it replaces, so the
            # consumers that follow it in the list still find their inputs.
            new_nodes = []
            other = mm.input[0]
            k_in = q_name
            if transpose is not None and kind == "key":
                # keep the transpose, but on the int8 tensor
                perm = [a.ints for a in transpose.attribute if a.name == "perm"][0]
                g.node.remove(transpose)
                new_nodes.append(helper.make_node(
                    "Transpose", [q_name], [q_name + "_T"],
                    name=transpose.name + "_int8", perm=list(perm)))
                k_in = q_name + "_T"
            # dynamic uint8 quantization of the float operand (Q or P)
            scale_in = s_name
            if kind == "key" and kmul is not None:
                # fold the constant of the K-side Mul into the per-head scale
                # whichever operand is not the transposed K carries the
                # factor; it may be an initializer or a Constant node output
                k_side = transpose.output[0] if transpose is not None else fp
                const = next(x for x in kmul.input if x != k_side)
                folded = s_name + "_folded"
                new_nodes.append(helper.make_node(
                    "Mul", [s_name, const], [folded], name=kmul.name + "_fold"))
                g.node.remove(kmul)
                scale_in = folded
            oq, os_, oz = other + "_u8", other + "_s", other + "_z"
            acc = mm.output[0] + "_i32"
            new_nodes += [
                helper.make_node("DynamicQuantizeLinear", [other],
                                 [oq, os_, oz], name=f"{other}_dyq"),
                helper.make_node("MatMulInteger", [oq, k_in, oz], [acc],
                                 name=mm.name + "_int"),
                helper.make_node("Cast", [acc], [acc + "_f"],
                                 name=mm.name + "_cast", to=1),
                helper.make_node("Mul", [os_, scale_in], [mm.output[0] + "_sc"],
                                 name=mm.name + "_scale"),
                helper.make_node("Mul", [acc + "_f", mm.output[0] + "_sc"],
                                 [mm.output[0]], name=mm.name + "_rescale"),
            ]
            pos = list(g.node).index(mm)
            g.node.remove(mm)
            for k_, nd in enumerate(new_nodes):
                g.node.insert(pos + k_, nd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst")
    ap.add_argument("--mode", choices=["dq", "int"])
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    model = onnx.load(args.src, load_external_data=False)
    if args.inspect:
        inspect(model)
        return
    if not (args.dst and args.mode):
        raise SystemExit("--dst and --mode required")
    model = rewrite(model, args.mode)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(f"saved {args.dst}  ({os.path.getsize(args.dst) / 1024 ** 2:.1f} MiB)")


if __name__ == "__main__":
    main()
