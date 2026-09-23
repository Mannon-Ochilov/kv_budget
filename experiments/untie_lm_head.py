"""Give the vocabulary projection its own weight, so quantization can reach it.

In the exported decoder the output projection is tied to the input embedding:
`/proj_out/MatMul` multiplies by a Transpose of `embed_tokens.weight` rather
than by an initializer of its own. `quantize_dynamic` only converts MatMuls
whose weight is an initializer, so proj_out stayed a float MatMul and the
tied table was handled by inserting a DequantizeLinear in front of it. ONNX
Runtime folds the Transpose but not the DequantizeLinear, so at run time the
51865 x 1024 table is expanded to FP32 on every decode step -- 10.5% of the
step at t = 30, and a 203 MiB write-then-read that the weight-only model does
not count.

Materialising the transposed table as a separate initializer costs 202 MiB
of duplicated FP32 on disk before quantization and nothing after it (the
quantizer then stores it once as int8). With that in place quantize_dynamic
turns proj_out into a MatMulInteger like every other projection, and both
the Transpose and the DequantizeLinear disappear from the graph.

Run this on the FP32 decoder, then quantize:

    python experiments/untie_lm_head.py --src decoder_with_past_model.onnx \
                                        --dst decoder_with_past_untied.onnx
"""

import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    model = onnx.load(args.src)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    producers = {o: n for n in g.node for o in n.output}

    proj = next((n for n in g.node if n.op_type == "MatMul"
                 and "proj_out" in n.name), None)
    if proj is None:
        raise SystemExit("no /proj_out/MatMul in graph")
    w_in = proj.input[1]
    src = producers.get(w_in)
    if src is None or src.op_type != "Transpose":
        raise SystemExit(f"proj_out weight is {src.op_type if src else 'an initializer'}, "
                         "not a Transpose of the embedding -- nothing to untie")
    emb_name = src.input[0]
    if emb_name not in inits:
        raise SystemExit(f"{emb_name} is not an initializer")

    emb = numpy_helper.to_array(inits[emb_name])          # (vocab, d_model)
    perm = next((a.ints for a in src.attribute if a.name == "perm"), [1, 0])
    untied = np.ascontiguousarray(np.transpose(emb, perm))
    new_name = "proj_out.weight"
    g.initializer.append(numpy_helper.from_array(untied, new_name))
    proj.input[1] = new_name

    # The Transpose is now dead; drop it if nothing else consumes it.
    if not any(w_in in n.input for n in g.node):
        g.node.remove(src)

    # Stays under the 2 GB protobuf limit (1.75 GB), so the file is written
    # self-contained; external data would drag in relative-location rules
    # that the checker tripped over on the first attempt.
    onnx.save(model, args.dst)
    print(f"untied: {emb_name} {emb.shape} -> {new_name} {untied.shape}")
    print(f"saved: {args.dst}")


if __name__ == "__main__":
    main()
