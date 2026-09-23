"""INT8 for the with-past decoder, with the same recipe as dec_int8.onnx.

Dynamic per-tensor weight quantization to QInt8 through onnxruntime, exactly
as the deployed decoder was produced, so the only difference between the two
INT8 decoders is the KV cache. Applied to whichever FP32 decoder is named.

Usage:  python experiments/quantize_with_past.py --src X.onnx --dst X_int8.onnx
"""

import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    quantize_dynamic(args.src, args.dst, weight_type=QuantType.QInt8)
    print(f"{os.path.basename(args.dst)}: "
          f"{os.path.getsize(args.dst) / 1024 ** 2:.1f} MiB")


if __name__ == "__main__":
    main()
