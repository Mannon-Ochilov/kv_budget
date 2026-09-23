"""Rebuild the Whisper-medium/uz decoders from the public checkpoint.

Same pipeline as the paper's models/whisper_with_past: optimum export with
past, ONNX Runtime dynamic INT8 for the first-step decoder, untie lm_head on
the with-past decoder and quantize it. The paper's cascade-compressed
encoder is not public; the FP32 encoder from the export is used instead
(the decoder-side measurements do not depend on it).

  models/whisper_medium_hf/            HF checkpoint (Kotib/uzbek_stt_v1)
  models/whisper_with_past/            decoder_model.onnx, decoder_model_int8.onnx,
                                       decoder_with_past_model.onnx,
                                       decoder_with_past_untied.onnx,
                                       decoder_with_past_untied_int8.onnx,
                                       encoder_model.onnx

Usage:  python experiments/prepare_medium_uz.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
HF = os.path.join(ROOT, "models", "whisper_medium_hf")
ONNX = os.path.join(ROOT, "models", "whisper_with_past")
PY = sys.executable
REPO = "Kotib/uzbek_stt_v1"


def run(*cmd):
    print("$", " ".join(os.path.basename(c) if os.sep in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    if not os.path.exists(os.path.join(HF, "config.json")):
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, local_dir=HF,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"])
    if not os.path.exists(os.path.join(ONNX, "decoder_with_past_model.onnx")):
        run(PY, "-m", "optimum.exporters.onnx", "--model", HF,
            "--task", "automatic-speech-recognition-with-past", ONNX)
    q = os.path.join(HERE, "quantize_with_past.py")
    u = os.path.join(HERE, "untie_lm_head.py")
    first = os.path.join(ONNX, "decoder_model_int8.onnx")
    if not os.path.exists(first):
        run(PY, q, "--src", os.path.join(ONNX, "decoder_model.onnx"), "--dst", first)
    untied = os.path.join(ONNX, "decoder_with_past_untied.onnx")
    if not os.path.exists(untied):
        run(PY, u, "--src", os.path.join(ONNX, "decoder_with_past_model.onnx"), "--dst", untied)
    step = os.path.join(ONNX, "decoder_with_past_untied_int8.onnx")
    if not os.path.exists(step):
        run(PY, q, "--src", untied, "--dst", step)
    print("tayyor:", ONNX)


if __name__ == "__main__":
    main()
