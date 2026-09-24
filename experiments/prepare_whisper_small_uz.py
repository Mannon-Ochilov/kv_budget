"""Fourth model: navai-uz/whisper-small-uzbek (local checkpoint, fine-tuned from openai/whisper-small).
No download and no audio step: the Uzbek Common Voice caches already exist.

Original E7 prep notes -- a second model and an open English test set.

Everything so far is Whisper-medium fine-tuned for Uzbek on one CPU. A
reviewer's first question is whether the padding-sink effect (E5) and the
precision headroom (E1) are properties of that one model. This brings in
openai/whisper-small (12 decoder layers, d = 768, the same 1500-position
encoder cache) and LibriSpeech test-clean / dev-clean, in exactly the file
formats the existing scripts read:

  models/whisper_small_hf/            HF checkpoint
  models/whisper_small_onnx/          optimum export, with past
      decoder_model.onnx, decoder_with_past_model.onnx, encoder_model.onnx
      decoder_model_int8.onnx, decoder_with_past_untied_int8.onnx
  models/_calib_cache/ls_test_clean.npz    300 utterances, test-clean
  models/_calib_cache/ls_dev_clean.npz     100 utterances, dev-clean
                                           (same audio/lengths/texts layout)

Usage:  python experiments/prepare_whisper_small.py
"""

import io
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
HF = os.path.join(ROOT, "models", "whisper_small_uz_hf")
ONNX = os.path.join(ROOT, "models", "whisper_small_uz_onnx")
CACHE = os.path.join(ROOT, "models", "_calib_cache")
PY = sys.executable
SETS = [("test", "test", 300, "ls_test_clean.npz"),
        ("validation", "validation", 100, "ls_dev_clean.npz")]


def run(*cmd):
    print("$", " ".join(os.path.basename(c) if os.sep in c else c for c in cmd),
          flush=True)
    subprocess.run(cmd, check=True)


def checkpoint():
    if os.path.exists(os.path.join(HF, "model.safetensors")):
        return
    from huggingface_hub import snapshot_download
    snapshot_download("openai/whisper-small", local_dir=HF,
                      allow_patterns=["*.json", "*.txt", "model.safetensors"])


def export():
    if os.path.exists(os.path.join(ONNX, "decoder_with_past_model.onnx")):
        return
    run(PY, "-m", "optimum.exporters.onnx", "--model", HF,
        "--task", "automatic-speech-recognition-with-past", ONNX)


def quantize():
    q = os.path.join(HERE, "quantize_with_past.py")
    u = os.path.join(HERE, "untie_lm_head.py")
    first = os.path.join(ONNX, "decoder_model_int8.onnx")
    if not os.path.exists(first):
        run(PY, q, "--src", os.path.join(ONNX, "decoder_model.onnx"), "--dst", first)
    untied = os.path.join(ONNX, "decoder_with_past_untied.onnx")
    if not os.path.exists(untied):
        run(PY, u, "--src", os.path.join(ONNX, "decoder_with_past_model.onnx"),
            "--dst", untied)
    step = os.path.join(ONNX, "decoder_with_past_untied_int8.onnx")
    if not os.path.exists(step):
        run(PY, q, "--src", untied, "--dst", step)


def audio():
    from datasets import Audio, load_dataset
    os.makedirs(CACHE, exist_ok=True)
    for _, split, n, fname in SETS:
        out = os.path.join(CACHE, fname)
        if os.path.exists(out):
            continue
        # decode the FLAC bytes with soundfile; the datasets torchcodec
        # backend does not load on this machine
        ds = load_dataset("openslr/librispeech_asr", "clean", split=split,
                          streaming=True).cast_column("audio", Audio(decode=False))
        waves, texts = [], []
        for ex in ds:
            wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            assert sr == 16000
            waves.append(np.asarray(wav, dtype=np.float32))
            texts.append(ex["text"].lower())
            if len(waves) == n:
                break
        np.savez(out, audio=np.concatenate(waves),
                 lengths=np.array([len(w) for w in waves]),
                 texts=np.array(texts, dtype=object))
        print(f"  {fname}: {n} utterances, mean "
              f"{np.mean([len(w) for w in waves]) / 16000:.1f} s", flush=True)


if __name__ == "__main__":
    assert os.path.exists(os.path.join(HF, "model.safetensors")), HF
    export()
    quantize()
    print("tayyor")
