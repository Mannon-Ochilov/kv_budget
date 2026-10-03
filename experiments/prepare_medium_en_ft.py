"""R6 -- does fine-tuning make Whisper-medium's cross-attention compact enough
for PadSink-Track? Two English fine-tunes of openai/whisper-medium.

The 2x2 design of the paper confounds language with fine-tuning: both Uzbek
models are fine-tuned, both English models are original checkpoints, and only
the original medium fails (Table 6). medium_uz is itself a fine-tune of
openai/whisper-medium with the same six alignment heads, yet the share of
the current step's attention that falls on padding outside the sink and the
window (pad_out, track_coverage.py) is 17.3 % there against 34.8 % on the
original (all mass outside the tracked set: 23.2 % against 41.3 %). These two checkpoints hold the language fixed (English, the same
LibriSpeech subsets as medium_en) and vary only the fine-tuning:

  medium_en_ftk  Veronica1NW/en_whisper_nonstandard_medium (Apache-2.0):
                 Common Voice 17 + Kenyan English, accented speech
  medium_en_ftm  santhosh643/whisper-medium-english (Apache-2.0):
                 medical speech, 600 steps -- a light fine-tune

Protocol, unchanged from the four paper models: the split rule (f_r, f_p) ->
K_i is calibrated on the 100 dev-clean utterances (eviction_budget.py), rho
is the value with the lowest validation WER of PadSink-KV among
{0.8, 0.9, 0.95} (spar.py --phase valid; ties -> 0.9), the Track settings
(back 0.1, sink cap 0.25, alignment heads of openai/whisper-medium) are the
frozen ones, and the verdict is the same paired-bootstrap gate on the 300
test-clean utterances at eps = 0.20.

Pre-registered predictions (written before any of these models was run):
  F1  on both fine-tunes pad_out (track_coverage.py, 40 test utterances,
      0.5 K_i) is below the original medium's 34.8 %, and lower for the
      Kenyan fine-tune than for the light medical one;
  F2  PadSink-Track at 0.5 K_i passes the gate on a fine-tune whose pad_out
      is <= 25 %, and fails where it stays near the original's.
  Refutation: if both fine-tunes fail like the original, fine-tuning is not
  what makes the Uzbek medium work, and the paper says so.
  F3  (added after the Kenyan calibration showed a higher dev-clean WER,
      0.069 against 0.029 for the original, before any Track result) --
      a fine-tune with a higher WER gets a wider relative gate, so a pass
      could come from the margin alone. The Track result is therefore also
      read against the original medium's absolute delta (0.0073): F2 counts
      as supported only if the upper CI bound is below that as well, or the
      result is reported as a pass at the model's own margin only.

Usage:  python experiments/prepare_medium_en_ft.py --model ftk|ftm
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PY = sys.executable
MODELS = {
    "ftk": ("Veronica1NW/en_whisper_nonstandard_medium", "whisper_medium_en_ft_kenyan"),
    "ftm": ("santhosh643/whisper-medium-english", "whisper_medium_en_ft_medical"),
}
BASE_HF = os.path.join(ROOT, "models", "whisper_medium_en_hf")


def run(*cmd):
    print("$", " ".join(os.path.basename(c) if os.sep in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def checkpoint(repo, hf):
    if not os.path.exists(os.path.join(hf, "model.safetensors")):
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=hf, allow_patterns=["*.json", "*.txt", "model.safetensors"])
    # Tokenizer, feature extractor and alignment heads come from the base
    # checkpoint, as for medium_uz: the fine-tunes do not change the vocabulary
    # and the alignment heads are a property of the base model.
    import shutil
    for f in ("generation_config.json", "preprocessor_config.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json",
              "added_tokens.json", "special_tokens_map.json"):
        if not os.path.exists(os.path.join(hf, f)):
            shutil.copy(os.path.join(BASE_HF, f), hf)
            print(f"  {f} taken from openai/whisper-medium", flush=True)


def differs_from_base(hf):
    """Some hub uploads are copies of the base; report how far the weights moved."""
    from safetensors import safe_open
    out = {}
    with safe_open(os.path.join(hf, "model.safetensors"), "np") as a, \
            safe_open(os.path.join(BASE_HF, "model.safetensors"), "np") as b:
        for part in ("encoder", "decoder"):
            num = den = 0.0
            for k in a.keys():
                if not k.startswith(f"model.{part}.") or k not in b.keys():
                    continue
                x, y = a.get_tensor(k).astype(np.float64), b.get_tensor(k).astype(np.float64)
                num += float(((x - y) ** 2).sum())
                den += float((y ** 2).sum())
            out[part] = (num / den) ** 0.5
    print(f"  relative weight change vs openai/whisper-medium: encoder {out['encoder']:.4f}, "
          f"decoder {out['decoder']:.4f}", flush=True)
    if min(out.values()) < 1e-6:
        sys.exit("weights identical to the base checkpoint -- not a fine-tune")
    return out


def export(hf, onnx):
    if os.path.exists(os.path.join(onnx, "decoder_with_past_model.onnx")):
        return
    run(PY, "-m", "optimum.exporters.onnx", "--model", hf,
        "--task", "automatic-speech-recognition-with-past", onnx)


def quantize(onnx):
    q = os.path.join(HERE, "quantize_with_past.py")
    u = os.path.join(HERE, "untie_lm_head.py")
    first = os.path.join(onnx, "decoder_model_int8.onnx")
    if not os.path.exists(first):
        run(PY, q, "--src", os.path.join(onnx, "decoder_model.onnx"), "--dst", first)
    untied = os.path.join(onnx, "decoder_with_past_untied.onnx")
    if not os.path.exists(untied):
        run(PY, u, "--src", os.path.join(onnx, "decoder_with_past_model.onnx"), "--dst", untied)
    step = os.path.join(onnx, "decoder_with_past_untied_int8.onnx")
    if not os.path.exists(step):
        run(PY, q, "--src", untied, "--dst", step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    repo, name = MODELS[args.model]
    hf = os.path.join(ROOT, "models", name + "_hf")
    onnx = os.path.join(ROOT, "models", name + "_onnx")
    checkpoint(repo, hf)
    differs_from_base(hf)
    export(hf, onnx)
    quantize(onnx)
    print("tayyor", flush=True)


if __name__ == "__main__":
    main()
