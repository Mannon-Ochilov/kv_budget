"""Does the with-past decoder produce the same transcript as the deployed one?

A faster decoder is worth nothing if it decodes differently. Both pipelines
are run greedily on the same evaluation utterances and their token sequences
compared. The with-past pipeline takes its first step through
decoder_model.onnx, which returns the initial `present.*` cache, and every
later step through decoder_with_past_model.onnx; the deployed pipeline is the
no-cache decoder re-run on the growing prefix, as in wer_cer_whole_network.

Both arms use the same precision (FP32 by default) so that what is compared
is the export, not the quantization. Token ids are compared exactly, and WER
is reported for both against the reference transcript.

Usage:  python experiments/verify_with_past.py [--n 6] [--int8]
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir("D:/DSc/ISH/nnopt")                 # calib_utils uses relative paths
sys.path.insert(0, "D:/DSc/ISH/nnopt/experiments")
from calib_utils import MODEL_DIR, TARGET_SR, load_audio  # noqa: E402
from wer_cer_whole_network import (EOT, SOT, error_rate,  # noqa: E402
                                   greedy_decode, normalize)
from transformers import WhisperFeatureExtractor, WhisperTokenizer  # noqa: E402

EXPORT = os.path.join(HERE, "..", "models", "whisper_with_past")
ENCODER_FP32 = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_FP32 = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
MAX_NEW = 96


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def greedy_with_past(first, step, enc, prompt_ids):
    """Greedy decode: one no-cache step, then cached steps."""
    ids = [SOT, *prompt_ids]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": enc})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    nxt = int(np.argmax(logits[0, -1]))
    step_in = [i.name for i in step.get_inputs()]
    step_out = [o.name for o in step.get_outputs()]
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in step_in:
            if n.startswith("past_key_values"):
                feed[n] = present["present" + n[len("past_key_values"):]]
            elif n == "encoder_hidden_states":
                feed[n] = enc
            elif n == "use_cache_branch":
                feed[n] = np.array([True])
        res = step.run(None, feed)
        logits = res[step_out.index("logits")]
        # A with-past export may omit the encoder-side cache from its outputs
        # because it never changes; keep the previous tensor in that case.
        for n, v in zip(step_out, res):
            if n.startswith("present"):
                present[n] = v
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt_ids):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--int8", action="store_true",
                    help="compare the INT8 decoders instead of FP32")
    args = ap.parse_args()

    if args.int8:
        base = session(DECODER_INT8)
        first = session(os.path.join(EXPORT, "decoder_model_int8.onnx"))
        step = session(os.path.join(EXPORT, "decoder_with_past_model_int8.onnx"))
    else:
        base = session(DECODER_FP32)
        first = session(os.path.join(EXPORT, "decoder_model.onnx"))
        step = session(os.path.join(EXPORT, "decoder_with_past_model.onnx"))

    enc_sess = session(ENCODER_FP32)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt = [t for _, t in tok.get_decoder_prompt_ids(language="uz",
                                                       task="transcribe")]
    waves, texts = load_audio(12, args.n)

    same, wer_base, wer_past = 0, [], []
    for wav, ref in zip(waves, texts):
        f = fe(wav, sampling_rate=TARGET_SR,
               return_tensors="np").input_features.astype(np.float32)
        enc = enc_sess.run(None, {"input_features": f})[0].astype(np.float32)
        a = greedy_decode(base, enc, prompt)
        b = greedy_with_past(first, step, enc, prompt)
        same += a == b
        rn = normalize(ref).split()
        wer_base.append(error_rate(rn, normalize(tok.decode(a, skip_special_tokens=True)).split()))
        wer_past.append(error_rate(rn, normalize(tok.decode(b, skip_special_tokens=True)).split()))
        print(f"  {'same' if a == b else 'DIFF'}  "
              f"len {len(a):>3}/{len(b):<3}  "
              f"WER {wer_base[-1]:.3f}/{wer_past[-1]:.3f}", flush=True)

    print(f"\nidentical token sequences: {same}/{args.n}")
    print(f"mean WER  no-cache {np.mean(wer_base):.4f}   "
          f"with-past {np.mean(wer_past):.4f}")


if __name__ == "__main__":
    main()
