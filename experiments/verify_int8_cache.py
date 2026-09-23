"""Do the in-graph int8-cache decoders decode the same as the FP32-cache one?

The `dq` and `int` graphs from int8_cache_graph.py take the cross-attention
cache as int8 plus a per-head scale. This feeds all three variants the same
cache -- quantized once, symmetric per head, the same grouping E1 measures as
`int8_head` -- and compares their greedy transcripts to the FP32-cache
decoder on a few evaluation utterances.

`dq` should match `int8_head` exactly, since it dequantizes the very tensor
E1 dequantizes in NumPy. `int` additionally quantizes Q and P dynamically to
uint8 per step, so small differences from `dq` are expected there; what is
not acceptable is a transcript that diverges. Both are reported.

Usage:  python experiments/verify_int8_cache.py [--n 3]
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir("D:/DSc/ISH/nnopt")
sys.path.insert(0, "D:/DSc/ISH/nnopt/experiments")
from calib_utils import MODEL_DIR, TARGET_SR, load_audio  # noqa: E402
from wer_cer_whole_network import EOT, SOT, error_rate, normalize  # noqa: E402
from transformers import WhisperFeatureExtractor, WhisperTokenizer  # noqa: E402

EXPORT = os.path.join(HERE, "..", "models", "whisper_with_past")
ENCODER = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
FIRST = os.path.join(EXPORT, "decoder_model_int8.onnx")
STEPS = {
    "fp32-cache": os.path.join(EXPORT, "decoder_with_past_untied_int8.onnx"),
    "dq": os.path.join(EXPORT, "decoder_with_past_cache_dq.onnx"),
    "int": os.path.join(EXPORT, "decoder_with_past_cache_int.onnx"),
}
N_LAYERS, HEADS = 24, 16
MAX_NEW = 96


def session(path, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def q_head(x):
    """Symmetric int8 per head: returns (int8 tensor, scale [1,H,1,1])."""
    s = np.max(np.abs(x), axis=(2, 3), keepdims=True) / 127.0
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    q = np.clip(np.round(x / s), -127, 127).astype(np.int8)
    return q, s


def decode(first, step, enc, prompt, int8_cache):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": enc})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    # Fixed cross-attention cache for the whole utterance.
    cross = {}
    for i in range(N_LAYERS):
        for kind in ("key", "value"):
            src = f"present.{i}.encoder.{kind}"
            dst = f"past_key_values.{i}.encoder.{kind}"
            if int8_cache:
                q, s = q_head(present[src])
                cross[dst + "_int8"], cross[dst + "_scale"] = q, s
            else:
                cross[dst] = present[src]
    step_in = [i.name for i in step.get_inputs()]
    step_out = [o.name for o in step.get_outputs()]
    # the dq graph declares 1-D scales, the int graph [1, H, 1, 1]
    declared = {i.name: [d if isinstance(d, int) else -1 for d in i.shape]
                for i in step.get_inputs() if i.name.endswith("_scale")}
    for n in list(cross):
        if n in declared:
            cross[n] = np.ascontiguousarray(cross[n].reshape(declared[n]))
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in step_in:
            if n in cross:
                feed[n] = cross[n]
            elif n.startswith("past_key_values"):
                feed[n] = present["present" + n[len("past_key_values"):]]
        res = step.run(None, feed)
        logits = res[step_out.index("logits")]
        for n, v in zip(step_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    waves, texts = load_audio(12, args.n)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt = [t for _, t in tok.get_decoder_prompt_ids(language="uz",
                                                       task="transcribe")]
    enc_sess, first = session(ENCODER), session(FIRST)
    steps = {k: session(p) for k, p in STEPS.items() if os.path.exists(p)}

    encs = []
    for wav in waves:
        f = fe(wav, sampling_rate=TARGET_SR,
               return_tensors="np").input_features.astype(np.float32)
        encs.append(enc_sess.run(None, {"input_features": f})[0].astype(np.float32))
    del enc_sess

    outs = {k: [decode(first, s, e, prompt, k != "fp32-cache")
                for e in encs] for k, s in steps.items()}
    base = outs["fp32-cache"]
    for k, seqs in outs.items():
        same = sum(a == b for a, b in zip(seqs, base))
        wer = np.mean([error_rate(normalize(r).split(),
                                  normalize(tok.decode(s, skip_special_tokens=True)).split())
                       for s, r in zip(seqs, texts)])
        print(f"  {k:<11} identical to fp32-cache: {same}/{args.n}   "
              f"WER {wer:.4f}   lengths {[len(s) for s in seqs]}")


if __name__ == "__main__":
    main()
