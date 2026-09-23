"""E3 -- end-to-end response time and RTF with the with-past decoder.

The paper's Table 18 was measured with the deployed decoder, which carries no
KV cache; its best real-audio RTF was 1.09 at eight threads. This repeats the
same measurement -- the same six real recordings, feature extraction plus
encoder plus full greedy decode, one and eight threads -- with each decoder
variant, so the table can be extended by rows rather than replaced.

Arms are named by (encoder, decoder). The encoder is the paper's cascade
(tau = 0.99 + GPTQ) throughout; only the decoder changes:

  deployed     the no-cache int8 decoder of the paper
  with-past    KV cache, tied lm_head (dequantized per step)
  untied       KV cache, lm_head as int8 MatMulInteger

Timing scope is identical to rtf_endtoend.py in nnopt: the decoder time is
the sum of every decode step, including the first no-cache step of the cached
pipelines. Results append to a JSON keyed by arm and thread count.

Usage:  python experiments/rtf_with_past.py [--threads 1,8] [--arms a,b]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir("D:/DSc/ISH/nnopt")
sys.path.insert(0, "D:/DSc/ISH/nnopt/experiments")
from calib_utils import MODEL_DIR, TARGET_SR, load_audio  # noqa: E402
from wer_cer_whole_network import EOT, SOT, error_rate, normalize  # noqa: E402
from transformers import WhisperFeatureExtractor, WhisperTokenizer  # noqa: E402

EXPORT = os.path.join(HERE, "..", "models", "whisper_with_past")
ENCODER = "models/_gptq/enc_gptq_pruned.onnx"          # paper's cascade encoder
ARMS = {
    "deployed": ("models/_whole_net/dec_int8.onnx", None),
    "with-past": (os.path.join(EXPORT, "decoder_model_int8.onnx"),
                  os.path.join(EXPORT, "decoder_with_past_model_int8.onnx")),
    "untied": (os.path.join(EXPORT, "decoder_model_int8.onnx"),
               os.path.join(EXPORT, "decoder_with_past_untied_int8.onnx")),
}
OUT_JSON = os.path.join(HERE, "results_rtf_with_past.json")
N_UTT, SKIP = 6, 12          # the same six recordings as Table 18
MAX_NEW = 96
WINDOW_S = 30.0


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def decode_nocache(dec, enc, prompt):
    ids = [SOT, *prompt]
    t0 = time.perf_counter()
    for _ in range(MAX_NEW):
        logits = dec.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                                "encoder_hidden_states": enc})[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == EOT:
            break
        ids.append(nxt)
    return ids[1 + len(prompt):], (time.perf_counter() - t0) * 1000


def decode_cached(first, step, enc, prompt):
    ids = [SOT, *prompt]
    t0 = time.perf_counter()
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": enc})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    step_in = [i.name for i in step.get_inputs()]
    step_out = [o.name for o in step.get_outputs()]
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in step_in:
            if n.startswith("past_key_values"):
                feed[n] = present["present" + n[len("past_key_values"):]]
        res = step.run(None, feed)
        logits = res[step_out.index("logits")]
        for n, v in zip(step_out, res):
            if n.startswith("present"):
                present[n] = v
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):], (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    waves, texts = load_audio(SKIP, N_UTT)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt = [t for _, t in tok.get_decoder_prompt_ids(language="uz",
                                                       task="transcribe")]
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}

    for threads in [int(t) for t in args.threads.split(",")]:
        enc_sess = session(ENCODER, threads)
        for arm in args.arms.split(","):
            first_p, step_p = ARMS[arm]
            first = session(first_p, threads)
            step = session(step_p, threads) if step_p else None
            rows = []
            for wav, ref in zip(waves, texts):
                t0 = time.perf_counter()
                f = fe(wav, sampling_rate=TARGET_SR,
                       return_tensors="np").input_features.astype(np.float32)
                t_feat = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                enc = enc_sess.run(None, {"input_features": f})[0].astype(np.float32)
                t_enc = (time.perf_counter() - t0) * 1000
                if step is None:
                    ids, t_dec = decode_nocache(first, enc, prompt)
                else:
                    ids, t_dec = decode_cached(first, step, enc, prompt)
                hyp = normalize(tok.decode(ids, skip_special_tokens=True))
                rows.append({"audio_s": len(wav) / TARGET_SR, "tokens": len(ids),
                             "feature_ms": t_feat, "encoder_ms": t_enc,
                             "decoder_ms": t_dec,
                             "total_ms": t_feat + t_enc + t_dec,
                             "wer": error_rate(normalize(ref).split(), hyp.split())})
            del first, step
            tot = float(np.mean([r["total_ms"] for r in rows]))
            audio = float(np.mean([r["audio_s"] for r in rows]))
            summary = {
                "threads": threads, "n": N_UTT,
                "feature_ms": float(np.mean([r["feature_ms"] for r in rows])),
                "encoder_ms": float(np.mean([r["encoder_ms"] for r in rows])),
                "decoder_ms": float(np.mean([r["decoder_ms"] for r in rows])),
                "total_ms": tot, "audio_s": audio,
                "tokens": float(np.mean([r["tokens"] for r in rows])),
                "wer": float(np.mean([r["wer"] for r in rows])),
                "rtf_real_audio": tot / 1000 / audio,
                "rtf_window": tot / 1000 / WINDOW_S,
                "per_utterance": rows,
            }
            results[f"{arm}|{threads}"] = summary
            json.dump(results, open(OUT_JSON, "w"), indent=1)
            print(f"{arm:<10} {threads} thr  enc {summary['encoder_ms']:7.0f}  "
                  f"dec {summary['decoder_ms']:7.0f}  total {tot:7.0f} ms  "
                  f"RTF {summary['rtf_real_audio']:.2f} / {summary['rtf_window']:.2f}"
                  f"  WER {summary['wer']:.3f}", flush=True)
        del enc_sess
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
