"""System-level composition: static encoder compression x dynamic decoder KV.

The weight paper compresses the encoder's static working set (tau-based
redundancy removal + INT8 + low-rank, cache-anchored); this paper compresses
the decoder's runtime working set (the cross-attention cache). They act on
different states of the same model, so the natural question is whether they
compose. Four configurations, Whisper-medium/uz:

  A  FP32 encoder        full FP32 cache
  B  cascade encoder     full FP32 cache          (the weight paper)
  C  FP32 encoder        calibrated split, FP32   (this paper's retention)
  D  cascade encoder     calibrated split, FP32   (both)

WER on the 300 test utterances (A, B, D exist in earlier result files; C is
decoded here), and end-to-end timing on the same six real recordings as
E3/Table 18 (feature extraction + encoder + full greedy decode), 1 and 8
threads. The decoder KV arm is the retention rule in FP32 because that is
the configuration whose latency ONNX Runtime can execute today; the int4
column is WER-only and is reported in the paper's Table 6.

Usage:  python experiments/system_composition.py --phase wer|timing|both
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from eviction_budget import keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, NNOPT, SEED, SETUPS, SOT, error_rate,
                   paired_ci, prompt_ids, real_positions, session, text_norm,
                   with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "results_system_composition.json")
ENCODERS = {"fp32": os.path.join(NNOPT, "models", "uzbek_stt_v1_onnx", "encoder_model.onnx"),
            "cascade": os.path.join(NNOPT, "models", "_gptq", "enc_gptq_pruned.onnx")}
FP32_STATES = os.path.join(ROOT, "models", "enc_states_test300_fp32.npy")   # E1's memmap
CONFIGS = [("A", "fp32", "full"), ("B", "cascade", "full"),
           ("C", "fp32", "split"), ("D", "cascade", "split")]
N_UTT, SKIP = 6, 12
WINDOW_S = 30.0


def rule():
    e6 = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_uz.json")))
    return e6["choice"]["rule"][1], e6["choice"]["rule"][2]


def greedy_timed(first, step, enc, prompt, keep, n_layers):
    """decode_cached of rtf_with_past.py with optional retention; returns ids, ms, kept."""
    t0 = time.perf_counter()
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc)})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    kept = ENC_POS
    if keep is not None:
        mass = np.zeros(ENC_POS)
        for n, v in zip(names, out):
            if "encoder_attn" in n and n.endswith("Softmax_output_0"):
                mass += v.sum(axis=(0, 1, 2))
        idx = keep(mass)
        kept = len(idx)
        for i in range(n_layers):
            for kn in (f"present.{i}.encoder.key", f"present.{i}.encoder.value"):
                present[kn] = np.ascontiguousarray(present[kn][:, :, idx, :])
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
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):], (time.perf_counter() - t0) * 1000, kept


def phase_wer(res, setup, tok, prompt, norm):
    """Config C: FP32 encoder states + split rule, 300 test utterances."""
    if "wer" in res and "C" in res["wer"]:
        return
    sys.path.insert(0, os.path.join(NNOPT, "experiments"))
    from calib_utils import SPLIT_CACHE, load_audio
    os.chdir(NNOPT)
    waves, texts = load_audio(0, 300, source=SPLIT_CACHE["test"])
    states = np.load(FP32_STATES, mmap_mode="r")
    refs = [norm(t).split() for t in texts]
    f_r, f_p = rule()
    first, step = session(with_attention(setup.first)), session(setup.step)
    wers, kept = [], []
    t0 = time.time()
    for i in range(300):
        ids, _, k = greedy_timed(first, step, states[i][None], prompt,
                                 keep_split(f_r, f_p, real_positions(waves[i])), setup.n_layers)
        hyp = norm(tok.decode(ids, skip_special_tokens=True))
        wers.append(error_rate(refs[i], hyp.split()))
        kept.append(k)
        if (i + 1) % 100 == 0:
            print(f"  C {i + 1}/300  WER {np.mean(wers):.4f}  [{time.time() - t0:.0f}s]", flush=True)
    # A from E1 (fp32 tag), B and D from the E6 test run
    e1 = json.load(open(os.path.join(HERE, "results_cache_precision_wer_fp32.json")))["schemes"]["fp32"]
    e6 = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_uz.json")))["test"]
    sk = [k for k in e6 if k.startswith("split")][0]
    res["wer"] = {"A": {"wer": e1["wer"], "per_sample_wer": e1["per_sample_wer"], "kept": 1500},
                  "B": {"wer": e6["full"]["wer"], "per_sample_wer": e6["full"]["per_sample_wer"], "kept": 1500},
                  "C": {"wer": float(np.mean(wers)), "per_sample_wer": wers, "kept": float(np.mean(kept))},
                  "D": {"wer": e6[sk]["wer"], "per_sample_wer": e6[sk]["per_sample_wer"], "kept": e6[sk]["kept"]}}
    rng = np.random.default_rng(SEED)
    base = res["wer"]["A"]["per_sample_wer"]
    for k, r in res["wer"].items():
        r["delta_vs_A"] = list(paired_ci(r["per_sample_wer"], base, rng))
    json.dump(res, open(OUT, "w"), indent=1)
    for k, r in res["wer"].items():
        d = r["delta_vs_A"]
        print(f"  {k}: WER {r['wer']:.4f}  vs A {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")


def phase_timing(res, setup, tok, prompt, norm, threads_list):
    from transformers import WhisperFeatureExtractor
    sys.path.insert(0, os.path.join(NNOPT, "experiments"))
    from calib_utils import load_audio
    os.chdir(NNOPT)
    waves, texts = load_audio(SKIP, N_UTT)
    fe = WhisperFeatureExtractor.from_pretrained(setup.hf_dir)
    f_r, f_p = rule()
    tm = res.setdefault("timing", {})
    for threads in threads_list:
        encs = {k: session(p, threads) for k, p in ENCODERS.items()}
        first, step = session(with_attention(setup.first), threads), session(setup.step, threads)
        for cfg, enc_name, kv in CONFIGS:
            key = f"{cfg}|{threads}"
            if key in tm:
                continue
            rows = []
            for wav, ref in zip(waves, texts):
                t0 = time.perf_counter()
                f = fe(wav, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
                t_feat = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                enc = encs[enc_name].run(None, {"input_features": f})[0].astype(np.float32)
                t_enc = (time.perf_counter() - t0) * 1000
                keep = keep_split(f_r, f_p, real_positions(wav)) if kv == "split" else None
                ids, t_dec, kept = greedy_timed(first, step, enc, prompt, keep, setup.n_layers)
                hyp = norm(tok.decode(ids, skip_special_tokens=True))
                rows.append({"audio_s": len(wav) / 16000, "tokens": len(ids), "kept": kept,
                             "feature_ms": t_feat, "encoder_ms": t_enc, "decoder_ms": t_dec,
                             "total_ms": t_feat + t_enc + t_dec,
                             "wer": error_rate(norm(ref).split(), hyp.split())})
            tot = float(np.mean([r["total_ms"] for r in rows]))
            audio = float(np.mean([r["audio_s"] for r in rows]))
            tm[key] = {"config": cfg, "encoder": enc_name, "kv": kv, "threads": threads,
                       "feature_ms": float(np.mean([r["feature_ms"] for r in rows])),
                       "encoder_ms": float(np.mean([r["encoder_ms"] for r in rows])),
                       "decoder_ms": float(np.mean([r["decoder_ms"] for r in rows])),
                       "total_ms": tot, "audio_s": audio,
                       "tokens": float(np.mean([r["tokens"] for r in rows])),
                       "kept": float(np.mean([r["kept"] for r in rows])),
                       "wer6": float(np.mean([r["wer"] for r in rows])),
                       "rtf_real_audio": tot / 1000 / audio, "rtf_window": tot / 1000 / WINDOW_S,
                       "per_utterance": rows}
            json.dump(res, open(OUT, "w"), indent=1)
            s = tm[key]
            print(f"  {cfg} ({enc_name:7s}, {kv:5s}) {threads} thr  enc {s['encoder_ms']:6.0f}  "
                  f"dec {s['decoder_ms']:6.0f}  total {tot:6.0f} ms  RTF {s['rtf_real_audio']:.2f}",
                  flush=True)
        del encs, first, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=["wer", "timing", "both"])
    ap.add_argument("--threads", default="1,8")
    args = ap.parse_args()
    setup = SETUPS["medium_uz"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tok, prompt = prompt_ids(setup)
    norm = text_norm(setup)
    if args.phase in ("wer", "both"):
        phase_wer(res, setup, tok, prompt, norm)
    if args.phase in ("timing", "both"):
        phase_timing(res, setup, tok, prompt, norm, [int(t) for t in args.threads.split(",")])
    print(f"\nsaqlandi: {OUT}")


if __name__ == "__main__":
    main()
