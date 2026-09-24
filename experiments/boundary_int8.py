"""A1 -- the least-compressed EXECUTABLE point at the budget boundary.

Medium, alpha = 0.7, 24 MiB L3: the KV budget is L * (alpha*M_L3 - M_w) =
24 * (16.8 - 14.0) = 67.2 MiB. The executable int8 per-head cache is 70.3
MiB at full retention, so it fits once 1433 of the 1500 positions are kept
(r = 0.956). The hardware constraint holds per step, so the slot cap is the
same for every utterance: K = 1433. Retention: all audio positions plus the
padding positions with the highest first-step mass up to K (the audio never
exceeds 525 positions here).

WER on the 300 test utterances paired against the full FP32 cache, and the
decoder step at t = 30 for the int8-cache graph at 1500 and 1433 positions
plus the FP32 cache at 1500, interleaved, 7 rounds, one thread.

Usage:  python experiments/boundary_int8.py
"""

import json
import os
import time

import numpy as np

from cache_precision_wer import SCHEMES
from eviction_budget import EPS
from kvlib import (ENC_POS, SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)
from step_latency_evicted import feed_for
from step_latency_evicted import session as step_session

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
L, M_L3, ALPHA, W_LAYER = 24, 24.0, 0.7, 14.00


def main():
    setup = SETUPS["medium_uz"]
    budget = L * (ALPHA * M_L3 - W_LAYER)
    per_pos = cache_mib(setup, 1, 8)
    K = int(budget // per_pos)
    print(f"KV budget {budget:.1f} MiB, int8 {per_pos * ENC_POS:.1f} MiB at 1500 -> K = {K} "
          f"({cache_mib(setup, K, 8):.2f} MiB)")
    full = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_uz.json")))["test"]["full"]
    out_json = os.path.join(HERE, "results_boundary_int8_medium_uz.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}

    if "wer" not in res:
        states, waves, texts = encoder_states(setup, "test", 300)
        norm = text_norm(setup)
        refs = [norm(t).split() for t in texts]
        tok, prompt = prompt_ids(setup)
        first, step = session(with_attention(setup.first)), session(setup.step)
        wers, t0 = [], time.time()
        for i in range(300):
            rn = real_positions(waves[i])
            def keep(m, _mass_l=None, rn=rn):
                return np.sort(np.concatenate([np.arange(rn), rn + np.argsort(-m[rn:])[:K - rn]]))
            ids, k = greedy(setup, first, step, states[i], prompt, keep, SCHEMES["int8_head"])
            assert k == K
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/300  WER {np.mean(wers):.4f}  [{time.time() - t0:.0f}s]", flush=True)
        d = paired_ci(wers, full["per_sample_wer"], np.random.default_rng(SEED))
        delta = round(full["wer"] * EPS, 4)
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        res.update({"K": K, "budget_mib": budget, "mib": cache_mib(setup, K, 8), "wer": float(np.mean(wers)),
                    "per_sample_wer": wers, "delta_vs_full": list(d), "gate": v})
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"int8 head, K = {K}: WER {np.mean(wers):.4f}  vs full {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}")

    med = os.path.join(ROOT, "models", "whisper_with_past")
    arms = [("fp32 cache, 1500", f"{med}/decoder_with_past_untied_int8.onnx", ENC_POS),
            ("int8 cache, 1500", f"{med}/decoder_with_past_cache_int.onnx", ENC_POS),
            (f"int8 cache, {K}", f"{med}/decoder_with_past_cache_int.onnx", K),
            ("int8 cache, 393", f"{med}/decoder_with_past_cache_int.onnx", 393)]
    rng = np.random.default_rng(0)
    sess = {}
    for lab, p, k in arms:
        s = sess.setdefault(p, step_session(p))
        for _ in range(3):
            s.run(None, feed_for(s, 30, 16, k, rng))
    times = {a[0]: [] for a in arms}
    for _ in range(21):
        for lab, p, k in arms:
            s = sess[p]
            f = feed_for(s, 30, 16, k, rng)
            t0 = time.perf_counter()
            s.run(None, f)
            times[lab].append((time.perf_counter() - t0) * 1e3)
    res["latency"] = {lab: {"median_ms": float(np.median(v)), "min_ms": float(min(v)), "max_ms": float(max(v))}
                      for lab, v in times.items()}
    json.dump(res, open(out_json, "w"), indent=1)
    for lab, v in res["latency"].items():
        print(f"  step {lab:<18} {v['median_ms']:6.1f} ms  [{v['min_ms']:.1f}-{v['max_ms']:.1f}]")
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
