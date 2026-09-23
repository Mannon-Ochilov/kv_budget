"""E11 -- long-form Uzbek audio: does H2O collapse on medium too?

Common Voice clips are at most ~10 s (test max 10.5 s), so the medium
model was never tested where padding is scarce. LibriSpeech gave that
regime for whisper-small and there H2O's global ranking collapsed while the
split rule held. To test the same on medium without leaving the Uzbek
domain, 100 composite recordings of 15-30 s are built by concatenating 3-5
unused test utterances (indices 300-599, never used in E1-E10) with 0.3 s of
silence between them; the reference is the concatenated transcript. The
composites are stated as such in the paper.

Arms: full FP32; calibrated split (1.0, 0.1) FP32; H2O at the split rule's
per-utterance K_i; H2O 25%; int4 KIVI full; split + int4. Same bootstrap
and gate. MAX_NEW raised to 224 for the longer transcripts.

Usage:  python experiments/long_audio_uz.py [--n 100]
"""

import argparse
import json
import os
import time

import numpy as np

import kvlib
from cache_precision_wer import SCHEMES
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, Setup, cache_mib, error_rate, load_audio,
                   paired_ci, prompt_ids, real_positions, session, text_norm,
                   with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LONG_NPZ = os.path.join(ROOT, "models", "_calib_cache", "cv_uz_long100.npz")
OUT = os.path.join(HERE, "results_long_audio_medium_uz.json")
GAP = int(0.3 * 16000)
kvlib.MAX_NEW = 224


def build_long(n, seed=SEED):
    base = SETUPS["medium_uz"]
    z = np.load(base.audio["test"], allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths:
        waves.append(flat[off:off + int(ln)])
        off += int(ln)
    pool = list(range(300, len(waves)))          # never used before
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    out_w, out_t, i = [], [], 0
    while len(out_w) < n and i < len(pool):
        target = rng.uniform(15, 29.5) * 16000
        parts, txt, tot = [], [], 0
        while i < len(pool) and tot + len(waves[pool[i]]) + GAP <= target:
            parts.append(waves[pool[i]])
            txt.append(str(texts[pool[i]]))
            tot += len(waves[pool[i]]) + GAP
            i += 1
        if len(parts) < 2:
            continue
        sil = np.zeros(GAP, np.float32)
        w = np.concatenate([np.concatenate([p, sil]) for p in parts])[:-GAP]
        out_w.append(w.astype(np.float32))
        out_t.append(" ".join(txt))
    np.savez(LONG_NPZ, audio=np.concatenate(out_w),
             lengths=np.array([len(w) for w in out_w]),
             texts=np.array(out_t, dtype=object))
    secs = np.array([len(w) / 16000 for w in out_w])
    print(f"built {len(out_w)} composites: {secs.mean():.1f} s mean, {secs.min():.1f}-{secs.max():.1f} s, "
          f"real positions mean {np.mean([real_positions(w) for w in out_w]):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    if not os.path.exists(LONG_NPZ):
        build_long(args.n)
    base = SETUPS["medium_uz"]
    setup = Setup("medium_uz_long", base.hf_dir, base.encoder, base.first, base.step,
                  {"test": LONG_NPZ, "validation": base.audio["validation"]},
                  base.language, base.n_layers, base.d_model, base.normalizer)
    states, waves, texts = kvlib.encoder_states(setup, "test", args.n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)
    e6 = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_uz.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]

    def keeper(rule, rn):
        if rule == "full":
            return lambda m: None
        if rule == "split":
            return keep_split(f_r, f_p, rn)
        if rule == "h2o_k":
            k = len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS)))
            return lambda m: np.sort(np.argsort(-m)[:k])
        if rule == "h2o25":
            return lambda m: np.sort(np.argsort(-m)[:375])
        raise ValueError(rule)

    arms = [("full/fp32", "full", "fp32"), ("split/fp32", "split", "fp32"),
            ("h2o_Ki/fp32", "h2o_k", "fp32"), ("h2o25/fp32", "h2o25", "fp32"),
            ("full/int4_kivi", "full", "int4_kivi"), ("split/int4_kivi", "split", "int4_kivi")]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"n": args.n, "arms": {}}
    secs = [len(w) / 16000 for w in waves]
    res["audio_s_mean"] = float(np.mean(secs))
    res["real_positions_mean"] = float(np.mean([real_positions(w) for w in waves]))
    for name, rule, scheme in arms:
        if name in res["arms"]:
            continue
        wers, kept, toks = [], [], []
        t0 = time.time()
        for i in range(args.n):
            ids, k = kvlib.greedy(setup, first, step, states[i], prompt,
                                  keeper(rule, real_positions(waves[i])), SCHEMES[scheme])
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            kept.append(k)
            toks.append(len(ids))
        bits = 4 if "int4" in scheme else 32
        res["arms"][name] = {"rule": rule, "scheme": scheme, "wer": float(np.mean(wers)),
                             "per_sample_wer": wers, "kept": float(np.mean(kept)),
                             "tokens": float(np.mean(toks)),
                             "mib": cache_mib(setup, float(np.mean(kept)), bits)}
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"  {name:<18} WER {np.mean(wers):.4f}  kept {np.mean(kept):5.0f}  tokens {np.mean(toks):.0f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)

    base_ps = res["arms"]["full/fp32"]["per_sample_wer"]
    ref = res["arms"]["full/fp32"]["wer"]
    delta = round(ref * EPS, 4)
    rng = np.random.default_rng(SEED)
    print(f"\nlong-form Uzbek composites: {args.n} x {res['audio_s_mean']:.1f} s, "
          f"real positions {res['real_positions_mean']:.0f}; full WER {ref:.4f}, delta {delta}")
    print(f"{'arm':<18}{'kept':>6}{'MiB':>8}{'WER':>9}{'dWER vs full [95% CI]':>30}   gate")
    for name, r in res["arms"].items():
        d, lo, hi = paired_ci(r["per_sample_wer"], base_ps, rng)
        r["delta_vs_full"] = [d, lo, hi]
        v = "Accepted" if round(hi, 4) < delta else "Rejected" if round(lo, 4) > delta else "Inconclusive"
        print(f"{name:<18}{r['kept']:>6.0f}{r['mib']:>8.1f}{r['wer']:>9.4f}   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   {v}")
    sp, hk = res["arms"]["split/fp32"]["per_sample_wer"], res["arms"]["h2o_Ki/fp32"]["per_sample_wer"]
    d, lo, hi = paired_ci(hk, sp, rng)
    res["h2o_Ki_minus_split"] = [d, lo, hi]
    print(f"\nH2O(K_i) - split: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nsaqlandi: {OUT}")


if __name__ == "__main__":
    main()
