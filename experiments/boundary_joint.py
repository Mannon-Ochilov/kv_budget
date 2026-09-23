"""E12 -- (a) refine the search at the hardware boundary; (b) a joint q x r
grid on the validation split.

(a) The stopping rule picks the least-compressed measured candidate that
fits. Between grid points a less aggressive configuration may exist: at
alpha = 0.7 on medium the KV budget is 67.2 MiB, int8 at full retention is
70.3 MiB -- 4% too large -- so int8 with 95.6% retention would fit and is
less compressed (by M_KV) than int4 at full retention. For each precision q
the boundary retention r_max(q) = B_KV / M_KV(q, r = 1) is computed and the
split rule is run with f_r = 1 and f_p set so that the mean kept count
equals r_max * 1500 (if r_max * 1500 < the real positions, f_r < 1 would be
needed; those cells are skipped as the grid showed them to fail). Test
split, 300 utterances, so the verdict is comparable with Table 9.

(b) Joint grid on validation: q in {FP32, int8 KIVI, int4 KIVI} x r in the
gate-passing frontier of the retention grid. Fills the (q, r) plane so
that the selection is made over C = Q x R rather than over precision-only
and retention-only sweeps composed afterwards.

Usage:  python experiments/boundary_joint.py --phase boundary|joint|both
"""

import argparse
import json
import os
import time

import numpy as np

from cache_precision_wer import SCHEMES
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_boundary_joint_medium_uz.json")
M_L3, W_LAYER, L = 24.0, 14.00, 24
BITS = {"fp32": 32, "int8_kivi": 8, "int4_kivi": 4}


def run_arm(setup, first, step, states, waves, refs, tok, norm, prompt, f_r, f_p, scheme):
    wers, kept = [], []
    for i in range(len(waves)):
        ids, k = greedy(setup, first, step, states[i], prompt,
                        keep_split(f_r, f_p, real_positions(waves[i])), SCHEMES[scheme])
        hyp = norm(tok.decode(ids, skip_special_tokens=True))
        wers.append(error_rate(refs[i], hyp.split()))
        kept.append(k)
    return wers, float(np.mean(kept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=["boundary", "joint", "both"])
    args = ap.parse_args()
    setup = SETUPS["medium_uz"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tok, prompt = prompt_ids(setup)
    norm = text_norm(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)

    if args.phase in ("boundary", "both"):
        states, waves, texts = encoder_states(setup, "test", 300)
        refs = [norm(t).split() for t in texts]
        real_mean = float(np.mean([real_positions(w) for w in waves]))
        full = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_uz.json")))["test"]["full"]
        b = res.setdefault("boundary", {})
        arms = []
        for alpha in (0.6, 0.7, 0.8):
            B = L * (alpha * M_L3 - W_LAYER)
            for scheme in ("fp32", "int8_kivi", "int4_kivi"):
                mib_full = cache_mib(setup, ENC_POS, BITS[scheme])
                r_max = min(1.0, B / mib_full)
                k_target = r_max * ENC_POS
                if k_target < real_mean + 1:
                    continue                       # would need f_r < 1
                if r_max >= 1.0:
                    continue                       # full retention already fits
                f_p = round((k_target - real_mean) / (ENC_POS - real_mean), 3)
                arms.append((f"a{alpha}/{scheme}/fp{f_p}", alpha, scheme, f_p, B))
        t0 = time.time()
        for name, alpha, scheme, f_p, B in arms:
            if name in b:
                continue
            wers, kept = run_arm(setup, first, step, states, waves, refs, tok, norm, prompt, 1.0, f_p, scheme)
            rng = np.random.default_rng(SEED)
            d = paired_ci(wers, full["per_sample_wer"], rng)
            mib = cache_mib(setup, kept, BITS[scheme])
            b[name] = {"alpha": alpha, "scheme": scheme, "f_r": 1.0, "f_p": f_p, "budget_mib": B,
                       "kept": kept, "mib": mib, "fits": mib <= B + 1e-6, "wer": float(np.mean(wers)),
                       "per_sample_wer": wers, "delta_vs_full": list(d)}
            json.dump(res, open(OUT, "w"), indent=1)
            delta = round(full["wer"] * EPS, 4)
            v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
            print(f"  {name:<24} kept {kept:5.0f}  {mib:6.1f} MiB (budget {B:.1f}, fits={mib <= B + 1e-6})"
                  f"  WER {np.mean(wers):.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {v}  [{time.time() - t0:.0f}s]",
                  flush=True)

    if args.phase in ("joint", "both"):
        states, waves, texts = encoder_states(setup, "validation", 100)
        refs = [norm(t).split() for t in texts]
        calib = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_uz.json")))["calib"]
        ref = calib["full"]["wer"]
        delta = round(ref * EPS, 4)
        frontier = [(1.0, 0.1), (1.0, 0.25), (0.75, 0.25), (0.5, 0.25), (1.0, 0.05), (0.5, 0.1)]
        j = res.setdefault("joint", {})
        for f_r, f_p in frontier:
            j[f"fp32/{f_r}/{f_p}"] = {"scheme": "fp32", "f_r": f_r, "f_p": f_p,
                                      "wer": calib[f"split/{f_r}/{f_p}"]["wer"],
                                      "per_sample_wer": calib[f"split/{f_r}/{f_p}"]["per_sample_wer"],
                                      "kept": calib[f"split/{f_r}/{f_p}"]["kept"],
                                      "mib": cache_mib(setup, calib[f"split/{f_r}/{f_p}"]["kept"], 32)}
        t0 = time.time()
        for scheme in ("int8_kivi", "int4_kivi"):
            for f_r, f_p in [(1.0, 1.0)] + frontier:
                name = f"{scheme}/{f_r}/{f_p}"
                if name in j:
                    continue
                wers, kept = run_arm(setup, first, step, states, waves, refs, tok, norm, prompt, f_r, f_p, scheme)
                j[name] = {"scheme": scheme, "f_r": f_r, "f_p": f_p, "wer": float(np.mean(wers)),
                           "per_sample_wer": wers, "kept": kept, "mib": cache_mib(setup, kept, BITS[scheme])}
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  joint {name:<22} WER {np.mean(wers):.4f}  kept {kept:5.0f}  "
                      f"{j[name]['mib']:6.1f} MiB  [{time.time() - t0:.0f}s]", flush=True)
        rng = np.random.default_rng(SEED)
        base = calib["full"]["per_sample_wer"]
        print(f"\njoint grid, validation (100): full WER {ref:.4f}, delta {delta}")
        print(f"{'(q, r)':<24}{'kept':>6}{'MiB':>8}{'WER':>9}{'dWER [95% CI]':>28}   gate")
        for name, r in sorted(j.items(), key=lambda kv: -kv[1]["mib"]):
            d = paired_ci(r["per_sample_wer"], base, rng)
            r["delta_vs_full"] = list(d)
            v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
            print(f"{name:<24}{r['kept']:>6.0f}{r['mib']:>8.1f}{r['wer']:>9.4f}   {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]   {v}")
        json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nsaqlandi: {OUT}")


if __name__ == "__main__":
    main()
