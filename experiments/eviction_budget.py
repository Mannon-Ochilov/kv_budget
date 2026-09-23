"""E6 -- sink-aware, budget-anchored eviction of the cross-attention cache.

E5 established three facts on Whisper-medium: the decoder treats the padding
positions of the 30 s window as attention sinks (dropping them all: WER 2.7);
a fixed handful of sink positions (StreamingLLM) is not enough; ranking all
1500 positions by first-step attention mass (H2O) works down to about a
quarter of the cache. The method here turns those facts into a rule with one
calibrated knob per model:

  rank the real (audio) positions and the padding positions SEPARATELY by
  attention mass, keep a fraction f_r of the real ones and a fraction f_p of
  the padding ones, and choose (f_r, f_p) on a calibration split as the
  cheapest pair whose WER is within delta = eps x WER_ref of the full cache
  -- the same three-way gate the paper applies to weights, now applied to
  activations. The cache size that pair implies is the activation term of
  the cache-anchored target.

The alternative it must beat at equal cache size is H2O's global ranking,
where real and padding positions compete in one list. If the padding sink is
a distinct resource, splitting the budget explicitly should reach a smaller
cache at the same WER. Calibration on the validation split, the test split
untouched until the final rows.

Usage:  python experiments/eviction_budget.py --setup medium_uz
        [--calib 100] [--test 300] [--phase calib|test|both]
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from kvlib import (ENC_POS, SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.20
F_REAL = [1.0, 0.75, 0.5, 0.35, 0.25]
F_PAD = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def keep_split(f_r, f_p, real_n):
    def fn(mass):
        n_r = max(1, int(round(f_r * real_n)))
        n_p = int(round(f_p * (ENC_POS - real_n)))
        r = np.argsort(-mass[:real_n])[:n_r]
        p = real_n + np.argsort(-mass[real_n:])[:n_p]
        return np.sort(np.concatenate([r, p]))
    return fn


def keep_global(frac):
    def fn(mass):
        k = max(1, int(round(frac * ENC_POS)))
        return np.sort(np.argsort(-mass)[:k])
    return fn


def run_arm(setup, first, step, states, waves, prompt, tok, norm, refs, rule):
    """rule = ("full",) | ("split", f_r, f_p) | ("global", frac)."""
    wers, kept = [], []
    for i in range(len(waves)):
        rn = real_positions(waves[i])
        if rule[0] == "full":
            keep = lambda m: None  # noqa: E731
        elif rule[0] == "split":
            keep = keep_split(rule[1], rule[2], rn)
        else:
            keep = keep_global(rule[1])
        ids, k = greedy(setup, first, step, states[i], prompt, keep,
                        lambda a, b: (a, b))
        hyp = norm(tok.decode(ids, skip_special_tokens=True))
        wers.append(error_rate(refs[i], hyp.split()))
        kept.append(k)
    return wers, float(np.mean(kept))


def load(setup, split, n):
    states, waves, texts = encoder_states(setup, split, n)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    return states, waves, refs, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--phase", default="both", choices=["calib", "test", "both"])
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    out_json = os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}

    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)

    # ---------------------------------------------------------- calibration
    if args.phase in ("calib", "both"):
        states, waves, refs, norm = load(setup, "validation", args.calib)
        grid = res.setdefault("calib", {})
        arms = [("full",)] + [("split", fr, fp) for fr, fp in
                              itertools.product(F_REAL, F_PAD)]
        t0 = time.time()
        for rule in arms:
            key = "/".join(str(x) for x in rule)
            if key in grid:
                continue
            wers, kept = run_arm(setup, first, step, states, waves, prompt,
                                 tok, norm, refs, rule)
            grid[key] = {"rule": list(rule), "wer": float(np.mean(wers)),
                         "per_sample_wer": wers, "kept": kept,
                         "mib": cache_mib(setup, kept, 32)}
            json.dump(res, open(out_json, "w"), indent=1)
            print(f"  calib {key:<16} WER {np.mean(wers):.4f}  kept {kept:5.0f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
        ref = grid["full"]["wer"]
        delta = round(ref * EPS, 4)
        ok = [(v["mib"], k) for k, v in grid.items()
              if k != "full" and round(v["wer"], 4) <= round(ref, 4) + delta]
        choice = min(ok)[1]
        res["choice"] = {"rule": grid[choice]["rule"], "calib_wer_ref": ref,
                         "delta": delta, "calib_mib": grid[choice]["mib"],
                         "calib_kept": grid[choice]["kept"]}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"\ncalibration: full WER {ref:.4f}, delta {delta}; chosen {choice} "
              f"-> {grid[choice]['kept']:.0f} positions, "
              f"{grid[choice]['mib']:.1f} MiB", flush=True)

    # ------------------------------------------------------------------ test
    if args.phase in ("test", "both"):
        ch = res["choice"]
        fr, fp = ch["rule"][1], ch["rule"][2]
        states, waves, refs, norm = load(setup, "test", args.test)
        # H2O global at the same kept fraction as the chosen split rule
        frac_match = ch["calib_kept"] / ENC_POS
        arms = [("full",), ("split", fr, fp), ("global", frac_match),
                ("global", 0.5), ("global", 0.25)]
        test = res.setdefault("test", {})
        t0 = time.time()
        for rule in arms:
            key = "/".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                           for x in rule)
            if key in test and test[key]["n"] == args.test:
                continue
            wers, kept = run_arm(setup, first, step, states, waves, prompt,
                                 tok, norm, refs, rule)
            test[key] = {"n": args.test, "rule": list(rule),
                         "wer": float(np.mean(wers)), "per_sample_wer": wers,
                         "kept": kept, "mib": cache_mib(setup, kept, 32)}
            json.dump(res, open(out_json, "w"), indent=1)
            print(f"  test {key:<20} WER {np.mean(wers):.4f}  kept {kept:5.0f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
        base = test["full"]["per_sample_wer"]
        rng = np.random.default_rng(SEED)
        ref = test["full"]["wer"]
        delta = round(ref * EPS, 4)
        print(f"\n{'arm':<22}{'kept':>6}{'MiB':>8}{'WER':>9}"
              f"{'dWER vs full [95% CI]':>30}   gate (delta {delta})")
        for key, r in test.items():
            d, lo, hi = paired_ci(r["per_sample_wer"], base, rng)
            r["delta_vs_full"] = [d, lo, hi]
            verdict = ("Accepted" if round(hi, 4) < delta else
                       "Rejected" if round(lo, 4) > delta else "Inconclusive")
            print(f"{key:<22}{r['kept']:>6.0f}{r['mib']:>8.1f}{r['wer']:>9.4f}"
                  f"   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   {verdict}")
        json.dump(res, open(out_json, "w"), indent=1)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
