"""The retention framework: per-model calibration of the fallback threshold.

PadSink-Track at 0.5 K_i reads k positions per step; a step whose top-1
probability is below tau is redone on the full cross cache (adaptive_track.py).
One tau for every model is wasteful: at tau = 0.9 the whole-utterance saving
falls from 12-17 % to 0-14 % (e2e_latency.py, results_e2e_latency_*_fb.json),
and to zero on small_en, which needs no fallback at all. Here tau is a
calibrated quantity, like K_i and rho:

  calibration  300 validation utterances (build_calib.py); arms: no fallback,
               tau = 0.6, 0.7, 0.8, 0.9, 0.95; reference = the full cache on
               the same utterances; delta = 0.2 * full-cache WER.
  rule         the FIRST arm in that order (the cheapest) with
               (i) upper 95 % bound of dWER < delta and
               (ii) no utterance with WER > 1 that the full cache does not
               have. No such arm -> the model keeps the full cache.
  evaluation   the chosen arm on the usual 300 test utterances of every setup
               (the four paper models and the other-language setups: original
               medium and small, Common Voice ru / tr). For the paper models
               these test sets were already used for the tau sensitivity runs
               (adaptive_track.py), so the evaluation there is not blind; the
               calibration itself uses validation data only. K_i (split rule),
               rho and the Track settings are the frozen ones.

Pre-registered predictions (before any calibration run):
  K1  chosen arm: none or 0.6 for small_en and small_uz; 0.6-0.8 for
      medium_uz; 0.95 or the full cache for medium_en.
  K2  every model with a chosen arm is accepted at eps = 0.20 on its
      evaluation set.
  K3  other languages: the original small gets none or 0.6 and passes; the
      original medium needs tau >= 0.9 or stays on the full cache, in Russian
      and in Turkish as in English.

Usage:  python experiments/framework.py --setup S --stage calib --arms full,none
        python experiments/framework.py --setup S --stage select
        python experiments/framework.py --setup S --stage eval
"""

import argparse
import glob
import json
import os
import time

import numpy as np

import adaptive_track as T
import align_track as A
import diag_budget  # noqa: F401  (registers rho for the extra setups)
import kvlib
from eviction_budget import EPS, keep_split
from indep_eval import speaker_ci
from kvlib import (ENC_POS, ROOT, SEED, SETUPS, Setup, error_rate, greedy, paired_ci, prompt_ids,
                   real_positions, session, text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "models", "_calib_cache")
ORDER = ["none", "0.6", "0.7", "0.8", "0.9", "0.95"]
PAPER = ("medium_uz", "small_uz", "medium_en", "small_en")
N_CAL = 300


def derived(base, suffix, split, npz):
    return Setup(base.name + suffix, base.hf_dir, base.encoder, base.first, base.step, {split: npz},
                 base.language, base.n_layers, base.d_model, base.normalizer)


def calib_setup(base):
    if base.name.endswith("_cv"):
        f = f"cv_{base.language}_calib300.npz"
    else:
        f = "cv_uz_calib300.npz" if base.language == "uz" else "ls_dev_clean_calib300.npz"
    return derived(base, "", "calib", os.path.join(CACHE, f))


def run(base, states, waves, texts, arm, ctx):
    norm, tok, prompt, first, step, sa, heads, f_r, f_p = ctx
    wers, red, st = [], 0, 0
    for i in range(len(waves)):
        rn = real_positions(waves[i])
        if arm == "full":
            ids, _ = greedy(base, first, step, states[i], prompt, lambda m: None, lambda a, b: (a, b))
        else:
            k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
            ids, _, r, t = T.decode(base, first, sa, states[i], prompt, k, rn, heads,
                                    tau=None if arm == "none" else float(arm))
            red += r
            st += t
        wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
    return {"wer": float(np.mean(wers)), "per_sample_wer": wers, "redone_share": red / max(st, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--stage", required=True, choices=["calib", "select", "eval"])
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    base = SETUPS[args.setup]
    main_json = os.path.join(HERE, f"results_framework_{base.name}.json")
    res = json.load(open(main_json)) if os.path.exists(main_json) else {"calib": {}}

    if args.stage == "select":
        for p in sorted(glob.glob(os.path.join(HERE, f"_fw_{base.name}_*.json"))):
            res["calib"].update(json.load(open(p)))
            os.remove(p)
        full = res["calib"]["full"]["per_sample_wer"]
        delta = round(float(np.mean(full)) * EPS, 4)
        choice = None
        print(f"{base.name}: calibration, {len(full)} utterances, full WER {np.mean(full):.4f}, delta {delta}")
        for arm in ORDER:
            r = res["calib"][arm]
            d = paired_ci(r["per_sample_wer"], full, np.random.default_rng(SEED))
            new = int(sum(a > 1 and b <= 1 for a, b in zip(r["per_sample_wer"], full)))
            ok = round(d[2], 4) < delta and new == 0
            r.update({"delta_vs_full": list(d), "new_wer_gt1": new, "ok": bool(ok)})
            if ok and choice is None:
                choice = arm
            print(f"  {arm:<5} dWER {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  redone {r['redone_share']:.0%}  new WER>1: {new}"
                  f"  {'ok' if ok else '--'}{'   <- chosen' if choice == arm else ''}", flush=True)
        res.update({"delta_calib": delta, "choice": choice})
        json.dump(res, open(main_json, "w"), indent=1)
        print(f"  {base.name}: chosen arm = {choice if choice else 'FULL CACHE'}", flush=True)
        return

    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{base.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    tok, prompt = prompt_ids(base)
    ctx = (text_norm(base), tok, prompt, session(with_attention(base.first)), session(base.step),
           session(with_attention(base.step)), A.align_heads(base), f_r, f_p)

    if args.stage == "calib":
        cal = calib_setup(base)
        states, waves, texts = kvlib.encoder_states(cal, "calib", N_CAL)
        out, part = {}, os.path.join(HERE, f"_fw_{base.name}_{args.arms.replace(',', '_')}.json")
        for arm in args.arms.split(","):
            if arm in res["calib"]:
                continue
            t0 = time.time()
            out[arm] = run(base, states, waves, texts, arm, ctx)
            json.dump(out, open(part, "w"))
            print(f"  {base.name} calib {arm:<5} WER {out[arm]['wer']:.4f}  redone {out[arm]['redone_share']:.0%}  [{time.time() - t0:.0f}s]", flush=True)
        return

    # ------------------------------------------------------------ evaluation
    arm = res.get("choice")
    if arm is None:
        print(f"  {base.name}: full cache (no arm chosen) -- nothing to evaluate", flush=True)
        return
    states, waves, texts = kvlib.encoder_states(base, "test", 300)
    spk, full, name = None, e6["test"]["full"]["per_sample_wer"], "test 300"
    r = run(base, states, waves, texts, arm, ctx)
    wf = float(np.mean(full))
    d = (speaker_ci(r["per_sample_wer"], full, spk, np.random.default_rng(SEED)) if spk
         else paired_ci(r["per_sample_wer"], full, np.random.default_rng(SEED)))
    verdict = lambda e: "Q" if round(d[2], 4) < round(wf * e, 4) else "R" if round(d[1], 4) > round(wf * e, 4) else "N"   # noqa: E731
    r.update({"arm": arm, "set": name, "full_wer": wf, "delta_vs_full": list(d),
              "verdicts": {str(e): verdict(e) for e in (0.10, 0.15, 0.20)},
              "new_wer_gt1": int(sum(a > 1 and b <= 1 for a, b in zip(r["per_sample_wer"], full)))})
    res["eval"] = r
    json.dump(res, open(main_json, "w"), indent=1)
    print(f"  {base.name} EVAL ({name}) arm {arm}: full WER {wf:.4f}  dWER {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  "
          f"redone {r['redone_share']:.0%}  eps .10/.15/.20: " + "/".join(r["verdicts"].values())
          + f"  new WER>1: {r['new_wer_gt1']}", flush=True)


if __name__ == "__main__":
    main()
