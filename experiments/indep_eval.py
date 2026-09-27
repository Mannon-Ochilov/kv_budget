"""R4 -- frozen PadSink-Track on the independent sets (build_independent_sets.py).

Nothing is tuned here: the split calibration (f_r, f_p -> K_i), rho and the
Track settings are the ones fixed before. For each model: full cache and
PadSink-Track at k = 0.5 K_i on 500 unseen utterances of unseen speakers.

Uncertainty: speaker-level (cluster) paired bootstrap -- speakers are resampled
with replacement and all their utterances follow; utterance-level CIs are
reported too. Verdicts at eps = 0.10 / 0.15 / 0.20 (delta = eps * WER_full).

Pre-registered predictions (before the run), at eps = 0.20 with the speaker
bootstrap:
  I1  PadSink-Track passes on medium_uz, small_uz and small_en (as on the
      original test sets) and fails on medium_en;
  I2  the verdicts on these three models do not change between the
      utterance-level and the speaker-level bootstrap.

Usage:  python experiments/indep_eval.py --setup medium_uz
"""

import argparse
import json
import os
import time

import numpy as np

import align_track as A
import kvlib
from eviction_budget import keep_split
from kvlib import (ENC_POS, SEED, SETUPS, Setup, error_rate, greedy, paired_ci, prompt_ids,
                   real_positions, session, text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
N = 500
EPSS = (0.10, 0.15, 0.20)


def speaker_ci(a, b, spk, rng, n_boot=10000):
    d = np.asarray(a) - np.asarray(b)
    groups = {}
    for i, s in enumerate(spk):
        groups.setdefault(s, []).append(i)
    keys = list(groups)
    sums = np.array([d[groups[k]].sum() for k in keys])
    cnts = np.array([len(groups[k]) for k in keys])
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    args = ap.parse_args()
    base = SETUPS[args.setup]
    npz = os.path.join(ROOT, "models", "_calib_cache",
                       "cv_uz_indep500.npz" if base.language == "uz" else "ls_test_clean_indep500.npz")
    ind = Setup(base.name + "_indep", base.hf_dir, base.encoder, base.first, base.step,
                {"test": npz, "validation": base.audio["validation"]}, base.language,
                base.n_layers, base.d_model, base.normalizer)
    states, waves, texts = kvlib.encoder_states(ind, "test", N)
    spk = np.load(npz, allow_pickle=True)["speakers"].tolist()[:N]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{base.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    heads = A.align_heads(base)
    norm = text_norm(base)
    tok, prompt = prompt_ids(base)
    first, step = session(with_attention(base.first)), session(base.step)
    sa = session(with_attention(base.step))
    out_json = os.path.join(HERE, f"results_indep_{base.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"arms": {}}
    res.update({"n": N, "speakers": len(set(spk))})
    for arm in ("full", "track/s0.5"):
        if arm in res["arms"]:
            continue
        wers, t0 = [], time.time()
        for i in range(N):
            rn = real_positions(waves[i])
            if arm == "full":
                ids, _ = greedy(base, first, step, states[i], prompt, lambda m: None, lambda a, b: (a, b))
            else:
                k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
                ids, _ = A.track_greedy(base, first, sa, states[i], prompt, k, rn, heads, False)
            wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
        res["arms"][arm] = {"wer": float(np.mean(wers)), "per_sample_wer": wers}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {base.name} indep {arm:<11} WER {np.mean(wers):.4f}  [{time.time() - t0:.0f}s]", flush=True)
    full = res["arms"]["full"]["per_sample_wer"]
    tr = res["arms"]["track/s0.5"]["per_sample_wer"]
    wf = float(np.mean(full))
    du = paired_ci(tr, full, np.random.default_rng(SEED))
    ds = speaker_ci(tr, full, spk, np.random.default_rng(SEED))
    verdict = lambda ci, dl: "Q" if round(ci[2], 4) < dl else "R" if round(ci[1], 4) > dl else "N"   # noqa: E731
    res["summary"] = {"full_wer": wf, "delta_utt": list(du), "delta_spk": list(ds),
                      "verdicts_spk": {str(e): verdict(ds, round(wf * e, 4)) for e in EPSS},
                      "verdicts_utt": {str(e): verdict(du, round(wf * e, 4)) for e in EPSS}}
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"  {base.name}: {N} utt, {len(set(spk))} speakers, full WER {wf:.4f}; Track dWER "
          f"{ds[0]:+.4f}  speaker CI [{ds[1]:+.4f}, {ds[2]:+.4f}]  utterance CI [{du[1]:+.4f}, {du[2]:+.4f}]  "
          f"verdicts (spk) eps .10/.15/.20: " + "/".join(res["summary"]["verdicts_spk"].values()), flush=True)


if __name__ == "__main__":
    main()
