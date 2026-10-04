"""R8 confirmation -- PadSink-Track with the full-cache fallback, tau = 0.9
frozen, on the independent sets (build_independent_sets.py: 500 utterances,
speakers disjoint from every earlier set).

tau = 0.9 is the only value of the grid {0.5, 0.7, 0.9} accepted on validation
for all four models (adaptive_track.py); the rule "one tau for all models" was
formulated after the first test results, so this run is the confirmation:
nothing is tuned here, k = 0.5 K_i, Track settings and tau as frozen.

Pre-registered predictions (before the run), speaker-level bootstrap:
  C1  all four models are accepted at eps = 0.20, medium_en included (plain
      PadSink-Track on this set: medium_en +0.0232, rejected);
  C2  the share of redone steps is 3-20 % per model.
  Refutation: medium_en not accepted -> the fallback does not make the method
  universal at tau = 0.9.

Usage:  python experiments/indep_fb.py --setup medium_en
"""

import argparse
import json
import os
import time

import numpy as np

import adaptive_track as T
import align_track as A
import kvlib
from eviction_budget import keep_split
from indep_eval import EPSS, N, ROOT, speaker_ci
from kvlib import (ENC_POS, SEED, SETUPS, Setup, error_rate, paired_ci, prompt_ids,
                   real_positions, session, text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
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
    prev = json.load(open(os.path.join(HERE, f"results_indep_{base.name}.json")))
    full = prev["arms"]["full"]["per_sample_wer"]
    heads, norm = A.align_heads(base), text_norm(base)
    tok, prompt = prompt_ids(base)
    first, sa = session(with_attention(base.first)), session(with_attention(base.step))
    wers, red, st, t0 = [], 0, 0, time.time()
    for i in range(N):
        rn = real_positions(waves[i])
        k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
        ids, _, r, t = T.decode(base, first, sa, states[i], prompt, k, rn, heads, tau=TAU)
        wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
        red += r
        st += t
    wf = float(np.mean(full))
    du = paired_ci(wers, full, np.random.default_rng(SEED))
    ds = speaker_ci(wers, full, spk, np.random.default_rng(SEED))
    verdict = lambda ci, dl: "Q" if round(ci[2], 4) < dl else "R" if round(ci[1], 4) > dl else "N"   # noqa: E731
    out = {"tau": TAU, "n": N, "speakers": len(set(spk)), "full_wer": wf, "wer": float(np.mean(wers)),
           "per_sample_wer": wers, "redone_share": red / max(st, 1), "delta_utt": list(du), "delta_spk": list(ds),
           "verdicts_spk": {str(e): verdict(ds, round(wf * e, 4)) for e in EPSS},
           "new_wer_gt1": int(sum(a > 1 and b <= 1 for a, b in zip(wers, full)))}
    json.dump(out, open(os.path.join(HERE, f"results_indep_fb_{base.name}.json"), "w"), indent=1)
    print(f"  {base.name} indep fb/{TAU}: full WER {wf:.4f}  dWER {ds[0]:+.4f}  speaker CI [{ds[1]:+.4f}, {ds[2]:+.4f}]  "
          f"redone {out['redone_share']:.0%}  verdicts eps .10/.15/.20: " + "/".join(out["verdicts_spk"].values())
          + f"  new WER>1: {out['new_wer_gt1']}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
