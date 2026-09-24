"""A -- SPAR inside the deployed pipeline (medium, graph-executable int8 cache).

Both operating points of Section 4.5 are rebuilt with SPAR-dyn (rho = 0.9)
as the retention rule instead of the grid rule / 'audio + heaviest padding':

  spar_int8_Ki     per-utterance K_i of the calibrated rule (mean 393) --
                   the latency option
  spar_int8_1433   fixed K = 1433 -- the hardware-selected point

int8 per-head cache (the executable graph's scheme), 300 test utterances,
paired against the full FP32 cache and against the corresponding non-SPAR
arm (split + int8 head; boundary int8 K = 1433).

Usage:  python experiments/spar_pipeline.py
"""

import json
import os
import time

import numpy as np

from cache_precision_wer import SCHEMES
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)
from spar import spar_rule

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    setup = SETUPS["medium_uz"]
    e6 = json.load(open(os.path.join(HERE, "results_eviction_budget_medium_uz.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    full = e6["test"]["full"]
    combo = json.load(open(os.path.join(HERE, "results_eviction_combo_medium_uz.json")))["arms"]["split/int8_head"]
    bound = json.load(open(os.path.join(HERE, "results_boundary_int8_medium_uz.json")))
    out_json = os.path.join(HERE, "results_spar_pipeline_medium_uz.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}

    states, waves, texts = encoder_states(setup, "test", 300)
    norm = text_norm(setup)
    refs = [norm(t).split() for t in texts]
    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)
    delta = round(full["wer"] * EPS, 4)

    arms = [("spar_int8_Ki", None, combo), ("spar_int8_1433", bound["K"], bound)]
    for name, k_fixed, other in arms:
        if name in res:
            continue
        wers, kept, t0 = [], [], time.time()
        for i in range(300):
            rn = real_positions(waves[i])
            k_i = k_fixed or len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS)))
            ids, k = greedy(setup, first, step, states[i], prompt,
                            spar_rule(k_i, rn, 0.9, setup.n_layers, dynamic=True, fill=True), SCHEMES["int8_head"])
            hyp = norm(tok.decode(ids, skip_special_tokens=True))
            wers.append(error_rate(refs[i], hyp.split()))
            kept.append(k)
        rng = np.random.default_rng(SEED)
        d = paired_ci(wers, full["per_sample_wer"], rng)
        do = paired_ci(wers, other["per_sample_wer"], rng)
        v = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
        res[name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "kept": float(np.mean(kept)),
                     "mib": cache_mib(setup, float(np.mean(kept)), 8), "delta_vs_full": list(d),
                     "delta_vs_nonspar": list(do), "gate": v}
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {name:<16} kept {np.mean(kept):6.0f}  {res[name]['mib']:5.1f} MiB  WER {np.mean(wers):.4f}  "
              f"vs full {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  vs non-SPAR {do[0]:+.4f} [{do[1]:+.4f}, {do[2]:+.4f}]"
              f"  {v}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
