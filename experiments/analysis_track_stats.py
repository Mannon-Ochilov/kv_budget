"""L3/L4 -- paired comparisons and length strata for PadSink-Track (from saved per-sample WER).

(5) track v2 minus each one-shot rule and minus the oracle at the same budget
    (0.5 and 0.25 K_i), paired bootstrap on the 300 test utterances.
(9) WER by utterance length (audio positions, tertiles of the test split)
    for the full cache, track v2 at 0.5 K_i and the best one-shot rule there.

Usage:  python experiments/analysis_track_stats.py
"""

import json
import os

import numpy as np

from kvlib import SEED, SETUPS, load_audio, paired_ci, real_positions

HERE = os.path.dirname(os.path.abspath(__file__))


def J(n):
    return json.load(open(os.path.join(HERE, n)))


def main():
    out = {}
    for m, setup in SETUPS.items():
        tr = J(f"results_align_track_{m}.json")["arms"]
        dg = J(f"results_diag_budget_{m}.json")["arms"]
        full = J(f"results_eviction_budget_{m}.json")["test"]["full"]["per_sample_wer"]
        rng = np.random.default_rng(SEED)
        print(f"\n== {m}")
        res = {"paired": {}, "strata": {}}
        for s in ("0.5", "0.25"):
            t = tr[f"track/s{s}"]["per_sample_wer"]
            for b in ("split", "h2o_layer", "pyramidkv", "padsink", "oracle"):
                d = paired_ci(t, dg[f"{b}/s{s}"]["per_sample_wer"], rng)
                res["paired"][f"track-{b}/s{s}"] = list(d)
                sig = "track better" if d[2] < 0 else "track worse" if d[1] > 0 else "n.s."
                print(f"  s{s:<5} track - {b:<10} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  {sig}")
        # length strata
        waves, _ = load_audio(setup.audio["test"], 300)
        rn = np.array([real_positions(w) for w in waves])
        edges = np.percentile(rn, [100 / 3, 200 / 3])
        strata = np.digitize(rn, edges)
        one = {b: np.mean(dg[f"{b}/s0.5"]["wer"]) for b in ("split", "h2o_layer", "pyramidkv", "padsink")}
        best = min(one, key=one.get)
        arms = {"full": full, "track 0.5": tr["track/s0.5"]["per_sample_wer"],
                f"{best} 0.5": dg[f"{best}/s0.5"]["per_sample_wer"]}
        names = ["short", "medium", "long"]
        print(f"  length strata (positions): <{edges[0]:.0f} / <{edges[1]:.0f} / rest  (seconds x0.02)")
        for a, w in arms.items():
            w = np.array(w)
            row = [float(w[strata == j].mean()) for j in range(3)]
            res["strata"][a] = row
            print(f"    {a:<16}" + "".join(f"{names[j]} {row[j]:.4f}   " for j in range(3)))
        out[m] = res
    json.dump(out, open(os.path.join(HERE, "results_track_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
