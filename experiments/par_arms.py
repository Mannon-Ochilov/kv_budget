"""Run a subset of eviction_budget.py arms in a separate process, then merge.

eviction_budget.py runs its arms one after another on one thread each, so the
grid can be split across processes: every worker writes its own part file and
--merge copies the parts into results_eviction_budget_{setup}.json under the
same keys eviction_budget.py uses. A later `eviction_budget.py --phase ...`
then skips the merged arms and computes the choice / CIs exactly as before.
WER results do not depend on how the arms are scheduled.

Usage:
  python experiments/par_arms.py --setup S --phase calib --f-real 0.75
  python experiments/par_arms.py --setup S --phase test --arm 2
  python experiments/par_arms.py --setup S --merge
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

from eviction_budget import F_PAD, F_REAL, load, run_arm
from kvlib import ENC_POS, SETUPS, cache_mib, prompt_ids, session, with_attention

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--phase", choices=["calib", "test"])
    ap.add_argument("--f-real", type=float, help="calib: run the full row of f_pad for this f_real (or 'full' with -1)")
    ap.add_argument("--arm", type=int, help="test: index into the five test arms")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    main_json = os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")
    res = json.load(open(main_json)) if os.path.exists(main_json) else {}

    if args.merge:
        for p in sorted(glob.glob(os.path.join(HERE, f"_part_{setup.name}_*.json"))):
            part = json.load(open(p))
            for sec, arms in part.items():
                res.setdefault(sec, {}).update(arms)
            os.remove(p)
        json.dump(res, open(main_json, "w"), indent=1)
        print(f"merged -> {main_json}: calib {len(res.get('calib', {}))}, test {len(res.get('test', {}))}")
        return

    tok, prompt = prompt_ids(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)
    out, tag = {}, None
    if args.phase == "calib":
        states, waves, refs, norm = load(setup, "validation", 100)
        rules = [("full",)] if args.f_real < 0 else [("split", args.f_real, fp) for fp in F_PAD]
        tag = "calib_full" if args.f_real < 0 else f"calib_{args.f_real}"
        done = res.get("calib", {})
        for rule in rules:
            key = "/".join(str(x) for x in rule)
            if key in done:
                continue
            wers, kept = run_arm(setup, first, step, states, waves, prompt, tok, norm, refs, rule)
            out.setdefault("calib", {})[key] = {"rule": list(rule), "wer": float(np.mean(wers)),
                                                "per_sample_wer": wers, "kept": kept,
                                                "mib": cache_mib(setup, kept, 32)}
            json.dump(out, open(os.path.join(HERE, f"_part_{setup.name}_{tag}.json"), "w"))
            print(f"  calib {key:<16} WER {np.mean(wers):.4f}  kept {kept:5.0f}", flush=True)
    else:
        ch = res["choice"]
        fr, fp = ch["rule"][1], ch["rule"][2]
        arms = [("full",), ("split", fr, fp), ("global", ch["calib_kept"] / ENC_POS),
                ("global", 0.5), ("global", 0.25)]
        rule = arms[args.arm]
        key = "/".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in rule)
        if key in res.get("test", {}):
            return
        states, waves, refs, norm = load(setup, "test", 300)
        wers, kept = run_arm(setup, first, step, states, waves, prompt, tok, norm, refs, rule)
        out["test"] = {key: {"n": 300, "rule": list(rule), "wer": float(np.mean(wers)),
                             "per_sample_wer": wers, "kept": kept, "mib": cache_mib(setup, kept, 32)}}
        json.dump(out, open(os.path.join(HERE, f"_part_{setup.name}_test{args.arm}.json"), "w"))
        print(f"  test {key:<20} WER {np.mean(wers):.4f}  kept {kept:5.0f}", flush=True)


if __name__ == "__main__":
    main()
