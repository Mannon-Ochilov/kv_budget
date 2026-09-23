"""Where the first-step attention mass sits, and what H2O's global ranking
keeps because of it.

E6 found the split rule matching H2O on Whisper-medium and beating it on
whisper-small. The explanation offered was that a global ranking spends its
budget differently on the two models. This checks that with numbers: for
each test utterance, the share of total mass on padding positions, and the
real/padding composition of what a global ranking at 25% and 50% keeps,
against what the calibrated split rule keeps.

Usage:  python experiments/mass_analysis.py --setup small_en [--n 300]
"""

import argparse
import json
import os

import numpy as np

from kvlib import (ENC_POS, SETUPS, SOT, encoder_states, prompt_ids,
                   real_positions, session, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))


def first_step_mass(first, enc, prompt):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    mass = np.zeros(ENC_POS, np.float64)
    for n, v in zip(names, out):
        if "encoder_attn" in n and n.endswith("Softmax_output_0"):
            mass += v.sum(axis=(0, 1, 2))
    return mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="small_en", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    chosen = json.load(open(os.path.join(
        HERE, f"results_eviction_budget_{setup.name}.json")))["choice"]["rule"]
    f_r, f_p = chosen[1], chosen[2]

    states, waves, _ = encoder_states(setup, "test", args.n)
    _, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))

    rows = []
    for i in range(args.n):
        rn = real_positions(waves[i])
        m = first_step_mass(first, states[i], prompt)
        pad_share = m[rn:].sum() / m.sum()
        # padding mass concentration: how many padding positions hold 90% of it
        pm = np.sort(m[rn:])[::-1]
        n90 = int(np.searchsorted(np.cumsum(pm) / pm.sum(), 0.9)) + 1
        order = np.argsort(-m)
        row = {"real": rn, "pad_share": pad_share, "pad_n90": n90}
        for frac in (0.25, 0.5):
            k = int(round(frac * ENC_POS))
            kept = order[:k]
            row[f"g{int(frac * 100)}_real"] = int((kept < rn).sum())
            row[f"g{int(frac * 100)}_pad"] = int((kept >= rn).sum())
        row["split_real"] = int(round(f_r * rn))
        row["split_pad"] = int(round(f_p * (ENC_POS - rn)))
        # H2O at exactly the split rule's slot count for this utterance
        k_i = row["split_real"] + row["split_pad"]
        kept = order[:k_i]
        row["gk_real"] = int((kept < rn).sum())
        row["gk_pad"] = int((kept >= rn).sum())
        rows.append(row)

    R = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    print(f"{setup.name}, {args.n} test utterances, chosen split f_r={f_r} f_p={f_p}")
    print(f"  real positions            {R['real']:6.0f}  (padding {ENC_POS - R['real']:.0f})")
    print(f"  mass on padding           {R['pad_share']:6.1%}")
    print(f"  padding positions for 90% of padding mass  {R['pad_n90']:.0f}")
    print(f"  H2O 25% keeps  real {R['g25_real']:5.0f}  pad {R['g25_pad']:5.0f}"
          f"   ({R['g25_real'] / R['real']:.0%} of the audio)")
    print(f"  H2O 50% keeps  real {R['g50_real']:5.0f}  pad {R['g50_pad']:5.0f}"
          f"   ({R['g50_real'] / R['real']:.0%} of the audio)")
    print(f"  H2O K_i-matched real {R['gk_real']:5.0f}  pad {R['gk_pad']:5.0f}"
          f"   ({R['gk_real'] / R['real']:.0%} of the audio)")
    print(f"  split keeps    real {R['split_real']:5.0f}  pad {R['split_pad']:5.0f}"
          f"   ({f_r:.0%} of the audio)")
    json.dump({"setup": setup.name, "n": args.n, "chosen": chosen, "mean": R,
               "rows": rows},
              open(os.path.join(HERE, f"results_mass_analysis_{setup.name}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
