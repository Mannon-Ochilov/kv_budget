"""R8 -- two extensions that let the budget follow the model instead of a fixed
0.5 K_i (sink_drift.py: the sink does not move, but on the original medium
the padding attention is diffuse -- ~96 positions for 90 % of it against ~74
on medium_uz -- and the 25 % sink cap holds only 57 % of it).

  adapt/rho   sink = the smallest padding set holding rho of the first-step
              padding mass, NOT capped; window = the PadSink-Track window at
              0.5 K_i (0.75 * 0.5 K_i positions). k = |sink| + window is an
              outcome, set by the model. rho in {0.8, 0.9, 0.95}.
  fb/tau      PadSink-Track at 0.5 K_i unchanged; a step whose top-1
              probability is below tau is redone on the full cache (which is
              in DRAM anyway) and the full-cache result is used.
              tau in {0.5, 0.7, 0.9}. Cost: a redone step is a Track step plus
              a full step.

Protocol: every arm on the 100 validation utterances, gate delta = 0.2 * the
validation full-cache WER. Per model and family the arm taken to the test set
is fixed in advance: adapt -> the accepted arm with the smallest mean k;
fb -> the accepted arm with the smallest tau. No accepted arm -> the model
stays on the full cache and nothing is tested. The chosen arm is run once on
the 300 test utterances.

Pre-registered predictions (before any run):
  A1  adapt: medium_en gets an accepted arm on validation with mean k below
      K_i, and it passes on test; the three models that already pass keep
      passing with a mean k within 20 % of 0.5 K_i.
  A2  fb: medium_en passes on test at tau <= 0.7 with fewer than 30 % of the
      steps redone; on the other three the redone share is below 15 %.
  Refutation: no accepted arm for medium_en in a family -> that extension
  does not make the method universal.

Usage:  python experiments/adaptive_track.py --setup medium_en --family adapt|fb
"""

import argparse
import json
import os
import time

import numpy as np

import align_track as A
from diag_budget import RHO
from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, SEED, SETUPS, SOT, encoder_states, error_rate,
                   paired_ci, prompt_ids, real_positions, session, text_norm, with_attention)
from spar import sink_set

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"adapt": (0.8, 0.9, 0.95), "fb": (0.5, 0.7, 0.9)}


def decode(setup, first, sa, enc, prompt, k, rn, heads, rho=None, tau=None):
    """rho: uncapped rho-mass sink + fixed window; tau: Track + full-cache redo."""
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    cross = {n: v for n, v in present.items() if ".encoder." in n}
    att = {int(n.split("layers.")[1].split("/")[0]): v[0, :, -1, :] for n, v in zip(names, out)
           if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    L = setup.n_layers
    cap = max(1, int(A.TRACK_CAP * k))
    sinks, wins = [], []
    for l in range(L):
        m = att[l].sum(0)
        if rho is not None:
            s = rn + sink_set(m[rn:], rho)
            wins.append(max(1, k - cap))
        else:
            s = (rn + sink_set(m[rn:], RHO[setup.name]))[:cap]
            wins.append(max(1, k - len(s)))
        sinks.append(np.sort(s).astype(int))
    c = int(np.argmax(sum(att[l][h, :rn] for l, h in heads)))
    s_in = [i.name for i in sa.get_inputs()]
    s_out = [o.name for o in sa.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    nxt = int(np.argmax(logits[0, -1]))
    kept, redone, steps = [], 0, 0
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        steps += 1
        keep = {}
        for l in range(L):
            lo = int(np.clip(c - int(A.BACK * wins[l]), 0, max(0, ENC_POS - wins[l])))
            keep[l] = np.unique(np.concatenate([np.arange(lo, min(ENC_POS, lo + wins[l])), sinks[l]]).astype(int))

        def run(full):
            feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
            for n in s_in:
                if n.startswith("past_key_values"):
                    pn = "present" + n[len("past_key_values"):]
                    if ".encoder." in pn:
                        feed[n] = cross[pn] if full else np.ascontiguousarray(cross[pn][:, :, keep[int(pn.split(".")[1])], :])
                    else:
                        feed[n] = present[pn]
            return sa.run(None, feed)

        res, full = run(False), False
        if tau is not None:
            z = res[s_out.index("logits")][0, -1].astype(np.float64)
            z -= z.max()
            if 1.0 / np.exp(z).sum() < tau:
                res, full = run(True), True
                redone += 1
        logits = res[s_out.index("logits")]
        for n, v in zip(s_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        score = np.zeros(ENC_POS)
        for l, h in heads:
            a = res[att_idx[l]][0, h, -1, :]
            if full:
                score += a
            else:
                np.add.at(score, keep[l], a[:len(keep[l])])
        score[rn:] = 0
        c = max(c, int(np.argmax(score)))
        kept.append(np.mean([len(x) for x in keep.values()]))
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):], (float(np.mean(kept)) if kept else float(k)), redone, steps


def run_arm(s, split, n, fam, val, ctx):
    states, waves, texts = encoder_states(s, split, n)
    norm, tok, prompt, first, sa, heads, f_r, f_p = ctx
    wers, kept, red, st = [], [], 0, 0
    for i in range(n):
        rn = real_positions(waves[i])
        k = max(1, int(round(0.5 * len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS))))))
        ids, kk, r, t = decode(s, first, sa, states[i], prompt, k, rn, heads,
                               **({"rho": val} if fam == "adapt" else {"tau": val}))
        wers.append(error_rate(norm(texts[i]).split(), norm(tok.decode(ids, skip_special_tokens=True)).split()))
        kept.append(kk)
        red += r
        st += t
    return wers, float(np.mean(kept)), red / max(st, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SETUPS))
    ap.add_argument("--family", required=True, choices=list(ARMS))
    args = ap.parse_args()
    s, fam = SETUPS[args.setup], args.family
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{s.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    tok, prompt = prompt_ids(s)
    ctx = (text_norm(s), tok, prompt, session(with_attention(s.first)), session(with_attention(s.step)),
           A.align_heads(s), f_r, f_p)
    out_json = os.path.join(HERE, f"results_adaptive_{fam}_{s.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {"valid": {}, "test": {}}
    res["k_i_valid"] = e6["choice"]["calib_kept"]

    def gate(wers, full):
        delta = round(float(np.mean(full)) * EPS, 4)
        d = paired_ci(wers, full, np.random.default_rng(SEED))
        return list(d), ("Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"), delta

    vfull = e6["calib"]["full"]["per_sample_wer"]
    for val in ARMS[fam]:
        name = f"{fam}/{val}"
        if name not in res["valid"]:
            t0 = time.time()
            wers, kept, share = run_arm(s, "validation", 100, fam, val, ctx)
            d, v, delta = gate(wers, vfull)
            res["valid"][name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "kept": kept,
                                  "redone_share": share, "delta_vs_full": d, "gate": v, "delta": delta}
            json.dump(res, open(out_json, "w"), indent=1)
        r = res["valid"][name]
        print(f"  {s.name} valid {name:<11} k {r['kept']:5.0f} (K_i {res['k_i_valid']:.0f})  redone {r['redone_share']:.0%}  "
              f"{r['delta_vs_full'][0]:+.4f} [{r['delta_vs_full'][1]:+.4f}, {r['delta_vs_full'][2]:+.4f}]  {r['gate']} (delta {r['delta']})", flush=True)
    ok = [n for n, r in res["valid"].items() if r["gate"] == "Accepted"]
    if not ok:
        res["choice"] = None
        json.dump(res, open(out_json, "w"), indent=1)
        print(f"  {s.name} {fam}: no accepted arm on validation -> full cache", flush=True)
        return
    choice = min(ok, key=(lambda n: res["valid"][n]["kept"]) if fam == "adapt" else (lambda n: float(n.split("/")[1])))
    res["choice"] = choice
    if choice not in res["test"]:
        wers, kept, share = run_arm(s, "test", 300, fam, float(choice.split("/")[1]), ctx)
        d, v, delta = gate(wers, e6["test"]["full"]["per_sample_wer"])
        res["test"][choice] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "kept": kept, "redone_share": share,
                               "delta_vs_full": d, "gate": v, "delta": delta, "catastrophic": int(sum(w > 1 for w in wers))}
        json.dump(res, open(out_json, "w"), indent=1)
    r = res["test"][choice]
    print(f"  {s.name} TEST  {choice:<11} k {r['kept']:5.0f}  redone {r['redone_share']:.0%}  "
          f"{r['delta_vs_full'][0]:+.4f} [{r['delta_vs_full'][1]:+.4f}, {r['delta_vs_full'][2]:+.4f}]  {r['gate']} (delta {r['delta']})"
          f"  WER>1: {r['catastrophic']}", flush=True)


if __name__ == "__main__":
    main()
