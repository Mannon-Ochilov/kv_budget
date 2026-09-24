"""E14 -- PadSink-KV (Padding Sink-Preserving Adaptive Retention; earlier working name SPAR).

Two measured facts drive the rule. The padding sink is a threshold
resource (Table 4 / Figure 5: a small set of padding positions carries 90%
of the padding mass and dropping below it is a cliff) while the audio is a
graceful one; and no static ranking wins on both models (Table 5a), because
each spends the budget on padding and audio without knowing the balance.

SPAR, per utterance and per layer l, with the same per-utterance budget K_i
as the calibrated split rule (so every comparison is slot-matched):

  1. sink guarantee: S_l = the smallest set of padding positions holding a
     fraction rho (default 0.9) of layer l's padding mass, ranked by that
     layer's own first-step mass;
  2. audio: the remaining K_i - |S_l| slots go to the audio positions with
     the highest layer-l mass; if S_l alone exceeds 60% of K_i it is capped
     there so the audio is never starved.

  spar_dyn adds a layer-adaptive budget: the total L*K_i audio slots are
  reallocated across layers in proportion to each layer's audio-mass share,
  so layers that attend to the audio get more of it (PyramidKV's idea, with
  the measured distribution instead of a fixed pyramid).

rho is the only parameter; it is a mass fraction, not a model-specific
grid cell, and its sensitivity (0.8 / 0.9 / 0.95) is measured on the
validation split.

Usage:  python experiments/spar.py --setup medium_uz --phase test|valid|both
"""

import argparse
import json
import os
import time

import numpy as np

from eviction_budget import EPS, keep_split
from kvlib import (ENC_POS, SEED, SETUPS, cache_mib, encoder_states, greedy,
                   error_rate, paired_ci, prompt_ids, real_positions, session,
                   text_norm, with_attention)

HERE = os.path.dirname(os.path.abspath(__file__))
SINK_CAP = 0.6


def sink_set(m_pad, rho):
    order = np.argsort(-m_pad)
    cum = np.cumsum(m_pad[order]) / max(m_pad.sum(), 1e-12)
    n = int(np.searchsorted(cum, rho)) + 1
    return order[:n]


def spar_rule(k_i, real_n, rho, n_layers, dynamic=False, fill=False):
    def fn(mass, mass_l):
        keep, audio_need = {}, []
        sinks = []
        for l in range(n_layers):
            s = real_n + sink_set(mass_l[l, real_n:], rho)
            s = s[:max(1, int(SINK_CAP * k_i))]
            sinks.append(s)
            audio_need.append(max(k_i - len(s), 1))
        if dynamic:
            share = np.array([mass_l[l, :real_n].sum() / max(mass_l[l].sum(), 1e-12) for l in range(n_layers)])
            share = share / share.sum()
            total = sum(audio_need)
            audio_need = np.clip(np.round(share * total).astype(int), 1, real_n)
        for l in range(n_layers):
            a = np.argsort(-mass_l[l, :real_n])[:min(int(audio_need[l]), real_n)]
            kl = np.concatenate([a, sinks[l]])
            if fill and len(kl) < k_i:
                # budget larger than audio + sink: spend the rest on the next-heaviest padding
                rest = np.setdiff1d(real_n + np.argsort(-mass_l[l, real_n:]), kl, assume_unique=False)
                order = real_n + np.argsort(-mass_l[l, real_n:])
                rest = order[~np.isin(order, kl)][:k_i - len(kl)]
                kl = np.concatenate([kl, rest])
            keep[l] = np.sort(kl)
        return keep
    return fn


def run_arm(setup, first, step, states, waves, refs, tok, norm, prompt, f_r, f_p, rho, dynamic):
    wers, kept = [], []
    for i in range(len(waves)):
        rn = real_positions(waves[i])
        k_i = len(keep_split(f_r, f_p, rn)(np.zeros(ENC_POS)))
        ids, k = greedy(setup, first, step, states[i], prompt,
                        spar_rule(k_i, rn, rho, setup.n_layers, dynamic), lambda a, b: (a, b))
        hyp = norm(tok.decode(ids, skip_special_tokens=True))
        wers.append(error_rate(refs[i], hyp.split()))
        kept.append(k)
    return wers, float(np.mean(kept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--phase", default="both", choices=["test", "valid", "both"])
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    out_json = os.path.join(HERE, f"results_spar_{setup.name}.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    tok, prompt = prompt_ids(setup)
    norm = text_norm(setup)
    first, step = session(with_attention(setup.first)), session(setup.step)

    if args.phase in ("valid", "both"):
        states, waves, texts = encoder_states(setup, "validation", 100)
        refs = [norm(t).split() for t in texts]
        calib = e6["calib"]
        v = res.setdefault("valid", {})
        for rho in (0.8, 0.9, 0.95):
            for dyn in (False, True):
                name = f"spar{'_dyn' if dyn else ''}/rho{rho}"
                if name in v:
                    continue
                t0 = time.time()
                wers, kept = run_arm(setup, first, step, states, waves, refs, tok, norm, prompt, f_r, f_p, rho, dyn)
                d = paired_ci(wers, calib["full"]["per_sample_wer"], np.random.default_rng(SEED))
                v[name] = {"wer": float(np.mean(wers)), "per_sample_wer": wers, "kept": kept, "delta_vs_full": list(d)}
                json.dump(res, open(out_json, "w"), indent=1)
                print(f"  valid {name:<18} WER {np.mean(wers):.4f}  kept {kept:5.0f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
        sk = f"split/{f_r}/{f_p}"
        print(f"  valid full {calib['full']['wer']:.4f}, split {calib[sk]['wer']:.4f}")

    if args.phase in ("test", "both"):
        states, waves, texts = encoder_states(setup, "test", 300)
        refs = [norm(t).split() for t in texts]
        full = e6["test"]["full"]
        split = e6["test"][[k for k in e6["test"] if k.startswith("split")][0]]
        t = res.setdefault("test", {})
        for rho, dyn in ((0.9, False), (0.9, True)):
            name = f"spar{'_dyn' if dyn else ''}/rho{rho}"
            if name in t:
                continue
            t0 = time.time()
            wers, kept = run_arm(setup, first, step, states, waves, refs, tok, norm, prompt, f_r, f_p, rho, dyn)
            rng = np.random.default_rng(SEED)
            d = paired_ci(wers, full["per_sample_wer"], rng)
            ds = paired_ci(wers, split["per_sample_wer"], rng)
            t[name] = {"n": 300, "wer": float(np.mean(wers)), "per_sample_wer": wers, "kept": kept,
                       "mib": cache_mib(setup, kept, 32), "delta_vs_full": list(d), "delta_vs_split": list(ds)}
            json.dump(res, open(out_json, "w"), indent=1)
            delta = round(full["wer"] * EPS, 4)
            vd = "Accepted" if round(d[2], 4) < delta else "Rejected" if round(d[1], 4) > delta else "Inconclusive"
            print(f"  test  {name:<18} WER {np.mean(wers):.4f}  kept {kept:5.0f}  vs full {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
                  f"  vs split {ds[0]:+.4f} [{ds[1]:+.4f}, {ds[2]:+.4f}]  {vd}  [{time.time() - t0:.0f}s]", flush=True)
        # against the other baselines at the same K_i
        sb_path = os.path.join(HERE, f"results_sota_baselines_{setup.name}.json")
        if os.path.exists(sb_path):
            sb = json.load(open(sb_path))["arms"]
            rng = np.random.default_rng(SEED)
            for name in t:
                for b in ("h2o_layer", "pyramidkv"):
                    d = paired_ci(t[name]["per_sample_wer"], sb[b]["per_sample_wer"], rng)
                    t[name][f"delta_vs_{b}"] = list(d)
                    print(f"  {name} - {b}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
            json.dump(res, open(out_json, "w"), indent=1)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
