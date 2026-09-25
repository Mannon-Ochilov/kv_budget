"""L2 -- what a PadSink-Track step costs on the CPU, gather included.

A one-shot rule prunes the cache once; PadSink-Track keeps the full cache in
DRAM and assembles the k-position working set every step. The step itself
is the same graph at a shorter cache; the new cost is the per-step gather
(24 layers x K and V, k positions out of 1500) and, for track_sum, the
summary-slot update (overlap subtraction) and the bias feed.

Arms per model (decoder step at t = 30, 1 thread):
  full        FP32 cache, 1500 positions (reference)
  oneshot_Ki  FP32 cache pruned once to K_i (the paper's split point)
  oneshot_k   FP32 cache pruned once to k = 0.5 K_i (the step alone)
  track_k     gather k from the 1500-position cache + step
  track_sum   gather k-1 + summary slot + bias graph + step
The window advances 0-3 positions per step as in decoding (random walk),
so the gather reads a sliding contiguous run plus the sink positions.

Interleaved rounds (21), each arm STEPS consecutive steps per round, the
per-round mean is one sample; median [min-max] over rounds.

First run: track_k 42.7 ms vs full 36.0 (medium) -- the per-step fancy-index
gather (~18 ms) dominates; prediction refuted. Added track_ring (persistent
buffer, 0-3 slot updates per step). Prediction for it: within 10% of
oneshot_k.

Prediction (before the first run): track_k is within 10% of oneshot_k (the gather
of ~200 positions x 24 layers is ~5 MB of copies, well under a millisecond
per layer) and below full by at least 25% on medium.

Usage:  python experiments/track_latency.py [--rounds 21]
"""

import argparse
import json
import os
import time

import numpy as np

from kvlib import SETUPS, session, with_attention, with_cross_bias
from step_latency_evicted import feed_for

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 20
HEADS = {"medium_uz": 16, "small_en": 12}


def k_of(name, scale):
    r = json.load(open(os.path.join(HERE, f"results_eviction_budget_{name}.json")))
    key = [k for k in r["test"] if k.startswith("split")][0]
    return int(round(scale * r["test"][key]["kept"]))


def make_arms(name):
    setup = SETUPS[name]
    L, H = setup.n_layers, HEADS[name]
    rng = np.random.default_rng(0)
    s_plain = session(setup.step)
    s_bias = session(with_cross_bias(setup.step))
    s_attn = session(with_attention(setup.step))   # track needs the alignment-head attention out
    k_i, k = k_of(name, 1.0), k_of(name, 0.5)
    rn = 600
    n_sink = int(0.25 * k)
    sinks = np.sort(rng.choice(np.arange(rn, 1500), n_sink, replace=False))
    base = feed_for(s_plain, 30, H, 1500, rng)
    enc_names = [n for n in base if ".encoder." in n]
    full_cache = {n: base[n] for n in enc_names}
    sums = {n: full_cache[n][:, :, rn:, :].sum(axis=2, keepdims=True) for n in enc_names}
    state = {"c": 50}

    def pruned(kk):
        f = dict(base)
        for n in enc_names:
            f[n] = np.ascontiguousarray(full_cache[n][:, :, :kk, :])
        return f

    f_full, f_ki, f_k = base, pruned(k_i), pruned(k)

    def gather(w, summary):
        state["c"] = (state["c"] + int(rng.integers(0, 4))) % (rn - w)
        lo = state["c"]
        idx = np.concatenate([np.arange(lo, lo + w), sinks])
        f = {"input_ids": base["input_ids"]}
        for n in base:
            if n.startswith("past_key_values") and ".decoder." in n:
                f[n] = base[n]
        for n in enc_names:
            x = full_cache[n][:, :, idx, :]
            if summary:
                ov = idx[idx >= rn]
                m = (sums[n] - full_cache[n][:, :, ov, :].sum(axis=2, keepdims=True)) / (1500 - rn - len(ov))
                x = np.concatenate([x, m], axis=2)
            f[n] = np.ascontiguousarray(x)
        if summary:
            b = np.zeros((1, 1, 1, len(idx) + 1), np.float32)
            b[..., -1] = np.log(1500 - rn)
            for l in range(L):
                f[f"cross_bias.{l}"] = b
        return f

    # ring buffer: cross attention is order-free over key positions, so the
    # working set can live in a persistent buffer where each step overwrites
    # only the slots of the positions that left the window with the ones that
    # entered (0-3 per step) -- no per-step gather of the whole set
    w_r = k - n_sink
    ring = {"lo": 50, "slot_of": {}}
    rbuf = {}
    idx0 = np.concatenate([np.arange(50, 50 + w_r), sinks])
    for n in enc_names:
        rbuf[n] = np.ascontiguousarray(full_cache[n][:, :, idx0, :])
    for j, pos in enumerate(range(50, 50 + w_r)):
        ring["slot_of"][pos] = j
    f_ring = dict(base)
    f_ring.update(rbuf)

    def ring_step():
        d = int(rng.integers(0, 4))
        lo = ring["lo"]
        if lo + w_r + d > rn:
            d = 0
        for j in range(d):
            old, new = lo + j, lo + w_r + j
            sl = ring["slot_of"].pop(old)
            ring["slot_of"][new] = sl
            for n in enc_names:
                rbuf[n][:, :, sl, :] = full_cache[n][:, :, new, :]
        ring["lo"] = lo + d
        return s_attn.run(None, f_ring)

    return {
        "full (1500)": lambda: s_plain.run(None, f_full),
        f"oneshot_Ki ({k_i})": lambda: s_plain.run(None, f_ki),
        f"oneshot_k ({k})": lambda: s_plain.run(None, f_k),
        f"track_k ({k})": lambda: s_plain.run(None, gather(k - n_sink, False)),
        f"track_sum ({k})": lambda: s_bias.run(None, gather(k - n_sink - 1, True)),
        f"track_ring ({k})": ring_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=21)
    args = ap.parse_args()
    out = {"rounds": args.rounds, "steps": STEPS, "models": {}}
    for name in ("medium_uz", "small_en"):
        arms = make_arms(name)
        for fn in arms.values():
            for _ in range(3):
                fn()
        times = {a: [] for a in arms}
        for r in range(args.rounds):
            for a, fn in arms.items():
                t0 = time.perf_counter()
                for _ in range(STEPS):
                    fn()
                times[a].append((time.perf_counter() - t0) * 1e3 / STEPS)
        print(f"\n{name}: decoder step at t = 30, {args.rounds} interleaved rounds x {STEPS} steps, 1 thread")
        res = {}
        for a, v in times.items():
            v = np.array(v)
            res[a] = {"median_ms": float(np.median(v)), "min_ms": float(v.min()), "max_ms": float(v.max())}
            print(f"  {a:<22}{np.median(v):8.2f} ms   [{v.min():.2f}-{v.max():.2f}]")
        out["models"][name] = res
    json.dump(out, open(os.path.join(HERE, "results_track_latency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
