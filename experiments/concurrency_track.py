"""L10 -- system throughput when several decoder streams share one DRAM channel.

Each stream is a separate process with one intra-op thread running the
decoder step of track_latency.make_arms (t = 30, medium) in a loop; N streams
start together (barrier) and run for SECONDS. Aggregate throughput = total
steps / wall time. Arms: full FP32 cache (1500), one-shot K_i/2, PadSink-Track
K_i/2 with the ring buffer. N in {1, 2, 4, 8}, three repeats, arms interleaved.

The roofline bound for the aggregate is saturated DRAM read bandwidth / bytes
per step: 43.8 GB/s (8-thread read test) / 675 MiB (full) = 62 steps/s,
/ 431 MiB (K_i/2) = 97 steps/s.

Pre-registered predictions (before the run):
  C1  at N = 8, PadSink-Track's aggregate throughput is at least 1.35x the
      full cache's, and equal to the one-shot K_i/2 arm within 10 %;
  C2  at N = 8 both the full cache and PadSink-Track reach 70-105 % of their
      bandwidth bound (the aggregate is DRAM-bound).

Results (medians of 3): N=8 full 59.8, one-shot K_i/2 92.0, Track 91.2 steps/s;
Track/full 1.53x, Track/one-shot 0.99 (C1 confirmed); 97 / 95 / 94 % of the
DRAM bound (C2 confirmed).

Usage:  python experiments/concurrency_track.py [--seconds 10] [--repeats 3]
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("full", "oneshot_k", "track_ring")
BW_SAT = 43.8e9


def worker(model, arm, seconds, barrier, q):
    import sys
    sys.path.insert(0, HERE)
    from track_latency import make_arms
    fns = make_arms(model)
    fn = [f for a, f in fns.items() if a.startswith(arm)][0]
    for _ in range(3):
        fn()
    barrier.wait()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    q.put((n, time.perf_counter() - t0))


def cell(model, arm, n_streams, seconds):
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(n_streams), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(model, arm, seconds, barrier, q)) for _ in range(n_streams)]
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return sum(r[0] for r in res) / max(r[1] for r in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default="medium_uz")
    args = ap.parse_args()
    roof = json.load(open(os.path.join(HERE, "results_roofline.json")))
    byts = {p["arm"].split(" ")[0]: p["bytes"] for p in roof["points"] if p["model"] == args.model}
    out = {"model": args.model, "cells": {}}
    streams = (1, 2, 4, 8)
    for rep in range(args.repeats):
        for n in streams:
            for arm in ARMS:
                thr = cell(args.model, arm, n, args.seconds)
                out["cells"].setdefault(f"{arm}|{n}", []).append(thr)
                print(f"  rep {rep + 1} N={n} {arm:<11} {thr:6.1f} steps/s", flush=True)
                json.dump(out, open(os.path.join(HERE, f"results_concurrency_track_{args.model}.json"), "w"), indent=1)
    print(f"\n{'arm':<12}" + "".join(f"{'N=' + str(n):>10}" for n in streams) + "   bound (N=8)")
    summ = {}
    for arm in ARMS:
        med = [float(np.median(out["cells"][f"{arm}|{n}"])) for n in streams]
        bound = BW_SAT / byts[arm]
        summ[arm] = {"median": dict(zip(map(str, streams), med)), "bound": bound, "frac_of_bound_N8": med[-1] / bound}
        print(f"{arm:<12}" + "".join(f"{m:10.1f}" for m in med) + f"   {bound:5.1f} ({med[-1] / bound:.0%})")
    r = summ["track_ring"]["median"]["8"] / summ["full"]["median"]["8"]
    print(f"\nN=8: PadSink-Track / full = {r:.2f}x; Track / one-shot K_i/2 = "
          f"{summ['track_ring']['median']['8'] / summ['oneshot_k']['median']['8']:.2f}")
    out["summary"] = summ
    json.dump(out, open(os.path.join(HERE, f"results_concurrency_track_{args.model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
