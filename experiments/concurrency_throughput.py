"""B -- does fitting the layer-local budget pay off when the L3 is shared?

A single decoder stream owns the whole L3, so condition (1) cannot show a
benefit there (Table 8). The budget is meant for the case where the cache
is contended: N decoder streams run at once, one thread each, and every
stream's hot set competes for the same 24 MiB. If the budget matters, the
configuration that fits it should lose less throughput as N grows than the
full FP32 cache does.

Configurations (medium decoder step at t = 30, random feeds of the right
shapes, as in step_latency_evicted.py):
  fp32_1500   full FP32 cache                        (does not fit (1))
  int8_1433   int8 cache, K = 1433, hardware-selected (fits (1))
  int8_1500   int8 cache, full                       (just misses (1))
  fp32_393    FP32 cache, calibrated retention        (latency option)
  int8_393    int8 cache, calibrated retention

For N in {1, 2, 4, 8}: N processes, each with its own session (1 intra-op
thread), warm up, wait on a common barrier, then run steps for SECONDS.
Aggregate throughput = total steps / wall time; per-stream step time =
median of per-process means. Scaling efficiency = throughput(N) / (N *
throughput(1)). Three repeats per cell, configurations interleaved.

Usage:  python experiments/concurrency_throughput.py [--seconds 10] [--repeats 3]
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MED = os.path.join(ROOT, "models", "whisper_with_past")
CONFIGS = {
    "fp32_1500": (os.path.join(MED, "decoder_with_past_untied_int8.onnx"), 1500),
    "int8_1433": (os.path.join(MED, "decoder_with_past_cache_int.onnx"), 1433),
    "int8_1500": (os.path.join(MED, "decoder_with_past_cache_int.onnx"), 1500),
    "fp32_393": (os.path.join(MED, "decoder_with_past_untied_int8.onnx"), 393),
    "int8_393": (os.path.join(MED, "decoder_with_past_cache_int.onnx"), 393),
}


def worker(path, enc_len, seconds, barrier, out_q):
    import sys
    sys.path.insert(0, HERE)
    import onnxruntime as ort
    from step_latency_evicted import feed_for
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    feed = feed_for(s, 30, 16, enc_len, np.random.default_rng(os.getpid()))
    for _ in range(3):
        s.run(None, feed)
    barrier.wait()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        s.run(None, feed)
        n += 1
    out_q.put((n, time.perf_counter() - t0))


def run_cell(name, n_streams, seconds):
    path, enc_len = CONFIGS[name]
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n_streams)
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(path, enc_len, seconds, barrier, q)) for _ in range(n_streams)]
    for p in procs:
        p.start()
    res = [q.get() for _ in procs]
    for p in procs:
        p.join()
    steps = sum(r[0] for r in res)
    wall = max(r[1] for r in res)
    per_stream_ms = float(np.median([r[1] / r[0] * 1e3 for r in res]))
    return steps / wall, per_stream_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--streams", default="1,2,4,8")
    args = ap.parse_args()
    streams = [int(x) for x in args.streams.split(",")]
    out_json = os.path.join(HERE, "results_concurrency_throughput.json")
    res = {"seconds": args.seconds, "repeats": args.repeats, "cells": {}}
    for rep in range(args.repeats):
        for n in streams:
            for name in CONFIGS:
                thr, ms = run_cell(name, n, args.seconds)
                res["cells"].setdefault(f"{name}|{n}", []).append({"throughput": thr, "step_ms": ms})
                json.dump(res, open(out_json, "w"), indent=1)
                print(f"  rep {rep + 1} N={n} {name:<10} {thr:7.1f} steps/s  per-stream {ms:6.1f} ms", flush=True)
    print(f"\n{'config':<11}" + "".join(f"{'N=' + str(n):>22}" for n in streams))
    summary = {}
    for name in CONFIGS:
        base = np.median([c["throughput"] for c in res["cells"][f"{name}|{streams[0]}"]])
        row = []
        for n in streams:
            thr = np.median([c["throughput"] for c in res["cells"][f"{name}|{n}"]])
            ms = np.median([c["step_ms"] for c in res["cells"][f"{name}|{n}"]])
            eff = thr / (n / streams[0] * base)
            summary[f"{name}|{n}"] = {"throughput": float(thr), "step_ms": float(ms), "efficiency": float(eff)}
            row.append(f"{thr:6.1f}/s {ms:5.1f}ms {eff:4.0%}")
        print(f"{name:<11}" + "".join(f"{x:>22}" for x in row))
    res["summary"] = summary
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"\nsaqlandi: {out_json}")


if __name__ == "__main__":
    main()
