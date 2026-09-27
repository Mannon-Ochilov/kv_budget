"""L9 -- roofline view of one decoder step.

Machine ceilings, single thread (as every timing in the paper):
  * DRAM read bandwidth: max-reduction of a 1 GiB int32 array (read once;
    same method as the multi-thread measurement). Earlier variants: OpenBLAS sdot of a 1 GiB float32 vector with itself
    (reads it once; far larger than the 24 MiB L3), median of 7. The first
    run used numpy's sum, which is compute-limited (9.8 GB/s, below the
    18-20 GB/s the decoder step itself sustains) -- replaced by sdot
    (24.5 GB/s; an int32 max reduction gives 25.2 GB/s);
  * FP32 peak: OpenBLAS sgemm 2048^3, 1 thread;
  * INT8 peak: ONNX Runtime MatMulInteger (uint8 x int8, 256x4096x4096), 1 thread.

Per decoder step (t = 30), exact from the graphs:
  bytes  = streamed weights (all initializers except the embedding table, of
           which one row is gathered) + cross-attention K/V of the k positions
           read + the decoder's own K/V at t = 30
  FLOPs  = 2 * weight MACs + 4 * L * d * (k + t)     (QK^T and PV, cross + self)

Measured step times: results_track_latency.json (one session: full, one-shot
K_i, one-shot K_i/2, PadSink-Track ring buffer; both models).

Pre-registered predictions (before the run):
  RF1  every configuration lies far left of the ridge point (arithmetic
       intensity < 5 FLOP/B): the step is memory-bound.
  RF2  step time ~ bytes / BW: with the measured single-thread bandwidth the
       prediction is within 25 % of every measured FP32-cache step, and the
       predicted full / PadSink-Track ratio is within 10 % of the measured one.

Results: machine 24.3 GB/s read, 108 GFLOP/s FP32, 426 GOP/s INT8 (ridge
4.5 FLOP/B). RF1 confirmed: every configuration at 1.3-1.9 FLOP/B. RF2 split:
bytes/BW under-predicts by 20-26 % (the step sustains 18-20 GB/s, ~75-80 % of
the read roof) -- 6 of 8 within 25 %, so strictly refuted; the full/Track ratio
is predicted within 6-7 % (confirmed). Per model, t = a + bytes/BW_eff fits
all four points with R^2 = 1.000 (a = 4.0 / 1.5 ms, BW_eff = 21.9 / 22.0 GB/s).

Usage:  python experiments/roofline.py
"""

import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

from kvlib import SETUPS, session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MiB = 2 ** 20
T_STEP = 30
HEADS = {"medium_uz": 16, "small_en": 12}


def read_bw(threads=1):
    """Read bandwidth: max-reduction over a 1 GiB int32 array split into
    `threads` chunks reduced concurrently (numpy releases the GIL). The same
    method is used for 1-8 threads (43.8 GB/s at 8, the platform limit VTune
    reports as 44 GB/s); OpenBLAS sdot, used first, does not scale with threads."""
    from concurrent.futures import ThreadPoolExecutor
    a = np.ones(2 ** 28, np.int32)
    parts = np.array_split(a, threads)
    ex = ThreadPoolExecutor(threads)
    run = lambda: list(ex.map(lambda p: p.max(), parts))   # noqa: E731
    run()
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t0)
    return a.nbytes / np.median(ts) / 1e9


def fp32_peak():
    n = 2048
    a, b = np.random.rand(n, n).astype(np.float32), np.random.rand(n, n).astype(np.float32)
    a @ b
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    return 2 * n ** 3 / np.median(ts) / 1e9


def int8_peak():
    m, k, n = 256, 4096, 4096
    g = helper.make_graph([helper.make_node("MatMulInteger", ["A", "B"], ["Y"])], "g",
                          [helper.make_tensor_value_info("A", TensorProto.UINT8, [m, k])],
                          [helper.make_tensor_value_info("Y", TensorProto.INT32, [m, n])],
                          [helper.make_tensor("B", TensorProto.INT8, [k, n],
                                              np.random.randint(-127, 127, (k, n)).astype(np.int8).tobytes(), raw=True)])
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    p = os.path.join(ROOT, "models", "_mmi_bench.onnx")
    onnx.save(mdl, p)
    s = session(p)
    x = {"A": np.random.randint(0, 255, (m, k)).astype(np.uint8)}
    s.run(None, x)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        s.run(None, x)
        ts.append(time.perf_counter() - t0)
    return 2 * m * k * n / np.median(ts) / 1e9


def graph_stats(path):
    m = onnx.load(path)
    init = {i.name: i for i in m.graph.initializer}
    size = {n: int(np.prod(i.dims)) * helper.tensor_dtype_to_np_dtype(i.data_type).itemsize for n, i in init.items()}
    gathered = {n.input[0] for n in m.graph.node if n.op_type == "Gather"}
    streamed = sum(s for n, s in size.items() if n not in gathered)
    macs = sum(init[n.input[1]].dims[0] * init[n.input[1]].dims[1] for n in m.graph.node
               if n.op_type in ("MatMul", "MatMulInteger") and n.input[1] in init and len(init[n.input[1]].dims) == 2)
    return streamed, macs


def main():
    bws = [read_bw(1) for _ in range(5)]
    bw, f32, i8 = float(np.median(bws)), fp32_peak(), int8_peak()
    print(f"machine (1 thread): DRAM read {bw:.1f} GB/s, FP32 {f32:.1f} GFLOP/s, INT8 {i8:.1f} GOP/s")
    lat = json.load(open(os.path.join(HERE, "results_track_latency.json")))["models"]
    out = {"bw_runs_GBs": bws, "bw_GBs": bw, "fp32_GFLOPs": f32, "int8_GOPs": i8, "points": []}
    for name in ("medium_uz", "small_en"):
        s = SETUPS[name]
        wbytes, macs = graph_stats(s.step)
        L, d = s.n_layers, s.d_model
        self_kv = L * 2 * T_STEP * d * 4
        print(f"\n{name}: streamed weights {wbytes / MiB:.1f} MiB, {macs / 1e6:.0f} M MACs/token")
        rows = []
        for arm, v in lat[name].items():
            if arm.startswith("track_k") or arm.startswith("track_sum"):
                continue                                           # per-step gather / summary slot: not streaming-bound
            k = 1500 if arm.startswith("full") else int(arm.split("(")[1].rstrip(")"))
            byts = wbytes + L * 2 * k * d * 4 + self_kv
            flops = 2 * macs + 4 * L * d * (k + T_STEP)
            t = v["median_ms"] / 1e3
            pred = byts / (bw * 1e9)
            rows.append((arm, k, byts, flops, t, pred))
            out["points"].append({"model": name, "arm": arm, "k": k, "bytes": byts, "flops": flops,
                                  "t_ms": t * 1e3, "pred_ms": pred * 1e3, "ai": flops / byts,
                                  "gflops": flops / t / 1e9, "eff_bw": byts / t / 1e9})
            print(f"  {arm:<22} k {k:5d}  {byts / MiB:7.1f} MiB  AI {flops / byts:5.2f} FLOP/B  "
                  f"measured {t * 1e3:6.1f} ms  bytes/BW {pred * 1e3:6.1f} ms  ({(pred - t) / t:+.0%})  "
                  f"effective {byts / t / 1e9:5.1f} GB/s")
        full = [r for r in rows if r[0].startswith("full")][0]
        ring = [r for r in rows if r[0].startswith("track_ring")][0]
        print(f"  full / PadSink-Track: measured {full[4] / ring[4]:.2f}, predicted {full[5] / ring[5]:.2f}")
        # two-parameter fit t = a + bytes / BW_eff over the configurations of this model
        X = np.array([[1, r[2]] for r in rows])
        y = np.array([r[4] for r in rows])
        (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.sum((X @ [a, b] - y) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  fit t = {a * 1e3:.1f} ms + bytes / {1 / b / 1e9:.1f} GB/s   (R^2 {r2:.3f})")
        out[name] = {"fit_offset_ms": a * 1e3, "fit_bw_GBs": 1 / b / 1e9, "fit_r2": r2,
                     "ratio_measured": full[4] / ring[4], "ratio_predicted": full[5] / ring[5]}
    ridge = f32 / bw
    print(f"\nridge point (FP32) {ridge:.1f} FLOP/B, (INT8) {i8 / bw:.1f} OP/B")
    out["ridge_fp32"] = ridge
    json.dump(out, open(os.path.join(HERE, "results_roofline.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
