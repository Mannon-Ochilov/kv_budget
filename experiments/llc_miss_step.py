"""LLC misses per decoder step, full cache against the calibrated length.

The capacity argument predicts that a smaller cross-attention cache means
fewer lines crossing from DRAM on every step. This measures it with the
same VTune event set the weight paper used (MEM_LOAD_RETIRED.L3_MISS /
L3_HIT, LONGEST_LAT_CACHE.MISS): each arm is profiled twice, once with the
timed loop and once with seconds = 0 (load + warm-up only), and the
difference per iteration is one step with setup removed.

Arms: FP32 cache at 1500 and at the calibrated length for both models, and
the integer-attention int8 cache graph for medium. Single thread, t = 30,
random feeds as in step_latency_evicted.py.

Usage:  python experiments/llc_miss_step.py             (driver, needs VTune)
        python experiments/llc_miss_step.py --runner MODEL HEADS ENC_LEN SECONDS
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VTUNE = r"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
PYTHON = sys.executable
RESULT_ROOT = r"D:\tmp\vtune_kv_runs"
OUT_JSON = os.path.join(HERE, "results_llc_miss_step.json")
EVENTS = ("MEM_LOAD_RETIRED.L3_MISS", "MEM_LOAD_RETIRED.L3_HIT",
          "LONGEST_LAT_CACHE.MISS")
SECONDS = 15.0


def runner(model, heads, enc_len, seconds):
    import numpy as np
    sys.path.insert(0, HERE)
    from step_latency_evicted import feed_for, session
    s = session(model)
    rng = np.random.default_rng(0)
    feed = feed_for(s, 30, heads, enc_len, rng)
    for _ in range(3):
        s.run(None, feed)
    it, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        s.run(None, feed)
        it += 1
    el = time.perf_counter() - t0
    print(f"{it} iteratsiya, {el * 1e3 / max(it, 1):.2f} ms/iter", flush=True)


def arms():
    from step_latency_evicted import build_arms
    return build_arms()


def profile(model, heads, enc_len, seconds, result_dir):
    shutil.rmtree(result_dir, ignore_errors=True)
    cmd = [VTUNE, "-collect-with", "runsa", "-knob", "event-config=" + ",".join(EVENTS),
           "-r", result_dir, "--", PYTHON, os.path.abspath(__file__), "--runner",
           model, str(heads), str(enc_len), str(seconds)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    counts = {}
    for ev in EVENTS:
        m = re.search(re.escape(ev) + r"\s+(\d+)\s", text)
        counts[ev] = int(m.group(1)) if m else None
    it = re.search(r"(\d+) iteratsiya", text)
    ms = re.search(r"([\d.]+) ms/iter", text)
    if any(v is None for v in counts.values()):
        print(f"    OGOHLANTIRISH: hodisalar o'qilmadi -- {text[-400:]}")
    return counts, int(it.group(1)) if it else 0, float(ms.group(1)) if ms else None


def main():
    sys.path.insert(0, HERE)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    rows = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for label, path, heads, k in arms():
        if label in rows:
            continue
        tag = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        print(f"[{label}]", flush=True)
        base, _, _ = profile(path, heads, k, 0.0, os.path.join(RESULT_ROOT, tag + "_base"))
        full, iters, ms = profile(path, heads, k, SECONDS, os.path.join(RESULT_ROOT, tag + "_full"))
        if iters < 1 or any(full[e] is None or base[e] is None for e in EVENTS):
            print("  XATO: natijasiz")
            continue
        per = {e: (full[e] - base[e]) / iters for e in EVENTS}
        miss, hit = per["MEM_LOAD_RETIRED.L3_MISS"], per["MEM_LOAD_RETIRED.L3_HIT"]
        rows[label] = {"model": os.path.basename(path), "positions": k, "iterations": iters,
                       "ms_per_iter": ms, "llc_load_misses_per_step": miss,
                       "llc_load_hits_per_step": hit,
                       "llc_all_misses_per_step": per["LONGEST_LAT_CACHE.MISS"],
                       "llc_load_miss_rate": miss / (miss + hit) if miss + hit > 0 else None}
        json.dump(rows, open(OUT_JSON, "w"), indent=1)
        print(f"  {iters} it | {ms:.1f} ms | L3 load misses/step {miss:,.0f} | "
              f"all misses/step {per['LONGEST_LAT_CACHE.MISS']:,.0f} | miss rate "
              f"{rows[label]['llc_load_miss_rate']:.3f}", flush=True)
    print(f"\n{'arm':<28}{'ms':>7}{'L3 load miss/step':>19}{'all miss/step':>15}{'rate':>7}")
    for label, r in rows.items():
        print(f"{label:<28}{r['ms_per_iter']:>7.1f}{r['llc_load_misses_per_step']:>19,.0f}"
              f"{r['llc_all_misses_per_step']:>15,.0f}{r['llc_load_miss_rate']:>7.3f}")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--runner":
        runner(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
    else:
        main()
