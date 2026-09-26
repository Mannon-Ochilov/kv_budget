"""L6 -- L3 misses per decoder step for PadSink-Track (ring buffer), VTune.

Same protocol as llc_miss_step.py (runsa, MEM_LOAD_RETIRED.L3_MISS / L3_HIT,
LONGEST_LAT_CACHE.MISS; each arm profiled with the timed loop and with
seconds = 0, the difference per iteration is one step). The arms come from
track_latency.make_arms: the full 1500-position FP32 cross cache stays
allocated (DRAM) and the step reads a k = 0.5 K_i ring buffer updated by
0-3 slots per step; oneshot_k is the same step on a cache pruned once.
Results are appended to results_llc_miss_step.json next to the full-cache
and K_i rows measured before.

Prediction (before the run): track_ring's L3 load misses per step are
within 15% of oneshot_k's and below the full cache's by at least the
factor measured between full and K_i on medium (the working set, not the
stored cache, sets the misses).

Usage:  python experiments/llc_track.py            (driver, needs VTune)
        python experiments/llc_track.py --runner SETUP ARM SECONDS
"""

import json
import os
import re
import sys
import time

from llc_miss_step import EVENTS, OUT_JSON, RESULT_ROOT, SECONDS

HERE = os.path.dirname(os.path.abspath(__file__))


def runner(setup, arm, seconds):
    sys.path.insert(0, HERE)
    from track_latency import make_arms
    arms = make_arms(setup)
    fn = [f for a, f in arms.items() if a.startswith(arm)][0]
    for _ in range(3):
        fn()
    it, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        it += 1
    print(f"{it} iteratsiya, {(time.perf_counter() - t0) * 1e3 / max(it, 1):.2f} ms/iter", flush=True)


def profile(setup, arm, seconds, result_dir):
    import shutil
    import subprocess
    from llc_miss_step import PYTHON, VTUNE
    shutil.rmtree(result_dir, ignore_errors=True)
    cmd = [VTUNE, "-collect-with", "runsa", "-knob", "event-config=" + ",".join(EVENTS),
           "-r", result_dir, "--", PYTHON, os.path.abspath(__file__), "--runner", setup, arm, str(seconds)]
    text = subprocess.run(cmd, capture_output=True, text=True)
    text = (text.stdout or "") + (text.stderr or "")
    counts = {}
    for ev in EVENTS:
        m = re.search(re.escape(ev) + r"\s+(\d+)\s", text)
        counts[ev] = int(m.group(1)) if m else None
    it = re.search(r"(\d+) iteratsiya", text)
    ms = re.search(r"([\d.]+) ms/iter", text)
    return counts, int(it.group(1)) if it else 0, float(ms.group(1)) if ms else None


def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)
    rows = json.load(open(OUT_JSON))
    for setup, pre in (("medium_uz", "medium "), ("small_en", "small  ")):
        for arm in ("oneshot_k", "track_ring"):
            label = f"{pre} {arm} (0.5 K_i)"
            if label in rows:
                continue
            tag = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
            base, _, _ = profile(setup, arm, 0.0, os.path.join(RESULT_ROOT, tag + "_base"))
            full, iters, ms = profile(setup, arm, SECONDS, os.path.join(RESULT_ROOT, tag + "_full"))
            if iters < 1 or any(full[e] is None or base[e] is None for e in EVENTS):
                print(f"  XATO: {label} natijasiz")
                continue
            per = {e: (full[e] - base[e]) / iters for e in EVENTS}
            miss, hit = per["MEM_LOAD_RETIRED.L3_MISS"], per["MEM_LOAD_RETIRED.L3_HIT"]
            rows[label] = {"model": setup, "positions": arm, "iterations": iters, "ms_per_iter": ms,
                           "llc_load_misses_per_step": miss, "llc_load_hits_per_step": hit,
                           "llc_all_misses_per_step": per["LONGEST_LAT_CACHE.MISS"],
                           "llc_load_miss_rate": miss / (miss + hit) if miss + hit > 0 else None}
            json.dump(rows, open(OUT_JSON, "w"), indent=1)
            print(f"  {label}: {ms:.1f} ms | L3 load misses/step {miss:,.0f} | all {per['LONGEST_LAT_CACHE.MISS']:,.0f}",
                  flush=True)
    for k, r in rows.items():
        print(f"{k:<34}{r['ms_per_iter']:>7.1f} ms {r['llc_load_misses_per_step'] / 1e6:7.2f} M load"
              f"{r['llc_all_misses_per_step'] / 1e6:7.2f} M all")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--runner":
        runner(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        main()
