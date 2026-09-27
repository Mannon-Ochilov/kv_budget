"""L11 -- DRAM bytes per decoder step from the memory-controller counters.

The roofline model counts the bytes a step must read (weights streamed +
cross K/V of the k positions + the decoder's own K/V). Here the integrated
memory controller's free-running CAS counters (UNC_MC0/1_RDCAS/WRCAS_COUNT_FREERUN,
64 B each) give the bytes actually moved from/to DRAM. Each arm runs in the
llc_track.py runner (full cache, one-shot K_i/2, PadSink-Track ring buffer;
t = 30, 1 thread) twice -- with the timed loop and with 0 s -- and the
difference per iteration is one step. The counters are package-wide, so
background activity adds a small positive bias.

Prediction (before the run):
  D1  DRAM read bytes per step are within 15 % of the model's bytes for the
      full cache and for PadSink-Track (the step streams its working set
      once; the 24 MiB L3 cannot hold it across steps);
  D2  PadSink-Track reads at least 30 % fewer DRAM bytes per step than the
      full cache (model: 36 %).

Results: DRAM read / model = 1.04, 1.03, 1.04 (medium: full, one-shot, Track)
and 1.00, 0.97, 0.97 (small); D1 confirmed. Track reads 36 % (medium) and 40 %
(small) fewer DRAM bytes per step than the full cache; D2 confirmed.

Usage:  python experiments/dram_vtune.py
"""

import json
import os
import re
import shutil
import subprocess

from llc_miss_step import PYTHON, VTUNE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_RUNS = r"D:\tmp\vtune_dram_runs"
EVENTS = ("UNC_MC0_RDCAS_COUNT_FREERUN", "UNC_MC1_RDCAS_COUNT_FREERUN",
          "UNC_MC0_WRCAS_COUNT_FREERUN", "UNC_MC1_WRCAS_COUNT_FREERUN")
SECONDS = 15.0


def run(model, arm, seconds, rdir):
    shutil.rmtree(rdir, ignore_errors=True)
    cmd = [VTUNE, "-collect-with", "runsa", "-knob", "event-config=" + ",".join(EVENTS), "-r", rdir, "--",
           PYTHON, os.path.join(HERE, "llc_track.py"), "--runner", model, arm, str(seconds)]
    txt = subprocess.run(cmd, capture_output=True, text=True)
    txt = (txt.stdout or "") + (txt.stderr or "")
    it = re.search(r"(\d+) iteratsiya", txt)
    rep = subprocess.run([VTUNE, "-report", "hw-events", "-r", rdir, "-format", "csv", "-csv-delimiter", "comma",
                          "-group-by", "package"], capture_output=True, text=True).stdout.splitlines()
    head, row = rep[0].split(","), rep[1].split(",")
    counts = {e: float(row[head.index("Uncore Event Count:" + e)]) for e in EVENTS}
    return counts, int(it.group(1)) if it else 0


def main():
    os.makedirs(ROOT_RUNS, exist_ok=True)
    roof = json.load(open(os.path.join(HERE, "results_roofline.json")))
    out = {}
    for model in ("medium_uz", "small_en"):
        model_bytes = {p["arm"].split(" ")[0]: p["bytes"] for p in roof["points"] if p["model"] == model}
        for arm in ("full", "oneshot_k", "track_ring"):
            tag = f"{model}_{arm}"
            base, _ = run(model, arm, 0.0, os.path.join(ROOT_RUNS, tag + "_base"))
            full, iters = run(model, arm, SECONDS, os.path.join(ROOT_RUNS, tag + "_full"))
            rd = sum(full[e] - base[e] for e in EVENTS if "RDCAS" in e) * 64 / iters
            wr = sum(full[e] - base[e] for e in EVENTS if "WRCAS" in e) * 64 / iters
            mb = model_bytes[arm]
            out[tag] = {"iters": iters, "dram_read_B": rd, "dram_write_B": wr, "model_B": mb, "ratio": rd / mb}
            print(f"  {model:<10} {arm:<11} {iters:5d} it  DRAM read {rd / 2**20:7.1f} MiB/step  write {wr / 2**20:5.1f}"
                  f"  model {mb / 2**20:7.1f} MiB  read/model {rd / mb:.2f}", flush=True)
            json.dump(out, open(os.path.join(HERE, "results_dram_vtune.json"), "w"), indent=1)
        f, t = out[f"{model}_full"]["dram_read_B"], out[f"{model}_track_ring"]["dram_read_B"]
        print(f"  {model}: PadSink-Track reads {1 - t / f:.0%} fewer DRAM bytes per step than the full cache")


if __name__ == "__main__":
    main()
