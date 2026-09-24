"""L3 misses per step for the hardware-selected point (int8 cache, K = 1433),
same VTune protocol as llc_miss_step.py; appended to results_llc_miss_step.json."""
import json
import os
import re

from llc_miss_step import EVENTS, OUT_JSON, RESULT_ROOT, SECONDS, profile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
K = json.load(open(os.path.join(HERE, "results_boundary_int8_medium_uz.json")))["K"]
label = f"medium  int8 cache, {K}"
path = os.path.join(ROOT, "models", "whisper_with_past", "decoder_with_past_cache_int.onnx")
rows = json.load(open(OUT_JSON))
os.makedirs(RESULT_ROOT, exist_ok=True)
tag = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
base, _, _ = profile(path, 16, K, 0.0, os.path.join(RESULT_ROOT, tag + "_base"))
full, iters, ms = profile(path, 16, K, SECONDS, os.path.join(RESULT_ROOT, tag + "_full"))
per = {e: (full[e] - base[e]) / iters for e in EVENTS}
miss, hit = per["MEM_LOAD_RETIRED.L3_MISS"], per["MEM_LOAD_RETIRED.L3_HIT"]
rows[label] = {"model": os.path.basename(path), "positions": K, "iterations": iters, "ms_per_iter": ms,
               "llc_load_misses_per_step": miss, "llc_load_hits_per_step": hit,
               "llc_all_misses_per_step": per["LONGEST_LAT_CACHE.MISS"],
               "llc_load_miss_rate": miss / (miss + hit)}
json.dump(rows, open(OUT_JSON, "w"), indent=1)
for k, r in rows.items():
    print(f"{k:<28}{r['llc_load_misses_per_step'] / 1e6:7.2f} M load{r['llc_all_misses_per_step'] / 1e6:7.2f} M all  rate {r['llc_load_miss_rate']:.3f}")
