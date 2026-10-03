#!/bin/sh
# R6 chain (prepare_medium_en_ft.py docstring): one process at a time.
#   prepare -> split calibration + full/split test -> rho on validation
#   -> attention coverage (F1) -> PadSink-Track 0.5 K_i on test (F2)
set -e
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export KV_FT=1 PYTHONIOENCODING=utf-8
for m in ${*:-ftk ftm}; do
  s=medium_en_$m
  echo "=== $s"
  $PY prepare_medium_en_ft.py --model "$m"
  $PY eviction_budget.py --setup "$s" --phase both
  $PY spar.py --setup "$s" --phase valid
  $PY track_coverage.py --setup "$s" --n 40
  $PY align_track.py --setup "$s" --n 300 --only track/s0.5
done
echo "R6 tugadi"
