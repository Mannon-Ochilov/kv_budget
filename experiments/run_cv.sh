#!/bin/sh
# Original whisper-medium and whisper-small on Common Voice English only
# (build_cv_en.py, predictions F4/F5). One process at a time.
set -e
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export KV_FT=1 PYTHONIOENCODING=utf-8
for s in medium_en_cv small_en_cv; do
  echo "=== $s"
  $PY eviction_budget.py --setup "$s" --phase both
  $PY spar.py --setup "$s" --phase valid
  $PY track_coverage.py --setup "$s" --n 40
  $PY align_track.py --setup "$s" --n 300 --only track/s0.5
done
echo "CV tugadi"
