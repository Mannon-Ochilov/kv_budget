#!/bin/sh
# R8 (adaptive_track.py): both families on the four paper models, in parallel.
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
for m in medium_en medium_uz small_en small_uz; do
  for f in adapt fb; do
    $PY adaptive_track.py --setup $m --family $f > "_log_r8_${f}_$m.txt" 2>&1 &
  done
done
wait
echo "R8 tugadi"
