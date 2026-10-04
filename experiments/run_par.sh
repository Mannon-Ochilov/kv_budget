#!/bin/sh
# Parallel pipelines for the setups given as arguments (after preparing the
# deepdml fine-tune). From run_cv.sh, parallel: decoding is single-threaded, so the calibration grid
# and the test arms run as separate processes (par_arms.py) and the two models
# run side by side. Results are identical to the sequential run.
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export KV_FT=1 PYTHONIOENCODING=utf-8

pipeline() {
  s=$1
  echo "=== $s: encoder states"
  $PY -c "from kvlib import SETUPS, encoder_states as e; s=SETUPS['$s']; e(s,'validation',100); e(s,'test',300)" || return 1
  echo "=== $s: calibration grid, 6 processes"
  for fr in -1 1.0 0.75 0.5 0.35 0.25; do
    $PY par_arms.py --setup "$s" --phase calib --f-real "$fr" > "_log_${s}_calib_$fr.txt" 2>&1 &
  done
  wait
  $PY par_arms.py --setup "$s" --merge
  $PY eviction_budget.py --setup "$s" --phase calib || return 1
  echo "=== $s: test arms (5) + rho on validation, in parallel"
  for a in 0 1 2 3 4; do
    $PY par_arms.py --setup "$s" --phase test --arm "$a" > "_log_${s}_test_$a.txt" 2>&1 &
  done
  $PY spar.py --setup "$s" --phase valid > "_log_${s}_spar.txt" 2>&1 &
  wait
  $PY par_arms.py --setup "$s" --merge
  $PY eviction_budget.py --setup "$s" --phase test || return 1
  grep "valid spar" "_log_${s}_spar.txt"
  echo "=== $s: coverage + PadSink-Track, in parallel"
  $PY track_coverage.py --setup "$s" --n 40 > "_log_${s}_coverage.txt" 2>&1 &
  $PY align_track.py --setup "$s" --n 300 --only track/s0.5 > "_log_${s}_track.txt" 2>&1 &
  wait
  grep "mean over layers" "_log_${s}_coverage.txt"
  grep "track/s0.5" "_log_${s}_track.txt"
  echo "=== $s tugadi"
}

$PY prepare_medium_en_ft.py --model ftc || exit 1
for s in "$@"; do pipeline "$s" > "_log_$s.txt" 2>&1 & done
wait
echo "PAR tugadi"
