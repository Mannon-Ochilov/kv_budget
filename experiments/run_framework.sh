#!/bin/sh
# framework.py for the setups given as arguments: calibration arms in three
# processes per setup, then selection and the single evaluation run.
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export KV_FT=1 PYTHONIOENCODING=utf-8
one() {
  s=$1
  $PY -c "import framework as F, kvlib; kvlib.encoder_states(F.calib_setup(kvlib.SETUPS['$s']), 'calib', 300)" || return 1
  for a in full,none 0.6,0.7,0.8 0.9,0.95; do
    $PY framework.py --setup "$s" --stage calib --arms "$a" > "_log_fw_${s}_$a.txt" 2>&1 &
  done
  wait
  $PY framework.py --setup "$s" --stage select
  $PY framework.py --setup "$s" --stage eval
}
for s in "$@"; do one "$s" > "_log_fw_$s.txt" 2>&1 & done
wait
echo "FW tugadi"
