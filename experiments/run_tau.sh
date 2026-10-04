#!/bin/sh
# tau = 0.9 confirmation on the independent sets (indep_fb.py) and the tau
# sensitivity on the 300 test utterances (reporting only; tau stays 0.9).
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
for m in medium_en medium_uz small_en small_uz; do
  $PY indep_fb.py --setup $m > "_log_indep_fb_$m.txt" 2>&1 &
  ( for t in 0.5 0.6 0.7 0.8 0.95; do $PY adaptive_track.py --setup $m --family fb --force-test $t; done > "_log_tau_$m.txt" 2>&1 ) &
done
wait
echo "TAU tugadi"
