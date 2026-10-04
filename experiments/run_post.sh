#!/bin/sh
# After run_all_framework.sh: the original medium on the Uzbek sets (pad_out
# diagnostic) and long audio with the fallback, in parallel; then, alone on the
# machine, whole-utterance timing at the tau framework.py chose per model.
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
until [ -f fw2_done.txt ]; do sleep 30; done
sh run_lang.sh medium_uzorig > uzorig_log.txt 2>&1 &
for s in medium_uz_long small_en_long; do $PY long_fb.py --set $s > "_log_long_fb_$s.txt" 2>&1 & done
wait
for m in medium_uz medium_en small_uz small_en; do
  $PY e2e_latency.py --setup $m --n 100 --tau auto --only full,track_ring,track_tau --tag _fw > "_log_e2e_fw_$m.txt" 2>&1
done
echo done > post_done.txt
