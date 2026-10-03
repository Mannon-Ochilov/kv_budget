#!/bin/sh
# R6 re-ordered (build_cv_en.py): finish the Kenyan fine-tune on LibriSpeech,
# then the original medium and the Kenyan fine-tune on Common Voice English,
# then the original small on Common Voice English, then the medical fine-tune on LibriSpeech. One process at a time.
set -e
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export KV_FT=1 PYTHONIOENCODING=utf-8
steps() {
  $PY eviction_budget.py --setup "$1" --phase both
  $PY spar.py --setup "$1" --phase valid
  $PY track_coverage.py --setup "$1" --n 40
  $PY align_track.py --setup "$1" --n 300 --only track/s0.5
}
for s in medium_en_ftk medium_en_cv medium_en_ftk_cv small_en_cv; do
  echo "=== $s"
  steps "$s"
done
echo "=== medium_en_ftm"
$PY prepare_medium_en_ft.py --model ftm
steps medium_en_ftm
echo "R6 tugadi"
