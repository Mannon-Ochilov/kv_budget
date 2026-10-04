#!/bin/sh
# After the timing runs: calibration sets, the framework on the four paper
# models, the split / rho calibration and plain Track for the other-language
# setups, then the framework on those.
cd "$(dirname "$0")"
PY=${PY:-/d/DSc/ISH/nnopt/.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
until [ -f e2e_fbtau_done.txt ]; do sleep 20; done
$PY build_calib.py > _log_build_calib.txt 2>&1
sh run_framework.sh medium_uz small_uz medium_en small_en > fw1_log.txt 2>&1
echo done > fw1_done.txt
sh run_lang.sh medium_ru_cv medium_tr_cv small_ru_cv small_tr_cv > lang_log.txt 2>&1
sh run_framework.sh medium_en_cv small_en_cv medium_ru_cv medium_tr_cv small_ru_cv small_tr_cv > fw2_log.txt 2>&1
echo done > fw2_done.txt
