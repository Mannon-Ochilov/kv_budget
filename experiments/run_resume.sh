#!/bin/sh
# Resume after the corrupt encoder-state caches were removed: redo the test
# phase for the original medium in ru / tr, then the framework on the six
# Common Voice setups.
cd "$(dirname "$0")"
sh run_lang.sh medium_ru_cv medium_tr_cv > lang_log2.txt 2>&1
sh run_framework.sh medium_en_cv small_en_cv medium_ru_cv medium_tr_cv small_ru_cv small_tr_cv > fw2_log.txt 2>&1
echo done > fw2_done.txt
