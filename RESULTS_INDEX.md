# Results index

Where each reported number comes from: script -> result file(s) in `experiments/`. Table and figure numbers follow the manuscript (A = appendix).
All WER results: 300 test utterances unless noted; paired bootstrap, seed 20260916, 10 000 resamples.

## Tables

| Table | Content | Script | Result file(s) |
|---|---|---|---|
| 2 | models, full-cache WER and gate delta per model | `eviction_budget.py` | `results_eviction_budget_{model}.json` (`test.full`) |
| A6 | cache precision (FP16 / int8 / int4 KIVI) | `cache_precision_wer.py` | `results_cache_precision_wer_cascade.json` (medium_uz), `results_cache_precision_wer_fp32.json` (small_en) |
| 3 | sink causal test (sink / audio / rest of padding -> mean) | `sink_causal.py` | `results_sink_causal_{model}.json`; FP32-encoder control: `results_sink_causal_medium_uz_fp32enc.json` |
| A7 | all rules at K_i | `eviction_budget.py`, `sota_baselines.py`, `spar.py`, `track_ablation.py` | `results_eviction_budget_*`, `results_sota_baselines_*`, `results_spar_*` (`test`), `results_track_ablation_*` (`track/s1.0`) |
| 4 | all rules at 0.5 K_i | `diag_budget.py`, `snapkv_half.py`, `align_track.py`, `track_ablation.py` | `results_diag_budget_*` (split / h2o_layer / pyramidkv / padsink / snapkv / oracle `s0.5`), `results_align_track_*` (`track/s0.5`), `results_track_ablation_*` (`fixed_rate/s0.5`, `track_int8/s0.5`) |
| 4 (note) | ring-buffer implementation WER | `ring_wer.py` | `results_ring_wer_{model}.json` |
| 4 (text) | speaker-disjoint fresh small_en test set | `fresh_small_en.py` | `results_fresh_small_en.json` |
| 7 | long audio, fixed k (with the fallback rows: `long_fb.py`, `results_long_fb_*.json`) | `long_fixed_k.py` | `results_long_fixed_k_small_en_long.json`, `results_long_fixed_k_medium_uz_long.json` |
| 8 | ablation at 0.5 K_i | `track_ablation.py`, `align_track.py` | `results_track_ablation_*`, `results_align_track_*` (`track_sum/s0.5`) |
| 9, 10 | step time, L3 misses, whole-utterance ms/token | `track_latency.py`, `llc_track.py`, `llc_miss_step.py`, `e2e_latency.py` | `results_track_latency.json`, `results_llc_miss_step.json`, `results_e2e_latency_{model}.json` |
| 9 (text) | step time at 2 / 4 / 8 threads | `track_latency.py --threads N` | `results_track_latency_t{N}.json`, `results_thread_scaling.json` |
| A9 | minimal L3 per family (short and long audio) | `l3_sweep.py` | `results_l3_sweep.json` (`min_l3`, `long`) |
| A1 (fixed-rate row) | fixed-rate window rate r | `track_ablation.py` | `results_track_ablation_*` (`rate_validation`) |
| A2 | margin sensitivity (eps = 0.10-0.30) | `margin_sensitivity.py` | `results_margin_sensitivity.json` |
| A3 | DRAM bytes per step (memory-controller counters) | `dram_vtune.py` | `results_dram_vtune.json` |

## Figures

Paper versions are the `*_v2` files (colours of the schematics, black text); the files without suffix are the earlier renderings.

| Figure | Content | Script | Data |
|---|---|---|---|
| 1 | method schematic | `make_figure_method2.py` -> `fig_t1_method_v2` | schematic |
| 2 | framework schematic (calibration, decoding step, fallback) | `make_figure_framework4.py` -> `fig_t11_framework_v5` | schematic |
| 3 | padding cliff (validation) + sink causal test | `make_figures_v2.py` (`fig_padding`) -> `fig_t2_padding_v2` | `results_eviction_budget_*` (`calib`), `results_sink_causal_*` |
| 4 | dWER vs budget (K_i, K_i/2, K_i/4) | `make_figures_v2.py` (`fig_budget`) -> `fig_t3_budget_v2` | as Tables A7, 4 + `results_diag_budget_*` (`s0.25`, oracle) |
| 5 | tracked window over alignment-head attention | `trace_track.py`, `make_figure_trace2.py` -> `fig_t4_trace_v2` | `trace_track_medium_uz_22.npz` (rn = 443, k = 274) |
| 6 | tau: quality, redone steps, saving | `make_figures_v2.py` (`fig_tau`) -> `fig_t10_tau_v2` | `results_align_track_*`, `results_adaptive_fb_*`, `results_e2e_latency_*_fb*`, `results_framework_*` |
| 7 | long audio, fixed k | `make_figures_v2.py` (`fig_long`) -> `fig_t7_long_v2` | `results_long_fixed_k_*` |
| 8 | ablation | `make_figures_v2.py` (`fig_ablation`) -> `fig_t5_ablation_v2` | as Table 8 |
| 9 | step time / L3 misses / ms per token | `make_figures_v2.py` (`fig_efficiency`) -> `fig_t6_efficiency_v2` | as Table 9; panel (c) from `results_e2e_latency_{model}.json` (separate session; Table 10 reports the `_fw` session) |
| 10 | roofline and bytes-vs-time | `roofline.py`, `make_figures_v2.py` (`fig_roofline`) -> `fig_t8_roofline_v2` | `results_roofline.json` |
| 11 | DRAM bytes vs model; N-process throughput | `dram_vtune.py`, `concurrency_track.py`, `make_figures_v2.py` (`fig_system`) -> `fig_t9_system_v2` | `results_dram_vtune.json`, `results_concurrency_track_medium_uz.json` |

## Other numbers quoted in the text

| Number | Source |
|---|---|
| oracle at 0.25 K_i passes on all four models | `results_diag_budget_*` (`oracle/s0.25`) |
| 35 % / 17 % of attention on padding outside sink and window | `track_coverage.py` -> `results_track_coverage_medium_en.json`, `results_track_coverage_medium_uz.json` |
| paired track-vs-baseline comparisons, length strata | `analysis_track_stats.py` -> `results_track_stats.json` |
| window shift per step, 13-17 positions (7-19 % of the window), from the speech rate of the test sets | `analysis_track_stats.py`; one utterance (11 slots) in `trace_track_medium_uz_22.npz` (`wins`) |
| streamed weights 388 MiB = 24 x 14.06 + lm_head 50.6 | `roofline.py` (`graph_stats`) |
| medium_en checkpoint = original openai/whisper-medium (SHA-256) | `prepare_whisper_medium_en.py` (docstring) |

Figures are written to `figures/fig_t*.png|pdf`.

## Framework (calibrated fallback) and additional checkpoints / languages

Setups beyond the four paper models are registered in `kvlib.py` with `KV_FT=1`.

| Result | Script | Result file(s) |
|---|---|---|
| Framework: per-model tau calibration (300 validation utterances) and evaluation (300 test), 10 setups | `build_calib.py`, `framework.py` (`run_framework.sh`) | `results_framework_{setup}.json` |
| Fallback, fixed tau, tau sensitivity (0.5-0.95) on test; adaptive (uncapped) sink | `adaptive_track.py` | `results_adaptive_fb_{model}.json`, `results_adaptive_adapt_{model}.json` |
| tau = 0.9 on the independent 500-utterance sets | `indep_fb.py` | `results_indep_fb_{model}.json` |
| Whole-utterance timing: no fallback, tau = 0.5 / 0.7 / 0.9, calibrated tau | `e2e_latency.py` (`track_fb*`, `--tau auto`) | `results_e2e_latency_{model}_fb.json`, `_fb_tau.json`, `_fw.json` |
| Long audio with the fallback | `long_fb.py` | `results_long_fb_{set}.json` |
| Original medium / small on Common Voice English, Russian, Turkish: split calibration, rho, plain PadSink-Track, coverage | `build_cv_en.py`, `run_lang.sh` (`eviction_budget.py`, `spar.py`, `align_track.py`, `track_coverage.py`) | `results_eviction_budget_*_cv.json`, `results_spar_*_cv.json`, `results_align_track_*_cv.json`, `results_track_coverage_*_cv.json` |
| English fine-tune of whisper-medium (Common Voice 17), both English sets | `prepare_medium_en_ft.py --model ftc`, `run_par.sh` | `results_*_medium_en_ftc.json`, `results_*_medium_en_ftc_cv.json` |
| Sink drift diagnostic | `sink_drift.py` | `results_sink_drift_{model}.json` |
| Refresh baseline (R = 4 / 8 / 16), independent sets, medium_en validation diagnostic | `refresh_baseline.py`, `indep_eval.py`, `medium_en_diag.py` | `results_refresh_*.json`, `results_indep_*.json`, `results_medium_en_diag.json` |
| Figure: tau vs quality / redone steps / saving | `make_figure_tau.py` (earlier), `make_figures_v2.py` (paper) | `figures/fig_t10_tau*.*` |
| Figure: framework schematic | `make_figure_framework.py` … `make_figure_framework4.py` (paper: v5) | `figures/fig_t11_framework*.*` |

Pre-registered predictions are in the docstrings of `prepare_medium_en_ft.py` (F1-F9),
`build_cv_en.py` (F4-F6, L1-L3), `adaptive_track.py` (A1-A2), `indep_fb.py` (C1-C2), `framework.py` (K1-K3)
and `long_fb.py`.
