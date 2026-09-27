# Results index

Where each reported number comes from: script -> result file(s) in `experiments/`.
All WER results: 300 test utterances unless noted; paired bootstrap, seed 20260916, 10 000 resamples.

## Tables

| Table | Content | Script | Result file(s) |
|---|---|---|---|
| 2 | full-cache WER and gate delta per model | `eviction_budget.py` | `results_eviction_budget_{model}.json` (`test.full`) |
| 3 | cache precision (FP16 / int8 / int4 KIVI) | `cache_precision_wer.py` | `results_cache_precision_wer_cascade.json` (medium_uz), `results_cache_precision_wer_fp32.json` (small_en) |
| 4 | sink causal test (sink / audio / rest of padding -> mean) | `sink_causal.py` | `results_sink_causal_{model}.json`; FP32-encoder control: `results_sink_causal_medium_uz_fp32enc.json` |
| 5 | all rules at K_i | `eviction_budget.py`, `sota_baselines.py`, `spar.py`, `track_ablation.py` | `results_eviction_budget_*`, `results_sota_baselines_*`, `results_spar_*` (`test`), `results_track_ablation_*` (`track/s1.0`) |
| 6 | all rules at 0.5 K_i | `diag_budget.py`, `snapkv_half.py`, `align_track.py`, `track_ablation.py` | `results_diag_budget_*` (split / h2o_layer / pyramidkv / padsink / snapkv / oracle `s0.5`), `results_align_track_*` (`track/s0.5`), `results_track_ablation_*` (`fixed_rate/s0.5`, `track_int8/s0.5`) |
| 6 (note) | ring-buffer implementation WER | `ring_wer.py` | `results_ring_wer_{model}.json` |
| 6 (text) | speaker-disjoint fresh small_en test set | `fresh_small_en.py` | `results_fresh_small_en.json` |
| 7 | long audio, fixed k | `long_fixed_k.py` | `results_long_fixed_k_small_en_long.json`, `results_long_fixed_k_medium_uz_long.json` |
| 8 | ablation at 0.5 K_i | `track_ablation.py`, `align_track.py` | `results_track_ablation_*`, `results_align_track_*` (`track_sum/s0.5`) |
| 9 | step time, L3 misses, whole-utterance ms/token | `track_latency.py`, `llc_track.py`, `llc_miss_step.py`, `e2e_latency.py` | `results_track_latency.json`, `results_llc_miss_step.json`, `results_e2e_latency_{model}.json` |
| 9 (text) | step time at 2 / 4 / 8 threads | `track_latency.py --threads N` | `results_track_latency_t{N}.json`, `results_thread_scaling.json` |
| 10 | minimal L3 per family (short and long audio) | `l3_sweep.py` | `results_l3_sweep.json` (`min_l3`, `long`) |
| A1 | fixed-rate window rate r | `track_ablation.py` | `results_track_ablation_*` (`rate_validation`) |
| A2 | margin sensitivity (eps = 0.10-0.30) | `margin_sensitivity.py` | `results_margin_sensitivity.json` |
| A3 | DRAM bytes per step (memory-controller counters) | `dram_vtune.py` | `results_dram_vtune.json` |

## Figures

| Figure | Content | Script | Data |
|---|---|---|---|
| 1 | method schematic | `make_figures_track.py` (`fig_method_full`) | schematic |
| 2 | padding cliff (validation) + sink causal test | `make_figures_track.py` (`fig_padding`) | `results_eviction_budget_*` (`calib`), `results_sink_causal_*` |
| 3 | dWER vs budget (K_i, K_i/2, K_i/4) | `make_figures_track.py` (`fig_budget`) | as Tables 5-6 + `results_diag_budget_*` (`s0.25`, oracle) |
| 4 | tracked window over alignment-head attention | `trace_track.py`, `make_figures_track.py` (`fig_trace`) | `trace_track_medium_uz_22.npz` (rn = 443, k = 274) |
| 5 | long audio, fixed k | `make_figures_track.py` (`fig_long`) | `results_long_fixed_k_*` |
| 6 | ablation | `make_figures_track.py` (`fig_ablation`) | as Table 8 |
| 7 | step time / L3 misses / ms per token | `make_figures_track.py` (`fig_efficiency`) | as Table 9 |
| 8 | roofline and bytes-vs-time | `roofline.py`, `make_figures_track.py` (`fig_roofline`) | `results_roofline.json` |
| 9 | DRAM bytes vs model; N-stream throughput | `dram_vtune.py`, `concurrency_track.py`, `make_figures_track.py` (`fig_system`) | `results_dram_vtune.json`, `results_concurrency_track_medium_uz.json` |

## Other numbers quoted in the text

| Number | Source |
|---|---|
| oracle at 0.25 K_i passes on all four models | `results_diag_budget_*` (`oracle/s0.25`) |
| 35 % / 17 % of attention on padding outside sink and window | `track_coverage.py` -> `results_track_coverage_medium_en.json`, `results_track_coverage_medium_uz.json` |
| paired track-vs-baseline comparisons, length strata | `analysis_track_stats.py` -> `results_track_stats.json` |
| window shift in real decoding (~11 positions / step) | `trace_track_medium_uz_22.npz` (`wins`) |
| streamed weights 388 MiB = 24 x 14.06 + lm_head 50.6 | `roofline.py` (`graph_stats`) |
| medium_en checkpoint = original openai/whisper-medium (SHA-256) | `prepare_whisper_medium_en.py` (docstring) |

Figures are written to `figures/fig_t*.png|pdf`.
