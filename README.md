# PadSink-Track: a calibrated retention framework for Whisper's cross-attention KV cache on CPUs

Measurement code, result files and figures for *"PadSink-Track: A Calibrated
Framework for Padding-Sink-Preserving, Alignment-Tracked Cross-Attention
Key–Value Cache Retention in CPU-Based Whisper Speech Recognition"*
(manuscript, 2026). Companion to the cache-aware compression framework in
[nnopt](https://github.com/Mannon-Ochilov/nnopt), which handles the weights;
this study handles the activations the decoder re-reads at every step.

## The problem

Whisper's decoder re-reads the encoder's cross-attention key/value cache for
every token it emits. For Whisper-medium that cache is
24 layers × 2 tensors × 1500 positions × 1024 channels × 4 bytes ≈ **281 MiB**,
about 11.7× the 24 MiB L3 of the test machine. The decoder step is
memory-bound (arithmetic intensity 1.3–1.9 FLOP/byte against a ridge of 4.6),
so what the step *reads* is what it costs.

For short utterances 74–82 % of the 1500 positions are padding. The obvious
move, dropping the padding, fails: a few **padding-sink** positions carry a
large share of the attention mass, and removing them raises WER sharply in all
four models tested. The less obvious finding is that *one-shot* selection,
choosing a set of positions at the first step and keeping it (calibrated
split, H2O, SnapKV, PyramidKV), also fails once the budget is tight: at half
the calibrated budget every one-shot rule is rejected, while an oracle that
re-selects at every step passes even at a quarter of it. The positions the
decoder needs move through the audio as decoding proceeds.

## The method

**PadSink-Track** keeps the full cache in DRAM and reads a small, changing
working set per step:

- **padding sink** `S_ℓ`: per layer, the smallest set of padding positions
  holding a fraction ρ of the first-step padding attention mass, capped at
  25 % of the budget; chosen once, kept with its exact K/V;
- **tracked window** `W_t`: `w_ℓ = k − |S_ℓ|` consecutive audio positions,
  `[c_t − 0.1 w, c_t + 0.9 w)`, where the centre `c_t` follows the attention
  peak of Whisper's own **alignment heads** (the heads used for word
  timestamps) and advances monotonically; at the end of speech the window
  runs into the padding, which lets the decoder find the end;
- **ring buffer**: the selected K/V live in a fixed-size buffer; each step
  overwrites only the slots that left the window (13–17 of them on average,
  7–19 % of the window), so there is no per-step gather. Cross-attention
  does not depend on slot order.

The three settings tuned for the method (backward fraction 0.1, sink cap
0.25, alignment heads as the peak source) were chosen on the
Whisper-small/English validation set only and applied unchanged everywhere.

## The framework

Plain PadSink-Track passes the quality gate on the two Uzbek fine-tunes and on
the original Whisper-small, but not on the original Whisper-medium, in any of
four conditions (English on two corpora, Russian, Turkish). That failure is a
property of the checkpoint, not of the data or of fine-tuning: an English
fine-tune of whisper-medium fails the same way, and the share of attention
outside sink and window does not predict it.

The framework therefore calibrates three quantities per model on validation
data and adds a **confidence fallback**:

| Quantity | Rule |
|---|---|
| budget `K_i` | split rule `(f_r, f_p)`: smallest kept set within the gate |
| sink mass `ρ` | smallest padding set holding ρ of the first-step padding mass |
| fallback threshold `τ` | a step whose top-1 probability is below τ is redone on the full cache (still in DRAM); the peak is then searched over all audio |

`τ` is picked on 300 validation utterances: the arms *(no fallback, 0.6, 0.7,
0.8, 0.9, 0.95)* are tried in order of cost, and the first one is taken whose
ΔWER upper 95 % bound is below δ **and** which produces no utterance with
WER > 1 that the full cache does not; otherwise the model stays on the full
cache. The gate is non-inferiority: δ = 0.2 · WER_full, paired bootstrap over
utterances (10 000 resamples, seed 20260916), accept if U < δ, reject if
L > δ, inconclusive otherwise.

## Results

**Half the calibrated budget, 300 test utterances, ΔWER vs the full cache.**

| Method | medium/uz (k = 196) | small/uz (98) | medium/en (174) | small/en (174) |
|---|---|---|---|---|
| calibrated split | +0.40 | +0.14 | +0.17 | +0.26 |
| H2O per layer | +0.61 | +0.28 | +2.52 | +2.53 |
| SnapKV | +4.61 | +0.94 | +3.38 | +4.53 |
| PyramidKV | +0.43 | +0.53 | +3.00 | +3.40 |
| sink + one-shot audio | +0.20 | +0.26 | +0.41 | +1.30 |
| fixed-rate window (MURMUR-style) | +0.56 | +0.46 | +0.18 | +0.15 |
| **PadSink-Track** | **+0.0083** | **+0.013** | +0.022 (rejected) | **−0.0015** |
| oracle (not realizable) | +0.0049 | −0.0027 | +0.0014 | −0.0016 |

Everything above the PadSink-Track row is rejected at δ. The ablation shows
both components are necessary: without the sink ΔWER is +0.93 … +5.80,
without the moving window +1.48 … +4.62; a window that advances at a fixed
rate fails on all four models, and summing all heads instead of the alignment
heads is worse everywhere.

**Calibrated framework, ten setups (ε = 0.20).**

| Model | Corpus | Chosen arm | Test ΔWER [95 % CI] | Steps redone |
|---|---|---|---|---|
| medium (Uzbek fine-tune) | CV uz | no fallback | +0.0083 [−0.0031, +0.0207] | 0 % |
| small (Uzbek fine-tune) | CV uz | no fallback | +0.0131 [+0.0014, +0.0263] | 0 % |
| small (original) | LibriSpeech | no fallback | −0.0015 [−0.0036, +0.0006] | 0 % |
| small (original) | CV en | no fallback | +0.0165 [+0.0024, +0.0320] | 0 % |
| small (original) | CV ru | no fallback | +0.0020 [−0.0087, +0.0131] | 0 % |
| small (original) | CV tr | τ = 0.8 | +0.0118 [−0.0014, +0.0293] | 35 % |
| medium (original) | LibriSpeech | τ = 0.9 | +0.0039 [+0.0018, +0.0064] | 18 % |
| medium (original) | CV en | τ = 0.8 | +0.0112 [+0.0062, +0.0169] | 31 % |
| medium (original) | CV ru | τ = 0.7 | +0.0120 [+0.0043, +0.0218] | 13 % |
| medium (original) | CV tr | τ = 0.6 | +0.0174 [+0.0060, +0.0291] | 19 % |

All ten are accepted at ε = 0.20. The original medium never passes at
ε = 0.10, and a single τ for all models (0.9) costs most of the saving on the
models that did not need it — which is why τ is calibrated per model.

**Cost (Intel i7-11850H, one thread, ONNX Runtime, dynamic INT8 decoder).**

| | Whisper-medium (Uzbek) | Whisper-small (English) |
|---|---|---|
| decoder step, full cache → ring buffer | 36.2 → 24.5 ms (−32 %) | 13.0 → 8.6 ms (−34 %) |
| L3 load misses per step | 6.68 M → 3.42 M (−49 %) | 2.18 M → 1.11 M |
| DRAM read per step (counters) | 700 → 446 MiB (−36 %) | 240 → 144 MiB (−40 %) |
| whole utterance, ms/token, no fallback | 61.3 → 53.3 (−13 %) | 21.4 → 18.5 (−14 %) |
| eight single-thread decoders, aggregate | 59.8 → 91.2 steps/s (+53 %) | — |

A per-step gather is slower than the full cache (41.7 ms); the ring buffer is
what makes the tracked set cost the same as a one-shot set of the same size.
Step time is linear in the bytes read (≈ 22 GB/s effective, R² = 1.000), and
the byte ratio predicts the time ratio within 6–8 %. Where the fallback is
needed (original medium/English, 18 % of steps redone) the whole-utterance
saving drops to about 4 %: the fallback buys quality, the speed depends on
the model.

On longer audio (LibriSpeech ≥ 10 s; 15–30 s Uzbek composites) a fixed
working set of 200–400 positions per layer passes the gate where the
calibrated one-shot split needs 582 and 1022 positions.

## What did not work

Reported in the result files alongside what did:

- plain PadSink-Track on the original whisper-medium (four conditions) and
  on an English fine-tune of it (`results_align_track_medium_en*.json`,
  `*_medium_en_ftc*.json`);
- a larger or uncapped sink, stopping the window at the end of speech
  (`results_medium_en_diag.json`, `results_adaptive_adapt_*.json`); the sink
  does not drift (`results_sink_drift_*.json`);
- the attention share outside sink and window as a predictor of failure
  (`results_track_coverage_*.json`);
- RefreshKV-style periodic re-selection: R = 4 passes on one model only and
  is not faster than the full cache (`results_refresh_*.json`);
- a fixed τ = 0.9 for every model (`results_indep_fb_*.json`,
  `results_e2e_latency_*_fb.json`);
- the first calibration rule for τ (smallest accepted τ on 100 utterances),
  which picked τ = 0.5 for medium/Uzbek and was inconclusive on test; the
  300-utterance rule with the looping condition replaced it.

Predictions were written into the docstrings before the runs
(`prepare_medium_en_ft.py` F1–F9, `build_cv_en.py` F4–F6 and L1–L3,
`adaptive_track.py` A1–A2, `indep_fb.py` C1–C2, `framework.py` K1–K3,
`long_fb.py`).

## Reproducing

The scripts read the exported decoders and the encoder-state caches through
`experiments/kvlib.py`; they expect the `nnopt` checkout and its virtual
environment next to this folder (`PY` in the `run_*.sh` scripts). Model
checkpoints and audio are not included; a script that needs them prints the
expected path and exits.

```bash
# one setup end to end: encoder states, split calibration, rho, PadSink-Track
sh experiments/run_lang.sh small_en_cv

# per-model tau calibration (300 validation utterances) and test evaluation
sh experiments/run_framework.sh medium_en

# whole-utterance timing: full cache, no fallback, calibrated tau, one session
python experiments/e2e_latency.py --setup medium_en --tau auto

# figures, in the colours of the schematics
python experiments/make_figure_method2.py        # Figure 1
python experiments/make_figure_framework4.py     # Figure 2
python experiments/make_figure_trace2.py         # Figure 5
python experiments/make_figures_v2.py            # Figures 3, 4, 6-11
```

Python 3.12, onnxruntime 1.28, transformers 4.57, Optimum export with past
key/values, decoder in dynamic INT8 with the output layer kept separate.
Counters via Intel VTune. Measurements are 21 interleaved rounds after three
warm-ups; medians and [min–max] are in the result files.

Checkpoints: `Mannon/uzbek_stt_v1`, `Mannon/whisper-small-uzbek`,
`openai/whisper-medium`, `openai/whisper-small`,
`deepdml/whisper-medium-en-cv17` (Hugging Face). Data: Common Voice (Uzbek;
English, Russian, Turkish from Common Voice 17) and LibriSpeech.

## Layout

```
experiments/        scripts, their results_*.json, run_*.sh pipelines
figures/            fig_t1 … fig_t11 (PNG 600 dpi + PDF); *_v2 are the paper versions
RESULTS_INDEX.md    which script and result file each table and figure comes from
```

`RESULTS_INDEX.md` maps every number in the paper to its result file. Where a
quantity was measured in more than one session (whole-utterance timing, DRAM
counters), both are kept and the index says which one the paper reports.

## Citation

```bibtex
@article{ochilov_padsink_track,
  title  = {PadSink-Track: A Calibrated Framework for Padding-Sink-Preserving,
            Alignment-Tracked Cross-Attention Key--Value Cache Retention in
            CPU-Based Whisper Speech Recognition},
  author = {Ochilov, Mannon and Musaev, Muhammadjon and Kakharov, Shukrullo
            and Nabieva, Dilorom and Abdullaev, Sherzod and Kubayev, Ulugbek
            and Khujayarov, Ilyos},
  year   = {2026}
}
```
