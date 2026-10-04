"""R6b -- Common Voice English (CV 17, fixie-ai/common_voice_17_0 mirror) test
and validation sets for the fine-tune comparison.

The Kenyan fine-tune (medium_en_ftk) was trained on a Common Voice subset plus
Kenyan English and degrades on LibriSpeech (test WER 0.163 against 0.036 for
the original; empty and looping outputs). On LibriSpeech it is therefore not a
fair representative of a working fine-tune. Here both the original medium and
the fine-tune are run on the same in-domain data, so only the fine-tuning
differs. Common Voice splits are speaker-disjoint, so no test speaker was in
the fine-tune's training data.

  cv_en_test300.npz        first 300 test rows, <= 10 per speaker, >= 1 s
  cv_en_validation100.npz  first 100 validation rows, same filters

Runs on these sets (run_r6b.sh): medium_en_cv (original medium),
medium_en_ftk_cv (Kenyan fine-tune), small_en_cv (original small), each with
its own split calibration and validation rho, frozen Track settings.

Pre-registered predictions (written before any run on these sets):
  F4  the original medium fails at 0.5 K_i on Common Voice English as on
      LibriSpeech, and its pad_out stays high (>= 30 %) -- the failure is a
      property of the checkpoint, not of LibriSpeech;
  F5  the original small passes, as on LibriSpeech;
  F6  the Kenyan fine-tune has a lower pad_out than the original medium on
      the same utterances, and passes; F3 (absolute delta of the original
      medium on the same set) applies.

R9 (2026-10-04) -- other languages. openai/whisper-medium is one multilingual
checkpoint, so the question "is the failure specific to English?" needs only
audio: the same builder with --lang ru / tr gives cv_{lang}_test300.npz and
cv_{lang}_validation100.npz. Setups (KV_FT=1): medium_ru_cv, medium_tr_cv,
small_ru_cv, small_tr_cv (original checkpoints, own split calibration and rho,
frozen Track settings) and medium_uzorig = the original medium on the Uzbek
Common Voice sets of medium_uz (same language and utterances as the Uzbek
fine-tune; its WER is high, so it is read through pad_out, not the gate).

Pre-registered predictions (before any run in these languages):
  L1  the original medium fails at 0.5 K_i in Russian and in Turkish, with
      pad_out >= 30 % -- the failure belongs to the checkpoint, not to English;
  L2  the original small passes in both;
  L3  on the Uzbek utterances the original medium's pad_out is >= 30 %,
      against 17.3 % for the Uzbek fine-tune.
  If L1 fails (medium passes in ru / tr), the failure is specific to English.

Usage:  python experiments/build_cv_en.py [--lang en|ru|tr]
"""

import io
import os
from collections import Counter

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "models", "_calib_cache")
PER_SPK = 10


def to16k(b):
    wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    return wav[:30 * 16000]


def build(split, n, fname, lang="en"):
    out = os.path.join(CACHE, fname)
    if os.path.exists(out):
        return
    ds = load_dataset("fixie-ai/common_voice_17_0", lang, split=split,
                      streaming=True).cast_column("audio", Audio(decode=False))
    waves, texts, spk, count = [], [], [], Counter()
    for r in ds:
        if count[r["client_id"]] >= PER_SPK or not r["sentence"].strip():
            continue
        w = to16k(r["audio"]["bytes"])
        if len(w) < 16000:
            continue
        waves.append(w)
        texts.append(r["sentence"])
        spk.append(r["client_id"])
        count[r["client_id"]] += 1
        if len(waves) == n:
            break
    np.savez(out, audio=np.concatenate(waves), lengths=np.array([len(w) for w in waves]),
             texts=np.array(texts, dtype=object), speakers=np.array(spk, dtype=object))
    print(f"{fname}: {len(waves)} utterances, {len(count)} speakers, "
          f"mean {np.mean([len(w) for w in waves]) / 16000:.1f} s", flush=True)


if __name__ == "__main__":
    import sys
    lang = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else "en"
    build("test", 300, f"cv_{lang}_test300.npz", lang)
    build("validation", 100, f"cv_{lang}_validation100.npz", lang)
