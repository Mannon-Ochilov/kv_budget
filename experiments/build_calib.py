"""Calibration sets of 300 utterances for the fallback threshold tau
(framework.py). 100 utterances do not show rare failures (a looping utterance
is about 1 in 100-150), so tau is calibrated on 300. All sets come from the
validation splits, which share no speaker with the test or independent sets:

  ls_dev_clean_calib300.npz   LibriSpeech dev-clean, first 300 utterances
  cv_uz_calib300.npz          Common Voice Uzbek validation, first 300 rows
  cv_{en,ru,tr}_calib300.npz  Common Voice 17 validation (<= 10 per speaker)

The first 100 of each are the utterances already used to calibrate K_i and rho.

Usage:  python experiments/build_calib.py
"""

import io
import os

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from build_cv_en import CACHE, build, to16k

N = 300


def save(out, waves, texts):
    np.savez(out, audio=np.concatenate(waves), lengths=np.array([len(w) for w in waves]),
             texts=np.array(texts, dtype=object))
    print(f"{os.path.basename(out)}: {len(waves)} utterances, mean {np.mean([len(w) for w in waves]) / 16000:.1f} s", flush=True)


def librispeech():
    out = os.path.join(CACHE, "ls_dev_clean_calib300.npz")
    if os.path.exists(out):
        return
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation",
                      streaming=True).cast_column("audio", Audio(decode=False))
    waves, texts = [], []
    for ex in ds:
        wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        assert sr == 16000
        waves.append(np.asarray(wav, dtype=np.float32))
        texts.append(ex["text"].lower())
        if len(waves) == N:
            break
    save(out, waves, texts)


def uzbek():
    out = os.path.join(CACHE, "cv_uz_calib300.npz")
    if os.path.exists(out):
        return
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation",
                      streaming=True).cast_column("audio", Audio(decode=False))
    waves, texts = [], []
    for r in ds:
        waves.append(to16k(r["audio"]["bytes"]))
        texts.append(r["sentence"])
        if len(waves) == N:
            break
    save(out, waves, texts)


if __name__ == "__main__":
    librispeech()
    uzbek()
    for lang in ("en", "ru", "tr"):
        build("validation", N, f"cv_{lang}_calib300.npz", lang)
