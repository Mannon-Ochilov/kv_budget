"""R4 -- independent test sets: never used before, speaker-disjoint, >= 20 speakers.

English: LibriSpeech test-clean utterances whose speaker occurs neither in the
300-utterance test set, nor in the fresh 300-utterance set, nor in dev-clean
(the validation split). At most PER_SPK utterances per speaker, >= 20 speakers.

Uzbek: Common Voice (yakhyo/mozilla-common-voice-uzbek) test split beyond the
first 600 rows (0-299 = test set, 300-599 = long-audio composites), whose
client_id occurs neither in those 600 rows nor in the 100 validation rows.
Same per-speaker cap.

Each set: N utterances, stored with speaker ids next to audio and text.

Usage:  python experiments/build_independent_sets.py
"""

import io
import os
from collections import Counter

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "models", "_calib_cache")
N, PER_SPK = 500, 25


def to16k(b):
    wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    return wav[:30 * 16000]


def save(path, waves, texts, spk):
    np.savez(path, audio=np.concatenate(waves), lengths=np.array([len(w) for w in waves]),
             texts=np.array(texts, dtype=object), speakers=np.array(spk, dtype=object))
    c = Counter(spk)
    print(f"{os.path.basename(path)}: {len(waves)} utterances, {len(c)} speakers "
          f"(max {max(c.values())} per speaker), mean {np.mean([len(w) for w in waves]) / 16000:.1f} s", flush=True)


def english():
    out = os.path.join(CACHE, "ls_test_clean_indep500.npz")
    if os.path.exists(out):
        return
    used = set(np.load(os.path.join(CACHE, "ls_test_clean_fresh300.npz"), allow_pickle=True)["speakers"].tolist())
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True).cast_column("audio", Audio(decode=False))
    rows = list(ds)                                              # 2620 utterances
    used |= {r["speaker_id"] for r in rows[:300]}
    dev = load_dataset("openslr/librispeech_asr", "clean", split="validation",
                       streaming=True).cast_column("audio", Audio(decode=False))
    used |= {r["speaker_id"] for r in dev}
    by = {}
    for r in rows:
        if r["speaker_id"] not in used:
            by.setdefault(r["speaker_id"], []).append(r)
    rng = np.random.default_rng(20260928)
    picked = []
    for spk in sorted(by):
        rs = by[spk]
        idx = rng.permutation(len(rs))[:PER_SPK]
        picked += [rs[i] for i in sorted(idx)]
    rng.shuffle(picked)
    picked = picked[:N]
    save(out, [to16k(r["audio"]["bytes"]) for r in picked], [r["text"].lower() for r in picked],
         [str(r["speaker_id"]) for r in picked])


def uzbek():
    out = os.path.join(CACHE, "cv_uz_indep500.npz")
    if os.path.exists(out):
        return
    name = "yakhyo/mozilla-common-voice-uzbek"
    test = load_dataset(name, split="test", streaming=True).cast_column("audio", Audio(decode=False))
    val = load_dataset(name, split="validation", streaming=True).cast_column("audio", Audio(decode=False))
    used = set()
    for j, r in enumerate(val):
        if j >= 100:
            break
        used.add(r["client_id"])
    waves, texts, spk, count = [], [], [], Counter()
    for j, r in enumerate(test):
        if j < 600:
            used.add(r["client_id"])
            continue
        c = r["client_id"]
        if c in used or count[c] >= PER_SPK:
            continue
        waves.append(to16k(r["audio"]["bytes"]))
        texts.append(r["sentence"])
        spk.append(c)
        count[c] += 1
        if len(waves) == N:
            break
    save(out, waves, texts, spk)


if __name__ == "__main__":
    english()
    uzbek()
