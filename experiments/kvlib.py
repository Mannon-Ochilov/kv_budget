"""Shared pieces for the cache experiments, parameterized by model.

`cache_precision_wer.py` and `cache_eviction_wer.py` are tied to the Uzbek
Whisper-medium. E6/E7 run the same decoding loop on a second model, so the
model-specific paths, sizes and prompt live here, selected by name:

  medium_uz   Whisper-medium fine-tuned for Uzbek, the paper's model, with the
              paper's cascade encoder; Common Voice uz test / validation
  small_en    openai/whisper-small, FP32 encoder; LibriSpeech test-clean /
              dev-clean (prepare_whisper_small.py)

Both decoders are the untied with-past INT8 export; the first step is the
no-past INT8 decoder with its cross-attention Softmax outputs exposed.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NNOPT = "D:/DSc/ISH/nnopt"
sys.path.insert(0, os.path.join(NNOPT, "experiments"))
from wer_cer_whole_network import EOT, SOT, error_rate, normalize  # noqa: E402,F401

ENC_POS = 1500
SAMPLES_PER_POSITION = 320
SEED, RESAMPLES = 20260916, 10000
MAX_NEW = 96


@dataclass(frozen=True)
class Setup:
    name: str
    hf_dir: str          # tokenizer + feature extractor
    encoder: str
    first: str           # no-past INT8 decoder
    step: str            # with-past untied INT8 decoder
    audio: dict          # split -> npz cache
    language: str
    n_layers: int
    d_model: int
    normalizer: str      # "uz" (paper's) or "en" (Whisper's basic normalizer)


SETUPS = {
    "medium_uz": Setup(
        "medium_uz", os.path.join(NNOPT, "models", "hh"),
        os.path.join(NNOPT, "models", "_gptq", "enc_gptq_pruned.onnx"),
        os.path.join(ROOT, "models", "whisper_with_past", "decoder_model_int8.onnx"),
        os.path.join(ROOT, "models", "whisper_with_past", "decoder_with_past_untied_int8.onnx"),
        {"test": os.path.join(NNOPT, "models", "_calib_cache", "cv_uz_test.npz"),
         "validation": os.path.join(NNOPT, "models", "_calib_cache", "cv_uz_validation.npz")},
        "uz", 24, 1024, "uz"),
    "small_en": Setup(
        "small_en", os.path.join(ROOT, "models", "whisper_small_hf"),
        os.path.join(ROOT, "models", "whisper_small_onnx", "encoder_model.onnx"),
        os.path.join(ROOT, "models", "whisper_small_onnx", "decoder_model_int8.onnx"),
        os.path.join(ROOT, "models", "whisper_small_onnx", "decoder_with_past_untied_int8.onnx"),
        {"test": os.path.join(ROOT, "models", "_calib_cache", "ls_test_clean.npz"),
         "validation": os.path.join(ROOT, "models", "_calib_cache", "ls_dev_clean.npz")},
        "en", 12, 768, "en"),
    # navai-uz/whisper-small-uzbek: whisper-small fine-tuned for Uzbek (Apache-2.0),
    # FP32 encoder; same Common Voice uz splits as medium_uz
    "small_uz": Setup(
        "small_uz", os.path.join(ROOT, "models", "whisper_small_uz_hf"),
        os.path.join(ROOT, "models", "whisper_small_uz_onnx", "encoder_model.onnx"),
        os.path.join(ROOT, "models", "whisper_small_uz_onnx", "decoder_model_int8.onnx"),
        os.path.join(ROOT, "models", "whisper_small_uz_onnx", "decoder_with_past_untied_int8.onnx"),
        {"test": os.path.join(NNOPT, "models", "_calib_cache", "cv_uz_test.npz"),
         "validation": os.path.join(NNOPT, "models", "_calib_cache", "cv_uz_validation.npz")},
        "uz", 12, 768, "uz"),
    # openai/whisper-medium, original multilingual checkpoint, no fine-tuning:
    # same size as medium_uz, same language/benchmark as small_en
    "medium_en": Setup(
        "medium_en", os.path.join(ROOT, "models", "whisper_medium_en_hf"),
        os.path.join(ROOT, "models", "whisper_medium_en_onnx", "encoder_model.onnx"),
        os.path.join(ROOT, "models", "whisper_medium_en_onnx", "decoder_model_int8.onnx"),
        os.path.join(ROOT, "models", "whisper_medium_en_onnx", "decoder_with_past_untied_int8.onnx"),
        {"test": os.path.join(ROOT, "models", "_calib_cache", "ls_test_clean.npz"),
         "validation": os.path.join(ROOT, "models", "_calib_cache", "ls_dev_clean.npz")},
        "en", 24, 1024, "en"),
}


# ------------------------------------------------------------------ helpers
def session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def load_audio(path, n):
    z = np.load(path, allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths[:n]:
        waves.append(flat[off:off + int(ln)])
        off += int(ln)
    return waves, list(texts[:n])


def text_norm(setup):
    if setup.normalizer == "uz":
        return normalize
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    bn = BasicTextNormalizer()
    return lambda s: " ".join(bn(s).split())


def with_attention(first_path):
    """The first-step graph with every cross-attention Softmax as an output."""
    dst = first_path.replace(".onnx", "_attn.onnx")
    if not os.path.exists(dst):
        m = onnx.load(first_path, load_external_data=False)
        names = [n.output[0] for n in m.graph.node
                 if n.op_type == "Softmax" and "encoder_attn" in n.name]
        for nm in names:
            m.graph.output.append(onnx.helper.make_tensor_value_info(
                nm, onnx.TensorProto.FLOAT, None))
        onnx.save(m, dst)
    return dst


def encoder_states(setup, split, n, threads=8):
    """Encoder output for the first n items of a split, memory-mapped."""
    from transformers import WhisperFeatureExtractor
    path = os.path.join(ROOT, "models", f"enc_states_{setup.name}_{split}{n}.npy")
    waves, texts = load_audio(setup.audio[split], n)
    if os.path.exists(path):
        return np.load(path, mmap_mode="r"), waves, texts
    fe = WhisperFeatureExtractor.from_pretrained(setup.hf_dir)
    enc = session(setup.encoder, threads)
    st = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                   shape=(n, ENC_POS, setup.d_model))
    for i, wav in enumerate(waves):
        f = fe(wav, sampling_rate=16000,
               return_tensors="np").input_features.astype(np.float32)
        st[i] = enc.run(None, {"input_features": f})[0][0]
        if (i + 1) % 50 == 0:
            print(f"  encoder {i + 1}/{n}", flush=True)
    st.flush()
    del enc
    return np.load(path, mmap_mode="r"), waves, texts


def prompt_ids(setup):
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained(setup.hf_dir)
    return tok, [t for _, t in tok.get_decoder_prompt_ids(
        language=setup.language, task="transcribe")]


def real_positions(wave):
    return min(ENC_POS, -(-len(wave) // SAMPLES_PER_POSITION))


def paired_ci(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), (RESAMPLES, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def greedy(setup, first, step, enc, prompt, keep_fn, cache_fn):
    """One utterance. keep_fn(mass) -> kept index array or None;
    cache_fn(k, v) -> (k, v) applied to every cache tensor."""
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    mass = np.zeros(ENC_POS, np.float64)
    mass_l = np.zeros((setup.n_layers, ENC_POS), np.float64)
    for n, v in zip(names, out):
        if "encoder_attn" in n and n.endswith("Softmax_output_0"):
            layer = int(n.split("layers.")[1].split("/")[0])
            mass_l[layer] = v.sum(axis=(0, 1, 2))
            mass += mass_l[layer]
    # keep_fn(mass) -> one index set for every layer, or
    # keep_fn(mass, mass_l) -> {layer: index set} (per-layer baselines)
    try:
        idx = keep_fn(mass, mass_l)
    except TypeError:
        idx = keep_fn(mass)
    for i in range(setup.n_layers):
        kn, vn = f"present.{i}.encoder.key", f"present.{i}.encoder.value"
        k, v = present[kn], present[vn]
        idx_i = idx[i] if isinstance(idx, dict) else idx
        if idx_i is not None:
            k, v = k[:, :, idx_i, :], v[:, :, idx_i, :]
        present[kn], present[vn] = cache_fn(np.ascontiguousarray(k),
                                            np.ascontiguousarray(v))
    step_in = [i.name for i in step.get_inputs()]
    step_out = [o.name for o in step.get_outputs()]
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in step_in:
            if n.startswith("past_key_values"):
                feed[n] = present["present" + n[len("past_key_values"):]]
        res = step.run(None, feed)
        logits = res[step_out.index("logits")]
        for n, v in zip(step_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        for i in range(setup.n_layers):
            kn, vn = f"present.{i}.decoder.key", f"present.{i}.decoder.value"
            present[kn], present[vn] = cache_fn(present[kn], present[vn])
        nxt = int(np.argmax(logits[0, -1]))
    if idx is None:
        kept = ENC_POS
    elif isinstance(idx, dict):
        kept = float(np.mean([len(x) for x in idx.values()]))
    else:
        kept = len(idx)
    return ids[1 + len(prompt):], kept


def cache_mib(setup, positions, bits):
    return setup.n_layers * 2 * positions * setup.d_model * bits / 8 / 1024 ** 2
