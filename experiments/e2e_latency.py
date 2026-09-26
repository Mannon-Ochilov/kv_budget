"""L5 -- whole-utterance decoding time with the real PadSink-Track loop.

track_latency.py timed one synthetic step. Here the full greedy decode of
real test utterances is timed (first step + every later step, Python loop
included; the encoder is excluded -- it is the same for every arm):

  full        FP32 cross cache, 1500 positions (kvlib.greedy)
  oneshot_Ki  FP32 cache pruned once to the calibrated K_i (kvlib.greedy)
  track_ring  PadSink-Track v2 at 0.5 K_i with a persistent ring buffer:
              per layer a [1, H, k, dh] buffer; each step only the slots of
              positions that left the window are overwritten with the ones
              that entered (a jump larger than the window rebuilds it)

Arms are interleaved per utterance (rotating order), 1 thread. Reported:
decode ms per utterance, ms per output token, and the real-time factor of
the decoder (decode time / audio duration). track_ring must produce the
same tokens as align_track.track_greedy (same kept set; only the slot
order differs) -- the token agreement is reported as a check.

Prediction (before the run): per token, track_ring is within 10% of
oneshot_Ki and at least 25% below full on medium_uz.

Usage:  python experiments/e2e_latency.py --setup medium_uz --n 100
"""

import argparse
import json
import os
import time

import numpy as np

import align_track as A
from diag_budget import RHO
from eviction_budget import keep_split
from kvlib import (ENC_POS, EOT, MAX_NEW, SETUPS, SOT, encoder_states, greedy,
                   prompt_ids, real_positions, session, with_attention)
from spar import sink_set

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000


def track_ring(setup, first, sa, enc, prompt, k, rn, heads):
    ids = [SOT, *prompt]
    out = first.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                           "encoder_hidden_states": np.ascontiguousarray(enc[None])})
    names = [o.name for o in first.get_outputs()]
    logits = out[names.index("logits")]
    present = {n: v for n, v in zip(names, out) if n.startswith("present")}
    att = {int(n.split("layers.")[1].split("/")[0]): v[0, :, -1, :] for n, v in zip(names, out)
           if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    L = setup.n_layers
    c = int(np.argmax(sum(att[l][h, :rn] for l, h in heads)))
    # per layer: sink slots fixed, window slots in a ring
    sinks, slot_pos, buf = [], [], {}
    for l in range(L):
        s = rn + sink_set(att[l].sum(0)[rn:], RHO[setup.name])
        sinks.append(np.sort(s[:max(1, int(A.TRACK_CAP * k))]).astype(int))
    w = [max(1, k - len(sinks[l])) for l in range(L)]
    lo = [int(np.clip(c - int(A.BACK * w[l]), 0, ENC_POS - w[l])) for l in range(L)]
    for l in range(L):
        # sink positions inside the window would be duplicated; drop them from the sink list
        win = np.arange(lo[l], lo[l] + w[l])
        sk = sinks[l][~np.isin(sinks[l], win)]
        pos = np.concatenate([win, sk])
        slot_pos.append(pos)
        for kv in ("key", "value"):
            n = f"present.{l}.encoder.{kv}"
            buf[n] = np.ascontiguousarray(present[n][:, :, pos, :])
    full_cross = {n: v for n, v in present.items() if ".encoder." in n}
    s_in = [i.name for i in sa.get_inputs()]
    s_out = [o.name for o in sa.get_outputs()]
    att_idx = {int(n.split("layers.")[1].split("/")[0]): j for j, n in enumerate(s_out)
               if "encoder_attn" in n and n.endswith("Softmax_output_0")}
    nxt = int(np.argmax(logits[0, -1]))
    for _ in range(MAX_NEW):
        if nxt == EOT:
            break
        ids.append(nxt)
        for l in range(L):
            new_lo = int(np.clip(c - int(A.BACK * w[l]), 0, ENC_POS - w[l]))
            if new_lo == lo[l]:
                continue
            pos = slot_pos[l]
            leaving = np.arange(lo[l], min(new_lo, lo[l] + w[l])) if new_lo > lo[l] else \
                np.arange(max(new_lo + w[l], lo[l]), lo[l] + w[l])
            entering = np.setdiff1d(np.arange(new_lo, new_lo + w[l]), pos)
            slots = np.where(np.isin(pos, leaving))[0]
            m = min(len(slots), len(entering))
            if m:
                slots, entering = slots[:m], entering[:m]
                pos[slots] = entering
                for kv in ("key", "value"):
                    n = f"present.{l}.encoder.{kv}"
                    buf[n][:, :, slots, :] = full_cross[n][:, :, entering, :]
            lo[l] = new_lo
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64)}
        for n in s_in:
            if n.startswith("past_key_values"):
                pn = "present" + n[len("past_key_values"):]
                feed[n] = buf[pn] if ".encoder." in pn else present[pn]
        res = sa.run(None, feed)
        logits = res[s_out.index("logits")]
        for n, v in zip(s_out, res):
            if n.startswith("present") and ".decoder." in n:
                present[n] = v
        score = np.zeros(ENC_POS)
        for l, h in heads:
            np.add.at(score, slot_pos[l], res[att_idx[l]][0, h, -1, :])
        score[rn:] = 0
        c = max(c, int(np.argmax(score)))
        nxt = int(np.argmax(logits[0, -1]))
    return ids[1 + len(prompt):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="medium_uz", choices=list(SETUPS))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    heads = A.align_heads(setup)
    e6 = json.load(open(os.path.join(HERE, f"results_eviction_budget_{setup.name}.json")))
    f_r, f_p = e6["choice"]["rule"][1], e6["choice"]["rule"][2]
    states, waves, _ = encoder_states(setup, "test", 300)
    tok, prompt = prompt_ids(setup)
    first = session(with_attention(setup.first))
    step = session(setup.step)
    sa = session(with_attention(setup.step))

    def k_i(i):
        return len(keep_split(f_r, f_p, real_positions(waves[i]))(np.zeros(ENC_POS)))

    arms = {
        "full": lambda i: greedy(setup, first, step, states[i], prompt, lambda m: None, lambda a, b: (a, b))[0],
        "oneshot_Ki": lambda i: greedy(setup, first, step, states[i], prompt,
                                       keep_split(f_r, f_p, real_positions(waves[i])), lambda a, b: (a, b))[0],
        "track_ring": lambda i: track_ring(setup, first, sa, states[i], prompt,
                                           max(1, int(round(0.5 * k_i(i)))), real_positions(waves[i]), heads),
    }
    for fn in arms.values():      # warm-up
        fn(0)
    t = {a: [] for a in arms}
    ntok = {a: [] for a in arms}
    agree = []
    order = list(arms)
    for i in range(args.n):
        order = order[1:] + order[:1]
        for a in order:
            t0 = time.perf_counter()
            ids = arms[a](i)
            t[a].append(time.perf_counter() - t0)
            ntok[a].append(max(1, len(ids)))
            if a == "track_ring":
                ref_ids, _ = A.track_greedy(setup, first, sa, states[i], prompt,
                                            max(1, int(round(0.5 * k_i(i)))), real_positions(waves[i]), heads, False)
                agree.append(ids == ref_ids)
    audio_s = np.array([len(w) / SR for w in waves[:args.n]])
    out = {"n": args.n, "arms": {}, "token_agreement_track": float(np.mean(agree))}
    print(f"\n{setup.name}: {args.n} utterances, decoder only, 1 thread")
    for a in arms:
        tt = np.array(t[a])
        per_tok = tt * 1e3 / np.array(ntok[a])
        out["arms"][a] = {"ms_per_utt_median": float(np.median(tt) * 1e3), "ms_per_token_median": float(np.median(per_tok)),
                          "rtf_decoder": float(tt.sum() / audio_s.sum()), "tokens_mean": float(np.mean(ntok[a]))}
        print(f"  {a:<11} {np.median(tt) * 1e3:8.0f} ms/utt   {np.median(per_tok):6.1f} ms/token   "
              f"RTF(dec) {tt.sum() / audio_s.sum():.3f}   tokens {np.mean(ntok[a]):.1f}")
    print(f"  track_ring tokens identical to align_track: {np.mean(agree):.0%}")
    json.dump(out, open(os.path.join(HERE, f"results_e2e_latency_{setup.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
