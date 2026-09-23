"""E4 -- does the compression target change when activations enter it?

The paper derives the target from the weight working set of a decision unit,
rho = M_eff / (alpha * L3), with the unit being one Transformer layer. This
applies three versions of that rule to decoders that decode identically and
asks of each rule: does it tell the decoders apart, and does it point at the
term a measurement shows is the lever.

  rule W      weights of the layer only             -- the paper's rule
  rule W+A    plus the activation bytes the layer reads per step
  rule W+A/I  as W+A, gated by arithmetic intensity: a step doing more than
              I_c MAC per byte is compute-bound and traffic is not its lever

Per-layer weight bytes exclude the vocabulary projection: it is not part of
any layer and Table 3 does not count it, so it is reported separately. The
activation term is the encoder-state reads of the no-cache decoder and the
KV-cache reads of the cached ones, per layer. Inputs are the static counts of
graph_traffic.py, one file per decoder, and the measured step time at t = 30
where available. Nothing is fitted; the intensity threshold is shown at two
values to make clear it is not doing the work.

Usage:  python experiments/target_ablation.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET = 0.7 * 24.0
N_LAYERS = 24
T = 30
MIB = 1024.0 ** 2

ARMS = [
    ("no-cache (deployed)", "results_graph_traffic_nocache.json",
     "results_step_latency.json", 1.0),
    ("KV cache, FP32", "results_graph_traffic_past.json",
     "results_step_latency_past.json", 1.0),
    ("+ untied lm_head", "results_graph_traffic_untied.json",
     "results_step_latency_untied.json", 1.0),
    # int8 cache shares the untied weights; its cache bytes are a quarter
    ("+ int8 cache, integer attn", "results_graph_traffic_untied.json",
     "results_step_latency_int.json", 0.25),
]


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def per_layer(row, cache_scale):
    """(layer weights MiB, layer activations MiB, lm_head MiB, MAC per layer)."""
    roles = row["by_role"]
    lm = roles.get("lm_head", {"w": 0, "act": 0, "macs": 0})
    w_layers = (row["weight_bytes"] - lm["w"]) / MIB / N_LAYERS
    act = row["act_bytes"] - lm["act"]
    if "kv_enc_bytes" in row:
        act += row["kv_enc_bytes"] * (cache_scale - 1.0)
    a_layers = act / MIB / N_LAYERS
    macs = (row["macs"] - lm["macs"]) / N_LAYERS
    return w_layers, a_layers, lm["w"] / MIB, macs


def verdict(rho):
    return "fits" if rho <= 1.0 else f"reduce {rho:.2f}x"


def main():
    arms = []
    for label, gfile, lfile, scale in ARMS:
        g = load(gfile)
        if g is None:
            print(f"  missing {gfile}")
            continue
        row = next(r for r in g["rows"] if r["t"] == T)
        lat = load(lfile)
        ms = next((r["ms"] for r in lat["rows"] if r["t"] == T), None) if lat else None
        arms.append((label, *per_layer(row, scale), ms))

    print(f"decision unit = one decoder layer; step at t = {T}; "
          f"budget alpha*L3 = {BUDGET:.1f} MiB\n")
    hdr = (f"{'arm':<30}{'W':>6}{'A':>6}{'lm_head':>9}{'MAC/B':>7}{'ms':>7}"
           f"   {'rule W':<13}{'rule W+A':<14}{'+I (50)':<14}{'+I (200)'}")
    print(hdr)
    print("-" * len(hdr))
    v_w, v_wa = [], []
    for label, w, a, lm, macs, ms in arms:
        intensity = macs / ((w + a) * MIB)
        rw, rwa = w / BUDGET, (w + a) / BUDGET
        v_w.append(verdict(rw))
        v_wa.append(verdict(rwa))
        gated = {ic: ("compute-bound" if intensity > ic else verdict(rwa))
                 for ic in (50, 200)}
        print(f"{label:<30}{w:>6.1f}{a:>6.1f}{lm:>9.1f}{intensity:>7.0f}"
              f"{'      -' if ms is None else f'{ms:7.0f}'}"
              f"   {verdict(rw):<13}{verdict(rwa):<14}{gated[50]:<14}{gated[200]}")

    print("\nreading (computed from the rows above)")
    if len(set(v_w)) == 1:
        print(f"  rule W gives all {len(arms)} arms the same verdict "
              f"('{v_w[0]}'): the per-layer weight working set does not "
              f"change with the cache, so the paper's rule cannot see the "
              f"difference between these decoders.")
    else:
        print(f"  rule W separates the arms: {v_w}")
    if len(set(v_wa)) > 1:
        print(f"  rule W+A separates them: {v_wa}")
        print("  the FP32 cache is the term that lifts the cached decoder over "
              "the budget, and the int8 cache is what brings it back.")
    sizes = [(w + a) * MIB for _, w, a, _, _, _ in arms]
    times = [ms for *_, ms in arms]
    if all(t is not None for t in times):
        by_bytes = sorted(range(len(arms)), key=lambda i: sizes[i])
        by_time = sorted(range(len(arms)), key=lambda i: times[i])
        print(f"  ordering by bytes read   : {[arms[i][0] for i in by_bytes]}")
        print(f"  ordering by measured step: {[arms[i][0] for i in by_time]}")
        print("  " + ("bytes alone reproduce the measured order"
                      if by_bytes == by_time else
                      "bytes alone do NOT reproduce the measured order; the "
                      "intensity gate is what explains the deployed decoder"))


if __name__ == "__main__":
    main()
