"""Reproduce the hardware-dependent measurements on another machine.

The quality results (WER, gates, calibration) do not depend on the CPU and
are shipped in the results_*.json files. What does depend on the CPU is:

  1. the decoder step time, full cache against the calibrated length
     (step_latency_evicted.py)                               -- always
  2. L3 misses per step, if Intel VTune is installed
     (llc_miss_step.py)                                      -- optional
  3. the selection rule (2) with THIS machine's L3 size
     (hardware_select.py --l3)                               -- always
  4. end-to-end RTF of the four system configurations, if the
     Uzbek audio cache is present (system_composition.py)    -- optional

Models are not in the repository (24 GB). This script rebuilds the ones it
needs from public checkpoints with the same pipeline as the paper:

  openai/whisper-small     -> models/whisper_small_onnx/   (prepare_whisper_small.py)
  Kotib/uzbek_stt_v1       -> models/whisper_with_past/    (prepare_medium_uz.py)
  int8-cache graph          -> decoder_with_past_cache_int.onnx (int8_cache_graph.py)

Requirements (a venv):  pip install onnx onnxruntime optimum[exporters]
                        transformers torch soundfile datasets numpy

Usage:  python experiments/other_cpu.py                 (everything it can)
        python experiments/other_cpu.py --skip-medium   (whisper-small only)
        python experiments/other_cpu.py --l3 12         (override detection)

Output: experiments/results_other_cpu_<hostname>.json and the usual per-step
result files, which can be committed next to the originals.
"""

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
MED = os.path.join(ROOT, "models", "whisper_with_past")
SML = os.path.join(ROOT, "models", "whisper_small_onnx")


def run(script, *args, check=True):
    cmd = [PY, "-u", os.path.join(HERE, script), *args]
    print("$", " ".join(os.path.basename(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=check).returncode


def cpu_info():
    """name, cores, L3 MiB -- Windows (WMI), Linux (lscpu) or macOS (sysctl)."""
    name, cores, l3 = platform.processor(), os.cpu_count(), None
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,L3CacheSize | ConvertTo-Json"],
                                 capture_output=True, text=True).stdout
            d = json.loads(out)
            d = d[0] if isinstance(d, list) else d
            name, cores, l3 = d["Name"].strip(), int(d["NumberOfCores"]), int(d["L3CacheSize"]) / 1024
        elif platform.system() == "Linux":
            out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
            m = re.search(r"Model name:\s*(.+)", out)
            name = m.group(1).strip() if m else name
            m = re.search(r"L3 cache:\s*([\d.]+)\s*([KMG])i?B?", out)
            if m:
                l3 = float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2)]
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.l3cachesize"],
                                 capture_output=True, text=True).stdout.split("\n")
            name = out[0].strip()
            if len(out) > 1 and out[1].strip().isdigit():
                l3 = int(out[1]) / 2 ** 20
    except Exception as e:  # noqa: BLE001 -- detection is best effort
        print("  CPU detection:", e)
    return name, cores, l3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3", type=float, help="L3 size in MiB (overrides detection)")
    ap.add_argument("--skip-medium", action="store_true", help="whisper-small only")
    ap.add_argument("--skip-llc", action="store_true")
    ap.add_argument("--skip-rtf", action="store_true")
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()

    name, cores, l3 = cpu_info()
    l3 = args.l3 or l3
    host = socket.gethostname()
    print(f"CPU: {name} | cores {cores} | L3 {l3} MiB | host {host}")
    if not l3:
        raise SystemExit("L3 size not detected; pass --l3 <MiB>")
    summary = {"host": host, "cpu": name, "cores": cores, "l3_mib": l3,
               "os": platform.platform(), "python": platform.python_version()}
    try:
        import onnxruntime
        summary["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass

    # ---- models
    if not os.path.exists(os.path.join(SML, "decoder_with_past_untied_int8.onnx")):
        run("prepare_whisper_small.py")
    have_medium = False
    if not args.skip_medium:
        if not os.path.exists(os.path.join(MED, "decoder_with_past_untied_int8.onnx")):
            run("prepare_medium_uz.py")
        if not os.path.exists(os.path.join(MED, "decoder_with_past_cache_int.onnx")):
            run("int8_cache_graph.py", "--src", os.path.join(MED, "decoder_with_past_untied_int8.onnx"),
                "--dst", os.path.join(MED, "decoder_with_past_cache_int.onnx"), "--mode", "int", check=False)
        have_medium = os.path.exists(os.path.join(MED, "decoder_with_past_untied_int8.onnx"))

    # ---- 1. step latency
    run("step_latency_evicted.py", "--rounds", str(args.rounds))
    lat = json.load(open(os.path.join(HERE, "results_step_latency_evicted.json")))
    summary["step_latency"] = lat["arms"]

    # ---- 2. LLC misses (VTune)
    vtune = r"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
    if not args.skip_llc and os.path.exists(vtune):
        run("llc_miss_step.py", check=False)
        p = os.path.join(HERE, "results_llc_miss_step.json")
        if os.path.exists(p):
            summary["llc_misses"] = json.load(open(p))
    else:
        print("  VTune not found -- LLC misses skipped (use perf stat -e LLC-load-misses on Linux)")

    # ---- 3. the selection rule at this machine's L3
    run("hardware_select.py", "--l3", str(l3))
    summary["selection"] = json.load(open(os.path.join(HERE, "results_hardware_select.json")))

    # ---- 4. RTF of the four system configurations (needs the Uzbek audio cache)
    if not args.skip_rtf and have_medium and os.path.exists(os.path.join(HERE, "results_system_composition.json")):
        run("system_composition.py", "--phase", "timing", check=False)
        summary["system_timing"] = json.load(open(os.path.join(HERE, "results_system_composition.json"))).get("timing")

    out = os.path.join(HERE, f"results_other_cpu_{host}.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"\nsaqlandi: {out}")
    print("\nstep latency (median ms):")
    for k, v in summary["step_latency"].items():
        print(f"  {k:<28}{v['median_ms']:8.1f}")


if __name__ == "__main__":
    main()
