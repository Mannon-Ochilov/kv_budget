# Boshqa CPU'da o'lchash

Sifat natijalari (WER, darvoza, kalibratsiya) CPU'ga bog'liq emas va
`results_*.json` fayllarida keladi. CPU'ga bog'liq narsalar: dekoder qadam
vaqti, L3 miss'lar, tanlov qoidasi (2) shu mashinaning L3 hajmi bilan, RTF.

## O'rnatish

    python -m venv .venv
    .venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
    pip install onnx onnxruntime "optimum[exporters]" transformers torch soundfile datasets numpy

## Ishga tushirish

    python experiments/other_cpu.py            # hammasi (modellar avtomatik yuklanadi/eksport qilinadi)
    python experiments/other_cpu.py --skip-medium   # faqat whisper-small (tezroq, ~1 GB)
    python experiments/other_cpu.py --l3 12         # L3 aniqlanmasa yoki boshqa qiymat kerak bo'lsa

Birinchi ishga tushirishda modellar yuklanadi: openai/whisper-small (~1 GB)
va Kotib/uzbek_stt_v1 (~3 GB), optimum eksporti va INT8 kvantlash — 20–40 min.

## Natija

`experiments/results_other_cpu_<hostname>.json` — CPU nomi, yadro, L3,
qadam vaqtlari (to'liq vs kalibrlangan kesh), agar VTune bo'lsa L3 miss'lar,
va (2) qoidaning shu L3 uchun tanlovi. Shu faylni va
`results_step_latency_evicted.json` ni git'ga qo'shib yuborsangiz yetarli.

Linux'da LLC miss uchun VTune o'rniga:

    perf stat -e LLC-load-misses,LLC-loads python experiments/llc_miss_step.py --runner <model> <heads> <enc_len> 15
