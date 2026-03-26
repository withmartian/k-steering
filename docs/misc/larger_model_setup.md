## Large Model Support Guide

The table below provides approximate GPU memory requirements for transformer models at different parameter scales, helping you determine which models can run on Free Tier Colab versus requiring larger compute setups.

| Model Size | Params    | FP16 VRAM (Inference) | 4-bit VRAM (Inference) | Recommended GPU | Colab Free Feasible? |
| ---------- | --------- | --------------------- | ---------------------- | --------------- | -------------------- |
| Tiny       | 100M–300M | ~0.5–1 GB             | ~0.3–0.5 GB            | Any GPU         | ✅ Yes               |
| Small      | 500M–1B   | ~2–3 GB               | ~1–1.5 GB              | T4 / L4         | ✅ Yes               |
| Medium     | 2B–3B     | ~5–7 GB               | ~2–3 GB                | T4 (tight) / L4 | ❌ No                |
| Upper-Mid  | 7B        | ~14–16 GB             | ~4–6 GB                | L4 / A100       | ❌ No                |
| Large      | 13B       | ~26–28 GB             | ~8–10 GB               | A100 40GB       | ❌ No                |
| Very Large | 30B       | ~60+ GB               | ~18–22 GB              | Multi-GPU       | ❌ No                |
| Frontier   | 70B       | ~140+ GB              | ~35–40 GB              | Multi A100/H100 | ❌ No                |
