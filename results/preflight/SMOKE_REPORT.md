# Pre-flight Smoke Report

- **Overall:** PASS
- Device: `cuda`
- Frames run: 1000
- Training steps: 984
- Wall-clock: 60.3 s
- Memory bank entries: 1000
- FAISS top-1 self score: 1.000000
- Checkpoint saved: True (/mnt/c/Users/Jason/Desktop/Eridos/Weft/checkpoints/preflight.pt)

## Pass criteria

- `no_nan_inf`: **PASS**
- `loss_decreased`: **PASS**
- `memory_bank_size_within_5pct`: **PASS**
- `checkpoint_saved`: **PASS**
- `stop_gradient_assertions_ok`: **PASS**
- `grad_norm_median_finite`: **PASS**
- `grad_norm_median_in_range`: **PASS**
- `predicted_norm_within_half_to_two_x`: **PASS**
- `faiss_index_and_retrieve_ok`: **PASS**

## Loss trajectory (next-step MSE)
- Early-window mean (first 50 train steps): `3.992818`
- Late-window mean (last 50 train steps):  `0.104851`
- Loss decreased: **True**

## Gradient norms
- Median: `6.291672`
- Finite: **True** ; within `[1e-6, 100]`: **True**

## Embedding norms
- Encoder mean: `58.7605`
- Predictor mean: `44.0720`
- Ratio (pred / enc): `0.7500` ; within `[0.5, 2.0]`: **True**
