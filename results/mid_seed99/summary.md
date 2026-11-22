Hardware: NVIDIA GeForce RTX 5090 (driver 580.142), AMD Ryzen 9 9950X 16-Core Processor; torch 2.15.0+cu132, CUDA 13.2, cuDNN 92600, PyTorch eager, no torch.compile; dtype float32 (fp16 autocast in *_fp16 variants); TF32 matmul False, TF32 cuDNN True.
Model: DINO-WM pusht (model_latest.pth), encoder dinov2_vits14, predictor ViT depth 6 heads 16; context 3 frames x 196 tokens; DINO-WM commit 0a9492f.
Workload: CEM 30 iterations x 300 candidates (batch), horizon 5 (frameskip 5), top-30; 16 paired tasks, seed 99; warm-up 1 plan(s) discarded; 1 plan per task; time.perf_counter around planner.plan with torch.cuda.synchronize before and after.

| variant | median plan (s) | mean ± sd (s) | p95 (s) | ms / CEM iter | candidates/s | peak GPU mem (MiB) | speedup vs official [95% CI] | success | same outcome as official |
|---|---|---|---|---|---|---|---|---|---|
| official | 71.61 | 71.61 ± 0.04 | 71.67 | 2387 | 126 | 18507 | 1.00x [1.00, 1.00] | 1.00 (16/16) | 1.00 |
| cache | 68.43 | 68.43 ± 0.05 | 68.50 | 2281 | 132 | 18598 | 1.05x [1.05, 1.05] | 1.00 (16/16) | 1.00 |
| cache_sdpa | 40.21 | 40.21 ± 0.02 | 40.24 | 1340 | 224 | 5217 | 1.78x [1.78, 1.78] | 1.00 (16/16) | 1.00 |
| cache_sdpa_fp16 | 14.36 | 14.35 ± 0.03 | 14.38 | 479 | 627 | 3740 | 4.99x [4.98, 4.99] | 1.00 (16/16) | 1.00 |

Reference rows (no planning): dataset_actions success 1.00, zero_actions success 0.00

Phase share of one plan (instrumented pass):
- official: encode_obs 4.3%, predict 95.5%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.2%  (total 71.61 s)
- cache: encode_obs 0.0%, predict 99.8%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.2%  (total 68.48 s)
- cache_sdpa: encode_obs 0.0%, predict 99.7%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.3%  (total 40.20 s)
- cache_sdpa_fp16: encode_obs 0.0%, predict 99.1%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.8%  (total 14.41 s)

Planned-action agreement with official:
- official: max |action - official| 0.00e+00; mean final CEM loss 0.0880; mean state distance to goal 25.5
- cache: max |action - official| 1.06e+00; mean final CEM loss 0.0880; mean state distance to goal 26.0
- cache_sdpa: max |action - official| 1.06e+00; mean final CEM loss 0.0880; mean state distance to goal 26.0
- cache_sdpa_fp16: max |action - official| 1.84e+00; mean final CEM loss 0.0876; mean state distance to goal 27.1

GPU kernel time by kind, PyTorch Profiler over one plan (share of the kernel rows among the profiler's top rows):
- official: GEMM (linear layers, attention matmul) 59%, elementwise 24%, memcpy 8%, softmax 8%, layer norm 1%, fused attention kernel 1%  (71.16 s of kernels)
- cache: GEMM (linear layers, attention matmul) 58%, elementwise 24%, softmax 8%, memcpy 8%, layer norm 1%  (68.32 s of kernels)
- cache_sdpa: GEMM (linear layers, attention matmul) 60%, fused attention kernel 33%, elementwise 5%, layer norm 1%, other 0%  (40.26 s of kernels)
- cache_sdpa_fp16: GEMM (linear layers, attention matmul) 48%, fused attention kernel 35%, elementwise 12%, layer norm 4%, other 1%  (14.32 s of kernels)

GPU exclusivity: runs only start on a card with no compute process and are stopped if another appears on it. The other card was idle at the start and end of every variant.

Rollout vs the official rollout on identical inputs (one task, all candidates):

| compared with official | max abs feature diff | mean abs diff | max objective diff (spread across candidates) | CEM elite overlap | same best candidate |
|---|---|---|---|---|---|
| official_rerun | 0.0e+00 | 0.0e+00 | 0.0e+00 (1.37) | 1.00 | True |
| official_two_half_batches | 0.0e+00 | 0.0e+00 | 0.0e+00 (1.37) | 1.00 | True |
| cache | 4.8e-02 | 4.3e-04 | 5.3e-04 (1.37) | 1.00 | True |
| cache_sdpa | 4.8e-02 | 4.3e-04 | 5.3e-04 (1.37) | 1.00 | True |
| cache_sdpa_fp16 | 3.3e-01 | 2.2e-03 | 4.2e-03 (1.37) | 1.00 | True |

One rollout call (one CEM iteration's world-model cost) vs candidates K; horizon 5, warm-up 3 calls discarded, median of 20 calls, ms (peak GPU MiB):

| K | official | cache | cache_sdpa | cache_sdpa_fp16 |
|---|---|---|---|---|
| 1 | 10.3 (308) | 8.9 (309) | 8.4 (264) | 4.7 (342) |
| 2 | 17.5 (371) | 15.9 (372) | 13.0 (278) | 5.4 (355) |
| 4 | 32.5 (489) | 30.1 (490) | 20.5 (304) | 7.2 (372) |
| 8 | 61.3 (726) | 57.9 (729) | 36.2 (356) | 12.9 (416) |
| 16 | 122.5 (1202) | 117.4 (1207) | 71.4 (463) | 24.7 (494) |
| 32 | 249.6 (2153) | 240.5 (2163) | 143.9 (707) | 48.3 (650) |
| 64 | 505.6 (4053) | 487.4 (4073) | 288.0 (1195) | 100.1 (966) |
| 128 | 1011.7 (7856) | 974.0 (7895) | 570.3 (2166) | 202.0 (1603) |
| 256 | 2035.0 (15458) | 1952.7 (15535) | 1144.3 (4112) | 405.6 (2902) |
| 300 | 2384.7 (18071) | 2283.6 (18162) | 1343.0 (4781) | 479.0 (3347) |
