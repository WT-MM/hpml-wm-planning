Hardware: NVIDIA GeForce RTX 5090 (driver 580.142), AMD Ryzen 9 9950X 16-Core Processor; torch 2.15.0+cu132, CUDA 13.2, cuDNN 92600, PyTorch eager (torch.compile only in *_compile variants); dtype float32 (fp16 autocast in *_fp16 variants); TF32 matmul False, TF32 cuDNN True.
Model: DINO-WM pusht (model_latest.pth), encoder dinov2_vits14, predictor ViT depth 6 heads 16; context 3 frames x 196 tokens; DINO-WM commit 0a9492f.
Two processes on the same card, same tasks and noise: fp16_compile, fp16_reuse_compile come from run mid2c_seed99, the rest from mid2_seed99. Shared variant, median plan in each: cache_sdpa_fp16 14.277 s and 14.293 s.
torch.compile cold start (first rollout call, not part of any plan latency): fp16_compile 11.8 s, fp16_reuse_compile 9.3 s.
Workload: CEM 30 iterations x 300 candidates (batch), horizon 5 (frameskip 5), top-30; 16 paired tasks, seed 99; warm-up 1 plan(s) discarded; 1 plan per task; time.perf_counter around planner.plan with torch.cuda.synchronize before and after.

| variant | median plan (s) | mean ± sd (s) | p95 (s) | ms / CEM iter | candidates/s | peak GPU mem (MiB) | speedup vs official [95% CI] | success | same outcome as official |
|---|---|---|---|---|---|---|---|---|---|
| official | 71.36 | 71.35 ± 0.05 | 71.41 | 2379 | 126 | 18508 | 1.00x [1.00, 1.00] | 1.00 (16/16) | 1.00 |
| cache | 68.27 | 68.26 ± 0.06 | 68.32 | 2276 | 132 | 18599 | 1.05x [1.05, 1.05] | 1.00 (16/16) | 1.00 |
| cache_sdpa | 39.99 | 39.99 ± 0.03 | 40.02 | 1333 | 225 | 5217 | 1.78x [1.78, 1.78] | 1.00 (16/16) | 1.00 |
| cache_sdpa_tf32 | 27.72 | 27.72 ± 0.02 | 27.74 | 924 | 325 | 5217 | 2.57x [2.57, 2.57] | 1.00 (16/16) | 1.00 |
| cache_sdpa_fp16 | 14.28 | 14.27 ± 0.02 | 14.30 | 476 | 630 | 3740 | 5.00x [5.00, 5.00] | 1.00 (16/16) | 1.00 |
| fp16_compile | 13.50 | 13.50 ± 0.01 | 13.52 | 450 | 667 | 3704 | 5.28x [5.28, 5.29] | 1.00 (16/16) | 1.00 |
| fp16_reuse | 10.09 | 10.09 ± 0.02 | 10.11 | 336 | 892 | 9402 | 7.07x [7.07, 7.08] | 0.94 (15/16) | 0.94 |
| fp16_reuse_compile | 9.32 | 9.31 ± 0.01 | 9.33 | 311 | 966 | 8910 | 7.66x [7.66, 7.66] | 1.00 (16/16) | 1.00 |

Reference rows (no planning): dataset_actions success 1.00, zero_actions success 0.00

Phase share of one plan (instrumented pass):
- official: encode_obs 4.3%, predict 95.5%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.2%  (total 71.31 s)
- cache: encode_obs 0.0%, predict 99.8%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.2%  (total 68.20 s)
- cache_sdpa: encode_obs 0.0%, predict 99.7%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.3%  (total 40.03 s)
- cache_sdpa_tf32: encode_obs 0.0%, predict 99.6%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.4%  (total 27.75 s)
- cache_sdpa_fp16: encode_obs 0.0%, predict 99.1%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.8%  (total 14.31 s)
- fp16_compile: encode_obs 0.0%, predict 99.1%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.9%  (total 13.53 s)
- fp16_reuse: encode_obs 0.1%, predict 98.8%, objective 0.1%, other(sampling,topk,preprocess,act-encode) 1.0%  (total 10.13 s)
- fp16_reuse_compile: encode_obs 0.0%, predict 98.8%, objective 0.1%, other(sampling,topk,preprocess,act-encode) 1.1%  (total 9.33 s)

Planned-action agreement with official:
- official: max |action - official| 0.00e+00; mean final CEM loss 0.0880; mean state distance to goal 25.5
- cache: max |action - official| 1.06e+00; mean final CEM loss 0.0880; mean state distance to goal 26.0
- cache_sdpa: max |action - official| 1.06e+00; mean final CEM loss 0.0880; mean state distance to goal 26.0
- cache_sdpa_tf32: max |action - official| 2.06e+00; mean final CEM loss 0.0893; mean state distance to goal 23.2
- cache_sdpa_fp16: max |action - official| 1.84e+00; mean final CEM loss 0.0876; mean state distance to goal 27.1
- fp16_compile: max |action - official| 1.84e+00; mean final CEM loss 0.0870; mean state distance to goal 27.7
- fp16_reuse: max |action - official| 1.81e+00; mean final CEM loss 0.0871; mean state distance to goal 29.1
- fp16_reuse_compile: max |action - official| 1.85e+00; mean final CEM loss 0.0875; mean state distance to goal 29.1

GPU kernel time by kind, PyTorch Profiler over one plan (share of the kernel rows among the profiler's top rows):
- official: GEMM (linear layers, attention matmul) 59%, elementwise 24%, softmax 8%, memcpy 8%, layer norm 1%, fused attention kernel 1%, other 0%  (71.28 s of kernels)
- cache: GEMM (linear layers, attention matmul) 58%, elementwise 25%, softmax 8%, memcpy 8%, layer norm 1%, other 0%, fused attention kernel 0%  (68.18 s of kernels)
- cache_sdpa: GEMM (linear layers, attention matmul) 60%, fused attention kernel 33%, elementwise 5%, layer norm 1%, other 0%, memcpy 0%  (40.02 s of kernels)
- cache_sdpa_tf32: fused attention kernel 47%, GEMM (linear layers, attention matmul) 42%, elementwise 8%, layer norm 2%, other 0%, memcpy 0%  (27.73 s of kernels)
- cache_sdpa_fp16: GEMM (linear layers, attention matmul) 48%, fused attention kernel 35%, elementwise 12%, layer norm 4%, other 1%, memcpy 0%  (14.27 s of kernels)
- fp16_compile: GEMM (linear layers, attention matmul) 44%, fused attention kernel 33%, layer norm 13%, GELU 9%, other 1%  (15.17 s of kernels)
- fp16_reuse: GEMM (linear layers, attention matmul) 49%, fused attention kernel 22%, other 12%, elementwise 12%, layer norm 4%, memcpy 0%  (10.08 s of kernels)
- fp16_reuse_compile: GEMM (linear layers, attention matmul) 46%, fused attention kernel 21%, other 12%, layer norm 12%, GELU 9%, elementwise 0%  (10.55 s of kernels)

GPU exclusivity: runs only start on a card with no compute process and are stopped if another appears on it. The other card was idle at the start and end of every variant.

Rollout vs the official rollout on identical inputs (one task, all candidates):

| compared with official | max abs feature diff | mean abs diff | max objective diff (spread across candidates) | CEM elite overlap | same best candidate |
|---|---|---|---|---|---|
| official_rerun | 0.0e+00 | 0.0e+00 | 0.0e+00 (1.37) | 1.00 | True |
| official_two_half_batches | 0.0e+00 | 0.0e+00 | 0.0e+00 (1.37) | 1.00 | True |
| cache | 4.8e-02 | 4.3e-04 | 5.3e-04 (1.37) | 1.00 | True |
| cache_sdpa | 4.8e-02 | 4.3e-04 | 5.3e-04 (1.37) | 1.00 | True |
| cache_sdpa_tf32 | 8.4e-02 | 9.2e-04 | 1.0e-03 (1.37) | 1.00 | True |
| cache_sdpa_reuse | 4.8e-02 | 4.3e-04 | 5.3e-04 (1.37) | 1.00 | True |
| cache_sdpa_fp16 | 3.3e-01 | 2.2e-03 | 4.2e-03 (1.37) | 1.00 | True |
| fp16_reuse | 3.6e-01 | 2.2e-03 | 4.0e-03 (1.37) | 1.00 | True |
| fp16_compile | 3.3e-01 | 2.2e-03 | 4.1e-03 (1.37) | 1.00 | True |
| fp16_reuse_compile | 3.4e-01 | 2.2e-03 | 3.9e-03 (1.37) | 1.00 | True |

Against the same computation without the extra step (max abs feature diff): cache_sdpa_tf32 vs cache_sdpa 5.9e-02; cache_sdpa_reuse vs cache_sdpa 0.0e+00; fp16_reuse vs cache_sdpa_fp16 5.0e-02; fp16_compile vs cache_sdpa_fp16 4.0e-02; fp16_reuse_compile vs cache_sdpa_fp16 5.5e-02.

One rollout call (one CEM iteration's world-model cost) vs candidates K; horizon 5, warm-up 3 calls discarded, median of 20 calls, ms (peak GPU MiB):

| K | official | cache_sdpa | cache_sdpa_tf32 | cache_sdpa_fp16 | fp16_reuse |
|---|---|---|---|---|---|
| 1 | 10.2 (308) | 8.3 (264) | 5.9 (264) | 4.7 (342) | 4.1 (358) |
| 2 | 17.2 (371) | 12.9 (278) | 7.7 (278) | 5.4 (355) | 4.4 (391) |
| 4 | 32.2 (489) | 20.4 (304) | 13.6 (304) | 7.2 (372) | 5.6 (439) |
| 8 | 60.8 (726) | 35.9 (356) | 26.1 (356) | 12.8 (416) | 8.7 (549) |
| 16 | 121.2 (1202) | 70.9 (463) | 49.9 (463) | 24.6 (494) | 17.6 (768) |
| 32 | 247.4 (2153) | 143.0 (707) | 99.4 (707) | 48.1 (650) | 34.0 (1231) |
| 64 | 501.5 (4053) | 285.8 (1195) | 196.4 (1195) | 99.6 (966) | 69.5 (2178) |
| 128 | 1007.0 (7856) | 566.2 (2166) | 394.2 (2166) | 200.8 (1603) | 140.5 (4029) |
| 256 | 2027.4 (15458) | 1135.0 (4112) | 788.7 (4112) | 403.1 (2902) | 283.9 (7734) |
| 300 | 2375.4 (18071) | 1332.7 (4781) | 924.5 (4781) | 476.0 (3347) | 336.7 (9010) |
