Hardware: NVIDIA GeForce RTX 5090 (driver 580.142), AMD Ryzen 9 9950X 16-Core Processor; torch 2.15.0+cu132, CUDA 13.2, cuDNN 92600, PyTorch eager (torch.compile only in *_compile variants); dtype float32 (fp16 autocast in *_fp16 variants); TF32 matmul False, TF32 cuDNN True.
Model: DINO-WM pusht (model_latest.pth), encoder dinov2_vits14, predictor ViT depth 6 heads 16; context 3 frames x 196 tokens; DINO-WM commit 0a9492f.
torch.compile cold start (first rollout call, not part of any plan latency): fp16_compile 11.8 s, fp16_reuse_compile 9.3 s.
Workload: CEM 30 iterations x 300 candidates (batch), horizon 5 (frameskip 5), top-30; 16 paired tasks, seed 99; warm-up 1 plan(s) discarded; 1 plan per task; time.perf_counter around planner.plan with torch.cuda.synchronize before and after.

| variant | median plan (s) | mean ± sd (s) | p95 (s) | ms / CEM iter | candidates/s | peak GPU mem (MiB) | speedup vs official [95% CI] | success | same outcome as official |
|---|---|---|---|---|---|---|---|---|---|
| cache_sdpa_fp16 | 14.29 | 14.28 ± 0.04 | 14.31 | 476 | 630 | 3741 | 1.00x [1.00, 1.00] | 1.00 (16/16) | 1.00 |
| fp16_compile | 13.50 | 13.50 ± 0.01 | 13.52 | 450 | 667 | 3704 | 1.06x [1.06, 1.06] | 1.00 (16/16) | 1.00 |
| fp16_reuse_compile | 9.32 | 9.31 ± 0.01 | 9.33 | 311 | 966 | 8910 | 1.53x [1.53, 1.53] | 1.00 (16/16) | 1.00 |

Reference rows (no planning): dataset_actions success 1.00, zero_actions success 0.00

Phase share of one plan (instrumented pass):
- cache_sdpa_fp16: encode_obs 0.0%, predict 99.1%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.8%  (total 14.33 s)
- fp16_compile: encode_obs 0.0%, predict 99.1%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.9%  (total 13.53 s)
- fp16_reuse_compile: encode_obs 0.0%, predict 98.8%, objective 0.1%, other(sampling,topk,preprocess,act-encode) 1.1%  (total 9.33 s)

Planned-action agreement with official:
- cache_sdpa_fp16: max |action - official| 0.00e+00; mean final CEM loss 0.0876; mean state distance to goal 27.1
- fp16_compile: max |action - official| 1.35e+00; mean final CEM loss 0.0870; mean state distance to goal 27.7
- fp16_reuse_compile: max |action - official| 1.66e+00; mean final CEM loss 0.0875; mean state distance to goal 29.1

GPU kernel time by kind, PyTorch Profiler over one plan (share of the kernel rows among the profiler's top rows):
- cache_sdpa_fp16: GEMM (linear layers, attention matmul) 48%, fused attention kernel 35%, elementwise 12%, layer norm 4%, other 1%, memcpy 0%  (14.09 s of kernels)
- fp16_compile: GEMM (linear layers, attention matmul) 44%, fused attention kernel 33%, layer norm 13%, GELU 9%, other 1%  (15.17 s of kernels)
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
| fp16_compile | 3.3e-01 | 2.2e-03 | 4.1e-03 (1.37) | 1.00 | True |
| fp16_reuse | 3.6e-01 | 2.2e-03 | 4.0e-03 (1.37) | 1.00 | True |
| fp16_reuse_compile | 3.4e-01 | 2.2e-03 | 3.9e-03 (1.37) | 1.00 | True |

Against the same computation without the extra step (max abs feature diff): cache_sdpa_tf32 vs cache_sdpa 5.9e-02; cache_sdpa_reuse vs cache_sdpa 0.0e+00; fp16_compile vs cache_sdpa_fp16 4.0e-02; fp16_reuse vs cache_sdpa_fp16 5.0e-02; fp16_reuse_compile vs cache_sdpa_fp16 5.5e-02.
