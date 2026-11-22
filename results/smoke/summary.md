Hardware: NVIDIA GeForce RTX 5090 (driver 580.142), AMD Ryzen 9 9950X 16-Core Processor; torch 2.15.0+cu132, CUDA 13.2, cuDNN 92600, PyTorch eager, no torch.compile; dtype float32 (fp16 autocast in *_fp16 variants); TF32 matmul False.
Model: DINO-WM pusht (model_latest.pth), encoder dinov2_vits14, predictor ViT depth 6 heads 16; context 3 frames x 196 tokens; DINO-WM commit 0a9492f.
Workload: CEM 3 iterations x 300 candidates (batch), horizon 5 (frameskip 5), top-30; 2 paired tasks, seed 99; warm-up 1 plan(s) discarded; 1 plan per task; time.perf_counter around planner.plan with torch.cuda.synchronize before and after.

| variant | median plan (s) | mean ± sd (s) | p95 (s) | ms / CEM iter | candidates/s | peak GPU mem (MB) | speedup vs official [95% CI] | success | same outcome as official |
|---|---|---|---|---|---|---|---|---|---|
| official | 7.08 | 7.08 ± 0.01 | 7.08 | 2359 | 127 | 18507 | 1.00x [1.00, 1.00] | 0.50 (1/2) | 1.00 |
| cache | 6.79 | 6.79 ± 0.01 | 6.79 | 2262 | 133 | 18598 | 1.04x [1.04, 1.04] | 0.50 (1/2) | 1.00 |
| cache_sdpa | 4.00 | 4.00 ± 0.00 | 4.00 | 1332 | 225 | 5217 | 1.77x [1.77, 1.77] | 0.50 (1/2) | 1.00 |
| cache_sdpa_fp16 | 1.42 | 1.42 ± 0.00 | 1.42 | 474 | 632 | 3741 | 4.97x [4.97, 4.97] | 0.00 (0/2) | 0.50 |

Reference rows (no planning): dataset_actions success 1.00, zero_actions success 0.00

Phase share of one plan (instrumented pass):
- official: encode_obs 4.3%, predict 95.6%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.2%  (total 7.07 s)
- cache: encode_obs 0.1%, predict 99.7%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.2%  (total 6.80 s)
- cache_sdpa: encode_obs 0.1%, predict 99.6%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.3%  (total 4.00 s)
- cache_sdpa_fp16: encode_obs 0.3%, predict 98.8%, objective 0.0%, other(sampling,topk,preprocess,act-encode) 0.9%  (total 1.43 s)

Planned-action agreement with official:
- official: max |action - official| 0.00e+00; mean final CEM loss 0.2012; mean state distance to goal 50.1
- cache: max |action - official| 1.19e-07; mean final CEM loss 0.2012; mean state distance to goal 50.1
- cache_sdpa: max |action - official| 1.19e-07; mean final CEM loss 0.2012; mean state distance to goal 50.1
- cache_sdpa_fp16: max |action - official| 3.24e-01; mean final CEM loss 0.2084; mean state distance to goal 51.4

Top GPU ops, official (one plan): void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(cutlass_80_simt_sgemm_256x128_8x4_tn_align1::Params) 13%; aten::addmm 12%; aten::bmm 11%; void magma_sgemmEx_kernel<float, float, float, false, false, 6, 3, 5, 3, 3>(int, int, int, BatchedTensor, int, BatchedTensor, int, BatchedTensor, int, BatchedTensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) 6%; aten::copy_ 6%; aten::mm 6%

Top GPU ops, cache (one plan): void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(cutlass_80_simt_sgemm_256x128_8x4_tn_align1::Params) 13%; aten::bmm 12%; aten::addmm 11%; void magma_sgemmEx_kernel<float, float, float, false, false, 6, 3, 5, 3, 3>(int, int, int, BatchedTensor, int, BatchedTensor, int, BatchedTensor, int, BatchedTensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long) 7%; aten::copy_ 7%; aten::mm 6%

Top GPU ops, cache_sdpa (one plan): void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>(cutlass_80_simt_sgemm_256x128_8x4_tn_align1::Params) 23%; aten::addmm 19%; aten::_efficient_attention_forward 16%; fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) 16%; aten::mm 11%; void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(cutlass_80_simt_sgemm_128x256_8x4_tn_align1::Params) 7%

Top GPU ops, cache_sdpa_fp16 (one plan): aten::_efficient_attention_forward 17%; fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params) 17%; void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align2>(cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_tn_align2::Params) 16%; aten::addmm 15%; aten::mm 9%; void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_f16_64x256_32x4_tn_align2>(cutlass_80_tensorop_f16_s16816gemm_f16_64x256_32x4_tn_align2::Params) 5%

GPU exclusivity: runs only start on a card with no compute process and are stopped if another appears on it. The other card was idle at the start and end of every variant.

Rollout vs the official rollout on identical inputs (one task, all candidates):

| compared with official | max abs feature diff | mean abs diff | max objective diff (spread across candidates) | CEM elite overlap | same best candidate |
|---|---|---|---|---|---|
| official_rerun | 0.0e+00 | 0.0e+00 | 0.0e+00 (1.37) | 1.00 | True |
| official_two_half_batches | 0.0e+00 | 0.0e+00 | 0.0e+00 (1.37) | 1.00 | True |
| cache | 4.8e-02 | 4.3e-04 | 5.3e-04 (1.37) | 1.00 | True |
| cache_sdpa | 4.8e-02 | 4.3e-04 | 5.3e-04 (1.37) | 1.00 | True |
| cache_sdpa_fp16 | 3.3e-01 | 2.2e-03 | 4.2e-03 (1.37) | 1.00 | True |

One rollout call (one CEM iteration's world-model cost) vs candidates K; horizon 5, warm-up 3 calls discarded, median of 3 calls, ms (peak GPU MB):

| K | official | cache | cache_sdpa | cache_sdpa_fp16 |
|---|---|---|---|---|
| 1 | 10.2 (308) | 8.4 (309) | 8.3 (264) | 4.7 (342) |
| 300 | 2327.3 (18071) | 2237.7 (18162) | 1305.9 (4781) | 466.4 (3348) |
