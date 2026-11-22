# Mid-Point Report draft (content for HPML_Midpoint_Report_Template.pptx)

Status of this draft: all numbers are from the run in `results/mid_seed99/` (one RTX 5090, 2026-11-22 UTC); the full
tables are in `results/mid_seed99/summary.md`. Still open: teammates' UNIs.

## Title & Team

**Accelerating World-Model Planning with Reference-Conditioned Rollouts** (Project Team 28)

| Name | UNI | Role (from the proposal) |
|---|---|---|
| Huaiyu (Harris) Chen | [UNI] | Baseline and data |
| Wesley Maa | wm2505 | Predictor and training |
| Ethan Yang | [UNI] | Profiling and integration |
| Rebecca Yu | [UNI] | Evaluation and reproducibility |

## Project Summary

**Project goal.** A sampling planner (CEM) spends almost all of its time rolling a frozen world model forward for
hundreds of candidate action sequences that start from the same observation. Claim to test: run the full model on one
reference rollout (the CEM mean), then score the other K candidates with a small predictor conditioned on that
reference, and reach the same task success at >= 1.5x lower planning latency than the batched full model. The speedup
condition is T_F(1) + T_G(K) + T_reuse < T_F(K+1). We report the 1.5x target whether or not it is met.

**Background context.** Model: DINO-WM (Zhou et al. 2024), released PushT checkpoint: frozen DINOv2 ViT-S/14 encoder
(22.1 M parameters, 196 patch tokens per frame) and a 6-layer, 16-head ViT predictor (20.1 M parameters) over a
3-frame context (588 tokens). Benchmark: PushT (pymunk simulator), goal-reaching from validation-set start states,
official success test. Planner: the official CEM (30 iterations x 300 candidates, horizon 5 at frameskip 5, top-30).
Hardware: one NVIDIA RTX 5090 (32 GB), Ryzen 9 9950X; all paired runs on the same card in one process.

**Optimization / technical methodology.**
1. Strong baseline: batched full model with the start frame encoded once per plan (the released code re-encodes the
   identical frame for every candidate in every iteration).
2. Systems optimization: predictor attention moved from hand-written matmul + softmax to fused
   `scaled_dot_product_attention` with the same frame-causal mask; then fp16 autocast.
3. Method: reference-conditioned predictor G that predicts final-feature deltas for the K alternatives from the cached
   reference rollout. Controls: independent small predictor at equal cost, local linearization (Jacobian at the
   reference), and "copy the reference".
4. Optional extension (only after the base result is clear): planning-aware training with a ranking loss on CEM elites.

**Profiling, benchmarking & evaluation tools.** `bench/planbench.py` (paired per-task latency with
`torch.cuda.synchronize`, peak memory, per-phase timers, official PushT evaluator), PyTorch Profiler (per-op GPU time),
`bench/summarize.py` (paired bootstrap confidence intervals). Planned: Nsight Systems for the kernel-level view,
MLflow / W&B for run tracking.

**Project links.** GitHub: https://github.com/WT-MM/hpml-wm-planning (private; add teammates and course staff).
Experiment tracking dashboard (W&B): https://wandb.ai/wesley-maa-columbia-university/hpml-wm-planning (one run per variant, plus tables and figures; uploaded with `bench/wandb_upload.py`).

## Current Progress

**Baseline & experimental setup.** The official DINO-WM code (commit `0a9492f`, unmodified) with the released PushT
checkpoint runs end to end in our harness: 16 tasks sampled from the validation split as in the official
`goal_source='dset'` protocol, open-loop CEM plan, execution in the simulator. Sanity rows: the dataset's own actions
reach the goal on every task, zero actions on none. Every metric is reported with: RTX 5090, driver 580.142,
PyTorch 2.15.0 + CUDA 13.2, eager mode, fp32 (fp16 autocast only where named), TF32 off for matmul and on for cuDNN convolutions (the PyTorch defaults), batch = 300
candidates, context 3 frames x 196 tokens, warm-up = 1 discarded plan, 1 timed plan per task over 16 paired tasks,
seed 99, `time.perf_counter` around the planner call with `torch.cuda.synchronize` before and after, the other GPU
idle. (16 tasks rather than the planned 50: one official plan takes 71.6 s and the GPU slot was shared. Latency varies
by 0.06% across tasks, so timing is tight; the success rate is coarse.) Papers: DINO-WM (arXiv 2411.04983), DINOv2, CEM / MPC planning, speculative decoding (the analogy).

| Variant | Median plan (s) | Mean ± sd (s) | p95 (s) | ms / CEM iteration | Candidates/s | Peak GPU memory (MB) | Speedup vs official [95% CI] | Success |
|---|---|---|---|---|---|---|---|---|
| `official` (DINO-WM as released) | 71.61 | 71.61 ± 0.04 | 71.67 | 2387 | 126 | 18507 | 1.00x | 16/16 |
| `cache` (start frame encoded once) | 68.43 | 68.43 ± 0.05 | 68.50 | 2281 | 132 | 18598 | 1.05x [1.05, 1.05] | 16/16 |
| `cache_sdpa` (+ fused attention) | 40.21 | 40.21 ± 0.02 | 40.24 | 1340 | 224 | 5217 | 1.78x [1.78, 1.78] | 16/16 |
| `cache_sdpa_fp16` (+ fp16 autocast) | 14.36 | 14.35 ± 0.03 | 14.38 | 479 | 627 | 3740 | 4.99x [4.98, 4.99] | 16/16 |

Speedup is the ratio of mean latencies over the paired tasks, 95% CI by bootstrap over tasks. Figure:
`results/mid_seed99/summary.png`.

**Profiling / baseline analysis.**
- Phase split of one official plan (instrumented pass): predictor rollout 95.5%, observation encoding 4.3%, CEM
  sampling / top-k / preprocessing 0.2%, objective < 0.1%. The planner is the predictor; nothing else matters yet.
- PyTorch Profiler, GPU kernel time of one official plan (71.2 s of kernels): GEMM 59% (linear layers and the
  attention matmuls), elementwise 24%, memcpy 8%, softmax 8%, layer norm 1%. With fused attention (40.3 s of
  kernels): GEMM 60%, fused attention kernel 33%, elementwise 5%; the memcpy and softmax rows disappear.
- Memory: the hand-written attention materializes a 300 x 16 x 588 x 588 fp32 attention matrix (6.6 GB) per layer;
  fused attention does not, so peak memory drops from 18.5 GB to 5.2 GB.
- Rollout cost T_F(K) of one CEM iteration against the number of candidates K (warm-up 3 calls, median of 20 calls):

| K | `official` ms (peak MB) | `cache` | `cache_sdpa` | `cache_sdpa_fp16` |
|---|---|---|---|---|
| 1 | 10.3 (308) | 8.9 (309) | 8.4 (264) | 4.7 (342) |
| 16 | 122.5 (1202) | 117.4 (1207) | 71.4 (463) | 24.7 (494) |
| 64 | 505.6 (4053) | 487.4 (4073) | 288.0 (1195) | 100.1 (966) |
| 300 | 2384.7 (18071) | 2283.6 (18162) | 1343.0 (4781) | 479.0 (3347) |

  Cost is linear in K from K = 8 up (official: 7.7 ms per candidate at K = 16, 7.9 at K = 300), so one 5090 is
  already saturated by a handful of candidates and batching buys nothing more. Lower latency has to come from a
  cheaper per-candidate computation, which is the premise of the reference-conditioned rollout. Figure:
  `results/mid_seed99/scaling.png`.

**First optimization / implementation status.** Implemented and measured: cached start-frame encoding (1.05x), fused
attention (1.78x, 3.5x less memory), fp16 autocast (4.99x, 4.9x less memory), all at 16/16 success on the same tasks.
- Correctness against the official rollout on identical inputs (one task, 300 candidates): fp32 variants differ by at
  most 0.048 in a feature (mean 4.3e-4), which moves the planning objective by 5.3e-4 against a spread of 1.37 across
  candidates; fp16 by at most 0.33 (mean 2.2e-3), objective 4.2e-3. In all cases CEM selects the same 30 elites and
  the same best candidate.
- Over 30 CEM iterations those small differences compound, so the final plans are not identical to the official
  ones (mean absolute action difference 0.09 for fp32, 0.27 for fp16, normalized action units). Plan quality is the
  same: mean final CEM loss 0.0880 / 0.0880 / 0.0876 and mean final state distance to the goal 25.5 / 26.0 / 27.1
  (official / fp32 optimized / fp16). `cache` and `cache_sdpa` give bit-identical plans, so fused attention itself
  changes nothing.
- Status against the 1.5x target: against the released code the systems optimizations alone exceed it. The project's
  claim is stricter: the reference-conditioned planner must be 1.5x faster than the optimized full-model baseline
  (`cache_sdpa`, or `cache_sdpa_fp16`) at the same success. That is not attempted yet. The measured budget for it:
  with T_F(1) = 8.4 ms and T_F(300) = 1343 ms (fp32), the small predictor G must score 300 candidates in under
  about 885 ms, i.e. 3.0 ms per candidate against 4.5 ms for the full model; in fp16, under about 315 ms
  (1.05 ms against 1.6 ms), before any reuse overhead.

## Future Work

**Remaining work & experiments.**
- Train the reference-conditioned predictor G and the equal-cost independent predictor; measure ranking error
  (elite-set overlap, best-candidate agreement) and task success against planning latency.
- Jacobian and copy-the-reference controls.
- Closed-loop MPC evaluation (replanning), 3 seeds x >= 200 paired tasks, median and tail latency, throughput, memory.
- `torch.compile` and Nsight Systems trace of the optimized planner; batch and search-budget sweeps for the baseline.
- Experiment tracking for every run; reproduction README; final report and demo (original vs accelerated planner).

**Scope adjustments.** The baseline is running, so none are required. PointMaze stays optional; planning-aware training
stays optional until the base result is clear (per the proposal feedback).

## Blockers & Limitations

- The released code targets Python 3.9 / PyTorch 2.3 / CUDA 12.1, which cannot drive an RTX 5090. Resolved: Python
  3.12 + PyTorch 2.15 environment, with the PointMaze import (mujoco-py, d4rl) stubbed and no source edits.
- The released checkpoint pickles whole modules; it needs `weights_only=False`, the DINOv2 hub code on the path, and
  `accelerate` installed. Resolved in the harness.
- Shared workstation: timing needs an exclusive GPU. Runs start only on an idle card and stop if another job appears.
- The PushT validation split has only 21 episodes; tasks differ by start offset. Plan: add random-state goals and
  held-out training episodes split by source episode.
- On GPU the fp32 variants differ from the official rollout by up to 0.048 in feature space (1e-5 on CPU). The
  official rollout itself is exactly repeatable, also when the 300 candidates are split into two batches of 150, so
  this is not run-to-run noise. It enters where the start frame is encoded once at batch size 1 instead of 300; the
  cause is not isolated yet (suspect: a batch-size-dependent convolution kernel in the patch embedding).
- Success is saturated (16/16 for every variant) on 16 open-loop tasks, so it cannot show a small quality loss; final
  CEM loss and state distance are the finer measures for now. fp16 in particular needs the larger closed-loop
  evaluation before we call it quality-neutral.
