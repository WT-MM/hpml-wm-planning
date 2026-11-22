<!-- Slide number: 1 -->
# Accelerating World-Model Planning
with Reference-Conditioned Rollouts
Huaiyu (Harris) Chen (<UNI>) - Baseline and data
Wesley Maa (wm2505) - Predictor and training
Ethan Yang (<UNI>) - Profiling and integration
Rebecca Yu (<UNI>) - Evaluation and reproducibility
COMS 6998 High Performance Machine Learning, Project Team 28

<!-- Slide number: 2 -->
# Project Summary
Project Goal
Claim: a CEM planner with a frozen world model can run the full model on one reference rollout and score the other K candidates with a small predictor conditioned on that reference, reaching the same task success at 1.5x lower planning latency than the batched full model.
Speedup condition: T_F(1) + T_G(K) + T_reuse < T_F(K+1). We report the 1.5x target whether or not it is met.
Background Context
Model: DINO-WM (Zhou et al. 2024), released PushT checkpoint. Frozen DINOv2 ViT-S/14 encoder (22.1 M parameters), 6-layer 16-head ViT predictor (20.1 M), context 3 frames x 196 patch tokens.
Benchmark: PushT goal reaching from validation-set start states with the official success test. Official CEM planner: 30 iterations x 300 candidates, horizon 5 at frameskip 5, top-30 elites.
Hardware: one NVIDIA RTX 5090 (32 GB), Ryzen 9 9950X. All paired runs on the same card in one process.
Start frame and goal
DINOv2 encoder, once per plan
Full model F
1 reference rollout (the CEM mean)
Small predictor G
K candidates, conditioned on the reference rollout
CEM update
objective, top-30 elites, next iteration
Today the full model rolls out all 300 candidates: 95.5% of planning time.

<!-- Slide number: 3 -->
# Project Summary (continued)
Optimization / Technical Methodology
1  Strong baseline
Batched full model with the start frame encoded once per plan. The released code re-encodes it for every candidate in every iteration.
2  Systems optimization
Fused scaled_dot_product_attention with the same frame-causal mask, fp16 autocast or TF32, keys and values reused across rollout steps, torch.compile.
3  Draft and verify
A cheap draft scores all K candidates and the full model re-scores only the top few. Drafts: the reference-conditioned predictor G, a subset of the patch tokens, a shorter context.
4  Planning-aware training
Ranking loss on the CEM elites. Only after the base result is clear, per the proposal feedback.
Measured: 1.05x
Measured: up to 7.66x
Next
Optional
Profiling, Benchmarking & Evaluation Tools
bench/planbench.py: paired per-task latency with torch.cuda.synchronize, peak memory, per-phase timers, official PushT evaluator
PyTorch Profiler for per-kernel GPU time; bench/summarize.py for paired bootstrap confidence intervals; bench/import_wandb.py for tracking, with unit tests
Planned: Nsight Systems trace
Project Links
GitHub: github.com/WT-MM/hpml-wm-planning
Experiment tracking (W&B): wandb.ai/wesley-maa-columbia-university/hpml-wm-planning
One W&B run per variant: summary metrics, per-task and per-K series, the task, scaling and profiler tables, and the source JSON. The raw results and figures are versioned in the repository under results/.

<!-- Slide number: 4 -->
# Current Progress (1/5)
Baseline & Experimental Setup: official DINO-WM as released
71.4 s
median plan latency
2,379 ms
per CEM iteration
126
candidates per second
18.1 GiB
peak GPU memory
16/16
task success
Working baseline: the official DINO-WM code (commit 0a9492f, unmodified) with the released PushT checkpoint runs end to end in our harness.
Experiment design: 16 tasks from the validation split (official goal_source='dset' protocol), open-loop CEM plan, executed in the simulator, official success test. Every variant plans the same tasks with the same candidate noise.
Sanity rows: the dataset's own actions succeed on 16/16 tasks, zero actions on 0/16.
Papers: DINO-WM (arXiv 2411.04983), DINOv2, CEM / MPC planning, speculative decoding (the analogy); related planners: JEPA-WM, LeWorldModel, Sparse Imagination.
Conditions for every metric in this report
Hardware: RTX 5090 32 GB (driver 580.142), Ryzen 9 9950X; the other GPU idle
Backend and dtype: PyTorch 2.15.0 + CUDA 13.2, eager (torch.compile only where named); fp32, fp16 autocast only where named; TF32 off for matmul, on for cuDNN convolutions (PyTorch defaults)
Batch and context: 300 candidates per batch; 3 frames x 196 tokens
Warm-up: 1 plan discarded
Repetitions: 1 timed plan per task over 16 paired tasks, seed 99
Timer: time.perf_counter around the planner call, torch.cuda.synchronize before and after

<!-- Slide number: 5 -->
# Current Progress (2/5)
Profiling / Baseline Analysis: where one plan spends its time
95.5%
predictor rollout
4.3%
observation encoding
0.2%
sampling, top-k, objective

### Chart: GPU kernel time per plan by kind (s), PyTorch Profiler

| Category | GEMM | Fused attention | Elementwise | Softmax | Memcpy | Other |
| --- | --- | --- | --- | --- | --- | --- |
| official | 41.78 | 0.42 | 17.13 | 5.58 | 5.57 | 0.8 |
| cache_sdpa | 24.01 | 13.17 | 2.16 | 0.0 | 0.0 | 0.68 |
| cache_sdpa_fp16 | 6.9 | 4.94 | 1.74 | 0.0 | 0.0 | 0.7 |
| fp16_reuse | 4.99 | 2.23 | 1.18 | 0.0 | 0.0 | 1.67 |

Limiting phase: the predictor rollout. Encoding, sampling and the objective together are under 5% of an official plan.
GPU kernels (PyTorch Profiler, one official plan, 71.3 s of kernel time): GEMM (linear layers and attention matmuls) 59%, elementwise 24%, memcpy 8%, softmax 8%.
Limiting resource: the hand-written attention materializes a 300 x 16 x 588 x 588 fp32 matrix (6.2 GiB) per layer, hence 18.1 GiB peak memory and the memcpy and softmax kernels.
With fused attention (40.0 s of kernel time): GEMM 60%, fused attention kernel 33%; the memcpy and softmax kernels disappear and peak memory falls to 5.1 GiB. With fp16 and rollout reuse the kernels take 10.1 s.

<!-- Slide number: 6 -->
# Current Progress (3/5)
Profiling / Baseline Analysis: cost of one rollout call T_F(K) against the number of candidates K

### Chart: Rollout latency (ms) against candidates per call, log-log

| Category | official | cache_sdpa | cache_sdpa_tf32 | cache_sdpa_fp16 | fp16_reuse |
| --- | --- | --- | --- | --- | --- |

| K | official | cache_sdpa | cache_sdpa_fp16 | fp16_reuse |
| --- | --- | --- | --- | --- |
| 1 | 10.2 | 8.3 | 4.7 | 4.1 |
| 16 | 121.2 | 70.9 | 24.6 | 17.6 |
| 64 | 501.5 | 285.8 | 99.6 | 69.5 |
| 300 | 2,375.4 | 1,332.7 | 476.0 | 336.7 |

Table and chart: ms for one rollout call, which is one CEM iteration's world-model cost. Warm-up 3 calls discarded, median of 20 calls.
Linear in K from K = 8: the official rollout costs 7.6 ms per candidate at K = 16 and 7.9 ms at K = 300. A handful of candidates already saturates one 5090, so batching buys nothing more.
Consequence: lower latency has to come from a cheaper computation per candidate. Rollout reuse does that exactly (1.41x at K = 300); the reference-conditioned rollout aims to do it approximately.

<!-- Slide number: 7 -->
# Current Progress (4/5)
First Optimization / Implementation Status: 7 optimized variants against the official code

### Chart: Median plan latency (s)

| Category | median plan latency (s) |
| --- | --- |
| official | 71.4 |
| cache | 68.3 |
| cache_sdpa | 40.0 |
| cache_sdpa_tf32 | 27.7 |
| cache_sdpa_fp16 | 14.3 |
| fp16_compile | 13.5 |
| fp16_reuse | 10.1 |
| fp16_reuse_compile | 9.3 |

| Variant | What changes | Median plan (s) | ms / CEM iter | Peak mem (GiB) | Speedup [95% CI] | Success |
| --- | --- | --- | --- | --- | --- | --- |
| official | DINO-WM as released | 71.36 | 2,379 | 18.1 | 1.00x | 16/16 |
| cache | start frame encoded once | 68.27 | 2,276 | 18.2 | 1.05x [1.05, 1.05] | 16/16 |
| cache_sdpa | cache + fused attention | 39.99 | 1,333 | 5.1 | 1.78x [1.78, 1.78] | 16/16 |
| cache_sdpa_tf32 | cache_sdpa + TF32 matmul | 27.72 | 924 | 5.1 | 2.57x [2.57, 2.57] | 16/16 |
| cache_sdpa_fp16 | cache_sdpa + fp16 autocast | 14.28 | 476 | 3.7 | 5.00x [5.00, 5.00] | 16/16 |
| fp16_compile | cache_sdpa_fp16 + torch.compile | 13.50 | 450 | 3.6 | 5.28x [5.28, 5.29] | 16/16 |
| fp16_reuse | cache_sdpa_fp16 + rollout reuse | 10.09 | 336 | 9.2 | 7.07x [7.07, 7.08] | 15/16 |
| fp16_reuse_compile | fp16_reuse + torch.compile | 9.32 | 311 | 8.7 | 7.66x [7.66, 7.66] | 16/16 |

16 paired tasks. Speedup is the ratio of mean plan latencies, with a 95% bootstrap interval over tasks. The two torch.compile rows come from a second process on the same card with the same tasks and noise; cache_sdpa_fp16, run in both, measured 14.28 s and 14.29 s. Compiling costs a one-off 9 to 12 s, outside the plan latency.

<!-- Slide number: 8 -->
# Current Progress (5/5)
First Optimization / Implementation Status: what the optimizations change, and the 1.5x target
Same decisions on identical inputs: against the official rollout the fp32 variants move the planning objective by at most 0.0010 and the fp16 variants by 0.004, against a spread of 1.37 across candidates. Every variant picks the same 30 elites and the same best candidate.
Rollout reuse is exact: in fp32 it reproduces the plain computation with a maximum difference of 0; in fp16 it differs by rounding only.
Same plan quality, not identical plans: small differences compound over 30 iterations. Mean final CEM loss is 0.0880 for the official code and 0.0870 to 0.0893 for the optimized variants.
One task lost: fp16_reuse fails task 3, the task with the highest planning loss. Its plan ends at state distance 33.2; cache_sdpa_fp16 ends at 30.9 and passes. Success on 16 open-loop tasks is a coarse measure.
Cost of reuse: peak GPU memory rises from 3.7 to 9.2 GiB for the cached keys and values.
1.5x target: met against the released code by systems work alone. Not yet attempted against the optimized full model, which is the project's claim. Budget for a cheaper scorer: 0.73 ms per candidate against 1.12 ms for the full model with reuse (fp16).

| Variant | Max objective change vs official | Elites kept (of 30) | Max feature change vs the same computation without the step |
| --- | --- | --- | --- |
| cache | 0.0005 | 30 | n/a |
| cache_sdpa | 0.0005 | 30 | n/a |
| cache_sdpa_tf32 | 0.0010 | 30 | 0.059 vs cache_sdpa |
| cache_sdpa_reuse (check only) | 0.0005 | 30 | 0.000 vs cache_sdpa |
| cache_sdpa_fp16 | 0.0042 | 30 | n/a |
| fp16_compile | 0.0041 | 30 | 0.040 vs cache_sdpa_fp16 |
| fp16_reuse | 0.0040 | 30 | 0.050 vs cache_sdpa_fp16 |
| fp16_reuse_compile | 0.0039 | 30 | 0.055 vs cache_sdpa_fp16 |

<!-- Slide number: 9 -->
# Future Work
Remaining Work & Experiments
1.  Train G and the equal-cost independent predictor; measure ranking error (elite overlap, best-candidate agreement) and task success against planning latency.
2.  Draft and verify: a cheap draft scores all candidates and the full model re-scores only the draft's top few. First look on 16 tasks (results/draft1): 150 samples match 300 at 2.0x, which is the baseline to beat; a shorter-context draft keeps plan quality but reaches only 1.6x; random token subsets fail. So G has to rank like the shorter-context draft at about a tenth of the full model's cost.
3.  Controls: Jacobian linearization at the reference and copy-the-reference.
4.  Closed-loop MPC evaluation: 3 seeds x at least 200 paired tasks; median and tail latency, throughput, memory.
5.  Other systems work: candidates split across both GPUs (cost is linear in K), unmasked per-frame attention calls, Nsight Systems trace. CUDA Graphs are not planned: GPU kernels already fill 14.3 s of the 14.3 s fp16 plan.
6.  Tracking and delivery: experiment tracking for every run, reproduction README, final report and demo (original against accelerated planner).
Scope Adjustments
The baseline runs and the profile supports the premise, so nothing is cut.
The bar is higher: the comparison baseline is now the optimized full model, 1.78x (fp32) to 7.66x (fp16, reuse, compile) faster than the released code.
The method is widened to draft and verify: the reference-conditioned predictor G becomes one draft among several, and the full model verifies the draft's top candidates.
Optional second model: JEPA-WM, which follows the same recipe as DINO-WM, replaces PointMaze.
Planning-aware training stays optional until the base result is clear, per the proposal feedback.

<!-- Slide number: 10 -->
# Blockers & Limitations
Blockers & System Limitations

| Blocker or limitation | Status | How it was resolved, or the plan |
| --- | --- | --- |
| Released code targets Python 3.9 / PyTorch 2.3 / CUDA 12.1, which cannot drive an RTX 5090 | Resolved | Python 3.12 + PyTorch 2.15 environment; the unused PointMaze import (mujoco-py, d4rl) is stubbed; no source edits |
| Released checkpoint pickles whole modules | Resolved | Loaded with weights_only=False, the DINOv2 hub code on the path and accelerate installed; handled in the harness |
| torch.compile could not run in the sandbox: NixOS has no C compiler, openssl or ldconfig on the path | Resolved | Compiler and openssl taken from the existing Nix store and the driver library path given to Triton; no system changes |
| Shared workstation: timing needs an exclusive GPU | Resolved | Runs start only on an idle card and stop if another job appears; the other card was idle for every reported number |
| PushT validation split has only 21 episodes; tasks differ by start offset | Open | Add random-state goals and held-out training episodes, split by source episode |
| Optimized fp32 rollouts differ from the official one by up to 0.048 per feature on GPU (1e-5 on CPU) | Open | Enters where the start frame is encoded at batch size 1 instead of 300; the official rollout itself repeats exactly. Plan: isolate the cause (suspect a batch-size-dependent convolution kernel). CEM elites are unchanged |
| Success is a coarse measure: every variant scores 15/16 or 16/16 on 16 open-loop tasks | Open | Larger closed-loop evaluation before calling fp16 quality-neutral; final CEM loss and state distance are the finer measures meanwhile |
