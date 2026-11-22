# Mid-Point Report draft (content for HPML_Midpoint_Report_Template.pptx)

Status of this draft: every section is written except the GPU numbers, marked **[GPU RUN PENDING]**. They come from
`results/mid_seed99/summary.md` once the benchmark has run on the RTX 5090.

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
Experiment tracking dashboard: **[TO SET UP: W&B project or hosted MLflow]**.

## Current Progress

**Baseline & experimental setup.** The official DINO-WM code (commit `0a9492f`, unmodified) with the released PushT
checkpoint runs end to end in our harness: 50 tasks sampled from the validation split as in the official
`goal_source='dset'` protocol, open-loop CEM plan, execution in the simulator. Sanity rows: the dataset's own actions
reach the goal on every task, zero actions on none. Every metric is reported with: RTX 5090, driver 580.142,
PyTorch 2.15.0 + CUDA 13.2, eager mode, fp32 (fp16 autocast only where named), TF32 matmul off, batch = 300
candidates, context 3 frames x 196 tokens, warm-up = 1 discarded plan, 1 timed plan per task over 50 paired tasks,
seed 99. Papers: DINO-WM (arXiv 2411.04983), DINOv2, CEM / MPC planning, speculative decoding (the analogy).

**[GPU RUN PENDING]** latency table (median, mean ± sd, p95, ms per CEM iteration, candidates/s, peak memory, success).

**Profiling / baseline analysis.** **[GPU RUN PENDING]** phase split (encode / predictor / objective / other) and top
GPU ops; rollout cost against the number of candidates K (this is T_F(K) in the speedup condition).
CPU dry run (functional check only, not a result): the predictor is 96-99% of planning time.

**First optimization / implementation status.** Implemented and verified: cached start-frame encoding, fused
attention, fp16 autocast. On CPU the optimized rollouts match the official rollout to 1e-5 and CEM picks identical
elites. **[GPU RUN PENDING]** paired speedups with 95% confidence intervals and success at equal task set.

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
- On GPU the fp32 variants differ from the official rollout by up to ~0.05 in feature space (1e-5 on CPU). The
  benchmark now measures the backend's own batch-size noise floor next to it. **[GPU RUN PENDING]**
