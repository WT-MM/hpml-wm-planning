# Accelerating World-Model Planning with Reference-Conditioned Rollouts

COMS 6998 High Performance Machine Learning, Fall 2026, Project Team 28
(Huaiyu Chen, Wesley Maa, Ethan Yang, Rebecca Yu).

This repo holds the planning benchmark for the project: the official DINO-WM world model and CEM planner on PushT,
timed and profiled, plus the planner optimizations we compare against it.

## What is measured

One *plan* is one task: 30 CEM iterations x 300 candidate action sequences x horizon 5 (frameskip 5), top-30 elites,
objective = feature MSE to the goal frame at the last step. These are the CEM settings of the official
`conf/plan_pusht.yaml`. The plan is executed open loop in the PushT simulator and scored with the official success test.
Simulator evaluation and early stopping inside the CEM loop are off, so latency is planning only.

| variant | what changes |
|---|---|
| `official` | DINO-WM as released (commit `0a9492f`), unmodified |
| `cache` | start frame encoded once per plan instead of once per candidate per iteration |
| `cache_sdpa` | `cache` + predictor attention through `F.scaled_dot_product_attention` (same frame-causal mask) |
| `cache_sdpa_fp16` | `cache_sdpa` + fp16 autocast inside the rollout |

Every variant plans the same tasks with the same candidate noise, on the same GPU, in one process.
`bench/planbench.py` applies the variants by patching at run time; the DINO-WM source is never edited.

## Results

One RTX 5090, fp32 unless named, 16 paired tasks, warm-up 1 plan, seed 99 (`results/mid_seed99/`):

| variant | median plan (s) | peak GPU memory (MB) | speedup vs official [95% CI] | success |
|---|---|---|---|---|
| `official` | 71.61 | 18507 | 1.00x | 16/16 |
| `cache` | 68.43 | 18598 | 1.05x [1.05, 1.05] | 16/16 |
| `cache_sdpa` | 40.21 | 5217 | 1.78x [1.78, 1.78] | 16/16 |
| `cache_sdpa_fp16` | 14.36 | 3740 | 4.99x [4.98, 4.99] | 16/16 |

The predictor rollout is 95.5% of an official plan and its cost is linear in the number of candidates.
Full tables (phase split, profiler kernels, rollout cost vs candidates, equivalence checks):
`results/mid_seed99/summary.md`. The mid-point report is `docs/HPML_Midpoint_Report_Team28.pptx` (the course template, filled); its text is in `docs/midpoint_draft.md`.

![latency and phase split](results/mid_seed99/summary.png)

Each run folder under `results/` holds `results.json`, `summary.md`, figures, profiler tables and rollout images.\nThe same run is on Weights & Biases: https://wandb.ai/wesley-maa-columbia-university/hpml-wm-planning (uploaded with\n`python bench/wandb_upload.py results/mid_seed99/results.json --entity ENTITY`).

## Reproduce

Needs an NVIDIA GPU, Python 3.12, and the packages in `requirements.txt` (exact versions used: `results/env_freeze.txt`).

1. Clone DINO-WM at `0a9492fa12044b852ae9e001cc74604b79c8bb0c`; download `checkpoints/outputs.zip` and
   `datasets/pusht_noise.zip` from the DINO-WM OSF page (https://osf.io/bmw48/) and unzip them.
2. Lay them out under one folder `$HPML_W`: `dino_wm/`, `ckpt/outputs/pusht/`, `data/pusht_noise/`.
3. Run from inside `dino_wm/`:

```bash
HPML_W=/path/to/folder DATASET_DIR=$HPML_W/data TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SDL_VIDEODRIVER=dummy \
  python /path/to/bench/planbench.py --out runs/mid_seed99 --n-tasks 16 --phase-tasks 2 --profile --scaling
python bench/summarize.py runs/mid_seed99/results.json
```

`planbench.py` starts with a self-check that the optimized rollouts match the official rollout on the same inputs.
`--device cpu --n-tasks 2 --num-samples 8 --opt-steps 2` is a quick functional dry run.

## Credits

World model, planner, simulator and checkpoint: DINO-WM, Zhou et al. 2024 (https://github.com/gaoyuezhou/dino_wm, MIT).
