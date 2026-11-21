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

See `results/` (one folder per run: `results.json`, `summary.md`, `summary.png`, profiler tables, rollout images).

## Reproduce

Needs an NVIDIA GPU, Python 3.12, and the packages in `requirements.txt` (exact versions used: `results/env_freeze.txt`).

1. Clone DINO-WM at `0a9492fa12044b852ae9e001cc74604b79c8bb0c`; download `checkpoints/outputs.zip` and
   `datasets/pusht_noise.zip` from the DINO-WM OSF page (https://osf.io/bmw48/) and unzip them.
2. Lay them out under one folder `$HPML_W`: `dino_wm/`, `ckpt/outputs/pusht/`, `data/pusht_noise/`.
3. Run from inside `dino_wm/`:

```bash
HPML_W=/path/to/folder DATASET_DIR=$HPML_W/data TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 SDL_VIDEODRIVER=dummy \
  python /path/to/bench/planbench.py --out runs/mid_seed99 --n-tasks 50 --profile
python bench/summarize.py runs/mid_seed99/results.json
```

`planbench.py` starts with a self-check that the optimized rollouts match the official rollout on the same inputs.
`--device cpu --n-tasks 2 --num-samples 8 --opt-steps 2` is a quick functional dry run.

## Credits

World model, planner, simulator and checkpoint: DINO-WM, Zhou et al. 2024 (https://github.com/gaoyuezhou/dino_wm, MIT).
