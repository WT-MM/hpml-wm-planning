# Draft-and-verify scan, first look (`bench/draftscan.py`)

One RTX 5090, the same 16 tasks, seed and candidate noise as `results/mid2_all`; full model = `fp16_reuse` (336.5 ms per CEM iteration at 300 candidates); CEM 30 iterations, top 30. An exploration, not part of the benchmark tables.

Check before the run: our CEM loop reproduces the official planner's plan (max action difference 0) and a draft that keeps all tokens equals the full model (max objective difference 0).

## Does a cheap draft find the elites?

Full-model CEM; at every iteration each draft also scores the same 300 candidates. Recall@m = share of the true top 30 inside the draft's top m (chance = m / 300). 16 tasks x 30 iterations.

| draft | tokens/frame | context frames | cost vs full | Spearman | recall@30 | recall@45 | recall@60 | recall@90 | recall@150 | recall@45 in iterations 1-5 / 6-15 / 16-30 |
|---|---|---|---|---|---|---|---|---|---|---|
| tok0.5 | 98 | 3 | 0.43 | 0.32 | 0.11 | 0.18 | 0.24 | 0.40 | 0.72 | 0.54 / 0.17 / 0.06 |
| tok0.25 | 49 | 3 | 0.21 | 0.24 | 0.08 | 0.13 | 0.19 | 0.35 | 0.70 | 0.42 / 0.10 / 0.05 |
| tok0.125 | 24 | 3 | 0.10 | 0.06 | 0.05 | 0.08 | 0.13 | 0.25 | 0.56 | 0.21 / 0.05 / 0.06 |
| ctx1 | 196 | 1 | 0.49 | 0.96 | 0.76 | 0.90 | 0.94 | 0.98 | 1.00 | 0.98 / 0.95 / 0.83 |
| tok0.25+ctx1 | 49 | 1 | 0.10 | 0.30 | 0.10 | 0.16 | 0.22 | 0.39 | 0.72 | 0.46 / 0.16 / 0.05 |

`tokR` = the same predictor on a fixed random share R of the patch tokens; `ctx1` = a context window of 1 frame instead of 3.

## End to end

`fewN` = the full model with N samples (top N/10). `dv:D:mM` = draft D scores all 300, the full model re-scores the draft's top M, elites = the best 30 of those. Objective = the full model's objective for the final plan (lower is better).

| plan | median latency (s) | speedup | final objective | paired difference vs full (se) | success | mean state distance |
|---|---|---|---|---|---|---|
| full | 10.15 | 1.00x | 0.0882 | +0.0000 (0.0000) | 15/16 | 29.1 |
| few150 | 4.99 | 2.03x | 0.0849 | -0.0034 (0.0013) | 15/16 | 28.8 |
| few100 | 3.30 | 3.07x | 0.1035 | +0.0153 (0.0092) | 13/16 | 37.6 |
| few60 | 1.94 | 5.23x | 0.1412 | +0.0530 (0.0189) | 11/16 | 65.6 |
| dv:tok0.25:m45 | 3.57 | 2.84x | 0.4104 | +0.3221 (0.0334) | 2/16 | 58.5 |
| dv:tok0.25:m60 | 4.06 | 2.50x | 0.3320 | +0.2437 (0.0439) | 6/16 | 49.7 |
| dv:tok0.25:m90 | 5.05 | 2.01x | 0.2578 | +0.1696 (0.0323) | 8/16 | 45.3 |
| dv:tok0.125:m45 | 2.45 | 4.14x | 1.2813 | +1.1931 (0.1145) | 0/16 | 151.2 |
| dv:tok0.125:m60 | 2.95 | 3.45x | 1.0664 | +0.9781 (0.1021) | 0/16 | 136.0 |
| dv:tok0.125:m90 | 3.94 | 2.58x | 0.7903 | +0.7021 (0.0694) | 0/16 | 112.3 |
| dv:ctx1:m45 | 6.40 | 1.59x | 0.0914 | +0.0032 (0.0039) | 16/16 | 26.8 |
| dv:ctx1:m60 | 6.86 | 1.48x | 0.1065 | +0.0182 (0.0168) | 15/16 | 27.2 |
| dv:ctx1:m90 | 7.85 | 1.29x | 0.0889 | +0.0006 (0.0004) | 16/16 | 29.7 |

## Reading

- **Fewer samples is the baseline to beat, and here it wins.** 150 samples match 300 (objective -0.003, same 15/16) at 2.03x. The released budget of 300 is more than these 16 tasks need.
- **The shorter-context draft ranks well** (Spearman 0.96, 90% of the elites in its top 45) and keeps plan quality, but it costs 0.49 of the full model, so draft-and-verify reaches only 1.3x to 1.6x.
- **Random token subsets do not rank candidates on this checkpoint.** They carry signal in the first five iterations (0.2 to 0.5 at m = 45, chance 0.15) and fall below chance (about 0.05) once CEM has converged and candidates differ only slightly.
- **Ranking gets harder as CEM converges**, for every draft: even `ctx1` drops from 0.98 to 0.83.
- **What a trained draft has to reach:** for draft-and-verify to beat `few150` clearly (say 4x at no loss), a draft needs about the ranking quality of `ctx1` at about a tenth of the full model's cost. That is the target for the reference-conditioned predictor.

## Limits

- 16 open-loop tasks at horizon 5; success is coarse and the tasks are easy enough that half the samples suffice.
- One fixed random token subset per draft; the checkpoint was not trained with token dropout.
- The below-chance recall of the token-subset drafts late in a plan is not explained. The draft code is only checked against the full model with all tokens kept, so a fault specific to subsets is not ruled out.
- Draft-and-verify keeps 30 elites out of the M verified, so with a poor draft the selection pressure also drops; the two effects are not separated here.
- Not tried: drafts only in the early iterations, a trained draft, or draft-and-verify on top of 150 samples.
