# Observation / planning study — 2026-09-06

Follow-up to `RESULTS_FULLPAPER.md`. Question: is the ranking collapse (probe +0.82 with
true dynamics, +0.24 through L05ar) caused by the observation (H_obs) or by how the
planner queries the model (H_plan)? Everything here uses cologne8, the L05ar recipe, and the
same 24-episode corpus (regenerated with full observations; the base slice is bit-identical to
`traffic_data_cologne8`, checked episode by episode).

New code: `traffic/sumo_multi_env.py` (`SUMO_OBS_MODE` = base | link | raster | full),
`traffic/slice_obs.py`, `diag_obs_sufficiency.py`, `FactoredCEM` options in
`plan_cem_multi.py` (`--hold_prior --warm_start --min_hold --per_agent --legal_only`),
`--history` in `train_multi_sumo_ar.py` (+ `WM_HISTORY` at eval time), `summarize_planfix.py`.

## T1 — observation sufficiency (model-free)  ✅

Regressors on the scripted corpus predict per-signal halting from a 3-step history, the
neighbour mean, and the k-step action sequence. Targets: halting at t+1 and cumulative
halting over t+1..t+5 (the planner's cost). Ranking tests: (a) the 30x16 held-phase
counterfactual set; (b) 30 fresh anchors x 16 per-step random legal joint sequences (the
switch-heavy population a uniform CEM proposes).

| obs | F | ridge R² h1 / cum5 | GBM R² h1 / cum5 | held-phase rank ridge / GBM (top-1) | switch-heavy rank ridge / GBM (top-1) |
|---|--:|--:|--:|--:|--:|
| base | 9 | 0.993 / 0.976 | 0.952 / 0.947 | +0.46 / +0.78 (0.43) | +0.21 / +0.56 (0.53) |
| link | 19 | 0.994 / 0.977 | 0.946 / 0.938 | +0.47 / +0.84 (0.53) | +0.22 / +0.52 (0.40) |
| raster | 73 | 0.994 / 0.977 | 0.952 / 0.953 | +0.46 / +0.89 (0.70) | +0.22 / +0.57 (0.50) |
| full | 83 | 0.994 / 0.977 | 0.947 / 0.947 | +0.45 / +0.88 (0.63) | — |

Reference: L05ar's model-through rank on CEM populations is +0.24 (probe alone +0.82).

Read:
- One-step and 5-step *magnitude* prediction is saturated (R² 0.98–0.99) for every
  observation — persistence dominates; accuracy metrics cannot discriminate designs.
- A *linear* read-out ranks poorly (+0.46) whatever the observation; a non-linear GBM on the
  same base observation ranks held-phase plans at +0.78 and switch-heavy plans at +0.56 —
  more than twice the world model's +0.24 on a comparable population.
- Richer observations help on held-phase plans (top-1 0.43 → 0.70 with the raster) but
  **not** on switch-heavy plans (all ≈ +0.55). The observation is not what limits ranking in
  the regime the planner operates in. → H_plan (model + search) over H_obs.

Artifacts: `results/diag_obs_sufficiency.json`, `results/diag_obs_sufficiency_sw.json`.

## T4 — planning-side fixes on the existing L05ar  ✅

Closed loop, 5 seeds (777/101/202/303/404), MaxPressure warm-up 10 steps, 40 decision steps,
tail halting relative to MaxPressure. Oracle-CEM = identical search scored by SUMO
(24 samples x 3 iters). Brackets = 10th–90th percentile over seeds. "legal" masks phase codes
a signal does not have (5 of 8 cologne8 signals have 2 or 3 greens; the original search fed
the model never-seen action one-hots for them).

| variant | latent-CEM / MP | oracle-CEM / MP | latent / oracle | switch rate latent / oracle / MP |
|---|--:|--:|--:|--:|
| uniform, memoryless (original) | 3.33 [2.83, 3.81] | 0.84 | 4.0x | 0.74 / 0.63 / 0.21 |
| + hold prior 0.7 | 2.52 [1.70, 3.26] | 1.58 | 1.6x | 0.29 / 0.20 |
| + hold prior + warm start | 2.56 [1.92, 3.07] | 0.81 | 3.1x | 0.50 / 0.25 |
| + min-hold 2 (10 s) | 1.65 [1.21, 2.15] | 0.64 | 2.5x | 0.33 / 0.31 |
| legal mask only | 3.83 [3.51, 4.08] | 0.85 | 4.6x | 0.72 / 0.52 |
| legal + hold prior | 2.50 [2.29, 2.85] | 1.72 | 1.5x | 0.27 / 0.14 |
| legal + hold + warm | 2.89 [1.94, 4.08] | 0.63 | 4.5x | 0.52 / 0.21 |
| legal + min-hold 2 (10 s) | **1.54 [1.33, 1.83]** | 0.66 | 2.3x | 0.28 / 0.28 |
| legal + min-hold 4 (20 s, H=8) | 1.62 [1.53, 1.70] | 0.94 | **1.75x** | 0.17 / 0.18 |
| legal + per-agent search | 3.76 [3.23, 4.70] | 0.85 | 4.5x | 0.68 / 0.52 |

Read:
- **Minimum green is the planner-side fix that works**: 10 s holds take the latent planner from
  3.3x to 1.5x MaxPressure *and* give the best oracle controller (0.64–0.66). The constrained
  regime is better control, not a handicap. 20 s holds give the tightest seed spread and the
  smallest model/oracle ratio (1.75x) but a worse oracle (0.94).
- **Illegal action codes are not the driver** (legal mask alone: no change). Neither is the
  factored joint search (per-agent search: same gap). Neither is a memoryless proposal
  (warm start alone does not move the latent number).
- **The hold prior helps the model and hurts the oracle** (0.84 → 1.6–1.7) in both batches:
  under true dynamics a search that mostly proposes holding is a worse search; under the
  model it looks better only because it stops querying the model where it is wrong.
  Direct evidence that part of the original gap is *where the proposal distribution sends
  the model*.
- Residual under the best admissible regime: the latent planner is still 1.75–2.3x worse
  than the same search under true dynamics. That is the number the model-side tests
  (T2/T3 observation, T7 window) have to move.
- Note on switching: batch-1 rates count raw code flips; batch-2 rates count physical
  switches. Uniform CEM really executes ~0.72 switches per signal per 5 s step.

Artifacts: `results/control_pf_*.json`, `results/planfix_summary.json`, `logs/pf_*.log`.

## T2/T3 — richer observation world models  ✅

`L05ar_link` (F=19, base + mean neighbour link state) and `L05ar_raster` (F=73, base +
per-phase 25 m occupancy/speed raster to 200 m): identical recipe to L05ar — same episodes,
same seeds, same 80 epochs — with the observation as the only change end to end (corpus,
probe fit, closed-loop env).

| model | F | ridge probe MSE | CEM/MP | CEM/rand | probe rank A | model rank B | eps plan/beh | matched plan / rand |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| L05ar (base) | 9 | 21.4 | 3.46 | 0.48 | 0.82 | 0.24 | 4.48 | 4.32 / 4.11 |
| L05ar_ac_v1 (coverage post-train) | 9 | — | 2.10 | 0.29 | 0.79 | 0.35 | 2.91 | 2.99 / 2.74 |
| **L05ar_link** | 19 | **7.4** | **2.36** | 0.32 | **0.90** | 0.25 | 3.17 | 3.35 / 3.17 |
| **L05ar_raster** | 73 | **56.3** | **3.88** | 0.53 | **0.21** | 0.11 | 3.90 | 3.74 / 3.52 |

Read:
- **Link features help everywhere except plan ranking.** Probe MSE 21.4 → 7.4, probe-with-true-
  dynamics rank 0.82 → 0.90, closed loop 3.46x → 2.36x (as good as coverage post-training
  without any planner-matched data), regret drift flips negative. But the model's own plan
  rank barely moves (0.24 → 0.25). Better observation, better model, same ranking failure.
- **The raster model is worse on every planning metric** (3.88x, rank B 0.11) — and its *probe*
  collapses too (rank A 0.21, MSE 56.3), even though its 1-step val loss is the best of the
  three (tf 0.023 vs 0.047 for base). That is not an information problem: see the probe-
  capacity test below.

### Probe capacity — the read-out, not the latent (`diag_probe_capacity.py`)

Same latents, same target (per-phase halting), ridge (as used by the planner) vs a 2-layer MLP,
val MSE in vehicles² (target variance 1116):

| model | ridge val MSE (R²) | MLP val MSE (R²) |
|---|--:|--:|
| L05ar | 62.2 (0.944) | **1.0 (0.999)** |
| L05ar_link | 11.1 (0.990) | **1.5 (0.999)** |
| L05ar_raster | 167.7 (0.850) | **1.9 (0.998)** |

**The cost is fully present in every latent — including the raster model's — and the linear
probe throws most of it away.** With a richer observation the latent stores halting less
linearly (SIGReg shapes the latent isotropically, nothing ties the cost to a direction), so
the *planner's* cost function degrades exactly when the observation improves. This is the
value-equivalence point, measured: reward-free latents are not obliged to keep the control
cost linearly accessible, and the post-hoc read-out is where richer observations are lost.

Artifacts: `results/{compounding,probe_vs_model,plan_vs_beh,matched_switching}_L05ar_{link,raster}.json`,
`results/diag_probe_capacity.json`.

## T5-lite — non-linear read-out  ⏳ (base arm done)

`PROBE=mlp` swaps the planner's ridge probe for the 2-layer MLP above (same fit data, same
targets, no retraining of the world model). L05ar:

| metric | ridge | MLP |
|---|--:|--:|
| probe rank A (true dynamics) | 0.82 | **0.998** |
| probe top-1 A | 0.36 | **0.93** |
| model rank B | 0.24 | 0.31 |
| model top-1 B | 0.027 | 0.067 |
| model abs. error | 104 | 128 |
| **CEM/MP (5 seeds)** | 3.46 | **2.70** |
| CEM/random | 0.48 | 0.37 |
| regret model/random | 0.79 | 0.69 |
| regret drift | +22 | **−16** |

- Given true dynamics an accurate read-out picks the best candidate **93 %** of the time
  (ridge: 36 %). Through the model: **6.7 %**. Almost all remaining *ranking* failure is the
  latent rollout under planner actions.
- Absolute error through the model *rises* (104 → 128): a sharper read-out amplifies predictor
  error, because it is fitted on **encoder** latents and applied to **predictor** outputs.
- **Closed-loop control nevertheless improves 3.46x → 2.70x** with the regret drift flipping
  negative — comparable to what coverage post-training (2.10x) and the link observation (2.36x)
  bought, for the price of a probe fit. Top-1 is a harsh metric on a 64-candidate population of
  near-equivalent plans; a read-out that is right on average steers the closed loop better even
  when it rarely nominates the single best plan.
- → the cost read-out is a **first-class design axis**, and the cheapest intervention tested.

Link and raster arms pending.

## T7 — predictor context window  ⏳

`L05ar_h6`, `L05ar_h12`: base observation, context 6 / 12 steps (30 / 60 s) instead of 3.

(training)

## T7 — predictor context window  ⏳

`L05ar_h6`, `L05ar_h12`: base observation, context 6 / 12 steps (30 / 60 s) instead of 3.

(pending)
