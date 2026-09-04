# Full-paper experiment results — 2026-09-04

Run in worktree `expl-fullpaper` (branch `expl-fullpaper`), cologne8, model `L05ar`
(`weights_epoch_80.pt`) unless noted. SUMO via the RESCO benchmark cologne8 config
(`resco/resco_benchmark/environments/cologne8/cologne8.sumocfg`) — the original
`HMARL-TSC` path is gone; `plan_cem_multi._resolve_sumocfg()` now falls back to it.

Targets AAMAS 2027 **full paper** — abstract 1 Oct 2026, paper **8 Oct 2026**,
rebuttal 20–24 Nov, notification 21 Dec. Conference 3–7 May 2027, Hanoi.

---

## #1 matched-switching control — DONE ✅ (changes the #1 framing)

`diag_matched_switching.py`, L05ar, 3 seeds (777/101/202), 20 anchors/seed, H=8.
240 beh / 720 plan / 720 rand_matched / 720 plan_downsampled sequences.

| group | mean latent-err ratio vs beh | joint switch rate |
|---|--:|--:|
| beh (scripted: max_pressure / fixed_time / ε-hold) | 1.00 | 0.054 |
| **plan** (CEM elite) | **4.32×** | 0.720 |
| **rand_matched** (random joint @ the plan switch rate) | **4.11×** | 0.720 |
| plan_downsampled (CEM elite, switches thinned to scripted rate) | 0.44× | 0.002 |

**Read:** the 4.3× model-error inflation on the planner's action sequences (#1) is
**~95 % explained by joint phase-switching frequency alone**. Random joint
sequences that merely switch as often as the planner (0.72) are already 4.11× —
statistically indistinguishable from the planner's own elites (4.32×). Conversely,
taking the planner's elites and thinning their switches to the scripted rate drops
the error *below* the scripted baseline (0.44×; caveat: near-static sequences are
trivially "predict no change", so `rand_matched ≈ plan` is the load-bearing
comparison).

**Consequence for Paper 1:** retire "the planner selects a specific OOD joint
region the model is bad at." The supported claim is: *the model's multi-step
latent rollout error rises steeply with **joint switching frequency**, and the
planner operates at ~13× the switching rate of any scripted controller
(0.72 vs 0.054).* Still a planner-induced distribution shift (the planner drives
the switching rate up); still multi-agent (it is the **joint** rate — many
signals switching per step — that no single controller produces); and it connects
cleanly to Delta-JEPA / the JEPA slow-feature bias (the model cannot track fast
action-driven latent change). This is a cleaner, more mechanistic story than
"exploitation of a bespoke region."

Artifact: `results/matched_switching_L05ar.json` (+ the AC-model reruns from the
pipeline).

---

## Horizon-sensitivity sweep (#3 at H = 1…8) — DONE ✅ (kills the "just re-plan often" objection)

`diag_probe_vs_model.py --horizon H --iters 2 --S 96`, L05ar, 2 seeds (777/101),
25 decision steps each. NOTE: `cem_rollout`'s first predicted step is
action-independent by construction, so reported H = 1 + (effective
action-conditioned horizon). H=1 is degenerate (model cost constant across
candidates → ranking undefined); the usable sweep is effective H ∈ {1,2,3,5,8}.

| reported H | effective H | probe rank vs true (A) | **model rank vs true (B)** | model top-1 (chance ~0.01) |
|--:|--:|--:|--:|--:|
| 2 | 1 | +0.54 | **−0.02** | 0.00 |
| 3 | 2 | +0.79 | **+0.18** | 0.04 |
| 4 | 3 | +0.82 | **+0.24** | 0.06 |
| 6 | 5 | +0.86 | **+0.31** | 0.06 |
| 9 | 8 | +0.90 | **+0.38** | 0.04 |

50/50 decision steps had a well-defined model ranking at every H≥2.

**Read:** at **every** horizon the probe (given true dynamics) ranks plans well
(+0.54 → +0.86) and the world model **destroys the ranking** (−0.02 → +0.31). The
model's rank agreement is **monotonically increasing** with horizon — it is
*worst* at the shortest horizons, not best. There is **no horizon sweet spot**:
short-horizon + frequent re-planning does not rescue the model (the opposite —
at effective H=1 the model has essentially zero ability to rank plans), and even
at H=5 the model (+0.31) is far below the probe (+0.86) and nowhere near a
gateable level.

This closes the R3 MAJOR alternative-path objection ("the fix is short horizon +
frequent re-plan, not training"): **it is not.** Caveat to report: the probe's own
rank agreement is lower at short H (+0.54 at eff H=1) because with one action step
the true plan costs are close together; the model–probe *gap* is large at every H.

Artifacts: `results/horizon_sweep/H{2,3,4,6,9}.json`.

---

## #4-proper — seed-only bootstrap ensemble — DONE ✅ (CB2 negative holds, now powered)

K=4 identical L05ar models, seed-only variation (1001/2002/3003/4004), same data
as L05ar. `diag_detectability.py --ensemble L05ar_bs* --seeds 777 101 202 303 404
--n_steps 25 --tag proper` — **5 seeds, 125 decision steps**, bootstrap 95 % CIs.

| signal | within-step Spearman vs per-plan regret | 95 % CI |
|---|--:|--:|
| **ensemble disagreement** (true seed-only bootstrap — the MOPO/MOReL gate) | **+0.013** | [−0.01, +0.03] |
| plausibility (`frac_clipped`) | +0.013 | [−0.02, +0.04] |
| action structure (`n_switch0`) | +0.195 | [+0.15, +0.24] |
| displacement (Delta-JEPA-style) | +0.218 | [+0.19, +0.25] |
| density model (16-comp GMM) | +0.249 | [+0.22, +0.28] |
| **latent coverage (per-agent)** | **+0.269** | [+0.24, +0.30] |

**Read:** the negative is now properly powered. No signal reaches a gateable
level (≈0.5). The **canonical ensemble-disagreement gate — a *genuine* seed-only
bootstrap ensemble, i.e. the actual MOPO/MOReL mechanism — is the weakest signal
at +0.013, CI spanning 0.** This is *stronger* than the fast cut (0.04 with a
heterogeneous 4-checkpoint ensemble): the proper bootstrap ensemble carries
essentially no per-plan signal. Best signal (per-agent coverage, +0.27) has a
tight non-zero CI but is nowhere near actionable. **Threat C5 resolved; CB2 is
the headline the paper can lead with.**

Artifact: `results/diag_detectability_proper.json` (+ `_rows.csv`).

## #6 MA-specificity — DONE ✅ (NEGATIVE for the hoped-for framing — honest finding)

Same run. Per-agent-mean coverage vs joint-configuration coverage as distinct
predictors of per-plan regret, paired bootstrap CI on the difference:

| predictor | within-step Spearman vs regret | 95 % CI |
|---|--:|--:|
| `cov_agent` (per-agent-mean) | **+0.269** | [+0.24, +0.30] |
| `cov_joint` (joint config) | **+0.113** | [+0.09, +0.14] |
| paired difference (joint − agent) | **−0.156** | [−0.18, −0.13] — **excludes 0** |

**Read:** per-agent coverage predicts per-plan regret **better** than
joint-configuration coverage. The hypothesis "the model's failure is about the
*joint* configuration, not any agent in isolation" is **not supported on this
axis.**

### ⚠️ Consequence for the "multi-agent contribution" (full-paper concern)

Three MA-specific framings are now disconfirmed by our own experiments:
1. "factored search makes exploitation *stronger*" — MA scale check (regret/gain
   flat 0.70–0.78 across 8→21 signals);
2. "the planner selects a bespoke OOD joint *region*" — matched-switching (it is
   switching *frequency*, not a region);
3. "joint-configuration coverage is the locus of failure" — #6 above.

**What survives is the mechanism claim only:** the factored joint action space is
*why* the planner reaches a joint-switching regime (0.72) unreachable by any
single scripted controller (0.054), *why* the model's rollout degrades there, and
*why* single-agent detection tools — the MOPO ensemble gate especially — do not
transfer. For an AAMAS full paper this needs either (a) the honest reframe to
*"planning over a factored joint action space in a coupled network"* with the
methodology + the detectability negative as the lead, or (b) one more experiment:
the **single-agent large-action-space control** (one intersection, large
phase-plan space) — if the same non-detectability appears there, the effect is
large-action-space, not multi-agent; if not, the factored joint structure is
load-bearing. Roadmap item #6(a); **recommend running it before committing to the
full-paper MA framing.**

---

## #7 — AC post-training — RUNNING ⏳

Augmented corpus `traffic_data_cologne8_aug` built (`traffic/generate_augment_sumo.py`):
24 base + **12 planner-matched** train episodes (6 `random_joint`, 3 `cem_elite`
closed-loop DAgger, 3 `eps_phase@0.3`, 3 `eps_phase@0.5`); 6 + 3 val.

`train_multi_sumo_ar.py` gained `--init_from` (warm-start), `--freeze_encoder`
(train predictor + action/message heads only), `--displacement_w` (Delta-JEPA
latent-difference loss on the AR rollout).

Variants:
- **L05ar_ac_v1** — warm-start L05ar, freeze encoder, augmented corpus, no displacement loss.
- **L05ar_ac_v2** — as v1 + `--displacement_w 1.0`.

Then re-run on each: closed-loop `diag_compounding` (latent-CEM / MaxPressure —
**the pivotal number**), `diag_probe_vs_model` (#3), `diag_plan_vs_beh` (#1),
`diag_matched_switching`, `diag_detectability`.

**Decision rule:** latent-CEM / MaxPressure ≳ 2.5× ⇒ structural negative
("training-time can't save it either"); ≈ 1.3× ⇒ "here is the fix." Baseline L05ar
is ~3.43× (Add. 4, 3 seeds) — being refreshed to 5 seeds in the same run.

Pending: `results/compounding_L05ar*.json`, `results/*_L05ar_ac_v{1,2}.json`.
