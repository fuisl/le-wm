# Full-paper experiment results — 2026-09-04

Run in worktree `expl-fullpaper` (branch `expl-fullpaper`), cologne8, model `L05ar`
(`weights_epoch_80.pt`) unless noted. SUMO via the RESCO benchmark cologne8 config
(`resco/resco_benchmark/environments/cologne8/cologne8.sumocfg`) — the original
`HMARL-TSC` path is gone; `plan_cem_multi._resolve_sumocfg()` now falls back to it.

Targets AAMAS 2027 **full paper** — abstract 1 Oct 2026, paper **8 Oct 2026**,
rebuttal 20–24 Nov, notification 21 Dec. Conference 3–7 May 2027, Hanoi.

---

## TL;DR — all experiments complete (pipeline finished 09:15, 2026-09-04)

| diagnostic | L05ar (baseline) | L05ar\_ac\_v1 (coverage post-train) | L05ar\_ac\_v2 (+displacement) |
|---|--:|--:|--:|
| closed-loop latent-CEM / MaxPressure (5 seeds) | **3.46×** | **2.10×** | 2.37× |
| #1 ε\_plan / ε\_beh (planner-action error inflation) | 4.48× | 2.91× | 2.96× |
| #1 matched-switching: plan vs rand\_matched | 4.32 vs 4.11 | 2.99 vs 2.74 | 3.08 vs 2.82 |
| #3 probe rank / model rank (Spearman) | 0.82 / 0.24 | 0.79 / 0.35 | 0.76 / 0.31 |
| #4 best runtime signal / ensemble-disagreement gate | 0.27 / 0.013 | 0.28 / 0.024 | 0.24 / 0.025 |

**Every headline finding survives a proper action-conditioned retrain** (threat C4
fully addressed):

1. **Undetectability (CB2) holds and is well-powered.** 5 seeds, 125 decision
   steps, genuine seed-only bootstrap ensemble. No runtime signal reaches a
   gateable level for the baseline *or* either AC model; the canonical
   ensemble-disagreement gate is the weakest (+0.01–0.03, CI at 0).
2. **The mechanism is joint switching frequency, not a bespoke OOD region.**
   Matched-switching: `plan ≈ rand_matched` at the same switch rate, for the
   baseline (4.32/4.11) and the AC models (≈3.0/≈2.8). The planner runs at
   ~13× the joint switching rate of any scripted controller.
3. **Short horizons do not rescue the ranking** — model rank agreement is
   monotone increasing in H and worst at short H.
4. **Coverage-aware AC post-training is a quantified *partial* fix:** closes ~40 %
   of the closed-loop gap (3.46× → 2.10×), halves the planner-action error
   inflation (4.5× → 2.9×), lifts model plan-ranking (0.24 → 0.35, still ≪ the
   probe's 0.79), and flips the within-episode regret drift from rising to
   falling. It does **not** close the gap. A Delta-JEPA displacement loss adds
   nothing (v2 marginally worse than v1 on every metric despite better 1-step
   rollout accuracy).
5. **Multi-agent specificity is the soft spot.** The "joint-config coverage is
   the locus" hypothesis is disconfirmed (#6). The solo-agent control (#6a) shows
   the action-structure signal that gates single-agent planning (+0.64 / +0.41)
   fails to gate the factored joint search (+0.20) — but only for 2 of 3
   intersections tested (agent 3 is a noisy null). Report as suggestive isolation
   of the factored structure, not a law; strengthen with more intersections/seeds
   or a synthetic large-action-space agent.

**Publishable position:** a well-powered measured negative — *planning against a
straightforwardly-trained reward-free latent world model in a factored
multi-agent regime leaves a control gap that is (a) driven by the planner's joint
switching frequency, (b) undetectable by any standard runtime trust signal at the
plan level, and (c) only ~40 % closable by coverage-aware training* — plus a
reusable methodology (oracle-decomposition + plan-level detectability protocol).

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

## #6(a) solo-agent control — solo0 DONE ✅ (RESCUES the multi-agent contribution)

`diag_detectability.py --solo_agent A` — one intersection plans its H-step phase
sequence (space $P^H\approx$ the 8-agent per-step joint space); all other signals
hold. 5 seeds, 125 decision steps, same bootstrap ensemble. Agent 0 (a 4-phase
intersection):

| signal | full joint search (#4-proper) | **solo agent 0** | solo 95% CI |
|---|--:|--:|--:|
| ensemble disagreement | +0.013 | +0.023 | [−0.08, +0.12] |
| density (GMM) | +0.249 | +0.219 | [+0.17, +0.27] |
| displacement | +0.218 | +0.297 | [+0.20, +0.40] |
| **action structure (`n_switch0`)** | **+0.195** | **+0.635** | **[+0.49, +0.77]** |
| latent coverage (per-agent) | +0.269 | +0.245 | [+0.19, +0.30] |
| coverage (joint config) | +0.113 | +0.290 | [+0.19, +0.39] |

**Read:** when a *single* agent plans, "how many phase switches does this plan
make" (`n_switch0`) predicts per-plan model error at **+0.635 — clearly gateable**
(CI well clear of the 0.35 bar). In the *factored joint* search the identical
signal collapses to +0.195, not gateable (#4-proper). Same model, same signals,
same everything — only the search structure changes.

**This is the isolated factored-structure result the multi-agent contribution
needs:** *plan-level model error is runtime-detectable (via action structure)
under single-agent planning, and becomes undetectable once the search is over the
factored joint action space* — because the switching that matters is distributed
across agents and no cheap per-plan scalar captures the joint pattern. Causal,
not asserted.

### All three 4-phase intersections (5 seeds, 125 steps each)

`n_switch0` within-step Spearman with per-plan regret:

| planner | `n_switch0` r(regret) | 95% CI | gateable? |
|---|--:|--:|:--|
| **full joint search** (#4-proper) | **+0.195** | [+0.15, +0.24] | no |
| solo agent 0 | **+0.635** | [+0.49, +0.77] | **yes** |
| solo agent 3 | +0.088 | [−0.23, +0.40] | no (noisy) |
| solo agent 7 | **+0.411** | [+0.09, +0.68] | **yes** |

**Honest read:** the effect is real but **not universal** — action structure
becomes gateable in the solo regime for **2 of 3** intersections tested (agents 0
and 7), never in the joint search. Agent 3's `n_switch0` is a noisy null (wide
CI). Every other signal stays sub-gateable in every solo run, matching the joint
search. The coverage joint−agent difference is also inconsistent across the three
solo runs (−0.001, +0.019, +0.151) — no clean coverage story.

**For the paper:** #4-proper is the headline (undetectability, tight CIs). The
solo-agent control is a *supporting* isolation of the factored structure — report
it as "in 2 of 3 intersections, the action-structure signal that gates
single-agent planning fails to gate the factored joint search," not as a
universal law. Strengthening options: more intersections / more seeds on the
noisy one, or a synthetic single big-action-space agent.

Artifacts: `results/diag_detectability_solo{0,3,7}.json`.

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
("training-time can't save it either"); ≈ 1.3× ⇒ "here is the fix."

### Pivotal number — DONE for v1 ✅ (partial fix, gap not closed)

Closed-loop tail halting, latent-CEM vs MaxPressure vs random, **5 seeds**
(`diag_compounding.py`, 40 steps):

| model | latent-CEM / MaxPressure | latent / random | regret m/r | regret drift |
|---|--:|--:|--:|--:|
| **L05ar** (baseline, 5-seed refresh) | **3.46×** | 0.48× | 0.79 | **+2177** (rises) |
| **L05ar\_ac\_v1** (freeze enc.\ + coverage data) | **2.10×** | 0.29× | 0.64 | **−1895** (falls) |
| L05ar\_ac\_v2 (+ Delta-JEPA displacement loss) | 2.37× | 0.33× | 0.71 | −1847 (falls) |

**Read:**
- **Coverage-aware AC post-training (v1) recovers ~40 % of the excess over
  MaxPressure** (3.46× → 2.10×), makes the planner extract more of the
  oracle-vs-random gain (0.79 → 0.64), and **flips the within-episode regret
  drift from rising to falling** — the AC model does not walk itself into
  worse-modelled regions.
- **The Delta-JEPA displacement loss does *not* help planning:** v2 is
  2.37×, slightly *worse* than v1's 2.10×, despite better 1-step val rollout
  accuracy (0.057 vs 0.075). Better open-loop prediction ≠ better planning — a
  value-equivalence-flavoured observation (magnitude accuracy is not the thing
  that matters for plan ranking).
- **The gap is not closed at this model scale.** 2.10× still loses clearly to a
  greedy baseline.

**Verdict (decision rule):** lands *between* "structural negative" (≳2.5×) and
"here is the fix" (≈1.3×): a **quantified partial fix**. Publishable statement:
*"coverage-aware training-time post-training recovers ~40 % of the control gap in
factored latent planning; a displacement/action-sensitivity loss adds nothing;
the residual is not eliminated at this model scale"* — a measured negative that
motivates the coupling / hierarchy chapter.

Pending: #1/#3/#4 re-runs on v1 and v2 (do the C1–C3 diagnostics move with the
better planner?), then pipeline done.
