# Continuous Trajectory PAM — Tiered v0 Specification

**Status:** Design phase — staged exploration plan  
**Version:** Tiered v0 (post-adversarial review synthesis)  
**Date:** April 2026  
**Guiding Principle:** Single-variable changes. Core claim validated first. New mechanisms introduced only as labelled experiments with explicit ablation plans and removal triggers.

---

## Executive Summary

This document defines a **staged, disciplined path** for Continuous Trajectory PAM.

**Tier A (Core)** validates the original thesis with one high-confidence upgrade (SIGReg).  
**Tier B** adds targeted experimental mechanisms only when specific problems are measured.  
**Tier C** contains deployment refinements once a working baseline exists.

The research claim remains pure and isolated:
> “A trajectory predictor trained on continuous visual experience produces cross-boundary associative retrieval where representational similarity scores near zero.”

Hybrid similarity + trajectory scoring is retained as a **deployment option**, not part of the core research claim or headline evaluation.

This structure directly addresses stagnation by enabling early implementation of a minimal viable system while preserving experimental integrity.

---

## Tier A — Core Architecture + One Substantive Upgrade (Validate First)

### Research Claim (Pure)
A trajectory predictor trained on continuous experience produces cross-boundary retrieval where similarity scores near zero. This is the distinctive, paper-validated contribution. Evaluation isolates the trajectory signal.

### Architecture (Tier A)

**Outward JEPA — Visual Encoder**
- ViT-style encoder (start with frozen V-JEPA 2 scaffolding or small trainable ViT-Tiny).
- Masked patch prediction (I-JEPA / V-JEPA protocol).
- **Strong candidate upgrade: SIGReg (Sketched-Isotropic-Gaussian Regularizer)**  
  - Project latents onto many random directions.  
  - Penalize deviation from normality on each 1D projection (Epps-Pulley statistic).  
  - Enforces isotropic Gaussian geometry on the full latent space (Cramér-Wold theorem).  
  - **Status**: Strong candidate for adoption in Tier A.  
  - **Ablation plan**: Train identical encoder with (a) classic stop-grad + EMA and (b) SIGReg. Compare training stability, embedding geometry (isotropy metrics), downstream Inward PAM performance, and cross-boundary retrieval quality.  
  - **Removal trigger**: If SIGReg destabilizes training, reduces cross-boundary performance, or shows no advantage over stop-grad + EMA after 3–5 runs, revert to stop-grad + EMA.

**Inward PAM — Trajectory Predictor**
- Transformer that takes window of N recent embeddings with masked temporal positions.
- Predicts embeddings at masked positions using MSE loss against stored targets (stop-gradient on targets).
- Learnable temporal position embeddings + mask tokens.
- **No recurrent LTI structure in Tier A** (deferred to Tier B with trigger).

**Memory Bank**
- Append-only store of every experienced frame embedding (or uniform random subsample initially).
- Simple management for Tier A (no surprise weighting or MLA compression yet).

**Inference & Retrieval (Tier A)**
- Live context → Inward PAM predicts target embedding at chosen position (next-step or random-window retrieval token — to be ablated in prototype).
- Nearest-neighbour lookup in memory bank using **pure cosine similarity on the predicted embedding**.
- **Multi-hop**: Single-best-hop only (K=1). Each hop uses the top retrieval as new context for the next prediction.
- **Headline evaluation metric**: Cross-boundary recall where cosine similarity to ground-truth associate is near zero. Secondary: specificity gain vs pure kNN baseline on same probes.

**Training Regime (Tier A)**
- Online, single-pass on continuous visual stream.
- **Curriculum Environment — Staged Progression (0a → 0b → 0c)**

  **Core Design Principle**: PAM learns that temporally adjacent experiences are associated even when embedding geometry suggests they should not be. In continuous reality this happens via smooth transition chains (doorway frames, biome morphs, rounding a corner). The predictor learns to traverse chains through transition zones rather than memorising direct pairs.

  **Staged Diagnostic Progression** (one environment family, three progressively richer setups):

  **Stage 0a — Single Config, Smooth Dynamics (Interpolation Baseline)**
  - Standard Push-T (one configuration, no visual changes).
  - Validates basic trajectory prediction and interpolation regime.
  - No cross-boundary evaluation yet.
  - Physical probing baselines from LeWM literature.
  - Gate: Predictor achieves stable low MSE on masked positions and smooth interpolation before proceeding.

  **Stage 0b — Multi-Config with Instant Scene Cuts (Simplified Cross-Boundary Test)**
  - 3–4 visually distinct configurations (example: blue/green/default, red/green/dimmer, grey/blue/default).
  - Task and dynamics identical across configs; only visual factors vary.
  - **Pre-gate**: Mean cosine similarity between configs < 0.8.
  - Instantaneous scene cuts (frame t-1 in config A, frame t in config B).
  - Sufficient block duration (500–2000 frames) and transition frequency so every config pair co-occurs many times.
  - **Purpose**: Validates whether the predictor can bridge temporal co-occurrence across embedding discontinuities at all (simplified diagnostic). If it fails here, it cannot succeed on the harder smooth-transition case.
  - Evaluation: Cross-boundary recall where cosine scores near zero (headline), within-config sanity checks, shuffle control, kNN baseline.

  **Stage 0c — Multi-Config with Smooth Morph Transitions (Real Capability — Chain Traversal)**
  - Same 3–4 configs as 0b.
  - Between configs the environment smoothly morphs over 100–300 frames (background colour interpolates, block texture crossfades, lighting shifts continuously).
  - The agent continues experiencing the environment throughout the morph — transition frames form a chain linking the two regions.
  - **Purpose**: Tests the full intended capability — learning general chain-traversal structure rather than specific pair memorisation. A new transition experienced later can build on the learned traversal mechanism without retraining on every pair.
  - Evaluation: Same cross-boundary probes as 0b, but now measuring retrieval across smooth transition chains (many-hops-removed associations). Additional diagnostic: does performance generalise to new transition pairs not seen in training?

  **Progression Rule**: Each sub-stage is gated on the previous. Stage 0a validates basic machinery. Stage 0b validates boundary-bridging in the simplified case. Stage 0c validates the realistic smooth-transition chain mechanism that matters for embodied agents in continuous reality.

  - Later stages (after 0c): Increase masking ratio and window size. Introduce richer continuous environments (AI2-THOR smooth navigation, procedural biome transitions, etc.) only after the full 0a–0b–0c progression is validated.
- Loss: MSE + (if SIGReg adopted) SIGReg term on Outward latents.
- No epochs, no revisiting.

**Scope (Tier A)**
- Not action-conditioned.
- Single visual stream.
- Memory/retrieval layer only (can later sit alongside a LeWM-style world model).

**Pre-Experiment Gates (Tier A)**
- Stage 0a–0c progression is implemented with proper gating (0a stable before 0b; 0b successful before 0c).
- Configs in 0b/0c are genuinely distinct (mean cosine < 0.8) with identical task/dynamics.
- Smooth morph transitions in 0c use 100–300 frames with sufficient co-occurrence window (τ = 5–15).
- Evaluation can isolate trajectory contribution (cross-boundary probes + shuffle control + chain-traversal diagnostics in 0c).
- Pure cosine/kNN baseline is beatable on cross-boundary probes.

**Goal of Tier A**
Establish that the core mechanism works and produces the claimed cross-boundary capability. This is the minimal viable system that can be implemented and measured quickly.

---

## Tier B — Experimental Mechanisms (Investigate After Tier A Baseline)

These are introduced **only** when a specific problem is measured in Tier A. Each has a clear trigger, ablation plan, and removal condition.

### B1. LTI-Recurrent Predictor Structure (from OpenMythos)
- Wrap Inward PAM in LTI injection + recurrent block with loop-index embeddings.
- Update rule: `h_{t+1} = A_disc · h_t + B · e + Transformer(h_t, e)` with guaranteed `ρ(A) < 1`.
- **Trigger for investigation**: If single-best-hop multi-hop retrieval produces divergent or unstable chains beyond depth ~3 (measured by retrieval quality collapse or exploding prediction error).
- **Ablation plan**: Compare (a) standard transformer predictor vs (b) LTI-recurrent version on stability, multi-hop depth, and cross-boundary performance. Ablate A_disc, B·e, and Transformer term individually.
- **Removal trigger**: If it adds complexity without measurable gain in stability or multi-hop quality once the trigger condition appears, do not adopt.

**Rationale for deferral**: PAM’s multi-hop is iterated prediction against a memory bank (grounded at each step), not single hidden-state evolution. We do not yet have evidence we need stable long recurrence.

### B2. MLA-Style Memory Compression (from OpenMythos)
- Store low-rank latent + RoPE keys instead of full embeddings; reconstruct for NN lookup.
- **Trigger for investigation**: When memory bank size creates measurable retrieval latency or memory pressure in Tier A runs.
- **Ablation plan**: Compare retrieval quality, speed, and memory usage with full embeddings vs MLA compression on the same probe set.
- **Removal trigger**: If reconstruction error degrades cross-boundary recall by >5% or implementation overhead outweighs benefits.

---

## Tier C — Deployment Refinements (After Tier A + Selected Tier B Items Validated)

These assume a working baseline exists. They improve practicality or performance but are not required to validate the core thesis.

- **Hybrid retrieval scoring** (`s_hybrid = α · cos(ẑ_query, z_mem) + (1-α) · sim_trajectory(...)`) with dynamic/gated α.  
  Kept as **deployment option**. Headline research evaluation remains pure cosine on predicted embedding + cross-boundary isolation test.

- Depth-wise LoRA adapters per curriculum stage or loop iteration.

- ACT-style halting for variable-depth prediction / multi-hop.

- Surprise / prediction-error signals for memory retention and eviction.

- Small meta-network for learning when to trust similarity vs trajectory signal (only if dynamic α proves insufficient).

**Status**: All Tier C items are deferred until Tier A (and any triggered Tier B items) demonstrate a stable, working system.

---

## Updated Open Design Questions & Process Notes

**Resolved in this tiered plan**
- Query token semantics: Ablate next-step vs random-window retrieval token inside Tier A prototype.
- MSE discrimination: Retain stop-grad targets in Tier A; SIGReg ablation tests whether geometry improves it.
- Memory management: Simple append-only in Tier A; MLA and surprise weighting deferred to Tier B/C with triggers.
- Encoder choice: SIGReg as strong candidate upgrade with clear ablation.

**Process Discipline Enforced**
- Every new mechanism is explicitly labelled (Core / Strong Candidate / Experimental with Trigger / Deferred).
- Every experimental item has a removal/ablation plan and a concrete trigger condition.
- The research claim and headline evaluation remain pure and isolated.
- Scope expansion is now deliberate and staged rather than bundled.

---

## Immediate Next Steps (Post v0 Review)

1. Final adversarial review of this tiered spec (primary + secondary reviewer).
2. Lock Tier A definition and begin minimal prototype:
   - Stage 0 curriculum on Push-T / Reacher-style environment.
   - Frozen V-JEPA 2 or small trainable encoder + SIGReg ablation.
   - Inward PAM as standard transformer predictor (no LTI yet).
   - Pure cosine retrieval + cross-boundary evaluation.
3. Run Tier A experiments. Measure whether core claim holds.
4. Only then evaluate Tier B triggers and decide on adoption.

---

## Appendix: Traceability to Source Discussions

- Original seed summary: Core two-component design, online training, curriculum, pure claim, open questions.
- LeWM / SIGReg analysis: Principled regularizer, isotropic geometry benefits, curriculum starting environments, strong candidate for Tier A.
- OpenMythos LTI analysis: Recurrent latent dynamics, stability tools — properly deferred with trigger (addresses “do we have this problem?” concern).
- Hybrid clarification: Hybrid scoring retained as deployment option; pure claim and isolated evaluation preserved.

---

*This tiered v0 specification restores process discipline while capturing the genuine value from recent advances. It enables early implementation to break stagnation while protecting the integrity of the core research claim.*

**End of Tiered v0 Spec**