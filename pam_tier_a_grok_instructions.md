# Continuous Trajectory PAM — Tier A Implementation Instructions

**Target:** Grok 4.20 Reasoning (in Cursor)
**Scope lock:** Tier A only. Tier B and Tier C items are explicitly out of scope for this work.
**Source spec:** `PAM_Tiered_v0_Spec.md` (canonical — this document implements, does not replace).
**Operating mode:** You are implementing a research system. Work is staged, instrumented, and stops at defined gates for human review.

---

## 0. Environment Header

- OS: Ubuntu 24.04 (local development) / Linux containers on vast.ai for scaled runs
- Python: 3.11
- PyTorch: 2.4+ with CUDA 12.4
- GPU (local): NVIDIA RTX 4080 Super, 16GB VRAM
- GPU (rented, when needed): vast.ai A100 or H100, 500GB+ disk provisioned
- Working directory: `~/projects/continuous_trajectory_pam/` (create if absent)
- Virtual environment: `.venv/` at project root, activated via `source .venv/bin/activate`
- Required envs: `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` (not applicable to Grok, but mirror: stay verbose in diagnostics, terse in production code)

Do not install packages globally. Do not operate outside the project directory. If a package is not in `requirements.txt`, add it there with a pinned version before importing it.

---

## 1. Operational Rules

These are non-negotiable. They exist because prior work produced specific failure modes they prevent.

1. **Never kill a running training process.** If a job is running, let it finish. Interrupting produces inconsistent state and corrupts logs.
2. **Use `nohup` for any script expected to run longer than 5 minutes.** Redirect to a timestamped log file. Poll the log rather than blocking on stdout.
3. **Stop after 5 failed tool calls in sequence.** Repeated failures indicate a structural problem. Report and wait for instruction.
4. **Operate in "away mode".** The user is not watching. Make reasonable decisions based on this spec, document them in `HANDOFF.md`, and flag them for later review. Do not stop to ask clarifying questions mid-task.
5. **Every number you report traces to a file.** No remembered numbers. No mental arithmetic. If the source file does not exist, the number is not quoted.
6. **Git commit after every completed task.** Descriptive messages. History reconstruction is impossible without this.
7. **Update `HANDOFF.md` at the end of every session.** Record what was attempted, what worked, what failed, what is in progress, and the next immediate action.
8. **STOP at gate failures.** Every stage has pass/fail gates. When a gate fails, stop and report. Do not silently proceed to the next stage.
9. **Single-variable changes only.** When testing an intervention, change exactly one thing. Bundled changes produce ambiguous attribution.
10. **"What NOT to change":** the scope lock in §2. Do not implement Tier B or Tier C items under any circumstances, even if they seem helpful. If you think a Tier B/C item is required, stop and report — do not implement.

---

## 2. Scope Lock — Tier A Only

### In scope for this work
- Outward JEPA visual encoder (frozen V-JEPA 2 for Stage 0a; SIGReg-trained small ViT-Tiny introduced as ablation from Stage 0b onwards)
- Inward PAM transformer trajectory predictor with masked temporal position prediction (MSE loss, stop-gradient on targets)
- Append-only memory bank (every frame, no subsampling, no compression)
- Pure cosine nearest-neighbour retrieval on predicted embeddings
- Single-best-hop multi-hop retrieval only (K=1 per hop)
- **Query mechanism for Tier A: next-step prediction.** The predictor produces an embedding at position t+1 given visible context [t-W+1, ..., t]. Retrieval uses this predicted embedding against the memory bank. The random-window retrieval token alternative is deferred.
  - **Potential follow-up (post-Tier A):** ablate next-step vs random-window retrieval token on the same Tier A checkpoint. Next-step may bias retrieval toward forward-prediction similarity; random-window may provide richer associative targets. Worth testing once the baseline is validated.
- Online, single-pass training with curriculum Stages 0a → 0b → 0c on Push-T variants

### Explicitly out of scope (do not implement)
- LTI-recurrent predictor structure (Tier B1)
- MLA-style memory compression (Tier B2)
- Hybrid retrieval scoring / dynamic α (Tier C)
- LoRA adapters per stage (Tier C)
- ACT halting (Tier C)
- Surprise-based memory retention (Tier C)
- Meta-network for signal trust (Tier C)
- Random-window retrieval token mechanism (deferred sub-ablation)
- Multi-step K>1 retrieval branching (deferred)
- Action conditioning of any kind (project scope)
- Multi-modal inputs (project scope)

If any of the out-of-scope items appear necessary during implementation, stop and report. Do not implement.

---

## 3. Architecture — Implementation Specification

### 3.1 Outward JEPA Visual Encoder

**Stage 0a: Frozen V-JEPA 2**
- Load V-JEPA 2 ViT-Large from the reference checkpoint (document the source and version hash in the code).
- Freeze all parameters.
- For each input frame (RGB, resized to 224×224): forward through V-JEPA 2, extract the CLS token from the final layer. This is the frame embedding.
- Embedding dimension: 1024 (V-JEPA 2 ViT-L default). Confirm this against the loaded checkpoint.
- **No fine-tuning, no projection head, no trainable parameters in this module for Stage 0a.**
- **Potential follow-up (post-Tier A):** once the trajectory predictor mechanism is validated against frozen V-JEPA 2, test whether a SIGReg-trained encoder (introduced in Stage 0b as ablation) outperforms the frozen baseline on cross-boundary recall. If so, SIGReg ViT-Tiny or SIGReg-finetuned V-JEPA 2 becomes a candidate for Tier B adoption.

**Stage 0b/0c: Frozen V-JEPA 2 remains primary; SIGReg-trained ViT-Tiny added as ablation arm**
- Introduce a second encoder path: small ViT-Tiny (~5M params) trained end-to-end with:
  - Standard I-JEPA masked patch prediction objective (MSE in latent space between predictor output and target encoder output on masked patches).
  - **SIGReg regularizer on the target encoder's output embedding** in place of stop-gradient + EMA. Implementation follows LeWorldModel (arXiv:2603.19312). Fetch the paper and reference implementation directly; implement from the source rather than from a summary.
  - Retain a 1-layer BatchNorm projection head after the encoder's final LayerNorm (LeWM §A2 pattern) to undo LayerNorm's distributional compression before SIGReg sees the embedding.
- Run Stage 0b and 0c with both encoder paths (frozen V-JEPA 2 and SIGReg ViT-Tiny) as an ablation. Report both.

**Design intent for SIGReg in our system (anchoring requirements — not implementation instructions):**

SIGReg is being used here as the anti-collapse mechanism that replaces stop-gradient + EMA for the trainable encoder. Its value to this specific architecture comes from three properties:

1. *Isotropic Gaussian embedding geometry.* Downstream retrieval uses cosine similarity on predicted embeddings. If the encoder produces embeddings with well-spread, isotropic structure, cosine similarity is meaningful across the full embedding space. Under LayerNorm compression (a known failure mode in prior work — score dynamic range as low as 0.000012 top1-top20 in spatially homogeneous environments), cosine similarity degenerates. SIGReg + BatchNorm projection head is the intended fix.

2. *Stability during training.* The trajectory predictor needs stable targets. If the encoder's representations drift non-smoothly (as can happen with aggressive EMA schedules), the memory bank contains embeddings that were produced by a now-stale encoder, and the predictor chases a moving target. SIGReg's regularizer is per-step and does not depend on a second network.

3. *Single hyperparameter (λ_sigreg).* Tunable by bisection. Contrast with the multi-term losses that produced training instabilities in related architectures (VICReg's 7-term loss being the canonical example). If λ requires extensive tuning to work at all, that is itself a signal to abandon SIGReg and revert.

**Required diagnostic logging for the SIGReg arm (not required for frozen V-JEPA 2):**
- Effective dimensionality (PCA 95% variance threshold) of the encoder's output distribution, logged every 5000 steps.
- Isotropy score (ratio of smallest to largest eigenvalue of embedding covariance, computed on a 1000-sample batch).
- Epps-Pulley statistic distribution across projection directions (histogram at checkpoint saves).
- Cosine similarity dynamic range (top1 vs top20 nearest neighbours on a held-out set of 100 embeddings).

These diagnostics are how we'll know whether SIGReg is producing the geometry we need. If isotropy and effective dimensionality do not increase meaningfully over the stop-grad + EMA baseline, SIGReg is not earning its complexity.

### 3.2 Inward PAM — Trajectory Predictor

- Architecture: standard transformer encoder (PyTorch `nn.TransformerEncoder`) with 4 layers, 8 heads, hidden dim 512, MLP dim 2048, GELU activations, pre-LayerNorm.
- Input: window of W frame embeddings. Tier A default W = 16.
- Each position in the window has:
  - A frame embedding (from §3.1), projected linearly to hidden dim 512.
  - A learnable temporal position embedding (one per position 0..W-1, plus one for the prediction position W, total W+1 learnable position vectors).
  - A learnable mask token vector (single shared vector) used at positions where the embedding is masked.
- Masking during training: select a set of positions to mask (masking schedule per §3.4). Replace their projected embeddings with the mask token. Temporal position embeddings are retained at all positions.
- Query position: append a single query position at index W (corresponds to t+1 in next-step prediction). This position always receives the mask token as input (no embedding to mask, it's the position being predicted).
- Output: the transformer output at the query position, projected linearly back to the original embedding dimension (1024 for frozen V-JEPA 2; whatever the ViT-Tiny produces for the SIGReg arm).
- **Loss**: MSE between the predicted output at the query position and the actual frame embedding at t+1 (which will be observed on the next step). **Stop-gradient on the target** — no gradients flow into the Outward encoder from this loss. During training, loss is also computed on the masked positions within the window using the same MSE + stop-gradient mechanism.

### 3.3 Memory Bank

- Data structure: numpy array or `torch.Tensor` on CPU, pre-allocated to maximum expected size (100k frames for Stage 0 curriculum; grow as needed).
- Entry per frame: frame index, embedding (float32), stage tag, config tag (which configuration the frame was experienced in), transition-zone flag (for Stage 0c).
- Append-only. No eviction, no subsampling, no compression in Tier A.
- Retrieval: exact nearest-neighbour cosine similarity via `faiss.IndexFlatIP` on L2-normalised embeddings. Index rebuilt periodically (every N=1000 new frames) — not on every insert, for efficiency.

### 3.4 Training Protocol

**Training loop pseudocode:**
```
initialize Outward encoder (frozen V-JEPA 2 for 0a; both encoders active from 0b)
initialize Inward predictor (random init, pre-LN transformer)
initialize memory bank (empty)
initialize frame buffer (ring buffer of size W)

for t in 0..T_total:
    frame = environment.next_frame()
    embedding = outward_encoder(frame)
    memory_bank.append(embedding, metadata)
    frame_buffer.push(embedding)

    if len(frame_buffer) < W:
        continue  # warmup

    # Training step
    context = frame_buffer.last_W_embeddings()  # [W, D]
    next_target = next_observed_embedding()  # from the PREVIOUS training step's "next" frame

    # Apply masking schedule (see below)
    mask_positions = sample_mask_positions(W, current_mask_ratio)
    predicted_next, predicted_masked = inward_predictor(context, mask_positions)

    loss_next = MSE(predicted_next, stop_grad(next_target))
    loss_masked = MSE(predicted_masked, stop_grad(context[mask_positions]))
    loss = loss_next + loss_masked

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % rebuild_interval == 0:
        memory_bank.rebuild_index()
```

**Masking schedule (curriculum-driven):**

Start with low mask ratio, grow with predictor competence. The mask ratio applies to positions within the window (not the query position at t+1, which is always predicted).

- Stage 0a initial: mask_ratio = 0.0625 (1 of 16 positions). Pure interpolation plus next-step.
- Grow trigger: mask loss plateau. When `loss_masked` has not decreased by more than 5% over the last 10,000 training steps, increase mask_ratio by one position (e.g., 1→2).
- Stage 0a target: reach mask_ratio ≥ 0.25 (4 of 16) with stable predictions before proceeding.
- Stage 0b/0c: mask ratio continues to grow. Target is 0.5 (8 of 16) or higher by end of Stage 0c — the trajectory regime where genuine temporal structure is required.

Log the current mask ratio, mask loss, and next-step loss at every training step. Log full distributions every 1000 steps.

**Optimizer:** AdamW, lr = 3e-4, weight_decay = 0.01, betas = (0.9, 0.95). Cosine warmup over first 5000 steps. No LR decay for Tier A — we want to see whether training plateaus naturally.

**Device placement:** Outward encoder and Inward predictor on GPU. Memory bank on CPU (use `faiss-gpu` only if retrieval becomes a bottleneck — profile first).

### 3.5 Retrieval at Inference

Retrieval is used during evaluation, not during training.

```
def retrieve(context_window, memory_bank, K=1):
    """
    context_window: [W, D] recent frame embeddings
    Returns: top-K nearest neighbours in memory bank to predicted t+1 embedding
    """
    predicted = inward_predictor(context_window, mask_positions=[])  # no masking at inference
    predicted_query = predicted[W]  # embedding at query position
    predicted_normalised = F.normalize(predicted_query, dim=-1)
    scores, indices = memory_bank.index.search(predicted_normalised.unsqueeze(0), K)
    return indices, scores
```

Multi-hop: for hop h ∈ [2, 3]:
- Take the retrieved frame at hop h-1.
- Construct a new context window ending at that retrieved frame (the W frames stored immediately before it in the memory bank, based on frame index).
- Run retrieval again to get hop h.
- Stop at hop 3 for Tier A. Report per-hop retrieval quality.

---

## 4. Curriculum — Push-T Staged Environment

### 4.1 Environment setup

Use the Push-T environment from `gym-pusht` (`pip install gym-pusht`). Render at 96×96 RGB, upscale to 224×224 for V-JEPA 2 input.

Agent policy: **uniform random actions** for Tier A. No learned policy, no expert demonstrations. Random action sampling every 4 environment steps with a held constant action in between (gives smoother trajectories than per-step random).

Frame rate: 10 Hz effective (every 4th environment step saved to the experience stream).

### 4.2 Stage 0a — Single Config

- One configuration only (default Push-T colours and block).
- Run for 50,000 frames.
**Gates for advancing to 0b (targets for calibration, not hard commitments):**

The Push-T environment is simplified relative to real visual experience; these thresholds may need adjustment based on what Stage 0a actually produces. Treat them as starting targets, observe the loss curves and prediction quality during the run, and **propose refined thresholds in `HANDOFF.md` before any Stage 0b run begins**.

  - Next-step MSE loss < 0.01 (normalised by embedding dim) for the last 5000 steps.
  - Masked-position MSE loss < 0.01 at current mask ratio for the last 5000 steps.
  - Mask ratio has grown to at least 0.25.
  - Qualitative inspection of predicted embeddings: they are close (cosine > 0.9) to actual next-step embeddings on a held-out stretch of stream.

If these gates cannot be met within 100,000 frames regardless of threshold calibration, **STOP and report**. Do not advance.

### 4.3 Stage 0b — Multi-Config, Instant Scene Cuts

**Configurations (minimum 3, target 4):**
- Config 1: default background, default T-block colour, default lighting.
- Config 2: red background, default T-block, dimmer lighting.
- Config 3: grey background, blue T-block, default lighting.
- Config 4 (optional): green background, yellow T-block, bright lighting.

All configs use identical physics, task, and dynamics. Only visual factors change.

**Pre-gate:** Encode 500 random frames from each config using the Outward encoder. Compute mean pairwise cosine similarity between configs. **All pairs must have mean similarity < 0.8.** If not, redesign configs for greater visual distinctness and retest.

**Training stream construction:**
- Block length per config: 1000 frames (configurable, but keep fixed within a run).
- Transition: instantaneous scene cut (frame t-1 in config A, frame t in config B, no interpolation).
- Order: cycle through configs, randomly shuffling the order within each full cycle. Ensure every ordered pair (A→B) appears at least 20 times in the total stream.
- Total stream length: sufficient for every ordered config pair to co-occur 20+ times. For 4 configs, that's 12 ordered pairs × 20 = 240 transitions minimum, × 1000 frames per block = 240,000 frames.

**Evaluation at end of 0b:**
- Held-out probe set: 100 frames per config, drawn from the middle of config blocks (at least 500 frames from any transition).
- For each probe, ground-truth associates are frames from the temporally adjacent configs (within τ=5 of a transition involving this probe's config) that the probe co-occurred with during training. **τ is SCAFFOLDING** (see §9) — fixed at 5 for Tier A, intended to become an adaptive parameter of the full architecture. Log retrieval quality stratified by temporal distance (1, 2, 3, 4, 5, 6+ frames from transition) so that post-Tier-A work can design the adaptive mechanism from real data.
- **Headline metric: cross-boundary recall@10 where cosine similarity of probe to ground-truth associate is < 0.3.** This isolates the cross-boundary regime.
- **Shuffle control:** repeat training with the temporal order of frames shuffled within the full stream (configs still identifiable, but temporal structure destroyed). Cross-boundary recall should collapse.
- **Baselines:** pure cosine kNN on the same probes. Should score near zero on cross-boundary by construction.

**Gates for advancing to 0c (targets for calibration, not hard commitments):**

Push-T's simplified visual structure may shift what "good" looks like in cross-boundary recall. Observe what Stage 0b actually produces across seeds and encoder arms, then **propose refined thresholds in `HANDOFF.md` before any Stage 0c run begins**. Shuffle-control and baseline thresholds are firmer because they are diagnostic rather than performance-driven.

  - Cross-boundary recall@10 ≥ 0.3 (predictor retrieves genuine temporal associates for ≥30% of cross-boundary probes). Target for calibration.
  - Shuffle control cross-boundary recall@10 ≤ 0.05 (proves the mechanism learned temporal structure, not geometry). Diagnostic, firm.
  - Within-config recall@10 ≥ 0.7 (sanity check that basic retrieval still works). Target for calibration.
  - Pure cosine baseline on cross-boundary probes < 0.05 (proves the probes are genuinely cross-boundary). Diagnostic, firm.

If performance gates cannot be met regardless of threshold calibration (e.g., cross-boundary recall stays near zero across all configurations and encoder arms), **STOP and report** with full diagnostic breakdown (per-config-pair recall, embedding geometry inspection, predictor confidence histograms). Do not advance.

### 4.4 Stage 0c — Multi-Config, Smooth Morph Transitions

Same configs as 0b. Transition mechanism changes from instant cut to smooth morph.

**Morph specification:**
- Transition length: 200 frames (configurable, range 100–300).
- Visual factors interpolated linearly over the transition:
  - Background colour: linear RGB interpolation from config A to config B.
  - T-block colour: linear RGB interpolation.
  - Lighting: linear interpolation of light intensity parameter.
- Agent continues acting throughout the morph (same random policy). Physics do not change during the morph — only rendering.
- **Transition-zone flag:** frames during a morph are tagged as transition-zone in the memory bank metadata.

**Training stream construction:**
- Same config durations and cycling order as 0b.
- Each transition is now 200 frames of morph instead of an instant cut.
- Total stream length adjusted to maintain same number of config-pair co-occurrences as 0b (≈ 280,000 frames with 200-frame morphs replacing instant cuts).

**Evaluation at end of 0c:**
- Same probe set construction as 0b (middle-of-config frames, ≥500 frames from any transition zone).
- Same headline metric: cross-boundary recall@10 where cosine < 0.3.
- **New diagnostic — chain-traversal test:** measure whether PAM can retrieve associates multiple hops away through the morph chain. For each probe in config A, do 3-hop retrieval and measure whether the final hop lands in config B (the temporally adjacent config) rather than staying in config A.
- **New diagnostic — generalisation probe:** withhold one config pair (e.g., A→C transitions never seen in training). At evaluation, test retrieval from A to C. If the mechanism is pair-memorisation, this should fail. If it's learned traversal through the chain via intermediate configs (A→B→C experienced), it should partially succeed.

**Gates for Tier A completion (targets for calibration, not hard commitments):**

Use the Stage 0b results as the anchor for refining these. Commit refined thresholds in `HANDOFF.md` before running final Stage 0c evaluation.

- Cross-boundary recall@10 in 0c ≥ 0.4 (higher than 0b target because smooth transitions provide more training signal via chain structure). Target for calibration.
- Shuffle control cross-boundary recall@10 ≤ 0.05. Diagnostic, firm.
- Chain-traversal test: 3-hop retrieval from config A probes lands in config B for ≥0.4 of probes. Target for calibration.
- Generalisation probe: retrieval to withheld config pair ≥0.15 (genuine generalisation is weaker than trained; anything meaningfully above shuffle-control baseline is signal). Target for calibration.

If gates pass, Tier A is complete. Report. **Do not proceed to any Tier B work without explicit instruction.**

If gates fail after threshold calibration, **STOP and report**. The failure mode is diagnostic information — do not attempt fixes without instruction.

---

## 5. Ablations Required Within Tier A

Run all of these as part of Tier A. They are not optional.

### 5.1 SIGReg vs Stop-Grad + EMA (Outward encoder)
- From Stage 0b onwards, maintain two encoder arms: frozen V-JEPA 2 (primary) and ViT-Tiny trained with SIGReg.
- Also train ViT-Tiny with classic stop-gradient + EMA (no SIGReg) as the third arm.
- Run full Stage 0b and 0c evaluation with each encoder. Report cross-boundary recall, embedding geometry metrics (isotropy, effective dimensionality), and training stability curves.
- **SIGReg removal trigger:** if SIGReg ViT-Tiny underperforms the stop-gradient ViT-Tiny on cross-boundary recall by >10% across three seeds, revert and do not adopt SIGReg for future stages. Report the removal decision.

### 5.2 Shuffle Control (temporal structure validation)
- Run for every stage's evaluation. Required, not optional.

### 5.3 Pure Cosine kNN Baseline
- Run for every stage's evaluation. Required, not optional.

### 5.4 Mask Ratio Sensitivity
- At end of Stage 0c, run evaluation with mask ratios {0.125, 0.25, 0.375, 0.5, 0.625} using the trained predictor (no retraining). Report cross-boundary recall at each.
- This diagnostic is cheap (no training, just evaluation at different masking during probe construction) and tells us whether the predictor is operating in the interpolation regime or the trajectory regime at test time.

---

## 6. Evaluation & Logging Protocol

### 6.1 Required metrics per stage
- Next-step MSE loss (training, held-out)
- Masked-position MSE loss (training, held-out) per mask ratio bin
- Cross-boundary recall@1, @5, @10 (for Stage 0b onwards)
- Within-config recall@1, @5, @10 (for Stage 0b onwards)
- Shuffle control equivalents of the above (required)
- Pure cosine kNN baseline equivalents (required)
- Predictor embedding norm statistics (detect collapse)
- Encoder embedding isotropy metrics (for SIGReg ablation)

### 6.2 Logging infrastructure
- Use TensorBoard for real-time curves.
- Dump per-probe retrieval results to JSON every evaluation checkpoint (probe frame index, ground-truth associate indices, retrieved indices, scores, cosine to retrieved, cross-boundary flag).
- **Every number that appears in a summary must correspond to a line in a JSON file on disk.** No exceptions.

### 6.3 Checkpointing
- Save predictor weights every 10,000 training steps.
- Save memory bank snapshot at end of each stage.
- Tag checkpoints with git commit hash in filename.

---

## 7. Repository Layout

```
continuous_trajectory_pam/
├── .venv/
├── requirements.txt
├── README.md
├── HANDOFF.md
├── PAM_Tiered_v0_Spec.md  # canonical spec (copy from source)
├── configs/
│   ├── stage_0a.yaml
│   ├── stage_0b.yaml
│   └── stage_0c.yaml
├── src/
│   ├── encoders/
│   │   ├── frozen_vjepa2.py
│   │   └── sigreg_vit_tiny.py
│   ├── predictor/
│   │   └── trajectory_predictor.py
│   ├── memory/
│   │   └── memory_bank.py
│   ├── env/
│   │   └── push_t_staged.py
│   ├── training/
│   │   └── online_loop.py
│   ├── evaluation/
│   │   ├── probe_construction.py
│   │   ├── cross_boundary.py
│   │   └── shuffle_control.py
│   └── utils/
├── scripts/
│   ├── run_stage_0a.sh
│   ├── run_stage_0b.sh
│   └── run_stage_0c.sh
├── results/
│   ├── stage_0a/
│   ├── stage_0b/
│   └── stage_0c/
└── checkpoints/
```

Git commit the spec, structure, and `requirements.txt` as the first commit before writing implementation code.

---

## 8. Execution Order

1. Set up repository structure and environment (`requirements.txt`, `.venv`, `HANDOFF.md` skeleton).
2. Implement and smoke-test the frozen V-JEPA 2 encoder path. Verify it produces sensible embeddings on 10 random frames.
3. Implement the Push-T environment wrapper (single config for now) with the random action policy and frame extraction.
4. Implement the trajectory predictor transformer. Unit test the forward pass with random inputs of correct shape.
5. Implement the memory bank with FAISS index. Unit test insert + retrieve.
6. Implement the online training loop. Run a 1000-step dry run to verify the end-to-end pipeline works. Commit.
7. **Run Stage 0a full training.** Log everything. Update `HANDOFF.md`. Commit.
8. Evaluate Stage 0a gates. **STOP if gates fail.** Report.
9. If Stage 0a passes, implement multi-config Push-T wrapper and config pre-gate (mean cosine similarity check). Commit.
10. Implement probe construction and cross-boundary evaluation. Unit test on synthetic data. Commit.
11. **Run Stage 0b full training.** Log everything. Update `HANDOFF.md`. Commit.
12. Evaluate Stage 0b gates. **STOP if gates fail.** Report.
13. If Stage 0b passes, implement SIGReg ViT-Tiny encoder and add as ablation arm. Commit.
14. Run Stage 0b ablation with all three encoder arms. Report.
15. Implement smooth morph transition wrapper for Push-T. Commit.
16. **Run Stage 0c full training.** Log everything. Update `HANDOFF.md`. Commit.
17. Evaluate Stage 0c gates. **STOP if gates fail.** Report.
18. Implement chain-traversal and generalisation probe diagnostics. Run on trained Stage 0c checkpoints.
19. Write Tier A final report: all metrics, all ablations, all gate outcomes, recommendations for Tier B triggers.

**Phase gate:** After Step 8, after Step 12, and after Step 17, stop completely and wait for instruction before continuing.

---

## 9. Decision Points Flagged for Documentation

These are design choices I'll make during implementation unless overridden. Every one gets documented in `HANDOFF.md` with my rationale:

- Specific V-JEPA 2 checkpoint version used (there are several releases)
- Random seed for action policy and random initialisations
- Exact FAISS index type (default: `IndexFlatIP` for exactness; switch to `IndexIVFFlat` only if latency becomes prohibitive)
- Epps-Pulley implementation details (use LeWM's reference if available; else implement from Cramér-Wold specification)
- BatchNorm projection head dimension for SIGReg arm (default: keep at encoder hidden dim)
- **Numerical gate thresholds** are starting targets, not hard commitments. Observe what each stage actually produces, propose refinements in `HANDOFF.md` at the end of each stage, and lock refined thresholds before running the next stage. The shuffle-control and baseline thresholds (≤ 0.05) are firm diagnostics and should not be relaxed; the performance thresholds (≥ 0.3, ≥ 0.4, ≥ 0.7, ≥ 0.15) are calibration targets that depend on what Push-T's simplified structure actually supports.
- **τ (temporal co-occurrence window), SCAFFOLDING.** Starting value τ=5 frames (0.5 seconds at 10 Hz effective frame rate). This is a fixed value for Tier A, but it is architecturally scaffolding standing in for an eventually-adaptive parameter: in the full architecture, the association window should expand or contract based on prediction error and experience density (narrow around sudden events, wide during routine exploration). For Tier A, observe Stage 0a retrieval behaviour and propose refinement in `HANDOFF.md` if the fixed value is producing obvious failure modes (too short: associations don't span meaningful transitions; too long: associations dilute into noise). Log retrieval quality stratified by temporal distance so that post-Tier-A work can design the adaptive mechanism from real data rather than speculation.

---

## 10. What to Do If You Get Stuck

If you hit any of the following conditions, **stop and report** rather than improvising:

- A gate fails and the spec's proposed investigation does not obviously apply.
- You believe a Tier B or Tier C item is required to proceed.
- An assumption in this document is contradicted by implementation reality (e.g., V-JEPA 2 checkpoint is unavailable, gym-pusht API has changed).
- You identify a cleaner approach than what's specified. Do not implement it silently.
- Training produces NaN losses or complete collapse (all embeddings identical).
- FAISS retrieval returns the probe itself as the top hit more often than expected (likely an indexing bug).

Report format: the issue, the evidence (file paths + line numbers + metric values), the options you see, and your recommendation. Do not proceed without instruction.

---

**End of instructions.**
