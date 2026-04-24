# Session Batch Instructions — Sessions 2 through 5

**Purpose:** Execute the implementation phases for Tier A autonomously. Build the full training pipeline through the end of Session 5, run the pre-flight smoke test, then stop for human review before launching Stage 0a.

**Scope:** Sessions 2, 3, 4, 5 only. Do not start Stage 0a training under any circumstances. Do not begin Round 2 planning. Stop at the pre-flight smoke test.

**Authority hierarchy applies:** `CODING_STANDARDS.md` governs operational discipline. `pam_tier_a_grok_instructions.md` is the implementation specification. `PAM_Tiered_v0_Spec.md` is the canonical research spec. This document coordinates the batch — it does not override the others.

---

## 0. Read First

Before starting any work, read in this order:

1. `CODING_STANDARDS.md` — operational discipline.
2. `PAM_Tiered_v0_Spec.md` — the canonical research spec.
3. `pam_tier_a_grok_instructions.md` — implementation specification.
4. `STAGE_GATES.md` — gate evaluation criteria.
5. `AUTONOMOUS_PROGRESSION.md` — when autonomous progression is allowed (note: not inside this batch).
6. `HANDOFF.md` — current state from Session 1.
7. This document.

If any of these are missing or their content conflicts with this batch plan, stop and report.

---

## 1. Scope Lock — Absolute

These rules apply across all four sessions. Violations mean stop and report, not silent proceed.

- **Tier A only.** Tier B and Tier C items are not implemented regardless of how natural the extension seems. This includes LTI-recurrent predictor structure, MLA compression, hybrid retrieval scoring, LoRA adapters, ACT halting, surprise-based retention, and meta-networks for signal trust.
- **No ZWM mechanisms.** Asymmetric masking, temporally-factored prediction, zero-shot perturbation prompting, developmental ordering, and any other ZWM-derived ideas are out of scope for this batch. They may surface as Round 2 arm candidates after Stage 0a completes.
- **No architectural improvisation.** If the spec is ambiguous on a specific implementation detail, document the ambiguity in HANDOFF.md, make the simplest reasonable choice, and flag it for review. Do not invent mechanisms to work around perceived problems.
- **No retry loops on failure.** If a session fails (crash, cannot complete), stop and report. Do not rerun with adjusted parameters.
- **Encoder: frozen V-JEPA 2 only for this batch.** Do not implement the SIGReg ViT-Tiny arm. It enters the pipeline at Stage 0b, not now.

---

## 2. Session Plan

Four sessions, sequential, with commit and HANDOFF update at each boundary. Each session is self-contained and produces a working, tested component.

### Session 2 — Outward Encoder Wrapper

**Goal:** Produce `src/encoders/frozen_vjepa2.py` that wraps the V-JEPA 2 HF model and provides a clean `encode_frame(frame) -> (B, D)` API.

**Implementation requirements:**
- Load V-JEPA 2 ViT-Large from the checkpoint identified in Session 1 bootstrap (`facebook/vjepa2-vitl-fpc64-256`, per HANDOFF.md).
- Freeze all parameters. Verify via parameter count: `sum(p.numel() for p in model.parameters() if p.requires_grad) == 0`.
- Accept input frames of shape `(C, H, W)` or `(B, C, H, W)`. Normalise to batched form internally.
- Handle the `pixel_values_videos` shape requirement by unsqueezing a T=1 dimension. The unsqueeze must be explicit and commented: this is a deliberate choice per Session 1 Open Decision #3, not an accident.
- Return CLS token from the final layer, shape `(B, D)` where D=1024.
- All forward passes under `torch.no_grad()` (encoder is frozen).

**Required comment at the unsqueeze site (verbatim):**
```python
# NOTE: T=1 chosen for Tier A to avoid cross-boundary blending in Stage 0b.
# V-JEPA 2 expects pixel_values_videos with shape (B, T, C, H, W).
# Consider T>1 sliding clip for Stage 0c smooth morphs (see HANDOFF.md).
videos = frame.unsqueeze(1)  # (B, T=1, C, H, W)
```

**Tests (required):**
- `tests/test_frozen_vjepa2.py` covering:
  - Forward pass produces expected output shape.
  - Parameter count confirms freeze (no trainable params).
  - Deterministic output for identical input (no dropout active).
  - Handles batched input (B > 1) and single-frame input (B=1) equivalently.
  - Rejects wrong-shape input with clear error.

**Commit message:** `feat(encoder): frozen V-JEPA 2 wrapper with T=1 video encoding`

**HANDOFF update:** Record the file path, API signature, any design decisions, and the git commit hash. Mark Session 2 complete.

### Session 3 — Inward Predictor

**Goal:** Produce `src/predictor/trajectory_predictor.py` that implements the transformer trajectory predictor per spec §3.2 of the Tier A instructions.

**Implementation requirements:**
- `nn.TransformerEncoder` with 4 layers, 8 heads, hidden dim 512, MLP dim 2048, GELU, pre-LayerNorm.
- Window size W = 16 (configurable but default 16).
- Frame embedding projection: linear from input dim (1024) to hidden dim (512).
- Learnable temporal position embeddings: W+1 positions (0..W-1 for context, W for query).
- Learnable mask token: single shared vector, projected to hidden dim.
- Query position: always at index W, always receives the mask token (no frame to mask).
- Output projection: linear from hidden dim (512) back to original embedding dim (1024).
- Final LayerNorm after output projection.

**Forward signature:**
```python
def forward(
    self,
    context: Tensor,  # (B, W, D) frame embeddings
    mask_positions: LongTensor,  # (B, K) indices in [0, W) to mask
) -> dict:
    """
    Returns:
    {
        "predicted_next": Tensor,  # (B, D) embedding at query position W
        "predicted_masked": Tensor,  # (B, K, D) embeddings at masked positions
    }
    """
```

**Tests (required):**
- `tests/test_trajectory_predictor.py` covering:
  - Forward pass produces expected output shapes for W=16, D=1024, K=4.
  - Handles K=0 (no masked positions in window, only query prediction).
  - Handles K=W-1 (all context positions masked except one).
  - Parameter count is reasonable (log the total; should be in the ~5–10M range).
  - Gradient flow works: backward from a dummy loss produces non-zero grads on all parameters except the input (stop-grad not applied here — that's handled in the training loop).
  - Output embedding dim matches input embedding dim (round-trip through the network).

**Commit message:** `feat(predictor): trajectory transformer for masked position prediction`

**HANDOFF update:** Record file path, parameter count, API signature, and commit hash.

### Session 4 — Memory Bank and Training Loop

**Goal:** Produce `src/memory/memory_bank.py` and `src/training/online_loop.py` that together implement the online training pipeline per spec §3.3 and §3.4.

**Memory bank requirements:**
- `src/memory/memory_bank.py` implementing:
  - `append(embedding, metadata)` — stores a frame embedding and its metadata (frame_idx, stage, config, transition_zone_flag).
  - `get_window(start_idx, window_size)` — retrieves a contiguous window of embeddings.
  - `rebuild_index()` — rebuilds the FAISS index after new additions.
  - `retrieve(query_embedding, k)` — returns top-k nearest neighbours with their metadata.
  - L2 normalisation applied before FAISS indexing. Explicit, tested, commented.
  - FAISS index type: `IndexFlatIP`. Do not switch to approximate indices in this batch.
  - Index rebuild frequency: every 1000 appends (configurable).
- Backing store: numpy arrays pre-allocated to a configurable max size (default 200k frames). Grow only if required; do not shrink.

**Training loop requirements:**
- `src/training/online_loop.py` implementing:
  - One training step per new frame arrival (online, single-pass).
  - Ring buffer of recent W frames for context window construction.
  - Masking schedule per spec §3.4: starts at `mask_ratio = 1/W`, grows on plateau trigger (no improvement > 5% over 10k steps), capped by stage config.
  - Loss: MSE on next-step prediction + MSE on masked positions within window.
  - **Stop-gradient on all targets.** Verify with an explicit assertion in the training step that `target.requires_grad == False`.
  - AdamW optimizer, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95), cosine warmup over first 5000 steps.
  - Gradient norm tracking. Log max, median, std every 100 steps.
  - Checkpoint save every 10,000 steps.
  - TensorBoard logging for all scalar metrics.
  - JSON dump of per-checkpoint statistics (loss distributions, mask ratio, gradient stats, embedding norms) to `results/<stage>/checkpoint_<step>.json`.

**Stop-gradient verification (required — this has been a failure mode in prior work):**

At the start of every training step, before computing loss, explicit assertion that targets have no grad:
```python
assert not target_next.requires_grad, "Stop-gradient not applied to next-step target"
assert not target_masked.requires_grad, "Stop-gradient not applied to masked-position targets"
```

Targets must be constructed via `.detach()` or equivalent, not by accident. This assertion must not be removed under any circumstance.

**Tests (required):**
- `tests/test_memory_bank.py`:
  - Append + retrieve round-trip on 1000 random embeddings.
  - L2 normalisation verified (retrieved embeddings have unit norm).
  - Metadata preserved through retrieval.
  - Index rebuild produces consistent results.
  - Retrieval of probe itself returns probe as top hit with score ≈ 1.0 (sanity check — FAISS correctness).

- `tests/test_online_loop.py`:
  - 100-step dry run with a dummy encoder (random embeddings) completes without error.
  - Stop-gradient assertions fire correctly when deliberately violated.
  - Checkpoint save/load round-trip preserves model state.
  - Gradient norms are finite (no NaN/Inf) across the dry run.
  - Mask ratio plateau detection triggers correctly on a constructed loss history.

**Commit messages:**
- `feat(memory): append-only memory bank with FAISS IndexFlatIP`
- `feat(training): online single-pass training loop with masked trajectory prediction`

**HANDOFF update:** Record both file paths, API signatures, test results, and commit hashes.

### Session 5 — Environment Wrapper, Curriculum, and Pre-Flight Smoke Test

**Goal:** Produce `src/env/push_t_staged.py`, the Stage 0a config, and a pre-flight smoke test that exercises the complete pipeline end-to-end.

**Environment wrapper requirements:**
- `src/env/push_t_staged.py` providing:
  - `PushTStagedEnv(config: str)` — wraps `gym-pusht` with staged behaviour.
  - Stage 0a: single default configuration, no visual variation.
  - Random action policy: uniformly sample from the action space every 4 environment steps, hold constant between.
  - Frame extraction: `next_frame() -> ndarray` returns RGB at the current timestep, upscaled from env render (96×96) to 224×224 for V-JEPA 2 input.
  - Effective frame rate: 10Hz (save every 4th env step).

**Stage 0a config:**
- `configs/stage_0a.yaml` with all hyperparameters: window size, mask ratio schedule, optimizer, lr, checkpoint frequency, total frames (50k for Stage 0a), log paths, seed.
- Seed: a fixed integer. Document it in HANDOFF.md.

**Pre-flight smoke test:**

Purpose: verify the full pipeline works end-to-end on a tiny budget before committing to the full Stage 0a run. This is a pipeline validator, not a gate evaluator.

- `scripts/preflight_smoke_test.py` running:
  - 1000 frames of Stage 0a configuration (approximately 2–3 minutes of wall-clock on the 4080 Super).
  - Full pipeline: env → encoder → memory bank → predictor → loss → optimizer step.
  - Real components, not mocks. This is the first integration test.

**Pre-flight pass criteria (all must hold):**
- No NaN or Inf in any logged metric across the 1000 steps.
- Next-step MSE loss at step 1000 is strictly lower than at step 100 (training is making progress).
- Memory bank contains the expected number of entries (within 5% of expected).
- Checkpoint save completed successfully at least once.
- Stop-gradient assertions did not fire.
- Gradient norm median is finite and in a reasonable range (log whatever it is, flag if > 100 or < 1e-6).
- Predicted embedding norms are within 0.5× to 2× of encoder embedding norms (no collapse, no explosion).
- FAISS index build and retrieve operations complete without error.

**Pre-flight output:**
- Writes `results/preflight/smoke_report.json` with all metric values.
- Writes `results/preflight/SMOKE_REPORT.md` with human-readable summary.
- Updates HANDOFF.md with pre-flight outcome.

**If pre-flight passes:** Stop. Commit everything. Update HANDOFF.md with "Pre-flight passed; ready for Stage 0a launch on explicit instruction." Wait.

**If pre-flight fails:** Stop. Commit everything (including the failure artifacts). Write `results/preflight/FAILURE_REPORT.md` with the specific failure mode, the evidence, and any diagnostic information. Update HANDOFF.md with the failure. Wait.

**Tests (required):**
- `tests/test_push_t_staged.py`:
  - Environment initialises and produces a frame.
  - Random action policy produces valid actions.
  - Frame rate subsampling behaves correctly (every 4th step).
  - Frame output has shape (224, 224, 3) and dtype uint8 in [0, 255].

**Commit messages:**
- `feat(env): Push-T staged environment wrapper, Stage 0a config`
- `test(preflight): end-to-end smoke test for full pipeline`

**HANDOFF update:** Record environment wrapper path, Stage 0a config path, smoke test path, smoke test outcome, and commit hashes. Clearly indicate that the pipeline is ready for Stage 0a launch but has not been launched.

---

## 3. Between-Session Protocol

At the boundary between any two sessions in this batch:

1. Commit all changes from the completed session.
2. Update `HANDOFF.md` with session outcome, decisions made, next action.
3. Run `git status` — working tree must be clean.
4. Start the next session by re-reading `HANDOFF.md` and the spec sections relevant to the next session.

If context is becoming heavy within a session, summarise state to HANDOFF.md and start a fresh session. Do not let context compaction happen mid-implementation — implementation details are where fabricated numbers appear.

---

## 3A. Orchestration and Context Economy

The four sessions in this batch span substantial implementation work across multiple context windows. Orchestration discipline prevents the main context from bloating with reference material, test output, and exploratory reads that belong elsewhere.

### 3A.1 Session boundaries are commit points, not context resets

Each numbered session (2, 3, 4, 5) ends with a commit, a HANDOFF update, and a clean re-read of the HANDOFF before starting the next session. This can happen within a single CC context — you do not need to end the context and start a new one at every session boundary.

What must happen at every session boundary:
- All work committed with a descriptive message.
- HANDOFF.md updated with session outcome, decisions made, next action.
- Git working tree clean.
- Before starting the next session's work, re-read HANDOFF.md and the relevant spec section for that session, even though they may still be in context. This treats the HANDOFF as the source of truth, which keeps the pattern consistent with fresh-context starts.

When you should end the context and wait for a new one:
- A context budget trigger in §3A.3 fires.
- A stop condition in §4 fires.
- The batch is complete (end of Session 5 with pre-flight passed).

The goal is continuous execution across all four sessions in a single CC run, with sub-agent delegation and disciplined commit hygiene keeping main context clean. Stopping mid-batch should happen only when context quality is genuinely threatened, not as a routine ceremony.

### 3A.2 Sub-agent delegation is default for bounded tasks

Spawn a sub-agent (not continue in main context) for:

- Running test suites and reporting pass/fail summaries. The sub-agent runs the tests, reads the output, and returns a terse result. Test output stays out of main context.
- File searches across the repo (grep, find). The sub-agent does the search and returns matches.
- Reading documentation pages, changelogs, or HF model cards. The sub-agent reads and returns a summary of the specific information needed.
- Verifying environment state (dependency versions, disk usage, GPU availability). The sub-agent runs the checks and returns a status line.

Rule of thumb: if a task consumes context but the detailed output isn't needed for downstream reasoning, delegate it. The main context keeps the information it needs (the summary) without the cost (the raw output).

Do not delegate actual implementation work to sub-agents. Implementation requires accumulated context on the spec and the codebase; sub-agents start fresh and will produce inconsistent code.

### 3A.3 Context budget triggers

Stop-and-handoff triggers within a session, even before the session is formally complete:

- You have read three or more large files (>500 lines each) in this context.
- You have been working in this context for more than approximately 2 hours of active work.
- You are about to read the full implementation instructions document for the second time in the same context.
- You have generated more than two significant implementation artifacts (modules + tests) without committing.
- The conversation length is making it harder to recall earlier decisions accurately.

When any of these trigger: stop, summarise current state to HANDOFF.md, commit work-in-progress to a branch if incomplete (or complete and commit to main if possible), and end the context.

### 3A.4 What stays in main context vs. what gets delegated

Main context should hold:
- The current session's implementation work.
- The relevant section of the spec for the current session.
- HANDOFF.md state from the previous session.
- Test scaffolding as it's being written.

Main context should NOT hold (delegate or discard):
- Full test output from prior test runs (keep pass/fail summary only).
- Reference documentation read for a single decision (keep the decision, discard the source).
- Exploratory code that didn't land in the final implementation.
- Multiple versions of the same file during iteration (git handles this).

### 3A.5 Pre-flight smoke test is a distinct phase

The pre-flight smoke test at the end of Session 5 is a meaningful context load: it runs the full pipeline, generates real outputs, produces a report. Treat it as a distinct phase with a clean re-read of HANDOFF before execution.

If Session 5 has been heavy with environment wrapper implementation, commit the wrapper work, update HANDOFF, then re-read HANDOFF and proceed to the smoke test as a separate logical phase within the same context. If context budget triggers fire (§3A.3), end the context and let the smoke test run in a fresh invocation — but this is the exception, not the default.

The structure is: 5a (env wrapper + smoke test script, commit) → HANDOFF update → 5b (run smoke test, evaluate, write report, commit). These can live in the same CC context if context is clean; they should be separated into two invocations if context is heavy.

### 3A.6 Do not run parallel CC instances

One active CC context per working tree at a time. Concurrent edits produce git conflicts that are expensive to untangle. If a long task is running (e.g., a test suite), wait for it rather than starting parallel work in a second context.

---

## 4. Stop Conditions During the Batch

Stop and report if any of the following occur:

- Any test fails and the cause is not immediately obvious.
- Implementation surprise: spec contradicts reality (V-JEPA 2 API different, dependency issue, hardware problem).
- Unable to complete a session within a reasonable context budget.
- Any Tier B/C mechanism would be natural to add but is out of scope — flag, don't implement.
- Pre-flight smoke test fails.
- Any assertion from `CODING_STANDARDS.md` is tested and found violated.
- Disk usage exceeds 80% of quota.
- You are about to make an implementation decision that the spec does not clearly cover and you cannot defer it with a scaffolding flag.

Stop format: see `AUTONOMOUS_PROGRESSION.md` §2.

---

## 5. What Success Looks Like

At the end of this batch, the state is:

- Four commits adding the four components (encoder, predictor, memory+training, env+smoke test).
- All tests green.
- Pre-flight smoke test passed.
- HANDOFF.md reflects "ready for Stage 0a launch" with full context on what was built, what decisions were made, and any flagged items.
- Clean git tree.
- No Stage 0a training has been launched.

The next action after this batch is a human review of the pre-flight report and an explicit instruction to launch Stage 0a. That launch is not part of this batch.

---

## 6. What This Batch Does NOT Do

Explicit list, for clarity:

- Does not launch Stage 0a training.
- Does not implement the SIGReg ViT-Tiny encoder arm.
- Does not implement any Tier B or Tier C mechanism.
- Does not design Round 2 arms.
- Does not evaluate Stage 0a gates (nothing to evaluate until Stage 0a runs).
- Does not modify `PAM_Tiered_v0_Spec.md`, `pam_tier_a_grok_instructions.md`, `CODING_STANDARDS.md`, `STAGE_GATES.md`, or `AUTONOMOUS_PROGRESSION.md`.
- Does not adjust gate thresholds.

---

*End of batch instructions.*
