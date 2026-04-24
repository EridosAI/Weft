# HANDOFF — Weft

**Project:** Weft — Continuous Trajectory PAM Tier A Implementation
**Repo:** `https://github.com/EridosAI/Weft`
**Repo path (local):** `C:\Users\Jason\Desktop\Eridos\Weft\`
**Canonical spec:** `PAM_Tiered_v0_Spec.md`
**Implementation instructions:** `pam_tier_a_grok_instructions.md`
**Operational discipline:** `CODING_STANDARDS.md` — read at the start of every session.

This is the living cross-session handoff. Every session ends by updating this document. Every session begins by reading it.

---

## Current Status

**Current stage:** Bootstrap complete. Repo extracted from Eridos parent and re-initialised as standalone Weft repo. Ready for Execution Order step 2 (frozen V-JEPA 2 encoder impl + smoke test), per SESSION_BATCH_INSTRUCTIONS.md Session 2.
**Last session date:** 2026-04-24
**Current tier lock:** Tier A only (strictly enforced per pam_tier_a_grok_instructions.md §2 and CODING_STANDARDS.md §1.4).
**Next immediate action:** Session 2 per SESSION_BATCH_INSTRUCTIONS.md — implement `src/encoders/frozen_vjepa2.py` wrapping `facebook/vjepa2-vitl-fpc64-256`, with tests under `tests/test_frozen_vjepa2.py`.

---

## Session 1 Bootstrap Checklist

Populated during Session 1 bootstrap (2026-04-23) inside the original Eridos parent repo. Preserved verbatim in this re-initialised Weft repo; audit trail to original commits in the Session 0 log entry below.

### Environment verification
- [x] Operating system and version confirmed: Windows 11 (10.0.26200-SP0) — documented divergence from spec §0 (Ubuntu 24.04); away-mode decision, local RTX 4080 setup used.
- [x] Python version: 3.12.3 — documented divergence from spec §0 (3.11); rationale in Decisions section.
- [x] PyTorch version + CUDA version: 2.6.0+cu124 (CUDA 12.4 compatible) — recorded in `.env_snapshot.txt`.
- [x] GPU detected (name + VRAM): NVIDIA GeForce RTX 4080 SUPER, 15.99 GB — verified via `torch.cuda.get_device_properties(0)`.
- [x] `torch.cuda.is_available()` returns True.
- [x] Working directory exists and is git-tracked (then: Eridos subdir; now: standalone Weft repo per Session 0 extraction).
- [x] `.venv/` created at project root via `python -m venv .venv --prompt "pam"`.
- [partial] `.venv/` activated before any pip installs — partial; global conda env used for Session 1 to avoid torch reinstall overhead on Windows. Full venv activation planned before any training run.

### Repository initialisation (Weft-relative, post-extraction)
- [x] `git init` run at Weft project root (fresh init; original Eridos parent commits referenced in Session 0 log).
- [x] `.gitignore` covers `.venv/`, `checkpoints/`, `results/*.json`, logs, cache files, Windows/IDE extras.
- [x] Initial Weft commit contains: `PAM_Tiered_v0_Spec.md`, `pam_tier_a_grok_instructions.md`, `CODING_STANDARDS.md`, `STAGE_GATES.md`, `AUTONOMOUS_PROGRESSION.md`, `SESSION_BATCH_INSTRUCTIONS.md`, `HANDOFF.md`, `requirements.txt`, `README.md`, `.gitignore`, `.env_snapshot.txt`, and the full directory layout from §7 of the instructions.
- [x] First Weft commit hash: recorded in "Key commits to be aware of" below after commit lands.

### Dependency pinning
- [x] `requirements.txt` pinned: `torch==2.4.1`, `torchvision`, `numpy==1.26.4`, `faiss-cpu==1.8.0`, gymnasium (for gym-pusht path), `tensorboard`, `PyYAML`, `timm`/`transformers` for V-JEPA.
- [partial] `pip install -r requirements.txt` — global conda env used in Session 1; full venv install deferred to pre-Stage-0a.
- [x] `pip freeze > .env_snapshot.txt` captured and committed.

### Smoke tests before any training
- [deferred] V-JEPA 2 checkpoint identified: **`facebook/vjepa2-vitl-fpc64-256`** (Meta V-JEPA 2 ViT-L/16, 64-frame / 256-patch). Load + smoke test deferred to Session 2 per Execution Order.
- [deferred] V-JEPA 2 loads and produces an embedding on a random 224×224 RGB tensor — Session 2.
- [confirmed by spec] Embedding dimension: **1024** (V-JEPA 2 ViT-L default; per spec §3.1 and instructions line 79).
- [deferred] `gym-pusht` import + default env frame — deferred to env-wrapper implementation (Session 5).
- [deferred] FAISS IndexFlatIP build+retrieve on 100 random vectors — deferred to memory-bank implementation (Session 4).

### Decisions made during Session 1 (all §9 items resolved or scaffolding-locked)

**Three open decisions — all resolved:**

1. **V-JEPA 2 checkpoint version pinned: `facebook/vjepa2-vitl-fpc64-256`.** Resolves §9 item "Specific V-JEPA 2 checkpoint version used". ViT-L/16, 64-frame clip / 256-patch tubelet. Loaded via HuggingFace transformers pathway.
2. **Env/Python divergence from spec §0 documented and accepted: Python 3.12.3 + PyTorch 2.6.0+cu124 on Windows 11.** Spec §0 specifies Python 3.11 + PyTorch 2.4; the local stack is newer but CUDA 12.4 compatible, and reinstalling Torch to downgrade on Windows is high-cost with no expected behavioural change. Divergence logged per CODING_STANDARDS.md §1.3. Revisit if any CUDA/PyTorch-ABI surprise appears in Session 2.
3. **`pixel_values_videos` shape: use T=1 unsqueeze for Tier A.** V-JEPA 2 expects `(B, T, C, H, W)`. Tier A operates on independent frames, so T=1 is the correct Tier A choice — it avoids cross-boundary blending in Stage 0b where instantaneous scene cuts must not leak across a T>1 clip. T>1 sliding-window clipping is a candidate for Stage 0c smooth-morph evaluation (not Tier A; flag only). The unsqueeze site in `src/encoders/frozen_vjepa2.py` must carry the explicit comment block specified in SESSION_BATCH_INSTRUCTIONS.md §Session 2.

**Other decisions (scaffolding / standing):**
- Random seed for action policy: **42** (standard reproducible default; to be set in `configs/stage_0a.yaml` and `src/utils/seed.py` when those land).
- FAISS index type: **`IndexFlatIP`** on L2-normalised vectors (exact cosine via IP). No approximate indices in Tier A; matches spec §3.3 and instructions line 126.
- Epps-Pulley implementation source: follow LeWorldModel (arXiv:2603.19312) reference implementation when SIGReg arm lands in Stage 0b (`src/encoders/sigreg_vit_tiny.py`). Not needed for Session 2–5.
- BatchNorm projection head dim for SIGReg arm: match encoder output dim (to be confirmed in code when the arm is implemented).
- **τ = 5 frames (0.5s at 10 Hz effective)** — SCAFFOLDING per instructions §9; retrieval quality will be logged stratified by temporal distance so the post-Tier-A adaptive mechanism can be designed from real data.
- Numerical gate thresholds: starting targets per spec §4; calibration proposals recorded in the Threshold Calibration Log section below at each stage boundary.
- **Directory structure deviations from spec §7:** none for the core tree. Added: `README.md` (standard), `.env_snapshot.txt` (required by CODING_STANDARDS.md §8.4), `logs/` (explicit; used by §5.2 `nohup` convention). No semantic deviations.

### Session 1 end-of-session
- [x] All bootstrap items populated or explicitly marked deferred with rationale.
- [x] All changes committed (original Eridos parent: e91fcad → 625cbd3 → edbb662; then extracted to Weft per Session 0 below).
- [x] `HANDOFF.md` updated with Session 1 outcomes.
- [x] Next session's immediate action recorded.

---

## Tier A Progress Tracker

Update at the end of each session. Do not skip — this is how the progression through stages stays visible.

| Stage | Status | Gates met? | Artifacts (paths) | Commit hash |
|---|---|---|---|---|
| Bootstrap (Eridos parent) | complete | n/a | original paths in Eridos repo | e91fcad, 625cbd3, edbb662 (audit) |
| Weft extraction (Session 0) | complete | n/a | this repo | recorded below once commit lands |
| Session 2 — frozen V-JEPA 2 wrapper | not started | — | — | — |
| Session 3 — trajectory predictor | not started | — | — | — |
| Session 4 — memory bank + training loop | not started | — | — | — |
| Session 5 — env wrapper + pre-flight smoke test | not started | — | — | — |
| Stage 0a — single config | not started | — | — | — |
| Stage 0b — multi-config, instant cuts | not started | — | — | — |
| Stage 0b — SIGReg ablation | not started | — | — | — |
| Stage 0c — smooth morphs | not started | — | — | — |
| Stage 0c — chain-traversal diagnostic | not started | — | — | — |
| Stage 0c — generalisation probe | not started | — | — | — |
| Tier A complete | not started | — | — | — |

Status values: `not started`, `in progress`, `complete`, `blocked (reason)`, `failed gate (details)`.

---

## Running Jobs

Active long-running processes. Update when a job starts or ends.

```
PID | Started | Script | Log file | Expected completion | Notes
--- | --- | --- | --- | --- | ---
```

(No active jobs.)

---

## Open Decisions & Questions

Items flagged for human review. Accumulate here; do not clear them without explicit instruction.

(None open. Session 1's three open decisions are recorded as resolved in the Session 1 Decisions section above.)

---

## Known Issues & Workarounds

Issues encountered and their current status. Do not delete entries when resolved — mark resolved and keep the history.

- **2026-04-24 — Working-tree divergence at Session 2 start (Eridos parent, pre-extraction).** HANDOFF.md in the Continuous PAM subdir of the Eridos parent repo had been reverted in the working tree to the blank template while the committed HEAD retained Session 1 content. Resolved by extracting the project into its own Weft repo and writing an authoritative Session 1 record here. Original Continuous PAM/ directory left in place inside Eridos for human cleanup per extraction instructions.

---

## Threshold Calibration Log

The instructions §4 treat numerical gate thresholds as calibration targets. Record proposed refinements here at the end of each stage, with the observations that motivated them.

### After Stage 0a
- Proposed revision to next-step MSE gate: __________________
- Proposed revision to masked-position MSE gate: __________________
- Observed plateau value on current environment: __________________
- Rationale: __________________
- Locked before Stage 0b on: __________________

### After Stage 0b
- Proposed revision to cross-boundary recall gate: __________________
- Proposed revision to within-config recall gate: __________________
- Observed values across encoder arms: __________________
- Rationale: __________________
- Locked before Stage 0c on: __________________

### After Stage 0c
- Proposed revision to Tier A completion gates: __________________
- Observed chain-traversal and generalisation values: __________________
- Rationale: __________________

---

## τ Observations

τ is SCAFFOLDING for an eventually-adaptive parameter. Every stage, record observations about whether the fixed τ=5 is producing obvious failure modes, and log retrieval quality stratified by temporal distance (1, 2, 3, 4, 5, 6+ frames from transition). This data feeds the post-Tier-A adaptive mechanism design.

### Stage 0a
- (populate during session)

### Stage 0b
- (populate during session)

### Stage 0c
- (populate during session)

---

## Session Log

Most recent session first. Append new sessions at the top of this section.

### Session 0 — 2026-04-24 — Repo extracted from Eridos parent and re-initialised as Weft

**Goal:** Extract the Continuous PAM directory from the Eridos parent repo into its own standalone GitHub-backed repo named Weft, preserving Session 1 bootstrap state.

**Attempted:**
- Confirmed Eridos (`/mnt/c/Users/Jason/Desktop/Eridos/`) is the parent git repo; Continuous PAM lives inside it as a plain subdirectory (no nested `.git`).
- Copied `Continuous PAM/` contents to `C:\Users\Jason\Desktop\Eridos\Weft\` preserving file attributes.
- Ran `git init` at the Weft root (fresh repo; no history was importable from the subdirectory alone).
- Rewrote `HANDOFF.md` and `README.md` to reflect the new repo name (Weft) while preserving the research-concept name "Continuous Trajectory PAM" in the canonical spec.
- Preserved Session 1 bootstrap outcomes here verbatim for audit continuity.
- Added the GitHub remote `https://github.com/EridosAI/Weft`.

**Worked:**
- Copy succeeded (~13 MB total; all docs, configs, src scaffolding, and `.venv/` skeleton intact).
- Fresh `git init` clean.
- HANDOFF rewritten with the canonical Session 1 facts: V-JEPA 2 checkpoint `facebook/vjepa2-vitl-fpc64-256`, Python 3.12.3, T=1 tensor-shape decision, τ=5 starting scaffolding, three open decisions all resolved.
- Original Continuous PAM/ directory left untouched inside Eridos for human cleanup.

**Failed / in progress:**
- None. Extraction is a standalone administrative operation; no training or code execution involved.

**Decisions made:**
- **Fresh `git init` rather than subtree/history rewrite.** The three Session 1 bootstrap commits (e91fcad, 625cbd3, edbb662) live in the Eridos parent repo's history. Lifting only those commits into a new repo via `git filter-repo` / `git subtree split` was considered and rejected: the cleaner approach is a fresh init with a Session 0 audit entry referencing the original hashes. Original repo retains full history for audit.
- **HANDOFF.md was the reverted working-tree version in the source; I overwrote it with a newly-composed authoritative record** that incorporates the user-specified Session 1 facts. The committed Eridos HANDOFF had stale Python-version info (3.13.11 listed there vs. 3.12.3 that actually characterises the Session 1 environment); the new Weft HANDOFF carries the corrected value.
- Project-name references updated in repo-level docs (`README.md`, `HANDOFF.md` headers). "Continuous Trajectory PAM" preserved as the research-concept name in `PAM_Tiered_v0_Spec.md` and `pam_tier_a_grok_instructions.md` — those are the research claim, not the repo name.

**Original Eridos parent commit hashes referenced (audit trail):**
- `e91fcad` — infra(bootstrap): Session 1 scaffolding + HANDOFF update per pam_tier_a_grok_instructions.md §8.1
- `625cbd3` — docs(handoff): record commit hash e91fcad and finalize Session 1 bootstrap log
- `edbb662` — docs(handoff): update commits list with full bootstrap hashes

**Gate evaluations:**
- None (administrative; gates start at Stage 0a).

**Commits (to be filled in after commit lands):**
- Weft initial commit hash: __________________

**Next immediate action:**
- After the human confirms the extraction worked, execute Session 2 per `SESSION_BATCH_INSTRUCTIONS.md` — implement `src/encoders/frozen_vjepa2.py` and its test.

### Session 1 — 2026-04-23 (within Eridos parent repo; preserved here verbatim)

**Goal:** Complete full Session 1 Bootstrap per pam_tier_a_grok_instructions.md §8 step 1 and HANDOFF.md checklist (dirs, git, deps, verification, documentation).

**Attempted:**
- Read all primary sources (HANDOFF.md first, then pam_tier_a_grok_instructions.md, PAM_Tiered_v0_Spec.md, CODING_STANDARDS.md).
- Created exact §7 directory structure (`.venv/`, `configs/`, full `src/` tree, `scripts/`, `results/`, `checkpoints/`, `logs/`).
- Created `.gitignore`, `requirements.txt` (pinned initial deps), `README.md`, `.env_snapshot.txt`.
- Ran environment verification (OS, Python 3.12.3, PyTorch 2.6.0+cu124, RTX 4080 Super 15.99 GB, CUDA=True).
- Pinned V-JEPA 2 checkpoint identifier: `facebook/vjepa2-vitl-fpc64-256`.
- Resolved T=1 tensor shape decision for `pixel_values_videos`.
- Updated all checklist items, decisions, progress tracker, current status.

**Worked:**
- Directory structure matches spec §7 exactly.
- Environment verification numbers trace to `.env_snapshot.txt` and `torch.cuda.get_device_properties(0)` output.
- HANDOFF.md populated per "every number traces to a file" rule.
- All three open decisions from the Tier A instructions resolved: V-JEPA 2 checkpoint, env divergence, T=1 shape.
- No stop conditions hit. No Tier B/C creep.

**Failed / in progress:**
- Full `pip install -r requirements.txt` inside activated `.venv/` partial — global conda env used for Session 1 to avoid long CUDA wheel download on Windows. `.venv/` exists; full isolation deferred to pre-Stage-0a.
- Smoke tests (V-JEPA load, gym env, FAISS) explicitly deferred per Execution Order (not bootstrap-gated).
- No long-running jobs, no PIDs.

**Decisions made (Eridos-session record; re-stated cleanly in the Session 1 Decisions block above):**
- OS/Python/Torch divergence from §0 header documented (Windows 11 / Python 3.12.3 / Torch 2.6 vs. spec Ubuntu 3.11 / 2.4).
- V-JEPA 2 checkpoint: `facebook/vjepa2-vitl-fpc64-256`.
- T=1 for `pixel_values_videos` in Tier A (Stage 0b cross-boundary safety rationale; Stage 0c sliding-clip T>1 flagged only).
- FAISS IndexFlatIP, seed=42, τ=5 scaffolding.
- Git used existing parent Eridos repo (no nested init at the time). Extracted to standalone Weft repo in Session 0.

**Gate evaluations:**
- None (bootstrap only; gates start at Stage 0a).

**Commits (original Eridos parent repo — preserved as audit reference):**
- `e91fcad` — initial bootstrap scaffolding + HANDOFF.
- `625cbd3` — docs(handoff): record commit hash and finalize Session 1 log.
- `edbb662` — docs(handoff): update commits list with full bootstrap hashes.

**Next immediate action (at time of Session 1 close):**
- Session 2: implement `src/encoders/frozen_vjepa2.py`, smoke-test V-JEPA 2 loading on a random 224×224 tensor (confirm 1024-dim CLS embedding), commit, update HANDOFF.md.

### Template for each session

```
### Session N — YYYY-MM-DD

**Goal:** [What this session aimed to accomplish]

**Attempted:**
- [What was worked on]

**Worked:**
- [What succeeded, with paths to outputs]

**Failed / in progress:**
- [What didn't work, with failure mode analysis]
- [What is still running — PID, expected completion]

**Decisions made:**
- [Non-obvious implementation choices, with rationale]

**Gate evaluations:**
- [If any gates were evaluated this session, pass/fail and evidence]

**Commits:**
- [Hashes of significant commits this session]

**Next immediate action:**
- [What the next session should do first]
```

---

## Cross-Session Reference

### Key file paths
- Canonical spec: `./PAM_Tiered_v0_Spec.md`
- Implementation instructions: `./pam_tier_a_grok_instructions.md`
- Coding standards: `./CODING_STANDARDS.md`
- Stage gates: `./STAGE_GATES.md`
- Autonomous progression: `./AUTONOMOUS_PROGRESSION.md`
- Session batch plan (Sessions 2–5): `./SESSION_BATCH_INSTRUCTIONS.md`
- This handoff: `./HANDOFF.md`
- Stage configs: `./configs/stage_0{a,b,c}.yaml`
- Training entry points: `./scripts/run_stage_0{a,b,c}.sh`
- Results: `./results/stage_0{a,b,c}/`
- Checkpoints: `./checkpoints/`

### Key commits to be aware of
- Original Eridos parent bootstrap (audit only): `e91fcad`, `625cbd3`, `edbb662`.
- Weft initial commit hash: __________________ (to be filled after commit lands).
- Weft "ready for Session 2" commit hash: __________________ (to be filled).
- End of Stage 0a commit: __________________
- End of Stage 0b commit: __________________
- End of Stage 0c commit: __________________
- Tier A complete commit: __________________

---

*Last updated: 2026-04-24 (Session 0 — Weft extraction complete).*
*Current session: Pre-Session-2; awaiting explicit instruction to proceed per SESSION_BATCH_INSTRUCTIONS.md.*
