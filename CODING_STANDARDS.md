# Coding Standards & Operational Discipline

**Purpose:** Operational rules for implementation work on this project. Read this at the start of every session before touching code. These rules are inherited from prior research work that accumulated specific failure modes; each rule prevents a concrete class of problem that has occurred before.

**Status:** Living document. If a rule proves wrong in practice, propose a change in `HANDOFF.md` with rationale — do not silently drift.

---

## 1. Before You Write Any Code

### 1.1 Read the canonical spec
The canonical specification is `PAM_Tiered_v0_Spec.md` at the repo root. The current implementation instructions are `pam_tier_a_grok_instructions.md`. Both must be present and current. If they disagree, the Tier A instructions are the immediate implementation authority; the tiered spec is the longer-term architecture.

### 1.2 Read `HANDOFF.md` from the previous session
Every session picks up from the previous session's handoff. The last entry in `HANDOFF.md` states what was in progress and what the next immediate action is. If that section is empty or unclear, stop and ask before assuming.

### 1.3 Verify the environment matches the header
The Tier A instructions §0 specifies exact versions for Python, PyTorch, CUDA. If the current environment diverges, either align it or document the divergence explicitly in `HANDOFF.md`. Do not silently train on a different stack.

### 1.4 Scope lock
This project has a strict tier lock. Tier A items are in scope. Tier B and Tier C items are explicitly out of scope and will not be implemented regardless of how tempting or obvious they seem. If you believe a Tier B or Tier C item is required, stop and report — do not implement.

---

## 2. Repository Hygiene

### 2.1 Directory structure is fixed, not evolving
The layout specified in §7 of the Tier A instructions is the layout. Do not reorganise, add parallel structures, or create new top-level directories without an explicit decision in `HANDOFF.md`. Flat directories with 50+ files are a failure mode from prior work — establish structure before the file count grows.

### 2.2 Current-phase files at the root; archive the rest
Only files relevant to the current phase live in active directories. Completed phases get moved to `archive/phase_Nx/` when the next phase begins. The `src/` tree should not accumulate files from multiple superseded approaches.

### 2.3 One source of truth per concept
Each concept has exactly one implementation file. If `memory_bank.py` exists, do not create `memory_bank_v2.py` alongside it. Either replace the original (committing the old version in git) or branch the work.

### 2.4 Never commit these
- `.venv/`
- `checkpoints/*.pt` (too large; use git-lfs only if explicitly configured)
- `results/*.json` beyond a small sample (accumulates rapidly)
- Large log files
- Any file containing secrets, API keys, or paths specific to your machine

`.gitignore` should handle all of these from session 1.

### 2.5 Git commit after every completed task
Not every change — every *task*. A completed bug fix, a completed module implementation, a completed experiment run. Each gets its own commit with a descriptive message that states what was done and why. History reconstruction without this is impossible.

### 2.6 Commit messages follow a convention
```
<type>(<scope>): <summary>

<body if needed — what and why, not how>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `exp`, `infra`. Examples:
- `feat(memory): append-only memory bank with FAISS index`
- `exp(stage_0a): first full training run, 50k frames, commit hash in results`
- `fix(predictor): stop-gradient was being applied to wrong tensor`

---

## 3. Development Workflow

### 3.1 Spec-driven, not exploration-driven
Every substantive addition corresponds to a spec item. If you find yourself writing code that doesn't map back to the Tier A instructions or a prior decision in `HANDOFF.md`, stop. Document what you're doing and why, then continue only if the divergence is justified.

### 3.2 Single-variable changes, not bundled fixes
When testing an intervention, change exactly one thing. Bundled changes produce ambiguous attribution. Prior work spent weeks untangling results from bundled fixes — this is not a principle we're free to ignore.

### 3.3 Fix what feeds the model before making the model bigger
If results look wrong, the first suspects are: data pipeline bugs, incorrect input normalisation, wrong tensor shapes, silent gradient issues. Scaling the model or adding capacity to compensate for upstream bugs wastes compute and obscures the real problem.

### 3.4 Run the simple baseline early, not late
The Tier A instructions require pure cosine kNN baselines as a gate. These are not afterthoughts — they get implemented and run alongside the main experiments, from day one. If the learned system doesn't beat the simple baseline, that's a finding, not a failure to hide.

### 3.5 Cheapest diagnostic first
Before committing to expensive compute, run the minimum test that can close the path. An hour of analysis on existing output often out-informs two hours of retraining. If a diagnostic requires a full training run, it is not a quick diagnostic — scope accordingly.

---

## 4. Orchestration & Sub-Agents

### 4.1 One main thread, targeted sub-agents for bounded tasks
The main Grok session orchestrates. For bounded, well-scoped tasks (run a test, grep for a pattern, summarise a log file), consider delegating to a sub-agent with minimal context. The sub-agent returns its result and its context is discarded. This extends effective context budget for long sessions.

### 4.2 Sub-agents get the minimum viable context
Do not dump the whole spec into a sub-agent prompt. Give it: the specific task, the specific file(s) it operates on, the specific success criterion. Anything more is context pollution.

### 4.3 Do not run parallel Cursor sessions on the same codebase
Concurrent edits produce race conditions and merge conflicts that are painful to untangle. One active session per working tree.

### 4.4 Context hygiene within a session
When a long session fills with exploratory output, summarise the state into `HANDOFF.md`, git commit, and start a new session. Compacted context can produce fabricated numbers — post-compaction summaries are not trustworthy without re-verifying against source files.

---

## 5. Running Training & Experiments

### 5.1 Never kill a running training process
Under any circumstances. If a training run is in progress, let it finish. Interruption produces inconsistent state and corrupts logs. If an experiment genuinely needs to be aborted, the decision is made explicitly, documented in `HANDOFF.md`, and the checkpoint state is captured before termination.

### 5.2 Use nohup for long-running scripts
Any script expected to run longer than 5 minutes is launched with:
```bash
nohup python -u scripts/run_stage_0a.py > logs/stage_0a_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/stage_0a.pid
```
Logs go to a timestamped file. PID goes to a file so the process can be monitored and (if truly necessary) terminated deliberately.

### 5.3 Poll logs, don't block on stdout
Check progress by reading the log file, not by blocking on the running process. `tail -f` or timed `cat` against the log file. The session continues to do other work while long jobs run.

### 5.4 Checkpoint aggressively
Save predictor weights every 10,000 training steps. Save memory bank snapshots at stage boundaries. Tag every checkpoint with the git commit hash at save time. A checkpoint you can't reproduce from a specific commit is a checkpoint you don't really have.

### 5.5 Every number traces to a file
Every number that appears in any summary, handoff, or report is verified against its source file before being quoted. No remembered numbers. No mental arithmetic. If the source file isn't findable, the number isn't used.

### 5.6 Stop at gate failures
Every stage has pass/fail gates. When a gate fails, the next stage does not run. Stop, report, wait for instruction. Do not attempt fixes without being asked.

---

## 6. Logging & Instrumentation

### 6.1 Log at three levels
- **Training-step level**: loss, gradient norm, current mask ratio, step index. Written to TensorBoard.
- **Checkpoint level**: aggregate metrics (MSE distributions, embedding norm stats, sample retrievals). Written to JSON on disk.
- **Session level**: what was run, what was observed, what the next step is. Written to `HANDOFF.md`.

### 6.2 JSON dumps have fixed schemas
Every JSON log file has a schema defined at the top of the logging module. Changing the schema is a code change that requires a commit. This prevents the pattern where downstream analysis breaks because logs from different runs have different fields.

### 6.3 Log distributions, not just means
Retrieval quality per probe, not mean retrieval quality. Per-instance MSE, not just batch mean. Failure modes hide in the tails; summary statistics conceal them.

### 6.4 Diagnostic logs are cheap — log more than you think you need
Disk is cheaper than re-running. The rule is: if a number might be useful for a diagnostic later, log it now. Tier A specifies required logs; add any diagnostic you think might help without asking permission.

### 6.5 Never log to stdout for long-running jobs
stdout in a backgrounded process is either redirected to a file (see §5.2) or lost. Never print() diagnostics that you expect to review — write them.

---

## 7. Tensor & Numerical Hygiene

### 7.1 Shape assertions at module boundaries
Every module's forward pass asserts input and output shapes. `assert x.shape == (B, W, D)` at the top of forward methods. Silent broadcasting bugs are expensive to debug after training has run for hours.

### 7.2 Verify stop-gradient and frozen parameters explicitly
After model initialisation, log `sum(p.numel() for p in model.parameters() if p.requires_grad)` and verify it matches expectation. If the Outward encoder is supposed to be frozen, that number should equal just the predictor's parameter count. Catching a frozen-not-frozen bug at init is trivial; catching it at epoch 20 because loss looks wrong is not.

### 7.3 Explicit device placement
No silent CPU↔GPU transfers. Tensors live on a specified device. Cross-device operations raise errors, not silent `.cpu()` calls. This is usually enforced by the framework, but verify.

### 7.4 Normalisation is explicit and tested
If embeddings are L2-normalised before FAISS indexing, there's a unit test verifying this. Cosine similarity on un-normalised vectors returns garbage that looks plausible. This is a well-documented failure mode from prior work.

### 7.5 No NaN/Inf silently
At checkpoint boundaries, assert no NaN or Inf in model parameters or key activations. NaN losses usually produce garbage logs that look acceptable on their face.

---

## 8. Package & Dependency Management

### 8.1 `requirements.txt` is pinned
Every package has a specific version pin. No `>=` or `~=`. Reproducibility requires determinism; determinism requires pinned dependencies.

### 8.2 New imports require a requirements update
Adding an `import foo` requires adding `foo==X.Y.Z` to `requirements.txt` in the same commit. No session should end with uncommitted dependency changes.

### 8.3 pip install uses `--break-system-packages` only in venv
Never install globally. If a package install requires system-level changes, stop and flag it in `HANDOFF.md`.

### 8.4 `pip freeze` at session end
End-of-session checklist includes `pip freeze > .env_snapshot.txt` committed to the repo. When a future session can't reproduce a result, the env snapshot is the first thing checked.

---

## 9. Away-Mode Operation

### 9.1 The user is not watching
Grok operates autonomously during sessions. Make reasonable decisions based on the spec and document them in `HANDOFF.md`. Do not stop to ask clarifying questions mid-task that can wait for the next review.

### 9.2 Document every non-obvious decision
When you make an implementation choice that isn't directly specified — default hyperparameter, library choice, code structure — document it in `HANDOFF.md` with rationale. The cost of over-documenting is minutes; the cost of a silent decision is days of later debugging.

### 9.3 Stop after 5 failed tool calls
Five attempts at the same operation is enough. Structural problems don't get solved by retrying. Stop, document the failure mode, wait for instruction.

### 9.4 Stop conditions beyond gate failures
Stop and report when any of these occur:
- Gate fails and spec's proposed investigation doesn't apply
- Training produces NaN losses or complete representational collapse
- FAISS retrieval returns the probe itself as top hit more often than expected
- You believe a Tier B or Tier C item is required
- The spec contradicts implementation reality (API changed, checkpoint unavailable)
- You identify a substantially cleaner approach — don't implement silently

Report format: the issue, the evidence (file paths + line numbers + metric values), the options you see, your recommendation. Then wait.

---

## 10. End-of-Session Protocol

At the end of every session, before ending the Cursor chat:

1. Git commit all changes with descriptive messages.
2. Update `HANDOFF.md` with:
   - What was attempted this session
   - What worked (with links to outputs)
   - What failed (with failure mode analysis)
   - What is in progress (running jobs, PIDs, log paths)
   - The next immediate action for the next session
3. `pip freeze > .env_snapshot.txt` and commit if changed.
4. Verify no uncommitted changes in working tree (`git status` clean).
5. If training is running, note the expected completion time and the PID.

A session that ends without these steps is a session whose state has to be reconstructed from git history and guesswork. Do not end sessions that way.

---

## 11. What This Document Does Not Cover

- Research methodology (which experiments to run, how to interpret results) — that's the spec's job.
- Architecture details (what the predictor looks like, what the loss is) — that's the instructions document's job.
- Theoretical rationale (why PAM works, why SIGReg matters) — that's in the canonical spec.

This document is purely operational. If something feels like it belongs here but is actually about the research itself, it goes in the spec or instructions instead.

---

*This document carries forward the operational discipline developed through the prior ALAN/PAM work, stripped of architecture-specific content and generalised to this project. Update it when a new discipline emerges from practice; do not update it with research findings.*
