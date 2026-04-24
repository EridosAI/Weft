# Autonomous Progression Rules

**Purpose:** Defines when CC may proceed autonomously between stages and when it must stop for human review. Written to enable efficient use of compute time when human review windows are infrequent.

**Authority:** This document governs cross-stage transitions. It works with `STAGE_GATES.md` (which defines per-stage evaluation) to produce a complete operational rulebook.

**Design principle:** CC progresses autonomously only when the decision is genuinely obvious. Anything requiring judgment stops for human review. Losing a few hours of wall-clock on an AMBIGUOUS-that-was-actually-fine is acceptable; progressing on a borderline baseline is not.

---

## 1. When CC Proceeds Autonomously

### 1.1 Stage 0a → Round 2 (parallel Stage 0b arms)

CC may launch Round 2 automatically if and only if:

- Stage 0a gate evaluation returns `overall_state: PASS` (all gates met per `STAGE_GATES.md` §0a).
- `recommended_action: autonomous_progression` in the gate evaluation output.
- No AMBIGUOUS flags are present.
- The Round 2 launch template (`ROUND_2_LAUNCH_TEMPLATE.md`) is present and current.
- The Round 2 arm configurations have been validated (see §3 below).
- Sufficient compute budget and disk quota remain.

On autonomous progression, CC:
1. Writes `STAGE_0A_COMPLETE.md` to `results/stage_0a/` with full gate values.
2. Updates `HANDOFF.md` with the autonomous transition, timestamp, and launch details.
3. Commits all Stage 0a artifacts (code, logs, results, checkpoints).
4. Launches Round 2 arms per the launch template.
5. Continues monitoring but does not progress further without human review.

### 1.2 Any other stage transition

**No other autonomous progression is supported in Tier A.** Specifically:

- Round 2 completion → Stage 0c: requires human review. Multiple arm results need comparison, and arm selection for Stage 0c is a judgment call.
- Stage 0c completion → Tier A complete: requires human review. Tier A completion is a project milestone; the decision to close Tier A and evaluate Tier B triggers is not autonomous.
- Any AMBIGUOUS result: requires human review regardless of stage.
- Any FAIL result: requires human review. FAIL is often diagnostic and the next action depends on root-cause analysis.

### 1.3 Autonomous retry / re-run is not supported

If Stage 0a fails, CC does not rerun with different parameters. CC stops. Retrying with adjusted parameters is a decision point that requires human judgment about what's likely to be wrong.

---

## 2. When CC Stops for Human Review

CC stops and waits at any of the following, independent of stage:

- Any stage gate evaluation returns FAIL or AMBIGUOUS.
- Any pre-experiment gate fails before training launches.
- Any implementation surprise: spec contradicts reality, dependency issue, hardware problem.
- Any training run produces NaN/Inf.
- Training exceeds 2× expected wall-clock time.
- Disk usage exceeds 80% of quota.
- Any scenario listed as a "Stop Condition" in the Tier A instructions §10.
- Any situation where CC is about to make a non-obvious implementation decision.

Stop format:
1. Complete any currently-running task cleanly (do not interrupt training).
2. Write a stop report to `STOP_REPORT.md` at project root with:
   - The triggering condition
   - The evidence (file paths, metric values, log excerpts)
   - CC's analysis of what it observes
   - Options CC sees
   - CC's recommendation (if any)
3. Update `HANDOFF.md` with the stop event.
4. Commit all artifacts.
5. Wait.

---

## 3. Round 2 Launch Validation

Before CC may autonomously launch Round 2 arms, the following validation runs:

- `ROUND_2_LAUNCH_TEMPLATE.md` exists at project root.
- The launch template specifies between 2 and 4 arms.
- Each arm has: a name, a config file path, a delta-from-baseline description, decision criteria, and expected wall-clock.
- The launch template's referenced config files exist.
- Each arm's config has been diff-checked against the baseline config — diffs are bounded (no more than 3 configuration lines changed per arm).
- Arms are orthogonal: no two arms change the same configuration dimension.
- Pre-gates for Stage 0b (see `STAGE_GATES.md` §0b pre-experiment gates) have been computed against the Stage 0a baseline and pass.

If any validation step fails, CC does not launch Round 2. It stops and reports.

---

## 4. Budget and Resource Management

### 4.1 Compute budget per round

- Round 2 has a per-arm compute budget of N GPU-hours (specified in launch template, typically 8–12 for Push-T scale).
- Total Round 2 budget = (per-arm budget × number of arms) + 20% margin.
- If Round 2 budget exceeds available compute, CC stops and requests budget confirmation.

### 4.2 Monitoring during parallel runs

CC polls log files of all running arms periodically (every 10 minutes). If any arm:
- Produces NaN/Inf: CC does not kill it (per Operational Rule — never kill a training run), but flags it in an ongoing status file, continues monitoring others, and includes the issue in the eventual stop report.
- Crashes with an exception: CC captures the traceback, notes the failure in the status file, continues monitoring remaining arms.
- Stalls (no log updates for > 30 minutes): CC notes the stall but does not intervene.

### 4.3 Round completion detection

A round is complete when all arms have either:
- Finished their expected step count and written final results, OR
- Crashed with documented failure, OR
- Been manually terminated by human instruction (logged in HANDOFF.md).

Round completion triggers the round evaluation (per `STAGE_GATES.md`) and report generation. Round completion never triggers autonomous progression to the next stage — Round 2 → Round 3 requires human review.

---

## 5. Session Boundary Protocol

Within an autonomous progression, session boundaries (end of active CC interaction) are handled as follows:

### 5.1 If a long training run is in progress
- Do not end the session.
- Write a status checkpoint to `HANDOFF.md`.
- Continue to poll logs per normal monitoring.
- The "session" is the training run, not a CC interaction.

### 5.2 If the system is between actions
- Write a session-end update to `HANDOFF.md` per `CODING_STANDARDS.md` §10.
- Commit all state.
- End the session cleanly.
- The next interaction resumes from HANDOFF state.

### 5.3 If the autonomous progression has triggered a next stage
- Complete the transition (close old stage, launch new stage).
- Write both the completion and launch to HANDOFF.md.
- Commit.
- Continue session if training is running; end session if idle.

---

## 6. What Autonomous Progression Does NOT Enable

To avoid ambiguity: CC is not authorised to, under any circumstance:

- Modify the canonical spec.
- Adjust gate thresholds to make a run pass.
- Introduce Tier B or Tier C mechanisms.
- Change the scope lock.
- Skip ablations specified in the Tier A instructions.
- Rerun a failed experiment with different parameters.
- Decide that a Stage gate was "close enough" and proceed.
- Change the directory structure or file naming conventions.
- Update `CODING_STANDARDS.md` or this document.

All of these require explicit human instruction.

---

## 7. Authority Hierarchy

When documents conflict:

1. `CODING_STANDARDS.md` — operational discipline, never overridden.
2. `AUTONOMOUS_PROGRESSION.md` (this document) — progression rules.
3. `STAGE_GATES.md` — evaluation criteria.
4. `pam_tier_a_grok_instructions.md` — implementation specification.
5. `PAM_Tiered_v0_Spec.md` — canonical research spec.
6. Other documents (`HANDOFF.md`, launch templates) — working state.

Higher-authority documents override lower-authority ones on operational questions. Lower-authority documents provide context and specification but cannot override operational rules.

---

*Last updated: pre-Stage-0a, Session 1 complete.*
