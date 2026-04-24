# Stage Gates — Programmatic Evaluation Criteria

**Purpose:** Objective, programmatically-checkable gate criteria for each Tier A stage. Gates are evaluated in three states: PASS, FAIL, AMBIGUOUS. CC uses these to determine whether to proceed autonomously or stop for human review.

**Authority:** This document is the authoritative gate reference. If the Tier A instructions and this document disagree on a specific threshold, this document wins. The instructions describe the research intent; this document defines the operational decision boundary.

**Design principle:** PASS requires gates met *with margin and without anomaly*. FAIL catches obvious breakage fast. AMBIGUOUS is broad and errs toward stopping for human review. The asymmetry is deliberate — unambiguous results should progress quickly, anything else should wait.

---

## Stage 0a — Single Config, Interpolation Regime

### Evaluation pipeline

Run at end of Stage 0a training (after 50k frames, or earlier if forced stop triggered):

1. Compute all metrics below from logged output files.
2. Evaluate each gate group (PASS / FAIL / AMBIGUOUS).
3. Write results to `results/stage_0a/gate_evaluation.json` with full metric values and PASS/FAIL/AMBIGUOUS per gate.
4. Write human-readable summary to `results/stage_0a/STAGE_0A_REPORT.md`.
5. Update `HANDOFF.md` with the stage outcome and decision path.
6. If overall state is PASS: proceed autonomously per `AUTONOMOUS_PROGRESSION.md`.
7. If overall state is FAIL or AMBIGUOUS: stop and wait for human review.

### PASS — all of the following must hold

**Loss trajectory:**
- `next_step_mse_final_5k_mean` ≤ 0.01
- `next_step_mse_monotonic_decrease`: linear regression slope on next-step MSE over the final 10k training steps is ≤ 0 (95% CI).
- `next_step_mse_total_reduction`: (initial_5k_mean − final_5k_mean) / initial_5k_mean ≥ 0.50
- `masked_position_mse_final_5k_mean` ≤ 0.01 at the current mask ratio.

**Curriculum progression:**
- `final_mask_ratio` ≥ 0.25
- `mask_ratio_growth_monotonic`: mask ratio increased through plateau triggers only, never decreased.

**Prediction quality:**
- `held_out_next_step_cosine_mean` ≥ 0.9 (on 1000-frame held-out stretch)
- `held_out_next_step_cosine_std` ≤ 0.1 (not high-variance)

**Numerical health:**
- `nan_inf_count` == 0 across all logged metrics for the full run.
- `grad_norm_spike_ratio` < 10 (max gradient norm in final 5k steps / median gradient norm in final 5k steps).
- `predictor_output_norm_ratio`: mean predictor output L2 norm in final 5k steps is within [0.8, 1.2] × mean encoder embedding L2 norm.

**Infrastructure health:**
- `memory_bank_size`: actual stored frames ≥ 0.95 × expected frames.
- `training_step_count`: final step count ≤ 1.5 × expected (catches efficiency regressions).

### FAIL — any of the following stops the run

- Any NaN or Inf appeared in loss or gradients during training.
- `next_step_mse_total_reduction` < 0.50 (trained but didn't improve meaningfully).
- Unhandled exception crashed the training loop.
- `final_mask_ratio` == starting value after 50k frames (no growth ever).
- `memory_bank_size` deviates > 10% from expected.
- `held_out_next_step_cosine_mean` < 0.5.
- Training did not complete within 2× expected wall-clock time (stuck or deadlocked).

### AMBIGUOUS — anything else stops the run

Any of:
- All PASS criteria met *except* 1–2 by a margin of less than 20% (e.g., mask ratio = 0.23 instead of 0.25, or cosine mean = 0.88 instead of 0.9).
- Mask ratio grew to ≥ 0.25 but non-monotonically (grew, shrank, regrew) — indicates instability.
- `training_step_count` between 1.5× and 2× expected.
- Any gate passes but the logged warnings file (`results/stage_0a/warnings.log`) is non-empty.
- `grad_norm_spike_ratio` between 5 and 10.
- Any computed metric has an expected-vs-actual mismatch that doesn't fit PASS or FAIL cleanly.

---

## Stage 0b — Multi-Config, Instant Scene Cuts

### Pre-experiment gates (evaluated before Stage 0b training begins)

These must pass before any Stage 0b training launches. Failure stops the pipeline.

- `config_pairwise_cosine_max` < 0.8: mean pairwise cosine similarity between every pair of configs (computed from 500 random frames per config through the Outward encoder) must be below 0.8 for all pairs.
- `config_count` ≥ 3.
- `frames_per_config_block` in [500, 2000] range.
- `total_stream_length` sufficient for every ordered config pair to co-occur ≥ 20 times.

If any pre-gate fails, CC writes to `results/stage_0b/pregate_failure.json` and stops.

### PASS — all of the following must hold

**Core metric:**
- `cross_boundary_recall_at_10` ≥ 0.3 on the headline probe set (probes ≥ 500 frames from any transition; ground-truth associates are frames co-occurring within τ=5 of a transition involving the probe's config).
- `cross_boundary_recall_at_10` is measured on probes where `probe_to_ground_truth_cosine` < 0.3 (isolates the cross-boundary regime).

**Diagnostic controls:**
- `shuffle_control_cross_boundary_recall_at_10` ≤ 0.05 (proves the mechanism learned temporal structure, not geometry).
- `pure_cosine_knn_cross_boundary_recall_at_10` ≤ 0.05 (proves probes are genuinely cross-boundary).

**Sanity checks:**
- `within_config_recall_at_10` ≥ 0.7 (basic retrieval still works).
- `cross_boundary_recall_at_10` > `shuffle_control_cross_boundary_recall_at_10` + 0.15 (meaningful margin above shuffle control).

**Numerical and infrastructure health:** same as Stage 0a (no NaN/Inf, no grad spikes, memory bank populated, training completed).

### FAIL — any of the following stops the run

- Any Stage 0a FAIL criterion applied to Stage 0b training.
- `cross_boundary_recall_at_10` < 0.1 (mechanism not working at all).
- `shuffle_control_cross_boundary_recall_at_10` ≥ `cross_boundary_recall_at_10` − 0.05 (mechanism might be learning geometry, not temporal structure).
- `within_config_recall_at_10` < 0.3 (basic retrieval broken).
- `pure_cosine_knn_cross_boundary_recall_at_10` ≥ 0.15 (probes are not genuinely cross-boundary; evaluation is invalid).

### AMBIGUOUS — anything else stops the run

- Cross-boundary recall between 0.1 and 0.3.
- Shuffle control between 0.05 and cross-boundary − 0.15.
- Within-config recall between 0.3 and 0.7.
- Any Stage 0a AMBIGUOUS criterion applied to Stage 0b training.
- Results vary significantly across seeds (if multiple seeds run): std across seeds > 0.1 on any headline metric.

### For parallel arms in Round 2

When Stage 0b is run as parallel arms, gate evaluation is per-arm. An arm is PASS, FAIL, or AMBIGUOUS independently. Round 2 progression to Round 3 requires human review of all arm results together — autonomous progression across a multi-arm round is not supported because arm comparison is a judgment call.

---

## Stage 0c — Multi-Config, Smooth Morph Transitions

### Pre-experiment gates

- All Stage 0b pre-gates met.
- `morph_transition_length_frames` in [100, 300] range.
- `transition_zone_flag` populated for all morph frames in memory bank metadata.
- At least one Stage 0b arm achieved PASS in the preceding round.

### PASS — all of the following must hold

**Core metric:**
- `cross_boundary_recall_at_10` ≥ 0.4 (higher than 0b because smooth transitions provide chain structure).

**Diagnostic controls:** same as Stage 0b.

**Chain-traversal diagnostic:**
- `chain_traversal_3hop_target_config_fraction` ≥ 0.4: for probes in config A, 3-hop retrieval lands in a temporally-adjacent config for ≥ 40% of probes.

**Generalisation diagnostic:**
- `withheld_pair_recall_at_10` ≥ 0.15 on a config pair withheld from training.
- `withheld_pair_recall_at_10` > `shuffle_control_withheld_pair_recall_at_10` + 0.1 (generalisation is real, not noise).

**Numerical and infrastructure health:** same as previous stages.

### FAIL — any of the following stops the run

- All Stage 0b FAIL criteria.
- `cross_boundary_recall_at_10` < 0.2.
- `chain_traversal_3hop_target_config_fraction` < 0.15.
- `withheld_pair_recall_at_10` not distinguishable from shuffle control.

### AMBIGUOUS — anything else stops the run

- Any PASS criterion missed by < 20% margin.
- Chain traversal between 0.15 and 0.4.
- Withheld pair recall between 0.05 and 0.15.

---

## Global AMBIGUOUS Triggers (apply to any stage)

CC also stops and reports AMBIGUOUS if any of the following occur regardless of gate outcome:

- Any implementation decision was made during the run that wasn't in the spec (documented in HANDOFF.md as "decision made autonomously").
- Disk usage exceeded 80% of allocated quota.
- A dependency upgrade happened mid-run.
- Git working tree was not clean at training start.
- More than 10 warnings logged during the run (warnings indicate unexpected conditions even if training completed).
- Metric log files have gaps (missing steps) > 1% of total steps.

These are "something unusual happened that deserves a human look, even if the numbers look fine."

---

## Gate Evaluation Code Contract

The gate evaluation function has a single entry point and a strict output format:

```python
def evaluate_stage_gates(
    stage: str,  # "0a", "0b", "0c"
    results_dir: Path,
) -> dict:
    """
    Returns:
    {
        "stage": "0a",
        "overall_state": "PASS" | "FAIL" | "AMBIGUOUS",
        "gates": {
            "gate_name_1": {"state": "PASS", "value": 0.023, "threshold": "<= 0.01", "margin": 0.77},
            ...
        },
        "failure_reasons": [...],  # list of strings, empty on PASS
        "ambiguity_reasons": [...],  # list of strings, empty on PASS or FAIL
        "recommended_action": "autonomous_progression" | "human_review_required",
        "evaluation_timestamp": "2026-04-24T14:30:00Z",
        "git_commit_hash": "...",
    }
}
```

The full output is written to `results/stage_{X}/gate_evaluation.json`. A human-readable summary is written to `STAGE_{X}_REPORT.md`.

---

## When Thresholds Need Calibration

The numerical thresholds in this document are starting values derived from the Tier A instructions. They may not be appropriate for Push-T specifically.

Calibration workflow:
1. If a Stage 0a run produces PASS comfortably (all gates met with > 30% margin), thresholds are appropriate.
2. If Stage 0a produces AMBIGUOUS due to one gate near threshold, CC stops for human review. Human decides whether to adjust the threshold and re-run, or to accept the current result.
3. Once thresholds are adjusted, commit the change to `STAGE_GATES.md` with rationale in the commit message.
4. Do not adjust thresholds post-hoc to move a run from FAIL to PASS. Threshold changes must be justified by reasoning about what the appropriate value should be, not by reverse-engineering a specific result.

---

*Last updated: Session 1 bootstrap complete, pre-Stage-0a.*
