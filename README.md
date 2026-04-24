# Weft

**Project:** Continuous Trajectory Predictive Associative Memory (PAM) — Tier A Implementation
**Repo:** `https://github.com/EridosAI/Weft`

Weft is the implementation repository for the Continuous Trajectory PAM research programme. "Continuous Trajectory PAM" is the research-concept name used throughout the canonical specification; "Weft" is this repository's name.

Canonical specification: `PAM_Tiered_v0_Spec.md`

Implementation instructions: `pam_tier_a_grok_instructions.md`

Operational standards: `CODING_STANDARDS.md`

Stage gates: `STAGE_GATES.md`

Autonomous progression rules: `AUTONOMOUS_PROGRESSION.md`

Session 2–5 batch plan: `SESSION_BATCH_INSTRUCTIONS.md`

Live handoff: `HANDOFF.md`

## Directory Structure
Follows §7 of the Tier A instructions exactly (see HANDOFF.md for any documented deviations).

## Setup
```bash
# On Windows (current environment)
python -m venv .venv
# Activate: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Bootstrap Status
Session 1 bootstrap completed (see HANDOFF.md for full log, decisions, and environment verification). Repo extracted from the Eridos parent to this standalone Weft repo in Session 0 (2026-04-24).

**Next:** Session 2 per `SESSION_BATCH_INSTRUCTIONS.md` — implement the frozen V-JEPA 2 encoder (`src/encoders/frozen_vjepa2.py`) and its tests.

---

*This repository implements Tier A only. All Tier B/C items are explicitly locked out until Tier A gates pass.*
