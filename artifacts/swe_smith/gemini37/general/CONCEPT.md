# CONCEPT: Meta-Harness Evolution — multi_repo / Gemini 3.7 Flash

Analysis of a 5-iteration meta-harness evolution run on the `multi_repo` bug-fixing
benchmark (50 issues, each scored `n=2` so per-issue rates take values `0.0 / 0.5 / 1.0`).
Executor model: `vertex_ai/gemini-3.7-flash` (a weak model — this shapes every finding).
Source: `evolution_summary.jsonl`, `trajectories/`, and per-iter `harness.py` / `SUMMARY.md`.

The frontier moved **once**: `iter1_structured_bugfix_workflow` (0.67) took the lead at
iteration 1 and was **never beaten** by iterations 2–5. This document is written for a
per-instance advisor that will sit on top of that iter1 frontier harness.

---

## 1. Evolution Scorecard

| Iter | Candidate | Train rate | Δ vs frontier | Became frontier? |
|-----:|-----------|-----------:|--------------:|:----------------:|
| 0 | baseline | 0.51 | — | — (start) |
| 1 | structured_bugfix_workflow | **0.67** | **+0.16** | ✅ **YES** |
| 2 | templated_commands_workflow | 0.57 | −0.10 | ❌ |
| 3 | simplified_phase_guidance | 0.64 | −0.03 | ❌ |
| 4 | git_aware_pressure_workflow | 0.01 | −0.66 | ❌ (catastrophic) |
| 5 | error_recovery_workflow | 0.58 | −0.09 | ❌ |

Frontier = 0.67 (iter1) for the entire run. Every subsequent change was a regression;
the run demonstrates a single large win followed by four failed attempts to extend it.

Supporting mechanical metrics (measured directly from `trajectories/`):

| Iter | Candidate | Avg steps | LimitsExceeded | Traj w/ format error | Total format errors |
|-----:|-----------|----------:|---------------:|---------------------:|--------------------:|
| 0 | baseline | 28.7 | 9 | 21 | 48 |
| 1 | structured_bugfix_workflow | 25.9 | 4 | 5 | 8 |
| 2 | templated_commands_workflow | 26.5 | 4 | 8 | 11 |
| 3 | simplified_phase_guidance | 26.2 | 7 | 9 | 10 |
| 4 | git_aware_pressure_workflow | 12.8 | 0 | 9 | 9 |
| 5 | error_recovery_workflow | 27.1 | 8 | 12 | 16 |

---

## 2. What Works

### iter1 — structured_bugfix_workflow (0.51 → 0.67, +0.16) — the only frontier gain

**Change:** override `instance_template` with an explicit 3-phase workflow —
`LOCATE` (max 3–4 search commands) → `UNDERSTAND` (read the exact file/function) →
`FIX` (surgical edits by line number, verify with `git diff`, test). Plus anti-pattern
warnings ("NEVER use broad `sed s/old/new/g` across a whole file") and a soft budget
note ("after 25 steps, prioritize completing the fix").

**Why it worked — two measured mechanisms:**

1. **It collapsed format errors.** Baseline emitted **48 multi-action format errors across
   21 of 50 trajectories** (the harness requires exactly ONE bash block per turn). iter1
   cut this to **8 errors across 5 trajectories** — a 6× reduction. Concretely, baseline
   `dask.pr_10042` failed at **4 steps** by emitting three `pytest` / `git diff` /
   `COMPLETE_TASK` blocks at once → `"found 3 actions"` rejection → wasted turn → premature
   submit, `resolved=False`. The per-phase "one action" discipline removes this failure mode.

2. **It cut step exhaustion.** `LimitsExceeded` (40-step timeouts) dropped **9 → 4** and
   average steps fell **28.7 → 25.9**. The LOCATE cap ("max 3–4 commands") stops the weak
   model from open-ended grepping until the budget is gone.

**Per-issue evidence:** iter1 improved **19 issues** over baseline
(`3,5,6,9,14,15,18,26,27,28,31,32,33,34,36,38,44,45,47`) and regressed only 6, all by −0.5
(`0,4,8,10,16,42`). Net +0.16. Standout recoveries: issue **5** and **6** went `0.0 → 1.0`/
`0.0 → 0.5`; issue **28** `0.0 → 1.0`; issue **18** `0.0 → 1.0`. iter1 is the unique best
harness on issues **5, 33, 34** and is (tied-or-better) best on **35 of 50** issues.

---

## 3. What Hurts

### iter4 — git_aware_pressure_workflow (0.67 → 0.01, −0.66) — catastrophic

The single most instructive failure. Built on iter1 but added: (a) "git-first" LOCATE that
tells the model to `git log`/`git checkout` to find and **revert the Bug Patch commit**;
(b) a "test restoration" pattern to `git checkout` deleted tests; (c) time-pressure
milestones injected at steps 15/25/35 via a `get_observation()` override.

**Why it failed (measured):**

- Average steps collapsed to **12.8** (vs 25.9 at iter1) with **0 LimitsExceeded** — i.e.
  the model stopped *early* everywhere, not late. This is premature submission, not timeout.
- **All 50/50 trajectories invoked `git checkout`** (vs ~0 at iter1). The "revert the bug
  patch / restore the tests" framing made the weak model treat a `git checkout` + a quick
  smoke check as "done." Example (`arrow.func_basic__ocbvf3dz`, 14 steps): the model ran a
  one-line repro, saw expected output, and concluded *"We restored the removed tests and
  reverted the buggy patch. We are ready to submit"* → `resolved=False`.
  Example (`gunicorn.combine_file`, 7 steps): it authored its **own** `tests/test_pidfile.py`,
  saw its self-written tests PASS, and submitted — the hidden F2P tests still failed.
- The `get_observation()` override also corrupted the message stream (stray `" Stanford\n\`\`\`"`
  fragments appear appended to assistant turns), further degrading behavior.

Result: only issue 12 scored non-zero (0.5); 49/50 issues went to 0.0. **Takeaway: never
instruct the model to "revert the bug patch" or self-restore tests, and never override
`get_observation()` to inject urgency — it trades real fixes for fast, false submissions.**

### iter2 — templated_commands_workflow (0.67 → 0.57, −0.10)

Kept iter1's 3 phases but piled on concrete command templates, repeated "ONE ACTION"
example blocks, and step-counter warnings at steps 10/20/30. **Regressed 11 issues vs
iter1** (`3,15,32,47` each by −1.0; `5,12,18,33,34,43,45` by −0.5). Format errors rose back
to 11 (from iter1's 8) and average steps rose to 26.5. The added prescription created
decision/cognitive overhead without buying accuracy — more words, not more fixes.

### iter5 — error_recovery_workflow (0.67 → 0.58, −0.09)

Added an in-line recovery prompt to `format_error_template` and a mandatory pre-submission
"run `pytest -xvs` and only submit if you see PASSED" checkpoint. Despite targeting real
failure modes, it **regressed 16 issues vs iter1** (`5,28,34` by −1.0; 13 more by −0.5) and
had the **most format errors of the non-broken runs (16)** plus 8 `LimitsExceeded`. The
mandatory verification loop consumed budget and, like iter2, the extra instruction volume
hurt the weak model more than the recovery text helped.

### iter3 — simplified_phase_guidance (0.67 → 0.64, −0.03) — a near-miss, not a hurt

Stripped iter2's templates/counters back toward lean iter1. Landed at 0.64 (essentially
iter1 minus noise). Confirms the diagnosis: iter2's loss came from *added bulk*, and
removing it recovers most of the gap — but adds nothing beyond iter1.

---

## 4. Agent Behavior Patterns (from raw trajectories)

- **Multi-action format violations are the dominant self-inflicted failure at baseline.**
  48 violations / 21 trajectories. The model batches `pytest` + `git diff` + `COMPLETE_TASK`
  into one turn. Any harness that enforces "one action per turn" up front largely removes it
  (iter1: 8 / 5).
- **Two distinct death modes:** (i) *timeout* — open-ended exploration hits the 40-step cap
  (`LimitsExceeded`: baseline 9, e.g. `arrow.func_basic__ocbvf3dz`, `getmoto.pr_7365`,
  `marshmallow.func_b*`); (ii) *premature submit* — the model declares victory after a
  superficial check (baseline `dask.pr_10042` at 4 steps; the entire iter4 run at avg 12.8).
  A good harness must fight **both** without over-correcting into the other (iter4 traded
  timeouts for a wall of premature submits).
- **The model trusts its own verification too readily.** In iter4 it accepted self-authored
  tests and one-line repros as proof of a fix (`gunicorn`, `arrow` above). Verification
  guidance only helps if it points at *pre-existing / hidden* tests, not the model's own.
- **Broad edits are a latent hazard** the frontier explicitly guards against
  ("NEVER `sed s/old/new/g` across a whole file"); iter1's surgical-edit rule is retained by
  every later candidate.
- **More instruction text ≠ better on a weak model.** Across iter2/iter5, added templates,
  counters, and mandates each *raised* format errors and/or steps and *lowered* the rate.
  Lean structure (iter1/iter3) dominates verbose structure.

---

## 5. Per-Issue Best-Harness Distribution

Counting, per issue, which harness achieves the max rate (ties credited to each; iter4
excluded as broken):

| Harness | # issues where it is (tied-)best of 50 |
|---------|---------------------------------------:|
| iter1 structured_bugfix_workflow | **35** |
| iter3 simplified_phase_guidance | 33 |
| iter5 error_recovery_workflow | 32 |
| iter2 templated_commands_workflow | 29 |
| iter0 baseline | 25 |

**Uniquely-best (only one harness wins the issue):**
- iter1 uniquely best: issues **5, 33, 34** (3)
- baseline uniquely best: issues **4, 42** (2) — i.e. 2 issues the structured workflow *hurt*

**Structural buckets:**
- **Always solved (rate 1.0 by every non-broken harness):** issues
  `1, 7, 13, 17, 30, 39, 40, 41, 49` (9 issues) — harness-insensitive; an advisor should not
  spend budget shaping these.
- **Never solved (rate 0.0 by everyone, incl. iter4):** issues `11, 23, 29, 35` (4 issues) —
  capability-bound; no prompt-level harness change recovered them.
- The remaining ~37 issues are **harness-sensitive**, and no single harness owns all of them
  (best harness varies: iter1 leads with 35 but iter0/iter3/iter5 each uniquely or jointly
  win subsets). **This spread — best harness differs per issue, with iter1 winning most but
  not all, and even baseline uniquely winning 2 — is the direct motivation for per-instance
  adaptation.**

---

## 6. Guidelines and Key Takeaways (for a per-instance advisor)

Every guideline is grounded in the data above.

1. **Default to the iter1 3-phase structure (LOCATE→UNDERSTAND→FIX).** It is the frontier
   (0.67, +0.16) and the best-of-breed on 35/50 issues. Do not discard it per-instance.

2. **Enforce one-action-per-turn; never batch commands.** Format errors were the #1 baseline
   failure (48 across 21 traj); the fix drove the +0.16 gain. If the advisor sees the model
   batching `pytest`/`git diff`/`COMPLETE_TASK`, intervene.

3. **Cap the LOCATE phase (~3–4 search commands).** This is what cut `LimitsExceeded` 9→4 and
   steps 28.7→25.9. On issues that timed out at baseline (e.g. 2, 4, 10 — `arrow`,
   `marshmallow`, `getmoto`), push toward the FIX phase sooner.

4. **Do NOT add verbose scaffolding.** Command templates (iter2, −0.10) and mandatory
   pre-submission verification loops (iter5, −0.09) both *raised* format errors/steps and
   *lowered* the rate on a weak model. Prefer lean guidance; when in doubt, less text.

5. **Never tell the model to "revert the bug patch," self-restore/author tests, or
   `git checkout` source/tests.** This caused the −0.66 iter4 collapse: 50/50 trajectories
   `git checkout`ed and submitted at avg 12.8 steps with 49/50 issues at 0.0. Verification
   must target pre-existing hidden tests, not the model's own or reverted state.

6. **Never override `get_observation()` to inject time-pressure/urgency.** In iter4 it
   corrupted the message stream and induced mass premature submission (0 timeouts, near-zero
   fixes). Time pressure without a real fix is worse than a timeout.

7. **Keep the surgical-edit rule** ("no broad `sed` across a file; edit by verified line
   number"). Retained by every candidate; cheap insurance against regressions.

8. **Budget by issue class.** Spend no adaptation on the 9 always-solved issues
   (`1,7,13,17,30,39,40,41,49`) and expect no recovery on the 4 never-solved issues
   (`11,23,29,35`); concentrate on the ~37 harness-sensitive issues.

9. **Consider light per-instance switching.** Baseline uniquely beats iter1 on issues
   **4, 42**, and iter3 (leaner) matches iter1 on many issues at 33/50 wins. Where the iter1
   structure appears to *over-constrain* (the 6 issues it regressed: `0,4,8,10,16,42`), an
   advisor may relax back toward the leaner baseline/iter3 behavior — but only for those
   specific instances, never globally.
