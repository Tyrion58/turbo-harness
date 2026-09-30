# CONCEPT: Meta-Harness Evolution for Gemini-3.7-Flash on SWE-bench Verified

Source data: `experiments/logs/meta_harness/verified_5iter_gemini37/` (evolution_summary.jsonl,
trajectories/) and `artifacts/verified/gemini-3-7-flash/verified_5iter_gemini37/iter*/`
(harness.py, SUMMARY.md). 251 train issues, student = gemini-3.7-flash.

Train rates (`avg_rate`) are the official metric from `evolution_summary.jsonl` (fractional,
multi-rollout). Behavioral counts below are computed from the single stored rollout per issue
(251 trajectory files/harness) with a regex edit-detector, so treat them as *trends*, not exact
resolve rates (they undercount python-file-write edits).

---

## 1. Evolution Scorecard

| iter | candidate | train rate (`avg_rate`) | delta vs frontier | became frontier? |
|-----:|-----------|------------------------:|------------------:|:-----------------|
| 0 | baseline | 0.2271 | — | (start) |
| 1 | early_test_driven_iteration | 0.2530 | +0.0259 | ✅ yes |
| 2 | progress_checkpoint_forcing | 0.2789 | +0.0259 | ✅ yes |
| 3 | exploration_budget_hard_cutoff | 0.3705 | +0.0916 | ✅ yes |
| 4 | **test_after_edit_forcing** | **0.3805** | **+0.0100** | ✅ **yes (final frontier)** |
| 5 | early_test_discovery_forcing | 0.3506 | −0.0299 | ❌ no (regressed) |

Net gain baseline → frontier: **+0.1534 (0.2271 → 0.3805, +67% relative)**. The single largest
jump is iter3 (+0.0916), which introduced *hard* command blocking. iter4 refined it (+0.0100);
iter5 over-constrained and regressed (−0.0299).

---

## 2. What Works

### iter1 — early_test_driven_iteration (0.2271 → 0.2530, +0.0259)
Reordered the `instance_template` workflow: added a step-0 "form a hypothesis," moved "analyze
codebase" *after* the first edit, and added timing guidance ("first edit within 10-15 steps").
- Evidence it helped: median first-edit step dropped 21 → 19; trajectories that made ≥1 edit rose
  118 → 134 (of 251); no-edit trajectories fell 63% → 61%.
- Per-issue: recovered issues like `#3` (0.5 → 1.0) and `#54` (0.5 → 1.0). Small because it is
  *soft* guidance — the model still ignored it on 61% of instances.

### iter2 — progress_checkpoint_forcing (0.2530 → 0.2789, +0.0259)
Injected `<checkpoint>` USER messages at steps 10/20/30 forcing hypothesis articulation.
- Evidence: median first-edit 19; no-edit trajectories fell to 59%; issues `#94` (0.0 → 1.0),
  `#122` (0.5 → 1.0) recovered.
- Still soft (a message, not a block) — modest, matching iter1's magnitude.

### iter3 — exploration_budget_hard_cutoff (0.2789 → 0.3705, +0.0916) ← biggest win
Overrode `get_observation()` to **block** non-edit commands after step 15 until an edit is made.
The read command is *not executed*; the agent sees only the block message.
- Evidence of mechanism firing: the "EXPLORATION BUDGET EXHAUSTED" message appears in **192/251**
  trajectories, and **40%** of those emit an edit command on the very next turn (more comply within
  2-3 turns via `python3 -c "with open(...,'w')"` writes the regex partially misses).
- Behavioral shift: no-edit trajectories dropped 59% → **41%**; avg edits/traj 1.34 → **1.96**;
  median first-edit 19 → **17**; Submitted status rose 80 → **107**.
- Per-issue recoveries: `#47` 0.0 → 1.0, `#51` 0.0 → 1.0, `#77` 0.0 → 1.0, `#104` 0.0 → 1.0,
  `#218` 0.5 → 1.0. Lesson: **for gemini-3.7-flash, hard enforcement >> soft prompting.**

### iter4 — test_after_edit_forcing (0.3705 → 0.3805, +0.0100) ← final frontier
Kept iter3's step-15 first-edit block and added a *continuous* edit→test cycle: after any edit,
enter `test_pending`, allow ≤5 exploration steps, then block until a test command runs.
- Evidence: the edit→test discipline is visible in trajectories — e.g. `django__django-11333`
  (baseline: LimitsExceeded, no successful fix) becomes **resolved in 38 steps** under iter4, with
  first edit at step 9 (`cat << 'EOF' > patch.py`) followed by repeated `runtests.py` validation
  at steps 24/30–33/36–37 before `COMPLETE_TASK_AND_SUBMIT`.
- Per-issue gains over iter3: `#8` 0.5 → 1.0, `#59` 0.5 → 1.0, `#62` 0.0 → 1.0, `#68` from mixed
  → 1.0, `#219` 0.0 → 1.0, `#233` 0.0 → 1.0. Submitted status rose to 121.
- Small delta because it mostly *converts near-misses* rather than opening new issues; avg edits
  actually fell 1.96 → 1.62 (less post-edit thrashing), which is the intended effect.

---

## 3. What Hurts

### iter5 — early_test_discovery_forcing (0.3805 → 0.3506, −0.0299) ← the regression
Added a step-12 block forcing a test command *before* the first edit (on top of iter4's logic).
- Why it failed: it spends scarce early-budget on test-syntax discovery the agent doesn't yet need,
  and over-forces action. avg edits/traj jumped to **2.46** (highest of all harnesses) and no-edit
  trajectories fell to **36%** (lowest) — yet the rate *dropped*. More forced edits ≠ better fixes;
  the extra edits were often premature/speculative churn.
- Per-issue casualties vs iter4: `#34` 0.5 → 0.0, `#57` 0.5 → 0.0, `#84` regressed on validation,
  `#242` 1.0 → 0.0, `#98` 0.5 → 0.0. It did win a few (`#20` → 1.0, `#93` → 1.0) but net-negative.
- Lesson: **there is a stopping point for forcing.** Chaining a third mandatory gate before the
  edit consumes exploration budget and pushes the model into low-quality edits. iter4's two gates
  (step-15 edit, post-edit test) are the sweet spot.

Note: iter1 and iter2 did not regress but delivered only ~1/4 of iter3's gain, confirming soft
prompts/messages are weak levers for this model.

---

## 4. Agent Behavior Patterns (from raw trajectories)

- **Analysis paralysis is the dominant baseline failure.** Baseline: **178/251 LimitsExceeded**
  (71%), **63% of trajectories make zero detected edits**, median first-edit step 21, avg steps
  36.7. The agent loops on `git grep`, `sed -n`, `cat` reads and never commits to a fix. Example
  `django__django-11333` baseline: 40 steps of grep/sed/runtests reads, `git diff` at the end,
  never converging.
- **Blocking converts exploration into action.** In iter3 the block fires on 192/251 trajectories;
  a representative response (`astropy__astropy-12907`) is the agent immediately switching to
  `python3 -c "with open('astropy/modeling/separable.py','r') as f: ..."` — a file write — on the
  next turn. Across iter3→iter5, LimitsExceeded fell from 178 (baseline) to 130–144 and Submitted
  rose from 73 to 107–121.
- **Post-edit drift is the second failure mode** iter4 targets: agents made a forced edit, then
  wandered back into reads until timeout. iter4's ≤5-step post-edit budget + test gate produces the
  observed edit→read-check→`runtests.py`→edit loops (e.g. `django-11333` steps 23–37).
- **Test-command fumbling is real but shouldn't be front-loaded.** Trajectories show wasted turns
  finding the right invocation (`./runtests.py` → `python tests/runtests.py` →
  `python -m unittest discover` → `chmod +x tests/runtests.py`, seen in `django-11333` steps 7–18).
  iter5 tried to pre-empt this at step 12 and regressed — the fix is cheaper *when* validation is
  actually needed, not before an edit exists.
- **Never-solvable core.** 98/251 issues (39%) are solved by *no* harness in any iteration —
  capability-bound, not harness-bound.

---

## 5. Per-Issue Best Harness Distribution

Computed from `per_issue` across all 6 harnesses (251 issues; best = max rate, ties shared):

- **153 issues solved by ≥1 harness; 98 never solved (39%).**
- **Sole (unique) best harness** — how often exactly one harness wins an issue:

  | harness | sole-best issues |
  |---------|-----------------:|
  | test_after_edit_forcing (iter4) | 17 |
  | exploration_budget_hard_cutoff (iter3) | 16 |
  | early_test_discovery_forcing (iter5) | 8 |
  | progress_checkpoint_forcing (iter2) | 4 |
  | baseline (iter0) | 3 |
  | early_test_driven_iteration (iter1) | 3 |

- **No harness dominates.** The final frontier (iter4) is *beaten by some other harness on 62 of
  the 153 solved issues*, and is uniquely best on only 17. Even the discarded iter5 is the sole
  winner on 8 issues; baseline itself is strictly best on 3 (issues where forcing hurt).

This is the core motivation for **per-instance adaptation**: picking iter4 globally leaves ~40% of
solvable-issue "wins" on the table where a different (often less-forcing or more-forcing) harness
would have solved it.

---

## 6. Guidelines and Key Takeaways for a Per-Instance Advisor

Every guideline is grounded in the evolution data above.

1. **Default to hard blocking, not prompting.** Soft levers (iter1 template reorder +0.0259, iter2
   checkpoint messages +0.0259) each moved the needle ~1/4 as much as iter3's hard block (+0.0916).
   For gemini-3.7-flash, mechanically preventing exploration commands is the highest-value action.

2. **Force the first edit around step 15.** iter3's step-15 cutoff cut no-edit trajectories 59% →
   41% and drove the single biggest gain. The advisor should ensure an edit attempt exists by
   ~step 15 on instances where the agent is still only reading.

3. **Enforce edit→test cycles, capped at ~5 post-edit reads.** iter4's post-edit test gate added
   +0.0100 and *reduced* avg edits (1.96 → 1.62) by killing post-edit drift. Require a test run
   within a few steps of each edit.

4. **Do NOT stack a pre-edit test-discovery gate.** iter5 added exactly this at step 12 and
   regressed −0.0299 while inflating edits to 2.46/traj. More mandatory gates before an edit exists
   waste exploration budget and produce speculative edits. Two gates (edit, then test) is the ceiling.

5. **More forced edits is not the objective.** iter5 had the most edits and fewest no-edit
   trajectories yet the worst rate among iters 3-5. Optimize for *validated* edits, not edit count.

6. **Relax forcing on issues where baseline/soft wins.** Baseline is strictly best on 3 issues and
   is among winners on 44; some instances are hurt by aggressive blocking (e.g. iter5 losses `#34`,
   `#242`, `#57`). An advisor should detect when an instance is progressing well under light
   guidance and *not* impose the step-15 block.

7. **Accept a hard capability floor.** 98/251 issues are unsolved by every harness — the advisor
   should not burn budget forcing edits on instances with no viable fix path; spend effort on the
   ~62 issues where harness choice flips the outcome.

8. **When choosing per instance, prefer iter4 as the base and iter3 as the high-exploration
   fallback.** iter4 and iter3 together hold 33 of 41 sole-best slots; they are the two policies an
   advisor should arbitrate between for the majority of instances, reserving softer policies for the
   handful of instances where forcing regresses.
