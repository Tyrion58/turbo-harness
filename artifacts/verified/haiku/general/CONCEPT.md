# CONCEPT: SWE-bench Verified Harness Evolution (student = Haiku 4.5)

Meta-harness evolution over 5 iterations against a frozen 251-issue Verified train
split, student model Haiku 4.5, 40-step budget. Train rate = `avg_rate` (2-rollout
mean) from `evolution_summary.jsonl`. Behavioral stats below are from the single
representative rollout stored per issue in `trajectories/` (n=251 each).

Frontier selected: **iter2 `understand_then_edit_checkpoint` (49.6%)** — this file
sits in its checkpoint dir. iter4 tied it (49.6%) but did not exceed, so the frontier
did not advance past iter2.

---

## 1. Evolution Scorecard

| iter | candidate | train rate | Δ vs frontier | became frontier? |
|------|-----------|-----------:|--------------:|:----------------:|
| 0 | baseline | 26.49% | — | (start) |
| 1 | early_submit_guidance | 48.61% | **+22.11** | ✅ yes |
| 2 | understand_then_edit_checkpoint | 49.60% | +1.00 | ✅ yes (marginal) |
| 3 | ignore_test_files | 44.22% | −5.38 | ❌ rejected |
| 4 | verify_edits_before_submit | 49.60% | 0.00 | ❌ tie, no advance |
| 5 | run_failing_test_before_submit | 35.26% | −14.34 | ❌ rejected |

One mechanism dominates the entire run: **iter1's early-submit guidance (+22.11)
accounts for ~96% of the total gain over baseline** (26.49 → 49.60). Everything after
iter1 moved the needle by ≤1 point or regressed.

---

## 2. What Works

### iter1 — early_submit_guidance (+22.11, 26.49 → 48.61) — the decisive lever
Removed baseline's "verify your fix / test edge cases" workflow; instructs the agent
to submit immediately after editing, with repeated 40-step-budget warnings.

Evidence (trajectory status counts):
- Baseline: **165/251 LimitsExceeded (65.7%)**, only 85 Submitted; mean steps 36.7,
  median 40 (i.e. the agent routinely ran the clock out).
- iter1: **LimitsExceeded collapses to 41/251 (16%)**, 210 Submitted; mean steps
  22.2, median 20; mean cost falls $0.478 → $0.203.
- Baseline burned steps testing: **74% of baseline trajectories ran a test command**
  (`pytest`/`manage.py test`); iter1 dropped this to **12%**.
- Issue-level: iter1 improved 85 issues vs baseline, regressed only 11.

Root cause it fixes: on a 40-step budget, Haiku's verify/test loops exhaust the budget
*after* a correct edit is already on disk. Cutting the loop converts
LimitsExceeded → Submitted. Baseline resolved trajectories averaged 30.0 steps vs 38.9
for unresolved — failures were the ones that ran long.

### iter2 — understand_then_edit_checkpoint (+1.00, 48.61 → 49.60) — marginal refinement
Kept iter1's early-submit, added a pre-edit checkpoint (state ROOT CAUSE / EXACT
LOCATION / SPECIFIC EDIT), anti-wandering guidance (">5 files → re-read the issue"),
and a "act decisively" reminder for Haiku.

Evidence: LimitsExceeded edges down further to 36/251; step profile essentially
unchanged (mean 22.0). The gain is real but small (+1.0 = ~2–3 issues on 2 rollouts),
within the range where noise matters. It became frontier because it did not regress
and slightly improved — but it did **not** reproduce the large per-SUMMARY target of
54–58%. The understanding checkpoint's benefit over pure early-submit is marginal for
this student.

---

## 3. What Hurts

### iter5 — run_failing_test_before_submit (−14.34, 49.60 → 35.26) — worst regression
Added "run the specific failing test before submit; if it fails, debug once." This
**re-introduced exactly the behavior iter1 removed**:
- Test-running jumps back to **82% of trajectories** (vs 11% in iter2).
- LimitsExceeded rockets back to **133/251 (53%)**, mean steps 35.1, median 40 — the
  budget-exhaustion regime returns.
- Cost doubles $0.197 → $0.416.
- Issue-level vs iter2: regressed 69 issues, improved only 17.

Lesson: the "run one failing test" instruction does not stay lightweight — Haiku
cannot reliably extract/scope a single test, so it drifts into full-suite runs and
multi-round debugging, spending the budget it needs to submit. Functional
verification is not affordable at 40 steps for this student.

### iter3 — ignore_test_files (−5.38, 49.60 → 44.22) — solved a non-problem
Forbade editing test files and added an early hypothesis checkpoint. But test-file
editing was **already rare** by iter2: precise `sed -i` on a test path occurred in only
**8/251 (3%)** of iter2 trajectories (baseline was 30/251 = 12%, already killed by
iter1's early-submit). The forbidden-block and extra checkpoint added prompt bulk and
constraint without addressing a live failure mode, and slightly hurt. Removing the
anti-wandering checkpoint here was also a change with no upside.

### iter4 — verify_edits_before_submit (0.00, tie at 49.60) — neutral, not adopted
Replaced "submit immediately" with "re-read the edited lines (5–10), then submit."
Single-rollout resolved was actually highest (49.0%), but the authoritative 2-rollout
`avg_rate` tied iter2 exactly (49.60), LimitsExceeded 33/251, mean steps 22.4. A
cheap +1-step re-read neither helps nor hurts meaningfully — it stays within the
early-submit regime, unlike iter5's test-run. Safe but not a gain.

---

## 4. Agent Behavior Patterns

From `trajectories/` (commands parsed from assistant bash blocks):

1. **The clock-runner (baseline dominant mode).** Median steps = 40, i.e. the modal
   baseline episode never submits — it explores, edits, then loops on
   verification/testing until LimitsExceeded (165/251). 74% ran a test command.
   Resolved episodes were shorter (30.0 steps) than unresolved (38.9): long = losing.

2. **The submitter (iter1/2/4 mode).** Once told to submit after editing, median steps
   drop to 18–20 and Submitted rate hits ~85% (210–218/251). This single shift is what
   moved the frontier; the later refinements only nudge it.

3. **Test-file editing is a minor, self-correcting pattern.** Precise `sed -i` on a
   test path: baseline 12% → iter1 3% → iter2 3%. It fell out naturally once the agent
   stopped chasing test output; it did not need an explicit prohibition (iter3).

4. **Verification does not stay "lightweight."** iter5's "run one failing test"
   instruction produced 82% test-running and 53% LimitsExceeded — the agent cannot bound
   the cost of testing, so any test-before-submit instruction reverts to the baseline
   clock-runner regime. iter4's "re-read edited lines" stayed bounded (11% test-running)
   because it involves no test execution.

5. **Capability ceiling.** 81/251 issues (32%) are solved by **no** harness in any
   iteration — a hard floor set by Haiku's reasoning, not addressable by harness prompt.

---

## 5. Per-Issue Best Harness Distribution

Over 251 issues (2-rollout `per_issue` rates):

- **81 issues (32%) never solved** by any of the 6 harnesses → capability-bound.
- **170 issues solved by ≥1 harness.**

Uniquely-best harness (strictly higher than all others, no tie) — shows no single
harness owns the solvable set:

| harness | uniquely best |
|---|---:|
| baseline | 2 |
| early_submit_guidance | 5 |
| understand_then_edit_checkpoint | 6 |
| ignore_test_files | 6 |
| verify_edits_before_submit | 7 |
| run_failing_test_before_submit | 4 |

Even the two *rejected* harnesses (iter3, iter5) are uniquely best on 6 and 4 issues
respectively — issues where a bit of extra verification or a test run flips the answer.
No harness exceeds 7 unique wins; the wins are spread across all six. This scatter is
the core motivation for **per-instance adaptation**: the best fixed harness (iter2)
leaves points on the table that other configurations would capture on specific issues.

---

## 6. Guidelines and Key Takeaways (for a per-instance advisor)

Grounded in the data above:

1. **Default to submit-after-edit.** The +22.11 jump from iter1 and the
   65.7%→16% LimitsExceeded drop show the dominant failure is budget exhaustion, not
   wrong fixes. The advisor's baseline stance should be: locate → one targeted edit →
   submit. (iter0 vs iter1.)

2. **Do not prescribe test execution under a tight budget.** iter5 (−14.34, test-run
   82%, LimitsExceeded 53%) proves functional verification is unaffordable for Haiku at
   40 steps and does not stay scoped. Never advise "run the failing test then submit"
   as a global rule.

3. **A pre-edit understanding checkpoint is a safe, tiny positive.** iter2's +1.00 and
   its unchanged step profile mean "state root cause + exact location before editing"
   costs ~nothing and helps slightly. Worth keeping; do not expect large gains.

4. **Bounded, execution-free verification is neutral, not a lever.** iter4's re-read
   (tie, still 11% test-running) neither helps nor hurts. If an advisor adds a
   verification step, keep it execution-free (re-read edited lines only) so it stays in
   the submit regime — but don't expect a gain from it.

5. **Don't legislate against rare behaviors.** iter3 forbade test-file edits that were
   already at 3%; result −5.38. The advisor should target *live* failure modes for the
   specific instance, not add blanket prohibitions that only inflate the prompt.

6. **Reserve extra steps only where they pay off.** Since iter3/iter5 are still
   uniquely best on 6/4 issues, a *selective* per-instance decision — spend more
   verification budget only when the fix is genuinely uncertain — could beat the fixed
   iter2 frontier. This is precisely where an advisor adds value over a static harness.

7. **Respect the ceiling.** ~32% of issues are unsolvable at this capability/budget; an
   advisor should not spend its complexity budget there. Gains are only available on the
   ~68% solvable set, and mostly on the margin between the top three harnesses.
