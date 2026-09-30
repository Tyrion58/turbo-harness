# CONCEPT — Multi-Repo Meta-Harness Evolution (Haiku student, Sonnet-4.5 proposer)

Domain: 50 SWE-bench-style issues across 25 Python repos (2 issues/repo), each scored with
n=2 rollouts (so per-issue rates ∈ {0.0, 0.5, 1.0}). Student model: Haiku, 40-step budget.
Frontier metric: `avg_rate` over the 50 issues. Baseline (default harness) = **0.49**.

Source data: `evolution_summary.jsonl` (rates/deltas/hypotheses) and
`trajectories/<agent>/*.jsonl` (single-rollout transcripts, used for behavior analysis).

---

## 1. Evolution Scorecard

| Iter | Candidate | Train rate | Δ vs frontier | Became frontier? |
|-----:|-----------|:----------:|:-------------:|:----------------:|
| 0 | baseline (default) | 0.49 | — | ✔ (start) |
| 1 | test_driven_debugging | 0.33 | −0.16 | ✘ |
| 2 | minimal_fix_fast_submit | 0.48 | −0.01 | ✘ |
| 3 | sliding_context_window | 0.48 | −0.01 | ✘ |
| 4 | **verify_once_submit** | **0.52** | **+0.03** | **✔ (new frontier)** |
| 5 | explore_before_edit | 0.32 | −0.20 | ✘ |

Only **iter4 (verify_once_submit)** improved the frontier (0.49 → 0.52). Iter5 was measured
against the new 0.52 frontier, hence its −0.20 delta. Two candidates (iter2, iter3) were
statistically flat (−0.01); two (iter1, iter5) regressed hard.

Trajectory-level status distribution (single rollout, n=50) corroborates the ranking:

| Agent | Submitted | LimitsExceeded | avg steps | resolved (1-roll) |
|-------|:---------:|:--------------:|:---------:|:-----------------:|
| baseline | 36 | 14 (28%) | 32.4 | 23 |
| test_driven_debugging | 47 | 3 | 15.0 | 19 |
| minimal_fix_fast_submit | 44 | 6 | 20.4 | 26 |
| sliding_context_window | 32 | 18 (36%) | 30.6 | 19 |
| **verify_once_submit** | 43 | 7 | 22.1 | **27** |
| explore_before_edit | 20 | **30 (60%)** | 37.1 | 12 |

---

## 2. What Works

### iter4 — verify_once_submit (+0.03, 0.49 → 0.52, the only frontier win)
**Change:** Override `get_observation()`. After the agent edits a source `.py` file (excluding
`test_*`/`tests/`), if the next command contains a verification keyword
(`test`/`pytest`/`python.*test`/`run`) **and** returns `returncode=0`, and ≥8 steps have
elapsed, inject a one-time strong directive: *"Your fix passed verification. You must submit
NOW … Next command MUST be: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"*. Pure behavioral
constraint, no config changes.

**Why it worked:** It targets the dominant baseline failure mode — **over-verification
timeouts (28% LimitsExceeded)** — without disturbing the exploration/fix phase. The injection
fired on **45/50** issues; the agent was resolved on **25** of those. Timeout rate dropped
28% → 14% and avg steps for submitted runs fell 29.5 → 19.2, while fix quality was preserved.

**Per-issue evidence (9 gains, +5.0 total):**
- #6 bottlepy__bottle `0.0 → 1.0`, #28 pallets__jinja `0.5 → 1.0`, #12 dask__dask `0.5 → 1.0`,
  #18 getmoto__moto `0.5 → 1.0`, #41 jsonschema `0.5 → 1.0`, #3 arrow-py `0.5 → 1.0`
- Partial recoveries: #5 gunicorn `0.0 → 0.5`, #21 faker `0.0 → 0.5`, #39 pygments `0.0 → 0.5`
- Cost: only 6 issues regressed (−3.5 total), e.g. #31 pandas `1.0 → 0.0`, #24 oauthlib
  `0.5 → 0.0` — net +5.0/−3.5 = **+1.5 raw** over 50 issues ≈ +0.03 rate. A **narrow** win.

### iter2 — minimal_fix_fast_submit (−0.01, effectively flat but directionally positive)
**Change:** Config-only. Added "make the smallest change that fixes the issue", removed the
"test edge cases to ensure your fix is robust" workflow step, and told the agent to submit
immediately after ONE verification.

**Why it partly worked:** Same over-verification target as iter4, via prompt rather than
runtime trigger. It produced the **same core gains** as iter4 — #3, #6, #12, #28, #39, #41 all
improved identically — and cut timeouts to 6 (avg steps 20.4). But the prompt is a weaker
forcing function than iter4's runtime injection, so it also lost ground elsewhere and netted
−0.01. This convergence (config guidance vs runtime injection hitting the *same* issues)
confirms the over-verification mechanism is real and that **a stronger stop-signal wins**.

### iter3 — sliding_context_window (−0.01, flat)
**Change:** Cap history at 38 messages (keep system+task, drop oldest exploration). Intended to
cut per-step processing and timeout. Result was flat: it did **not** reduce timeouts (36%
LimitsExceeded, *worse* than baseline's 28%) — dropping old exploration context did not stop
the agent from looping on verification, so the mechanism missed the actual cause.

---

## 3. What Hurts

### iter5 — explore_before_edit (−0.20, the worst regressor)
**Change:** Override `execute_action()` to **block all edits** until the agent has read ≥4
distinct non-test source files; on a blocked edit, return
`"Cannot edit yet. First explore the codebase by reading relevant source files…"`.

**Why it failed:** The hard gate starved the step budget. LimitsExceeded exploded to **30/50
(60%)** and avg steps rose to 37.1 — the agent burned its budget satisfying the read-quota
instead of fixing. The block message was observed on essentially every task early in the run.
**14 issues regressed (−9.0 total)**, wiping out tasks the baseline solved cleanly:
- #17 hydra `1.0 → 0.0`, #19 getmoto `1.0 → 0.0`, #22 marshmallow `1.0 → 0.0`,
  #31 pandas `1.0 → 0.0`; plus #30 pandas, #32 paramiko, #34 prettytable, #48 tornado each
  `→ 0.5`. Forcing exploration on already-easy bugs is pure overhead.

### iter1 — test_driven_debugging (−0.16, second-worst)
**Change:** Front-load a pre-exploration step: discover `test_*.py`/`*_test.py`, run pytest,
inject the failure output before the agent loop.

**Why it failed:** It over-corrected on steps (avg 15.0, only 3 timeouts) but the injected
test-discovery context was noisy/misleading and pushed the agent to submit too early on the
wrong fix. **18 issues regressed (−11.0 total)**, the most of any candidate:
- #7 bottlepy `1.0 → 0.0`, #31 pandas `1.0 → 0.0`, #34 prettytable `1.0 → 0.0`,
  #45 sqlfluff `1.0 → 0.0`; and #4, #15, #19, #22, #32, #43, #48 each dropped by 0.5.
The lesson mirrors iter5: **rigid front-loaded scaffolding that fires on every task destroys
baseline-solved cases.** Blindly running the repo's own test suite ≠ understanding the bug.

---

## 4. Agent Behavior Patterns

Patterns from raw trajectories (`trajectories/<agent>/*.jsonl`):

- **Over-verification spiral (the core baseline failure).** In the baseline yamllint timeout
  (#0, 40/40 steps, LimitsExceeded) the last turns are all re-runs:
  `pytest ./tests/rules/test_anchors.py -v` → `pytest tests/ -v | tail -50` → manual repro in
  `/tmp` → `cat -n anchors.py`. The agent had a candidate fix but kept re-testing instead of
  submitting. Baseline submitted-run avg = 29.5 steps vs verify_once = 19.2 — the ~10 saved
  steps are exactly these redundant re-verifications.

- **Late, spontaneous submit.** When the baseline *does* stop (e.g. pandas #31 trajectory), the
  final commands are cleanup + `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` at step 27 — the
  agent *can* stop, it just lacks a trigger. iter4's injection supplies that trigger; it fired
  on 45/50 tasks.

- **Quota-starvation under hard gates (iter5).** The `"Cannot edit yet…"` message appears
  repeatedly early in nearly every iter5 transcript; the agent spends turns reading files to
  clear the ≥4-file gate, then times out (60% LimitsExceeded) before landing an edit.

- **Premature submit under front-loaded context (iter1).** With pytest output pre-injected,
  iter1 submits fast (avg 15 steps, 47/50 Submitted) but resolves fewer (19) — it acts on the
  injected signal without independent root-causing, converting baseline wins into wrong-fix
  submissions.

- **Context trimming ≠ loop breaking (iter3).** Despite a 38-message window, iter3 still hit
  36% LimitsExceeded — trimming old exploration does not remove the *incentive* to keep
  verifying, so the loop persists.

---

## 5. Per-Issue Best-Harness Distribution

Best-harness counts across the 6 candidates (a tie credits every candidate at the max; 50
issues). "Unique wins" = issues where exactly one candidate strictly beat all others.

| Harness | Issues where it ties/holds the max | Unique wins |
|---------|:---------------------------------:|:-----------:|
| iter4 verify_once_submit | 42 | 1 |
| baseline | 39 | 2 |
| iter3 sliding_context_window | 38 | 1 |
| iter2 minimal_fix_fast_submit | 37 | 0 |
| iter1 test_driven_debugging | 29 | 1 |
| iter5 explore_before_edit | 26 | 0 |

Key observations:
- **No single harness dominates.** Even the frontier winner (iter4) is not best-or-tied on 8
  issues, and baseline still uniquely wins 2 (e.g. #31 pandas, which iter1/iter4/iter5 all
  break). This heterogeneity is the motivation for per-instance adaptation.
- **15 issues are unsolved by every harness** (max=0): #0 yamllint, #2 arrow, #8/#9 chardet,
  #10/#11 cookiecutter, #14 starlette, #16 hydra, #23 marshmallow, #26 click, #29 jinja,
  #33 paramiko, #35 prettytable, #42 scrapy, #46 dspy. These are **capability/knowledge-bound**
  — no harness change recovered them; an advisor should not spend budget forcing them.
- **4 issues are solved by every harness** (min=1): #1 yamllint, #13 dask, #47 dspy, #49
  tornado. Robust; leave them alone.
- The **contested middle** (~31 issues) is where harness choice matters — this is the
  addressable surface for a per-instance advisor.

---

## 6. Guidelines and Key Takeaways

For a per-instance advisor over this Haiku student, grounded in the data above:

1. **Default to injecting a stop-signal after the first green verification, not before.**
   iter4's reactive post-verification submit reminder was the *only* frontier gain (+0.03) and
   cut timeouts 28% → 14%. Trigger only *after* a source edit + `returncode=0` + ≥8 steps —
   never earlier.

2. **Never impose hard, always-on gates.** Both blanket scaffolds destroyed baseline wins:
   iter5's edit-block (−0.20, 60% timeouts, #17/#19/#22/#31 all `1.0 → 0.0`) and iter1's
   forced test-run front-load (−0.16, 18 regressions). Any intervention must be *conditional*
   and *skippable* on tasks that are already going well.

3. **Prefer the runtime injection over prompt-only nudges when the goal is to stop looping.**
   iter2 (config nudge) and iter4 (runtime injection) hit the *same* winning issues
   (#3/#6/#12/#28/#39/#41), but only iter4's stronger forcing function crossed into positive
   net (+0.03 vs −0.01).

4. **Do not front-load or blindly trust the repo's own test suite.** iter1 submitted fast but
   resolved fewer (19 vs baseline 23) by acting on injected pytest output without
   root-causing. Advise the agent to *understand* the bug before editing, but do not *mandate*
   a fixed exploration quota (iter5's ≥4-file rule was the failure mode of over-mandating).

5. **Budget triage: skip the 15 always-fail and 4 always-solve issues.** Spend intervention
   budget on the contested middle (~31 issues). On always-fail issues (#0, #2, #8–11, #14,
   #16, #23, #26, #29, #33, #35, #42, #46) extra steps yield nothing; on always-solve issues
   (#1, #13, #47, #49) intervention only risks regression (iter1/iter5 broke #48 tornado, a
   near-robust task).

6. **Watch for the specific over-verification signature.** Repeated `pytest …` / broad
   `pytest tests/ -v` re-runs after an edit already returned green (seen in baseline #0) are
   the highest-yield place to inject "submit now." Redundant re-verification, not fix failure,
   is the primary lost-budget mode (baseline 29.5 → verify_once 19.2 submitted-run steps).

7. **Aim for narrow, reversible edits to the harness.** The winning delta was small (net
   +1.5 raw issues) and easily erased by side-effects (iter4 still regressed 6 issues). A
   per-instance advisor should make the *minimum* intervention that addresses the observed
   failure mode and avoid changes that touch tasks not exhibiting it.
