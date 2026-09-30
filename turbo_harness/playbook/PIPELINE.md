# Playbook Advisor Pipeline

Trajectory-grounded contrastive learning for instance-specific harness adaptation.

## Pipeline Overview

```
Training data                                              Test time
─────────────                                              ─────────
evolution_summary.jsonl ─┐
trajectories/{harness}/  ─┤  Stage 1      Stage 2      Stage 3       Stage 4
artifacts/iter*/          ─┼→ Extract  →  Reflect  →  Curate   →   Advise
train_multi_repo.json  ──┘  (no LLM)    (Opus×50)   (Opus×1)    (Sonnet×N)

Output:                     experiences  reflections  playbook    patched
                            .jsonl       .jsonl       .json       harness
```

## Stage 1: Extract Contrastive Experience Pairs

**Module**: `turbo_harness/playbook/extract_experiences.py`

**Input**:

| Source | What it provides |
|---|---|
| `evolution_summary.jsonl` | Per-iteration per-issue pass rates (which harness passed which issue) |
| `trajectories/{harness}/{issue}.jsonl` | Full agent conversation (messages array) for each issue under each harness |
| `artifacts/iter*/harness.py` | Harness source code for computing diffs |
| `train_multi_repo.json` | Problem statements for each training issue |

**Process**: For each of the 50 training issues, loads all 6 trajectories (baseline + 5 iterations), classifies the issue as contrastive/all-pass/all-fail, and picks the best contrastive pair.

**Output**: `experiences.jsonl` — one record per issue (50 total):

```json
{
  "instance_id": "adrienverge__yamllint...",
  "contrast_type": "contrastive",
  "pass_harness": "verify_loop_enforcer",
  "fail_harness": "baseline",
  "pass_trajectory": "[system]\n...",
  "fail_trajectory": "[system]\n...",
  "pass_steps": 34,
  "fail_steps": 40,
  "fail_status": "LimitsExceeded",
  "harness_diff": "--- harness.py...",
  "problem_statement": "Please solve..."
}
```

- `contrast_type`: `"contrastive"` (mixed pass/fail across harnesses), `"all_pass"`, or `"all_fail"`
- `pass_trajectory` / `fail_trajectory`: FULL agent conversation, no summarization
- `harness_diff`: unified diff between the two harnesses in the contrastive pair

**Stats from our data**: 24 contrastive, 9 all-pass, 17 all-fail.

**LLM cost**: 0 (pure data loading and formatting).

## Stage 2: Contrastive Trajectory Reflection

**Module**: `turbo_harness/playbook/reflector.py`

**Input**: `experiences.jsonl` from Stage 1.

**Process**: For each experience record, sends the full trajectories to Opus and asks it to analyze the behavioral difference. Two prompt modes:

- **Contrastive** (24 issues): Shows BOTH full trajectories side-by-side + harness diff. Asks: "What did the agent do differently? Why did the harness difference matter for THIS issue?"
- **Uniform** (26 issues): Shows single trajectory. Asks: "Why does this issue consistently pass/fail regardless of harness?"

The reflector sees the **full reasoning trace** — the agent's complete chain-of-thought, commands, and outputs — not a summary.

**Output**: `reflections.jsonl` — one reflection per issue (50 total):

```json
{
  "instance_id": "adrienverge__yamllint...",
  "issue_characteristics": [
    "Bug requires implementing missing logic (AnchorToken handling)",
    "No existing test file for the anchors rule",
    "Agent can get trapped in test-writing rabbit holes"
  ],
  "strategy": "For issues where the fix involves implementing missing functionality and no existing tests cover it, the harness should nudge the agent to verify using lightweight ad-hoc validation scripts rather than writing formal test suites...",
  "rationale": "The verify_loop_enforcer's phase structure kept the PASS agent focused: it moved through REPRODUCE->LOCALIZE->FIX->VERIFY cleanly, using simple Python scripts to confirm...",
  "confidence": 0.92,
  "outcome": "pass"
}
```

Strategies are grounded in actual behavioral evidence (e.g., "the agent ran tests at step 12", "auto-retry loop forced re-edit at step 25").

**LLM cost**: ~50 Opus calls x ~35K input tokens = ~$26.

## Stage 3: LLM-Based Semantic Curation

**Module**: `turbo_harness/playbook/curator.py`

**Input**: `reflections.jsonl` from Stage 2.

**Process**: Sends ALL 50 reflections to a single Opus call. The curator:

1. Groups semantically similar strategies into clusters (e.g., "allocate more budget to verification" and "give more steps to test validation" merge into one)
2. For each cluster: produces one canonical strategy with merged conditions and aggregated helpful/harmful evidence counts
3. Identifies anti-patterns (strategies where harmful > helpful)
4. Filters vague/generic strategies

This replaces keyword-overlap dedup, which failed because reflector outputs use unique wording per issue.

**Output**: `playbook.json` — the final playbook (expected 5-15 entries):

```json
{
  "entries": [
    {
      "strategy_id": "ace_000",
      "condition": "Bug requires implementing missing or reverted functionality, No existing test files cover the buggy behavior",
      "strategy": "Use a phased workflow (REPRODUCE->LOCALIZE->FIX->VERIFY) with step budget awareness, and in the VERIFY phase, use lightweight ad-hoc validation scripts rather than formal test suites...",
      "evidence": {"helpful": 2, "harmful": 0},
      "confidence": 0.87,
      "strategy_type": "SCAFFOLD_MODIFICATION"
    }
  ],
  "general_notes": ["Soft nudges outperform hard gates..."],
  "anti_patterns": ["AVOID: ..."]
}
```

**LLM cost**: ~1 Opus call x ~15K tokens = ~$0.25.

## Stage 4: Advise (Test Time)

**Module**: `turbo_harness/playbook/memory_advisor.py` -> calls `turbo_harness/patch_advisor.py`

**Input**:

- `playbook.json` from Stage 3
- A test issue's problem statement
- The general harness source code (harness.py + SUMMARY.md + CONCEPT.md + RULES_MEMORY.md)

**Process**: Following the ACE design, the **entire playbook** is injected into the patch advisor's prompt — no retrieval, no keyword matching. The LLM (Sonnet/Opus) reads the harness code + issue description + full playbook and decides which strategies are relevant, then generates SEARCH/REPLACE edits to adapt the harness.

The playbook is small (~500 tokens for 5-15 entries), so it trivially fits in context alongside the harness code and issue description.

**Output**: A patched `harness.py` adapted for the specific test issue.

**LLM cost**: 1 Sonnet/Opus call per test issue.

## Cost Summary

| Stage | LLM calls | Approx. cost | When |
|---|---|---|---|
| Extract | 0 | $0 | Offline, once |
| Reflect | 50 (Opus) | ~$26 | Offline, once |
| Curate | 1 (Opus) | ~$0.25 | Offline, once |
| Advise | N (Sonnet) | ~$0.10/issue | Per test issue |

Total offline pipeline: ~$26 (one-time).
