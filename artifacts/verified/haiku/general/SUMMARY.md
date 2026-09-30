# Understand-then-Edit Checkpoint - Design Rationale

## Hypothesis
Haiku 4.5's weak reasoning causes (1) premature incorrect fixes from rushing without understanding deeply, and (2) wandering/looping when confused. Adding structured problem-decomposition guidance + explicit 'validate understanding before editing' checkpoint will reduce wrong-fix failures without reintroducing step-waste.

## Root Cause Analysis
Analysis of iter1 failures (48.6% pass rate) reveals two key failure modes:
1. **Premature fixes**: Agent makes edits based on superficial understanding, leading to incorrect fixes
2. **Exploration wandering**: Agent reads many files without clear focus, exhausting steps before finding the root cause

## Changes vs iter1 (early_submit_guidance)

### Inherited from iter1
- Keep immediate-submit guidance (no verification/testing)
- Maintain tight 40-step budget warnings
- Preserve all baseline command examples and format guidance

### New additions

#### 1. Mandatory Understanding Checkpoint (Phase 2)
Added explicit `<checkpoint>` block requiring agents to state before editing:
1. ROOT CAUSE: What is the bug/issue in one sentence?
2. EXACT LOCATION: Which file and lines need to change?
3. SPECIFIC EDIT: What exact change will you make?

This forces deliberate thinking before code changes, reducing premature incorrect fixes.

#### 2. Anti-Wandering Guidance
Added checkpoint in Phase 1: "If you've read >5 files without finding the bug location, STOP and re-read the issue description to refocus your search."

Prevents exploration loops that exhaust steps without finding the bug.

#### 3. Haiku-Specific Decisiveness Guidance
Added prominent reminder: "Think step-by-step but ACT decisively - one careful edit beats three hasty guesses."

Balances deliberation (understanding checkpoint) with action (don't over-analyze).

#### 4. Structured Workflow Phases
Organized workflow into clear phases with step budgets:
- Phase 1: Understand (5-10 steps)
- Phase 2: Validate Understanding (checkpoint)
- Phase 3: Execute Fix (1 step)
- Phase 4: Submit IMMEDIATELY (1 step)

Makes the workflow more concrete and measurable.

### Modified Templates

#### instance_template
- Added Phase 1-4 workflow structure
- Added mandatory `<checkpoint>` block
- Added anti-wandering guidance
- Added Haiku-specific decisiveness reminder at top
- Kept iter1's immediate-submit guidance

#### format_error_template
- Added reminder about understanding checkpoint (ROOT CAUSE, EXACT LOCATION, SPECIFIC EDIT)
- Added decisiveness reminder
- Kept iter1's immediate-submit reminder

## Implementation
Subclasses AgentConfig to override instance_template and format_error_template while preserving all other baseline templates. Self-contained with no external dependencies.

## Expected Impact
**Target**: 54-58% pass rate (current baseline: 48.6%)

By forcing structured problem understanding before editing, we expect to:
- Reduce incorrect-fix failures by 30-40% (more careful, targeted edits)
- Reduce wandering/looping failures by 20-30% (anti-wandering checkpoint)
- Maintain low LimitsExceeded rate from iter1 (keep immediate-submit guidance)

The understanding checkpoint adds ~1-2 thinking steps but should save 3-5 steps by preventing wrong-path exploration and incorrect edits that require recovery.
