# Structured Bugfix Workflow Harness

## Hypothesis
Adding explicit search-then-fix workflow guidance will reduce wasted exploration steps and improve fix accuracy on the first attempt, especially for the weak Gemini 3.7 Flash model.

## Changes vs Baseline

### 1. Explicit 3-Phase Workflow
- PHASE 1: LOCATE (max 3-4 commands) - Quick search using grep/git log/find
- PHASE 2: UNDERSTAND (max 5-7 commands) - Read specific files, identify exact lines
- PHASE 3: FIX (remaining) - Surgical edits with line numbers, verify with git diff

### 2. Anti-Pattern Warnings
- NEVER use broad sed patterns without line restrictions
- ALWAYS verify line numbers before editing
- Explicit GOOD vs BAD examples

### 3. Progress Guidance
- After 25 steps, prioritize completion over perfection
- Encourages trying different approaches when stuck

### 4. Structured Thinking
- Requires labeling workflow phase in each THOUGHT section

## Expected Impact
- Reduce average steps from 28.7 to ~20-23
- Improve pass rate from 51% to ~58-62% by reducing self-inflicted errors
- Higher first-attempt fix accuracy through better verification
