# Iteration 6: Tight Loop Breaker with Few-Shot

## Hypothesis

**Root cause**: iter5's few-shot (62%) solved most multi-step tasks, but 5+ tasks are stuck in EXTREMELY tight loops — repeating the same observation 35-47 times consecutively, burning all 50 steps without progress.

**Evidence from iter5 failed trajectories:**

**Task 5** ("look at pencil under desklamp"):
- Agent went to desk, saw pencil
- Got stuck in tight loop: repeatedly doing "look" action
- Saw "You are facing the desk 1. Next to it, you see nothing." **47 consecutive times**
- Never took pencil, never found desklamp
- Exhausted all 50 steps in the loop

**Task 51** ("put a cool apple in sidetable"):
- Agent searched for apple, didn't find it
- Tried "go to drawer 2" but env canonicalized it to cabinet 2 (drawer 2 doesn't exist)
- Got stuck alternating: "go to drawer 2" → cabinet 2 → "go to drawer 1" → "go to drawer 2" → cabinet 2 ...
- **30+ consecutive observations** of this loop
- Exhausted all 50 steps

**Tasks 54, 61** (similar to 51): 45 consecutive identical observations each
**Tasks 49, 69**: 35-44 consecutive identical observations

**Total**: At least 5 tasks with severe loops (35-47 consecutive repeats).

### Why didn't iter3's loop detection help?

iter3 (loop breaker) REGRESSED -2% (49% → 47%) because:
1. **Triggered too easily**: "3+ in last 5 observations" → false positives on valid exploration
2. **Generic nudge**: "Try a different action" → not actionable, just noise
3. **No few-shot foundation**: Applied to baseline without fixing the underlying multi-step understanding

### Why iter5 is strong but incomplete

iter5's few-shot examples:
- ✅ Fixed multi-step clean/heat/cool understanding (16%/15%/7% → much better)
- ✅ Improved from 49% → 62% (+13%)
- ❌ Has NO loop detection at all
- ❌ Leaves 5-10 tasks stuck in severe loops unsolved

## New Approach

**Combine the best of both**: Start from iter5's few-shot foundation + add MUCH TIGHTER loop detection.

**Key differences from iter3:**
1. **Tighter threshold**: Trigger only on 5+ CONSECUTIVE identical observations (not "3+ in last 5")
   - Avoids false positives on valid exploration
   - Targets only severe loops like Tasks 5, 51, 54, 61, 69
2. **Actionable nudge**: "You've seen this exact observation 5+ times in a row. You're stuck in a loop. Try exploring a completely different location or trying a fundamentally different action."
   - More specific than "try different action"
   - Explicitly calls out the loop
3. **Rate limiting**: Max 1 nudge per 10 steps
   - Prevents context pollution
   - Gives agent space to act on the nudge
4. **Few-shot foundation**: Keeps iter5's clean/heat/cool examples
   - Maintains the 62% baseline
   - Loop detection is an ADD-ON, not a replacement

## Changes from iter5 (Frontier)

### 1. Loop Detection Logic
- Track last 10 observations in a rolling buffer
- Detect "tight loop": when last 5 observations are ALL identical
- This catches severe loops (like 47 consecutive repeats) but NOT normal exploration

### 2. Actionable Loop-Breaking Nudge
- WHEN: Tight loop detected AND ≥10 steps since last nudge
- WHERE: Append to tool feedback (same message, after AVAILABLE ACTIONS)
- WHAT: "[Note: You've seen this exact observation 5+ times in a row. You're stuck in a loop. Try exploring a completely different location or trying a fundamentally different action.]"

### 3. Rate Limiting
- Track `steps_since_last_nudge`
- Only inject nudge if ≥10 steps passed since last one
- Prevents spamming the context with repeated warnings

### 4. Keep ALL of iter5's Few-Shot
- Still load few-shot examples for clean/heat/cool categories
- Still zero-shot for simple put/examine tasks
- Loop detection is purely additive

## Expected Impact

**Target improvements:**
- **Tasks 5, 49, 54, 61, 69** (currently looping 35-47 times): Break the loop with actionable nudge → **+5 tasks → 67%**
- **All other tasks**: Maintain iter5's 62% (few-shot unchanged, tight threshold avoids false positives)

**Why this won't regress like iter3:**
- Tighter threshold (5 consecutive vs 3 in 5) → fewer false positives
- Keeps iter5's few-shot foundation → maintains gains
- Rate-limited nudges → no context pollution
- More actionable guidance → when it triggers, it helps

**Validation on specific failures:**
- Task 5: After 5th consecutive "You are facing the desk 1" → nudge triggers → agent tries "go to sidetable 1" or "take pencil 1" → breaks loop
- Task 51: After 5th consecutive drawer 2 → cabinet 2 loop → nudge triggers → agent tries "go to diningtable 1" or another location → breaks loop

**Overall expected**: 62% → 67% (+5 tasks from loop-breaking, +0 regression)

## What's Kept General

- No hard-coded object/receptacle ids
- No task-specific logic
- Loop detection works across all task categories
- Observation comparison is normalized (strip + lowercase) to handle minor variations
- Works on any task instance, any environment layout
