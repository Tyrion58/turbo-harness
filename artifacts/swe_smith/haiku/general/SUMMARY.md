# Verify Once Submit Harness

## Design Rationale

### Problem
Baseline Haiku agents successfully fix bugs and verify them, but continue testing edge cases, re-running tests, or exploring alternative approaches until they hit timeout (28% LimitsExceeded rate). The issue is not verification failure, but lack of a concrete stopping signal after the first successful verification.

### Hypothesis
Haiku agents need a strong, explicit signal to submit immediately after successful verification. By detecting the first successful verification (returncode=0 on test/pytest/run commands after a source code edit) and injecting a prominent submit reminder, we can prevent over-verification spirals and reduce timeouts while maintaining fix quality.

### Changes vs Baseline

**Mechanism:**
- Overrides `get_observation()` to track agent state and inject submit reminders
- Tracks three state variables:
  1. `has_edited_source`: Set when agent edits a .py file (sed/cat/write, excluding test_*/tests/)
  2. `step_count`: Increments each step to enforce ≥8 step minimum before triggering
  3. `verification_triggered`: Ensures submit reminder is shown only once

**Detection logic:**
- Source edit: Matches sed/cat/write patterns on *.py files not in test directories
- Verification: Matches commands containing test/pytest/python.*test/run keywords
- Success: Checks returncode=0 in command output

**Injection:**
When all conditions are met (source edited + verification command + returncode=0 + ≥8 steps), appends:
```
<verification_success>
Your fix passed verification. You must submit NOW to avoid timeout.

Next command MUST be:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Do not run additional tests or checks. Submit immediately.
</verification_success>
```

**No config changes:**
- Uses standard templates (same as baseline/iter3)
- Pure behavioral constraint via observation injection
- Self-contained, minimal implementation

### Changes vs Prior Iterations

**vs Iter1 (test_driven_debugging):**
- Iter1: Front-loads test discovery before agent starts (proactive)
- Iter4: Reactive trigger after successful verification (intervention at decision point)
- Complementary: Iter1 helps find bugs faster, iter4 prevents post-fix over-testing

**vs Iter2 (minimal_fix_fast_submit):**
- Iter2: Template-level guidance ("submit immediately after verification")
- Iter4: Runtime intervention with concrete signal (stronger forcing function)
- Iter4 advantage: Explicit, unmissable prompt at the exact moment submission is appropriate

**vs Iter3 (sliding_context_window):**
- Iter3: Maintains focused context via message window (memory management)
- Iter4: Detects specific event and injects directive (behavioral trigger)
- Complementary: Iter3 prevents context bloat, iter4 prevents verification bloat

### Expected Outcome

- **Pass rate**: 52-54% (+3-5 pts vs baseline 49%)
- **Timeout rate**: ~10% (vs baseline 28%)
- **Mechanism**: Preserves fix quality while preventing unproductive verification loops
- **Avg steps**: 18-22 (reduced from baseline 25-30)
- **Key metric**: Ratio of verified-and-submitted to verified-and-timeout improves significantly
