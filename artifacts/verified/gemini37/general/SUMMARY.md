# Test-After-Edit Forcing Harness (iter4)

## Design Rationale

### Problem with iter3
Iter3's step-15 blocker successfully forced an early first edit (+9.2% improvement over baseline), but failed to maintain validation discipline afterward. Observed pattern in iter3 failures:
1. Agent makes forced edit at step ~15
2. Agent then explores freely for 15-20+ more steps
3. Either timeout occurs or agent submits without proper test validation
4. Edit was often correct but untested, or edit had bugs that tests would have revealed

The step-15 blocker proved the value of forcing action, but was a one-time intervention. After the first edit, agents reverted to unstructured exploration-heavy patterns.

### Solution: Continuous Edit→Test Cycle Enforcement
This harness extends the proven blocking mechanism from iter3 to enforce discipline throughout the entire episode:

**Retained from iter3:**
- Step-15 cutoff for first edit (proven effective +9.2%)

**New enforcement (test_after_edit_forcing):**
- After ANY edit is detected, activate `test_pending` mode
- In `test_pending` mode: allow up to 5 exploration steps (grep/cat/find/sed -n)
- After 5 exploration steps: hard block further exploration with message
- Blocking message: "EDIT VALIDATION REQUIRED. You made an edit but have not validated it with tests. Run pytest/test commands to verify your changes before further exploration."
- Test command detection: recognize pytest, python -m test, ./manage.py test, npm test, cargo test, etc.
- When test executed: clear `test_pending` flag, reset exploration counter
- Next edit: re-enter `test_pending` mode, repeat cycle

### Hypothesis
Forcing continuous edit→test cycles will:
1. Prevent post-edit exploration drift (the "edit then explore 20 more steps" pattern)
2. Make test failures guide iteration (rather than speculation)
3. Ensure edits are validated before timeout
4. Increase pass rate by 3-5 points (40-42% vs iter3's 37.1%)

The 5-step exploration budget allows quick verification reads (checking the edit was applied, viewing adjacent context) but blocks extended exploration expeditions.

### Implementation Details

**Three blocking modes:**
1. **Pre-edit blocking (step 15)**: Same as iter3 - force first edit
2. **Test-pending exploration limit**: After edit, allow 5 exploration commands
3. **Test-pending hard block**: After 5 explorations, block until test runs

**Command classification:**
- **Edit commands**: sed -i, cat <<EOF, >, python scripts with writes, tee, patch
- **Test commands**: pytest, python -m pytest/unittest, ./manage.py test, npm/yarn test, cargo test, go test, rspec, mvn/gradle test
- **Exploration commands**: grep, find, ls, cat (non-heredoc), head, tail, sed -n, nl, wc

**State machine:**
```
Initial → (step 15 passed, no edit) → Pre-edit blocking
       → (edit detected) → Test pending (counter = 0)
Test pending → (exploration command) → Increment counter
            → (counter > 5, not edit, not test) → Test validation blocking
            → (test command) → Clear test_pending, reset counter
            → (edit command) → Reset test_pending mode
```

### Expected Outcome
Target: 40-42% pass rate (+3-5 points from iter3's 37.1%)

The mechanism creates a forcing function for validation-driven iteration rather than exploration-driven timeout. Each edit must be validated before the agent can move to the next exploration phase.
