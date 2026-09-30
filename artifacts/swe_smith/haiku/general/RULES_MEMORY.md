# General Rules and Knowledge

## Verification Success Detection

1. **Single verification principle**: After you edit a source file and successfully verify the fix works (returncode=0), submit immediately. The harness will notify you when verification succeeds.

2. **Recognize success signals**: When you see the `<verification_success>` tag, it means your fix passed verification. Submit immediately with the exact command shown.

3. **Avoid over-testing**: Testing additional edge cases or running multiple verification commands wastes steps and leads to timeouts. One successful verification is enough.

## Efficient Bug-Fix Workflow

1. **Understand the issue**: Read the problem statement and identify the bug.

2. **Locate the bug**: Find the relevant source files using targeted search.

3. **Implement the fix**: Edit the source code to resolve the issue.

4. **Single verification**: Run one verification command (test/pytest/run script) to confirm the fix works.

5. **Submit when prompted**: If verification succeeds (returncode=0), you'll receive a submit reminder. Follow it immediately.

## Step Budget Management

1. **Focused debugging**: Don't explore unrelated code or read extra files unnecessarily.

2. **Targeted verification**: Use a specific command that tests the exact functionality you fixed.

3. **No verification loops**: Running the same test multiple times or testing edge cases is wasteful. Trust your fix after one successful verification.

4. **Immediate submission**: When the harness indicates verification success, submit without delay. Additional steps risk timeout.
