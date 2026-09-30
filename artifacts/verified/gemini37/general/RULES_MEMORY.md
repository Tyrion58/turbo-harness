# General Repository Rules

## Workflow Discipline

Software engineering requires systematic validation cycles. After making changes to code:
1. Run tests immediately to verify the change works as intended
2. Use test failures as specific guidance for what to fix next
3. Avoid extended exploration after edits - tests provide better signal than speculation

## Edit-Test Iteration Pattern

The most effective debugging pattern is:
1. Form a hypothesis about the fix
2. Make a targeted edit
3. Run tests to validate
4. Use test results to refine the next hypothesis
5. Repeat

Extended exploration after edits (reading more files, searching for more patterns) often leads to drift away from the immediate validation task.

## Test Command Recognition

Common test runners across different ecosystems:
- Python: `pytest`, `python -m pytest`, `python -m unittest`, `./manage.py test`
- JavaScript: `npm test`, `yarn test`, `pnpm test`
- Rust: `cargo test`
- Go: `go test`
- Ruby: `rspec`, `rake test`
- Java: `mvn test`, `gradle test`

## Exploration vs Validation

Exploration commands (grep, find, cat, ls) are valuable for understanding the codebase but should not dominate after edits are made. Once you've edited code, the fastest path to a solution is through test-driven iteration, not further exploration.

## File Modification Detection

Edit operations that modify the repository:
- `sed -i` (in-place file editing)
- `cat <<EOF > file` (heredoc writes)
- Output redirection (`>`, `>>`)
- Python scripts with file writes
- `tee` command writing to files
- `patch` or `git apply` commands

Read-only operations (safe for exploration):
- `grep`, `find`, `ls`
- `cat`, `head`, `tail`, `less`, `more`
- `sed -n` (print mode, no editing)
- `nl`, `wc`, `file`
