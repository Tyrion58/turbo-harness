# General SWE-bench Guidance

## Common Patterns

- Repository structure typically follows standard Python conventions (src/, tests/, setup.py, etc.)
- Test files often have naming patterns like test_*.py or *_test.py
- Configuration files may be in root directory or config/ subdirectories

## Best Practices

- Always read files before editing to understand context
- Use sed for precise line-based edits when possible
- Check return codes to verify command success

## Problem-Solving Strategy

- Focus search on files most likely related to the issue description
- If exploration becomes too broad (>5 files without finding bug location), refocus by re-reading the issue
- State your understanding clearly before making changes: root cause, exact location, specific edit
