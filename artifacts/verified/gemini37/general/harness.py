"""Test-after-edit forcing harness for SWE-bench tasks.

Hypothesis: Forcing one early edit is necessary but insufficient. Agents need continuous
edit-test cycle enforcement throughout the episode. After ANY edit, blocking exploration
until tests are run will prevent post-edit exploration drift and force validation-driven
iteration, increasing pass rate by 3-5 points.

This extends iter3's proven step-15 blocker with continuous edit→test cycle enforcement:
after any edit, allow max 5 exploration steps before blocking until tests are run.
"""

import re
from dataclasses import dataclass

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent


@dataclass
class TestAfterEditConfig(AgentConfig):
    """Config with standard workflow guidance (blocking enforced dynamically)."""

    system_template: str = """You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected. You will receive advice from time to time on your progress."""

    instance_template: str = """Please solve this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Edit the source code to resolve the issue
3. Verify your fix works by running your script again
4. Test edge cases to ensure your fix is robust
5. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```bash
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

{%- if system == "Darwin" -%}
<important>
You are on MacOS. For all the below examples, you need to use `sed -i ''` instead of `sed -i`.
</important>
{%- endif -%}

<important>
When editing specific lines with sed, ALWAYS verify the line numbers first by viewing the file.
Line numbers can shift after edits, so re-check before each edit operation.
</important>

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py

# BEST PRACTICE: View the exact lines before editing
sed -n '10,20p' filename.py  # View lines 10-20 first
sed -i '10,20s/old/new/g' filename.py  # Then edit those lines
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
```"""

    action_observation_template: str = """<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
{%- set elided_chars = output.output | length - 10000 -%}
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}"""

    format_error_template: str = """Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.
If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command.
Else, please format your response exactly as follows:

<response_example>
Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</response_example>

Note: In rare cases, if you need to reference a similar format in your command, you might have
to proceed in two steps, first writing TRIPLEBACKTICKSBASH, then replacing them with ```bash."""


class TestAfterEditAgent(DefaultAgent):
    """Agent that enforces edit→test cycles throughout the episode.

    Combines iter3's proven step-15 blocker with continuous test enforcement:
    - Before step 15: free exploration
    - After step 15 without edit: block exploration (iter3 behavior)
    - After ANY edit: enter test_pending mode
    - In test_pending mode: allow max 5 exploration steps, then block until tests run
    - After test execution: clear test_pending, allow new exploration
    """

    EXPLORATION_CUTOFF_STEP = 15
    MAX_EXPLORATION_AFTER_EDIT = 5

    def __init__(self, model: Model, env: Environment, step_limit: int = 0, cost_limit: float = 3.0):
        super().__init__(
            model=model,
            env=env,
            config_class=TestAfterEditConfig,
            step_limit=step_limit,
            cost_limit=cost_limit,
        )
        self.edit_made = False
        self.blocking_active = False
        self.test_pending = False
        self.exploration_count_after_edit = 0

    def _is_edit_command(self, command: str) -> bool:
        """Detect if command performs a file edit operation.

        Looks for:
        - sed -i (in-place edit)
        - cat << EOF (heredoc writes)
        - Python scripts with file writes (open/write mode)
        - echo/printf redirected to files (>)
        - tee command writing to files
        - patch/diff applications
        """
        cmd = command.strip()

        # sed in-place edit
        if re.search(r'sed\s+-i', cmd):
            return True

        # heredoc writes (cat << or cat <<')
        if re.search(r'cat\s*<<', cmd):
            return True

        # Redirect operators (>, >>)
        if re.search(r'>\s*[\w/.-]+', cmd):
            return True

        # Python inline scripts with write operations
        if 'python' in cmd and re.search(r'open\([^)]*["\']w|write\(', cmd):
            return True

        # tee command (writes to files)
        if re.search(r'\|\s*tee\s+', cmd):
            return True

        # patch/diff applications
        if re.search(r'\b(patch|git\s+apply)\b', cmd):
            return True

        # mv/cp to existing source files (conservative: only if writing to .py/.js/.java etc)
        if re.search(r'\b(mv|cp)\b.*\.(py|js|java|cpp|c|h|rb|go|rs|php|ts|tsx|jsx)$', cmd):
            return True

        return False

    def _is_test_command(self, command: str) -> bool:
        """Detect if command runs tests.

        Looks for:
        - pytest
        - python -m pytest
        - python -m unittest
        - python -m test
        - ./manage.py test
        - npm test / yarn test
        - cargo test
        - go test
        - Other common test runners
        """
        cmd = command.strip()

        # Pytest
        if re.search(r'\bpytest\b', cmd):
            return True

        # Python test modules
        if re.search(r'python.*-m\s+(pytest|unittest|test)', cmd):
            return True

        # Django tests
        if re.search(r'manage\.py\s+test', cmd):
            return True

        # Node.js tests
        if re.search(r'\b(npm|yarn|pnpm)\s+test\b', cmd):
            return True

        # Rust tests
        if re.search(r'\bcargo\s+test\b', cmd):
            return True

        # Go tests
        if re.search(r'\bgo\s+test\b', cmd):
            return True

        # Ruby tests
        if re.search(r'\b(rspec|rake\s+test)\b', cmd):
            return True

        # Java tests
        if re.search(r'\b(mvn|gradle)\s+test\b', cmd):
            return True

        return False

    def _is_exploration_command(self, command: str) -> bool:
        """Detect if command is a read-only exploration operation.

        These are allowed during test_pending mode (up to limit).
        """
        cmd = command.strip()

        # Read-only commands
        exploration_patterns = [
            r'\bgrep\b',
            r'\bfind\b',
            r'\bls\b',
            r'\bcat\b(?!.*<<)',  # cat but not heredoc
            r'\bhead\b',
            r'\btail\b',
            r'\bless\b',
            r'\bmore\b',
            r'\bsed\s+-n',  # sed with -n (no edit)
            r'\bnl\b',
            r'\bwc\b',
            r'\bfile\b',
            r'\bwhich\b',
            r'\bwhereis\b',
            r'\btree\b',
        ]

        for pattern in exploration_patterns:
            if re.search(pattern, cmd):
                return True

        return False

    def get_observation(self, response: dict) -> dict:
        """Execute action with edit→test cycle enforcement."""
        try:
            action = self.parse_action(response)
            command = action.get("action", "")

            current_step = self.model.n_calls

            # Check command type
            is_edit = self._is_edit_command(command)
            is_test = self._is_test_command(command)
            is_exploration = self._is_exploration_command(command)

            # Handle edit commands
            if is_edit:
                self.edit_made = True
                self.blocking_active = False  # Deactivate step-15 blocking once edit is made
                self.test_pending = True  # Enter test_pending mode
                self.exploration_count_after_edit = 0  # Reset exploration counter

            # Handle test commands
            elif is_test:
                self.test_pending = False  # Clear test_pending mode
                self.exploration_count_after_edit = 0  # Reset exploration counter

            # Handle exploration in test_pending mode
            elif self.test_pending and is_exploration:
                self.exploration_count_after_edit += 1

            # Activate step-15 blocking if we've passed the cutoff and no edit made yet
            if current_step > self.EXPLORATION_CUTOFF_STEP and not self.edit_made:
                self.blocking_active = True

            # BLOCKING LOGIC 1: Step-15 blocker (iter3 behavior)
            if self.blocking_active and not is_edit:
                blocking_message = """<exploration_budget_exhausted>
EXPLORATION BUDGET EXHAUSTED. You have taken more than 15 steps without making any file edits.

Your next command MUST be a file edit operation. Acceptable edit commands include:
- sed -i (in-place file editing)
- cat <<EOF (heredoc file creation/overwriting)
- Python scripts that write to files
- Redirecting output to files (>)

Commands that will be REJECTED:
- grep, find, ls (read-only exploration)
- cat, head, tail (file reading)
- Any other read-only operations

You must make your best hypothesis about the fix and attempt an edit. Test failures will provide more signal than continued exploration. Form a hypothesis and edit a file NOW.
</exploration_budget_exhausted>"""

                self.add_message("user", blocking_message)
                return {"output": "", "returncode": 0}

            # BLOCKING LOGIC 2: Test-after-edit enforcement (NEW)
            if (self.test_pending
                and self.exploration_count_after_edit > self.MAX_EXPLORATION_AFTER_EDIT
                and not is_edit
                and not is_test):

                blocking_message = """<edit_validation_required>
EDIT VALIDATION REQUIRED. You made an edit but have not validated it with tests yet.

You have explored for {count} steps after your last edit. Before continuing further exploration, you MUST run tests to verify your changes.

Acceptable test commands include:
- pytest (or pytest <specific_file>)
- python -m pytest
- python -m unittest
- python -m test
- ./manage.py test
- npm test, yarn test
- cargo test, go test
- Any other test runner for this repository

Commands that will be REJECTED until you run tests:
- grep, find, ls (continued exploration)
- cat, head, tail (reading more files)
- Any other exploration commands

Run tests NOW to validate your edit. Test results will guide your next iteration.
</edit_validation_required>""".format(count=self.exploration_count_after_edit)

                self.add_message("user", blocking_message)
                return {"output": "", "returncode": 0}

        except Exception:
            # If parsing fails, let normal flow handle it
            pass

        # Normal flow: execute and observe
        return super().get_observation(response)


def build_agent(model: Model, env: Environment, step_limit: int = 0, cost_limit: float = 3.0) -> TestAfterEditAgent:
    """Build a test-after-edit forcing agent.

    Args:
        model: Language model to use for generating actions
        env: Environment to execute actions in
        step_limit: Maximum number of steps (0 = unlimited)
        cost_limit: Maximum cost in dollars (0 = unlimited)

    Returns:
        Configured agent instance with continuous edit→test cycle enforcement
    """
    return TestAfterEditAgent(model=model, env=env, step_limit=step_limit, cost_limit=cost_limit)
