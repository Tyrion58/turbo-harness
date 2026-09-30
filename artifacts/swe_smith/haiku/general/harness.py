"""Verify-once-submit harness for SWE-smith bug-fix tasks.

Detects successful verification and injects a strong submit reminder to prevent
over-verification spirals that lead to timeouts.
"""

import re
from dataclasses import dataclass
from minisweagent.agents.default import DefaultAgent, AgentConfig


@dataclass
class VerifyOnceSubmitConfig(AgentConfig):
    """Config with standard SWE workflow."""

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
3. Verify your fix works
4. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
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


class VerifyOnceSubmitAgent(DefaultAgent):
    """Agent that detects successful verification and triggers immediate submission."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_edited_source = False
        self.verification_triggered = False
        self.step_count = 0
        self.last_action = ""

    def get_observation(self, response: dict) -> dict:
        """Execute action and inject submit reminder after successful verification.

        Tracks:
        1. Source file edits (sed/cat/write to *.py not in test_*/tests/)
        2. Verification commands (test/pytest/python.*test/run)
        3. Successful execution (returncode=0)

        After source edit + verification success, injects strong submit prompt.
        """
        # Parse and execute the action
        action_dict = self.parse_action(response)
        self.last_action = action_dict.get("action", "")
        self.step_count += 1

        # Check if this is a source file edit
        if self._is_source_edit(self.last_action):
            self.has_edited_source = True

        # Execute the action
        output = self.execute_action(action_dict)

        # Render the standard observation
        observation = self.render_template(self.config.action_observation_template, output=output)

        # Check for successful verification after source edit
        if (
            not self.verification_triggered
            and self.has_edited_source
            and self.step_count >= 8
            and self._is_verification_command(self.last_action)
            and output.get("returncode") == 0
        ):
            # Inject strong submit reminder
            submit_reminder = """

<verification_success>
Your fix passed verification. You must submit NOW to avoid timeout.

Next command MUST be:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Do not run additional tests or checks. Submit immediately.
</verification_success>"""
            observation += submit_reminder
            self.verification_triggered = True

        # Add observation to messages
        self.add_message("user", observation)
        return output

    def _is_source_edit(self, action: str) -> bool:
        """Detect if action edits a source file.

        Matches:
        - sed ... *.py (not test_*/tests/)
        - cat ... > *.py (not test_*/tests/)
        - Any write operation to .py files outside test directories
        """
        # Skip if it looks like a test file operation
        if re.search(r'\b(test_|tests/|_test\.py)', action):
            return False

        # Check for common edit patterns on .py files
        patterns = [
            r'sed\s+.*\.py',  # sed operations on .py files
            r'cat\s+.*>\s*\S+\.py',  # cat redirect to .py
            r'cat\s+<<.*>\s*\S+\.py',  # heredoc to .py
            r'echo\s+.*>\s*\S+\.py',  # echo to .py
        ]

        for pattern in patterns:
            if re.search(pattern, action):
                return True

        return False

    def _is_verification_command(self, action: str) -> bool:
        """Detect if action is a verification command.

        Matches:
        - pytest ...
        - python -m pytest ...
        - python test_*.py / python *_test.py
        - python ... (any test-related script)
        - Commands with 'test' keyword
        - Commands with 'run' keyword
        """
        verification_patterns = [
            r'\bpytest\b',
            r'\bpython\s+.*test',
            r'\bpython\s+-m\s+pytest\b',
            r'\bpython\s+-m\s+unittest\b',
            r'\btest\b',
            r'\brun\b.*\.py',
            r'\./.*test',
        ]

        for pattern in verification_patterns:
            if re.search(pattern, action, re.IGNORECASE):
                return True

        return False


def build_agent(model, env, step_limit: int, cost_limit: float):
    """Build verify-once-submit agent.

    Args:
        model: The model instance to use for queries
        env: The environment instance for command execution
        step_limit: Maximum number of steps allowed
        cost_limit: Maximum cost allowed in dollars

    Returns:
        VerifyOnceSubmitAgent instance with verification detection
    """
    return VerifyOnceSubmitAgent(
        model=model,
        env=env,
        config_class=VerifyOnceSubmitConfig,
        step_limit=step_limit,
        cost_limit=cost_limit,
    )
