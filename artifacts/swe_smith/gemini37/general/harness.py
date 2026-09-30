"""Structured bugfix workflow harness for SWE-smith tasks.

Adds explicit 3-phase workflow (LOCATE → UNDERSTAND → FIX) to reduce wasted
exploration and improve fix accuracy on first attempt for weak models.
"""

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent


STRUCTURED_INSTANCE_TEMPLATE = """Please solve this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

## 3-Phase Structured Workflow

Follow this workflow systematically to solve the bug efficiently:

### PHASE 1: LOCATE (max 3-4 commands)
Your goal is to quickly find the buggy code location.
- Use grep to search for error messages, function names, or relevant code patterns
- Use git log or git blame to understand recent changes that may have introduced the bug
- Use find or ls to locate relevant files
- Read directory structures with tree or ls -la

```bash
# Example: grep -r "error_pattern" .
```

### PHASE 2: UNDERSTAND (max 5-7 commands)
Once you've located the buggy file/function, understand the exact issue.
- Read the specific file sections using cat, head, tail, or nl
- View exact line numbers with: nl -ba filename.py | sed -n 'start,end p'
- Identify the EXACT lines that need to change
- Understand the context around those lines

<important>
Before making ANY edit, verify the exact line numbers and content.
Line numbers can shift after edits, so re-check before each operation.
</important>

### PHASE 3: FIX (remaining commands)
Make surgical, targeted edits.
- Use sed with SPECIFIC line numbers (e.g., sed -i '10,12s/old/new/g' file.py)
- OR use targeted patterns that match only the buggy lines
- NEVER use broad patterns like s/old/new/g across entire file without line restrictions
- After each edit, verify with git diff to ensure you changed exactly what you intended
- Test your fix by running the code/tests
- If the fix doesn't work, analyze the failure and iterate

```bash
# GOOD: Targeted edit with line numbers
sed -i '42s/old_value/new_value/' filename.py

# BAD: Broad pattern that might break other code
# sed -i 's/old_value/new_value/g' filename.py
```

## Progress Guidelines

- After 25 steps, prioritize completing the fix over perfect understanding
- If stuck, try a different approach rather than repeating failed commands
- Submit your changes when the bug is fixed: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Formatting your response

Include a THOUGHT section before your command where you explain your reasoning and which phase you're in.

<example_response>
THOUGHT: [PHASE 1: LOCATE] I need to find where the bug is occurring. Let me search for the error message in the codebase.

```bash
grep -r "KeyError" .
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
# Replace all occurrences in specific line range (RECOMMENDED)
sed -i '10,20s/old_string/new_string/g' filename.py

# Replace only first occurrence on specific line
sed -i '42s/old_string/new_string/' filename.py

# BEST PRACTICE: View the exact lines before editing
nl -ba filename.py | sed -n '10,20p'  # View lines 10-20 first
sed -i '10,20s/old/new/g' filename.py  # Then edit those lines
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'

# View file with all line numbers
nl -ba filename.py
```

### Any other command you want to run

```bash
anything
```
"""


def build_agent(model: Model, env: Environment, step_limit: int, cost_limit: float) -> DefaultAgent:
    """Build an agent with structured 3-phase bugfix workflow.

    Args:
        model: The language model to use
        env: The execution environment
        step_limit: Maximum number of steps allowed
        cost_limit: Maximum cost allowed

    Returns:
        DefaultAgent configured with structured workflow template
    """
    # Use custom config with structured instance template
    config = AgentConfig(
        system_template="""You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process and which workflow phase you're in.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected. You will receive advice from time to time on your progress.""",
        instance_template=STRUCTURED_INSTANCE_TEMPLATE,
        action_observation_template="""<returncode>{{output.returncode}}</returncode>
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
{%- endif -%}""",
        format_error_template="""Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.
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
to proceed in two steps, first writing TRIPLEBACKTICKSBASH, then replacing them with ```bash.""",
        step_limit=step_limit,
        cost_limit=cost_limit,
    )

    return DefaultAgent(model, env, config_class=lambda **kwargs: config)
