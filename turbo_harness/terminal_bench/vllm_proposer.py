"""Proposer that calls a local vLLM-served model instead of Claude Code CLI.

Gathers all context into a single prompt, calls the model once via OpenAI-
compatible API, parses the response into agent files + pending_eval.json.

Adapted for Terminal-Bench 2: agents are Terminus2 subclasses that run
terminal commands via native tool calling in Harbor sandboxes.
"""

import json
import re
import subprocess
import time
from pathlib import Path

import openai

EVOLVE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = EVOLVE_DIR / "agents"

SYSTEM_PROMPT = """\
You are an expert Python programmer designing agent scaffolds for Terminal-Bench 2.

## Task

Design 1 new agent scaffold that improves on the KIRA baseline for solving
complex terminal tasks (compiling code, reverse engineering, system admin, etc.).

The agent is a Python class named `AgentHarness` that subclasses
`harbor.agents.terminus_2.terminus_2.Terminus2`. It runs inside a Harbor
sandbox with a tmux terminal session.

## Architecture

The agent interacts with an LLM (Claude Opus) via `litellm.acompletion` using
native tool calling. Three tools are available:
- `execute_commands`: send keystrokes to the terminal with analysis/plan
- `task_complete`: mark the task as done
- `image_read`: analyze image files

Key methods you can override:
- `_call_llm_with_tools(messages)` — makes the litellm API call
- `_parse_tool_calls(tool_calls)` — converts tool call dicts to commands
- `_execute_commands(commands, session)` — runs commands on tmux
- `_run_agent_loop(initial_prompt, chat, original_instruction)` — main loop
- `_get_completion_confirmation_message(terminal_output)` — completion check
- `_get_prompt_template_path()` — path to system prompt template
- `_limit_output_length(output, max_bytes)` — truncation

## Constraints

- Class MUST be named `AgentHarness`
- MUST subclass `harbor.agents.terminus_2.terminus_2.Terminus2`
- ALL methods are async (use `await` for super() calls)
- No task-specific hints or hardcoded task names
- Focus on scaffold-level improvements: prompt engineering, tool design,
  command execution strategy, context management, error recovery

## Output Format

Output exactly 1 agent using this format:

### AGENT: snake_case_name
```python
<complete Python file — must be importable and contain AgentHarness class>
```

### PENDING_EVAL
```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "snake_case_name",
      "import_path": "agents.snake_case_name:AgentHarness",
      "hypothesis": "A falsifiable claim about what this will improve",
      "changes": "What was changed from the baseline"
    }
  ]
}
```

Follow this format exactly. The Python file must be complete and importable.\
"""


def _read_file(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def _gather_context(iteration: int, logs_dir: Path) -> str:
    """Read all state files and build the user prompt context."""
    parts = []

    parts.append(f"# Iteration {iteration}\n")
    parts.append("Design 1 new agent scaffold for this iteration.\n")

    # Evolution history
    summary_path = logs_dir / "evolution_summary.jsonl"
    if summary_path.exists():
        lines = summary_path.read_text().strip().split("\n")
        if len(lines) > 20:
            lines = lines[-20:]
        parts.append("## Past Results (evolution_summary.jsonl)\n")
        parts.append("\n".join(lines))
        parts.append("")

    # Frontier
    frontier_path = logs_dir / "frontier_val.json"
    if frontier_path.exists():
        parts.append("## Current Frontier (frontier_val.json)\n")
        parts.append(frontier_path.read_text())
        parts.append("")

    # Baseline agent source
    kira_path = AGENTS_DIR / "baseline_kira.py"
    if kira_path.exists():
        parts.append("## Baseline: agents/baseline_kira.py\n```python\n")
        parts.append(kira_path.read_text())
        parts.append("```\n")

    # Prompt template
    prompt_path = EVOLVE_DIR / "prompt-templates" / "terminus-kira.txt"
    if prompt_path.exists():
        parts.append("## Prompt Template: terminus-kira.txt\n```\n")
        parts.append(prompt_path.read_text())
        parts.append("```\n")

    # Top agents from frontier
    if frontier_path.exists():
        try:
            frontier = json.loads(frontier_path.read_text())
            best = frontier.get("_best", {})
            best_agent = best.get("agent", "")
            if best_agent and best_agent not in ("kira-baseline", "terminus2-baseline"):
                agent_path = AGENTS_DIR / f"{best_agent}.py"
                if agent_path.exists():
                    parts.append(
                        f"## Best Agent: agents/{best_agent}.py "
                        f"(pass_rate={best.get('avg_pass_rate', 0):.1%})\n```python\n"
                    )
                    source = agent_path.read_text()
                    if len(source) > 8000:
                        source = source[:8000] + "\n# ... truncated"
                    parts.append(source)
                    parts.append("```\n")
        except (json.JSONDecodeError, KeyError):
            pass

    return "\n".join(parts)


def _parse_response(text: str, iteration: int) -> tuple[dict, dict[str, str]]:
    """Parse the model's response into pending_eval dict and agent code files."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    agents = {}
    pending_eval = None

    agent_pattern = r"###\s*AGENT:\s*(\w+)\s*\n```python\n(.*?)```"
    for match in re.finditer(agent_pattern, text, re.DOTALL):
        name = match.group(1).strip()
        code = match.group(2).strip()
        agents[name] = code

    eval_pattern = r"###\s*PENDING_EVAL\s*\n```json\n(.*?)```"
    eval_match = re.search(eval_pattern, text, re.DOTALL)
    if eval_match:
        try:
            pending_eval = json.loads(eval_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    if pending_eval is None and agents:
        pending_eval = {
            "iteration": iteration,
            "candidates": [
                {
                    "name": name,
                    "import_path": f"agents.{name}:AgentHarness",
                    "hypothesis": "auto-generated",
                    "changes": "",
                }
                for name in agents
            ],
        }

    # Fallback: look for any code block with AgentHarness class
    if not agents:
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        for code in code_blocks:
            if "AgentHarness" in code and "Terminus2" in code:
                class_match = re.search(r"class\s+AgentHarness\s*\(", code)
                if class_match:
                    name = f"candidate_iter{iteration}"
                    agents[name] = code.strip()

        if agents and pending_eval is None:
            pending_eval = {
                "iteration": iteration,
                "candidates": [
                    {
                        "name": name,
                        "import_path": f"agents.{name}:AgentHarness",
                        "hypothesis": "auto-generated",
                        "changes": "",
                    }
                    for name in agents
                ],
            }

    return pending_eval or {"iteration": iteration, "candidates": []}, agents


def _validate_agent(name: str) -> tuple[bool, str]:
    """Import an agent and check it's a valid Terminus2 subclass."""
    import_path = f"agents.{name}:AgentHarness"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "meta_harness.py",
            "--validate-agent",
            import_path,
        ],
        capture_output=True,
        text=True,
        cwd=str(EVOLVE_DIR),
        timeout=30,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        return True, ""
    error = (result.stderr or result.stdout).strip()
    if len(error) > 500:
        error = error[-500:]
    return False, error


def propose_vllm(
    iteration: int,
    logs_dir: Path,
    pending_eval_path: Path,
    api_base: str,
    model_name: str,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> bool:
    """Propose new agent scaffolds via a vLLM-served model with validation retries.

    Returns True if pending_eval.json was written with at least one candidate.
    """
    context = _gather_context(iteration, logs_dir)

    client = openai.OpenAI(base_url=api_base, api_key="unused")

    print(f"    calling {model_name} ...", flush=True)
    t0 = time.time()

    try:
        models = client.models.list()
        served_model = models.data[0].id if models.data else model_name
    except Exception:
        served_model = model_name

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]

    all_agents = {}
    pending_eval = None

    for attempt in range(1 + max_retries):
        try:
            response = client.chat.completions.create(
                model=served_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            print(f"    vLLM call failed: {e}")
            return False

        elapsed = time.time() - t0
        text = response.choices[0].message.content or ""
        usage = response.usage
        print(
            f"    attempt {attempt + 1}: {len(text)} chars  "
            f"tokens={usage.completion_tokens if usage else '?'}  "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        parsed_eval, agents = _parse_response(text, iteration)

        if not agents:
            print("    no valid agents parsed from response")
            if attempt < max_retries:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ]
                continue
            return False

        for name, code in agents.items():
            agent_path = AGENTS_DIR / f"{name}.py"
            agent_path.write_text(code + "\n")
            all_agents[name] = code

        if parsed_eval and parsed_eval.get("candidates"):
            pending_eval = parsed_eval

        # Validate each agent
        failed = {}
        valid = []
        for name in agents:
            ok, err = _validate_agent(name)
            if ok:
                print(f"    {name}: OK")
                valid.append(name)
            else:
                print(f"    {name}: FAIL")
                failed[name] = err

        if not failed or attempt >= max_retries:
            break

        error_parts = [
            f"Fix the following {len(failed)} agent(s) that failed validation. "
            f"Output ONLY the corrected agents using the format:\n"
            f"### AGENT: name\n```python\n<code>\n```\n\n"
        ]
        for name, err in failed.items():
            code_preview = agents[name]
            if len(code_preview) > 3000:
                code_preview = code_preview[:3000] + "\n# ... truncated"
            error_parts.append(f"### {name}\nError:\n```\n{err}\n```\n")
            error_parts.append(f"Current code:\n```python\n{code_preview}\n```\n")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(error_parts)},
        ]
        print(f"    retrying {len(failed)} failed agent(s)...")

    # Filter pending_eval to only valid agents
    if pending_eval and valid:
        pending_eval["candidates"] = [
            c for c in pending_eval.get("candidates", [])
            if c.get("name") in valid or c.get("name") in all_agents
        ]

    if not pending_eval or not pending_eval.get("candidates"):
        if valid:
            pending_eval = {
                "iteration": iteration,
                "candidates": [
                    {
                        "name": name,
                        "import_path": f"agents.{name}:AgentHarness",
                        "hypothesis": "auto-generated",
                        "changes": "",
                    }
                    for name in valid
                ],
            }
        else:
            print("    no valid agents after retries")
            return False

    pending_eval_path.write_text(json.dumps(pending_eval, indent=2))
    print(f"    wrote pending_eval.json ({len(pending_eval.get('candidates', []))} candidates)")

    return pending_eval_path.exists() and len(pending_eval.get("candidates", [])) > 0
