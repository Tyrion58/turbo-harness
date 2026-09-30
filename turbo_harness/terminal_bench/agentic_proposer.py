"""A-Evolve-style agentic proposer for Terminal-Bench 2.

Single bash tool + converse loop. The model reads files, writes agent code,
and validates via bash commands. No JSON writing required — the engine
detects new agent files automatically.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

EVOLVE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = EVOLVE_DIR / "agents"
BASELINE_FILES = {"__init__.py", "baseline_kira.py", "baseline_terminus2.py"}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "workspace_bash",
        "description": (
            "Execute a bash command in the workspace directory. "
            "Use this to read/write files, run validation, list directories, etc. "
            "Examples: cat agents/baseline_kira.py, ls agents/, "
            "cat > agents/new_agent.py << 'PYEOF'\n...\nPYEOF"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute in the workspace directory.",
                },
            },
            "required": ["command"],
        },
    },
}


def _workspace_bash(command: str) -> str:
    """Execute a bash command in the workspace directory."""
    for banned in ["rm -rf /", "rm -rf ~", ":(){ :|:&"]:
        if banned in command:
            return "ERROR: dangerous command blocked"

    for bf in BASELINE_FILES:
        if bf in command and (">" in command or "tee" in command or "sed -i" in command):
            if f"agents/{bf}" in command:
                return f"ERROR: Cannot modify baseline file {bf}. Write new agents with different names."

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(EVOLVE_DIR),
            env=env,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > 10_000:
            output = output[:10_000] + "\n... (truncated)"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out."
    except Exception as exc:
        return f"ERROR: {exc}"


SYSTEM_PROMPT = """\
You are an expert Python programmer evolving agent scaffolds for Terminal-Bench 2.
You have one tool: workspace_bash, which runs bash commands in the project directory.

Use it to: read files (cat), write files (cat > file << 'PYEOF'), list dirs (ls), \
and run validation (uv run python meta_harness.py --validate-agent agents.<name>:AgentHarness).

Do all work yourself — do NOT delegate to subagents.

## WORKFLOW — follow ALL steps in order

### Step 1: Analyze previous results

Read ALL of these state files to understand what has been tried and what works. \
The task prompt provides the exact file paths — use those paths with cat.
- evolution_summary.jsonl  (one JSON per past candidate: agent, avg_pass_rate, hypothesis, delta)
- frontier_val.json  (current best agent per task + overall best)
- agents/baseline_kira.py  (baseline agent to build upon)

If frontier_val.json shows a top-performing agent, also read its code:
- cat agents/<best_agent>.py

Analyze: which scaffold changes improved pass rate? Which regressed? What \
failure modes are common? Identify gaps and opportunities.

### Step 2: Design 1 candidate

Based on your analysis, design 1 new agent scaffold. It MUST change a \
fundamental mechanism, not just tune parameters.

Good candidates: new tool design, improved prompt template, better command \
execution strategy, context management improvements, error recovery logic, \
task decomposition approach, verification workflows.

Bad candidates: same logic with different constants. If the change is only \
adjusting timeouts or string tweaks, redesign with a genuinely novel mechanism.

### Step 3: Implement and validate

1. Copy baseline_kira.py and modify it:
   cat > agents/<name>.py << 'PYEOF'
   # Hypothesis: <one-line falsifiable claim about what this improves>
   <complete Python code>
   PYEOF

2. Validate:
   uv run python meta_harness.py --validate-agent agents.<name>:AgentHarness
   If validation fails, read the error, fix the code, and retry until it passes.

You are DONE after implementing and validating the agent. Do NOT write any \
JSON files — the engine detects new agents automatically.

## Key Requirements

- Class MUST be named `AgentHarness`
- MUST subclass `harbor.agents.terminus_2.terminus_2.Terminus2`
- ALL methods are async (use `await` for super() calls)
- No task-specific hints or hardcoded task names
- Do NOT modify existing baseline files (baseline_kira.py, baseline_terminus2.py)
- If validation fails, FIX the code and retry until it passes
"""


def propose_agentic(
    task_prompt: str,
    iteration: int,
    logs_dir: Path,
    pending_eval_path: Path,
    api_base: str,
    model_name: str = "Qwen/Qwen3.5-9B",
    max_turns: int = 50,
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> bool:
    """Run an A-Evolve-style converse loop to propose new agent scaffolds.

    Returns True if pending_eval.json was written with at least one candidate.
    """
    agents_before = set(f.stem for f in AGENTS_DIR.glob("*.py")) - BASELINE_FILES

    client = openai.OpenAI(base_url=api_base, api_key="unused")

    try:
        models = client.models.list()
        served_model = models.data[0].id if models.data else model_name
    except Exception:
        served_model = model_name

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    # Set up session logging (mirrors claude_sessions/ layout)
    session_dir = logs_dir / "agentic_sessions" / f"{time.strftime('%Y%m%d_%H%M%S')}_iter{iteration}"
    session_dir.mkdir(parents=True, exist_ok=True)
    events_path = session_dir / "events.jsonl"
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": task_prompt,
        "model": model_name,
        "served_model": served_model,
        "iteration": iteration,
        "api_base": api_base,
        "max_turns": max_turns,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _log_event(event: dict):
        with open(events_path, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    _log_event({"type": "system", "content": SYSTEM_PROMPT})
    _log_event({"type": "user", "content": task_prompt})

    total_input_tokens = 0
    total_output_tokens = 0
    nudge_count = 0
    t0 = time.time()

    for turn in range(max_turns):
        try:
            response = client.chat.completions.create(
                model=served_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=[BASH_TOOL],
                tool_choice="auto",
            )
        except Exception as e:
            print(f"    turn {turn}: API error: {e}")
            break

        usage = response.usage
        if usage:
            total_input_tokens += usage.prompt_tokens or 0
            total_output_tokens += usage.completion_tokens or 0

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls = list(message.tool_calls or [])

        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)
        _log_event({"type": "assistant", "turn": turn, "content": content,
                     "tool_calls": assistant_msg.get("tool_calls"),
                     "input_tokens": usage.prompt_tokens if usage else None,
                     "output_tokens": usage.completion_tokens if usage else None})

        if not tool_calls:
            new_files = set(f.stem for f in AGENTS_DIR.glob("*.py")) - BASELINE_FILES - agents_before
            if not new_files and nudge_count < 3:
                nudge_count += 1
                print(f"    turn {turn}: model stopped — nudging ({nudge_count}/3)")
                messages.append({
                    "role": "user",
                    "content": (
                        "You haven't finished. Write 1 new agent file using workspace_bash "
                        "(cat > agents/<name>.py << 'PYEOF'), then validate with "
                        "uv run python meta_harness.py --validate-agent agents.<name>:AgentHarness. "
                        "Don't stop until the agent is written and validated."
                    ),
                })
                continue
            print(f"    turn {turn}: model finished")
            break

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            if fn_name == "workspace_bash":
                cmd = fn_args.get("command", "")
                cmd_short = cmd[:100].replace("\n", "\\n")
                result_text = _workspace_bash(cmd)
                result_short = result_text[:120].replace("\n", " ")
                print(f"    turn {turn}: bash({cmd_short}...) → {result_short}")
            else:
                result_text = f"ERROR: Unknown tool '{fn_name}'. Use workspace_bash."

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })
            _log_event({"type": "tool", "turn": turn, "tool_call_id": tc.id,
                         "function": fn_name, "command": fn_args.get("command", ""),
                         "result": result_text})

    elapsed = time.time() - t0
    print(
        f"    agentic loop: {turn + 1} turns, "
        f"{total_input_tokens} in / {total_output_tokens} out tokens, "
        f"{elapsed:.1f}s"
    )

    # Write response.md summary
    response_lines = []
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("content"):
            response_lines.append(msg["content"])
    (session_dir / "response.md").write_text("\n\n---\n\n".join(response_lines) if response_lines else "(no text output)")
    _log_event({"type": "summary", "turns": turn + 1, "elapsed": round(elapsed, 1),
                "input_tokens": total_input_tokens, "output_tokens": total_output_tokens})

    # Auto-detect new agent files, generate pending_eval.json
    agents_after = set(f.stem for f in AGENTS_DIR.glob("*.py")) - BASELINE_FILES
    new_agents = sorted(agents_after - agents_before)

    if not new_agents:
        return False

    candidates = []
    for name in new_agents:
        agent_path = AGENTS_DIR / f"{name}.py"
        hypothesis = ""
        try:
            source = agent_path.read_text()
            for line in source.split("\n"):
                stripped = line.strip().lstrip("#").strip().strip('"').strip("'")
                if stripped.lower().startswith("hypothesis:"):
                    hypothesis = stripped[len("hypothesis:"):].strip().rstrip(".")
                    break
        except Exception:
            pass
        candidates.append({
            "name": name,
            "import_path": f"agents.{name}:AgentHarness",
            "hypothesis": hypothesis or f"Novel agent scaffold: {name}",
            "changes": "",
        })

    pending_eval_path.write_text(
        json.dumps({"iteration": iteration, "candidates": candidates}, indent=2)
    )
    return True
