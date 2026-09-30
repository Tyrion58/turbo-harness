"""Advisor module: local LLM generates strategic advice for the proposer.

The advisor (e.g. Qwen3-8B served via vLLM) reads the evolution history
and current frontier, then produces natural-language advice that gets
injected into the proposer's task prompt. The proposer (Claude) still
does all the actual code generation.
"""

import json
import re
from pathlib import Path

import openai

ADVISOR_SYSTEM = """\
You are a research advisor analyzing the evolution history of agent scaffolds \
for Terminal-Bench 2 (TB2). TB2 is a benchmark of 89 complex, long-horizon \
terminal tasks (e.g. building compilers, reverse engineering, system admin). \
Agents are Terminus2 subclasses that interact with a terminal via tool calls.

Your job is to provide strategic advice to a code-writing agent who will \
design the next generation of agent scaffolds.

Analyze the evolution history and current frontier, then provide:
1. What scaffold changes have worked well (and why)
2. What changes have failed or regressed (and why)
3. Specific suggestions for the next iteration — novel approaches to try
4. What to avoid based on past failures

Focus on scaffold-level improvements: prompt engineering, tool design, \
command execution strategy, context management, error recovery, \
task decomposition, verification workflows.

Be concrete and actionable. Reference specific agents and their results. \
Keep your advice under 500 words. Do NOT include any thinking process \
or meta-commentary — go straight to the analysis."""


def generate_advice(
    evolution_summary_path: Path,
    frontier_path: Path,
    api_base: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen3-8B",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str | None:
    """Call the advisor LLM to generate advice based on evolution history.

    Returns the advice string, or None if the call fails or there's no history.
    """
    history_lines = []
    if evolution_summary_path.exists():
        for line in evolution_summary_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    history_lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    frontier = {}
    if frontier_path.exists():
        try:
            frontier = json.loads(frontier_path.read_text())
        except json.JSONDecodeError:
            pass

    if not history_lines and not frontier:
        return None

    user_content = "## Evolution History\n\n"
    if history_lines:
        for entry in history_lines:
            per_task = entry.get("per_task", {})
            n_tasks = len(per_task)
            passed = sum(1 for v in per_task.values() if v > 0)
            user_content += (
                f"- **{entry.get('agent', '?')}**: "
                f"avg_pass_rate={entry.get('avg_pass_rate', '?'):.1%}, "
                f"passed={passed}/{n_tasks} tasks, "
                f"hypothesis=\"{entry.get('hypothesis', '?')}\", "
                f"delta={entry.get('delta', '?')}\n"
            )
    else:
        user_content += "No previous iterations yet.\n"

    user_content += "\n## Current Frontier\n\n"
    if frontier:
        best = frontier.get("_best", {})
        if best:
            user_content += (
                f"Overall best: **{best.get('agent', '?')}** "
                f"@ {best.get('avg_pass_rate', 0):.1%}\n\n"
            )
        task_count = 0
        improved_tasks = []
        for task, info in sorted(frontier.items()):
            if task.startswith("_"):
                continue
            task_count += 1
            agent = info.get("best_agent", "?")
            rate = info.get("pass_rate", 0)
            if rate > 0:
                improved_tasks.append(f"{task} ({agent}, {rate:.0%})")

        user_content += f"Frontier covers {task_count} tasks.\n"
        if improved_tasks:
            user_content += f"Tasks with >0% pass rate: {', '.join(improved_tasks[:20])}\n"

    user_content += (
        "\nBased on this history, what specific advice do you have "
        "for designing the next generation of agent scaffolds?"
    )

    client = openai.OpenAI(base_url=api_base, api_key="unused")

    try:
        models = client.models.list()
        served_model = models.data[0].id if models.data else model_name
    except Exception:
        served_model = model_name

    try:
        response = client.chat.completions.create(
            model=served_model,
            messages=[
                {"role": "system", "content": ADVISOR_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        advice = response.choices[0].message.content or ""
        advice = re.sub(r"<think>.*?</think>\s*", "", advice, flags=re.DOTALL)
        if "</think>" in advice:
            advice = advice.split("</think>", 1)[1]
        usage = response.usage
        tokens_info = ""
        if usage:
            tokens_info = f" ({usage.prompt_tokens}in/{usage.completion_tokens}out)"
        print(f"    advisor generated {len(advice)} chars{tokens_info}")
        return advice.strip() if advice.strip() else None
    except Exception as e:
        print(f"    advisor error: {e}")
        return None
