"""Shared API for ALFWorld full-scaffold harnesses.

A harness is a `harness.py` exposing:
    run_episode(env, llm, max_steps) -> {"won": bool, "messages": [{role,content}], "steps": int, "status": str}

`env` is an `AlfEnv` (see env.py): env.reset() -> (obs, admissible); env.step(action) -> (obs, admissible, done, won).
`llm` is an `LLMClient` (below): a thin wrapper over the frozen student (vLLM, OpenAI-compatible).

These are lightweight helpers (no alfworld import) so both the runner and any harness can use them.
The proposer OWNS the whole loop in run_episode — prompt construction, memory, control flow,
retries, stop logic. Helpers here (system prompt, few-shot, action canonicalization, tool schema)
just make it easy to reproduce a strong ReAct baseline; override freely.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

# Verbatim AgentBench ALFWorld system instruction (so our baseline matches theirs).
SYSTEM_PROMPT = (
    "Interact with a household to solve a task. Imagine you are an intelligent agent in a "
    "household environment and your target is to perform actions to complete the task goal. At "
    "the beginning of your interactions, you will be given the detailed description of the current "
    "environment and your goal to accomplish. A tool will be provided for you to use to submit the "
    "action you want to take. This tool is the only tool you should and must take in order to "
    "operate any action in the environment. The way you perform action is to place the action "
    "chosen by you in the arguments field of your tool call. For each of your turn, you will be "
    "given a list of actions which you can choose one to perform in this turn. The action you would "
    'like to take should be offered in this format: "the name of your next action", and you should '
    "fill it in the argument field of your tool call. Note that you should always call a tool to "
    "operate an action from the given choices. After your each turn, the environment will give you "
    "immediate feedback based on which you plan your next few steps. if the environment output "
    '"Nothing happened", that means the previous action is invalid and you should try more options.\n'
    " Reminder:\n"
    "1. the action must be chosen from the given available actions. Any actions except provided "
    "available actions will be regarded as illegal.\n"
    "2. Always call the tool to hand in your next action and think when necessary."
)

# The single ALFWorld tool (matches AgentBench).
TAKE_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "take_action",
        "description": "Take an action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action you would like to take",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}

_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

_FEWSHOT_FILE = Path(__file__).resolve().parent / "prompts" / "alfworld_multiturn_plan_first.json"


def task_category(game_relpath: str) -> str:
    """Map a game path (e.g. 'json_2.1.1/.../pick_clean_then_place_in_recep-...') to a few-shot key."""
    name = game_relpath.split("/")[2] if "/" in game_relpath else game_relpath
    for prefix, cat in _PREFIXES.items():
        if name.startswith(prefix):
            return cat
    return "put"


def load_fewshot(category: str) -> list[dict]:
    """Return the 2-shot ReAct exemplar messages for a task category (role user/assistant text)."""
    try:
        data = json.loads(_FEWSHOT_FILE.read_text())
    except Exception:
        return []
    ex = data.get(category, [])
    out = []
    for m in ex:
        role = m.get("role", "user")
        role = "assistant" if role in ("agent", "assistant") else "user"
        out.append({"role": role, "content": str(m.get("content", ""))})
    return out


def available_actions_str(admissible: list[str]) -> str:
    return " AVAILABLE ACTIONS: " + "\n".join(admissible) + "\n"


def _bleu1(cand: str, ref: str) -> float:
    # cheap unigram overlap as a BLEU proxy for nearest-admissible matching
    cw, rw = cand.split(), ref.split()
    if not cw or not rw:
        return 0.0
    rset = {}
    for w in rw:
        rset[w] = rset.get(w, 0) + 1
    match = 0
    for w in cw:
        if rset.get(w, 0) > 0:
            match += 1
            rset[w] -= 1
    return match / len(cw)


def canonicalize(action: str, admissible: list[str]) -> str:
    """Map a model action string to the nearest admissible command (AgentBench-style)."""
    if not action:
        return action
    a = action.strip().lower().split("\n")[0]
    if not admissible:
        return a
    if a in admissible:
        return a
    scored = [
        (0.7 * _bleu1(a, c) + 0.3 * SequenceMatcher(None, a, c).ratio(), c)
        for c in admissible
    ]
    best_score, best = max(scored, key=lambda x: x[0])
    return best if best_score > 0.01 else a


def extract_turn(response):
    """Return (assistant_message_dict, tool_call_id_or_None, action_str) from a chat response.

    Preserves proper OpenAI tool-calling history: the assistant dict includes tool_calls so the
    follow-up tool message (with matching tool_call_id) is well-formed for the chat template.
    """
    msg = response.choices[0].message
    am = {"role": "assistant", "content": msg.content or ""}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        tc = tcs[0]
        am["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        ]
        action = ""
        try:
            a = (
                json.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments
            )
            if isinstance(a, dict):
                action = next((v for v in a.values() if isinstance(v, str)), "")
        except Exception:
            pass
        return am, tc.id, action
    content = msg.content or ""
    m = re.search(r"(?:action|ACTION)\s*[:=]\s*(.+)", content)
    return am, None, (m.group(1) if m else content).strip()


class LLMClient:
    """Thin wrapper over the frozen student (vLLM OpenAI endpoint)."""

    def __init__(
        self,
        base_url="http://127.0.0.1:8110/v1",
        model="Qwen3.5-9B",
        temperature=0.0,
        max_tokens=4096,
        enable_thinking=False,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url, api_key="EMPTY", timeout=120.0, max_retries=2
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.n_calls = 0

    def complete(self, messages, tools=None, tool_choice="auto", max_tokens=None):
        """Raw chat completion. Returns the OpenAI response object (or None on error)."""
        self.n_calls += 1
        kw = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        )
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice
        try:
            return self.client.chat.completions.create(**kw)
        except Exception:
            return None
