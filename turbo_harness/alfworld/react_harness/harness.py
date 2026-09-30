"""ReAct baseline (Yao et al. 2023) for ALFWorld — few-shot interleaved reasoning + acting.

Distinct from the zero-shot Default harness by the two canonical ReAct additions:
  (a) few-shot Thought->Action->Observation exemplars (AgentBench's shipped 2-shot, routed by task
      category via api.task_category / api.load_fewshot), and
  (b) a mandated explicit "Thought:" in the assistant content before/with each action (meaningful
      because the frozen student runs with thinking disabled).
Same tool-calling loop + nearest-admissible canonicalization as Default. A forced-action fallback
(tool_choice="required") guarantees an action on any turn where the model writes only a Thought.

    run_episode(env, llm, max_steps) -> {"won", "messages", "steps", "status"}
"""

from turbo_harness.alfworld import api

REACT_INSTRUCTION = (
    "\n\nUse the ReAct strategy: on EVERY turn, write a brief 'Thought:' in your message content "
    "(reason about the observation, your progress, and the single best next action) AND, in the SAME "
    "turn, call take_action with that action. Think, then act."
)

FORCE = "Now call the take_action tool with exactly one action from the AVAILABLE ACTIONS."


def _fewshot_block(category: str) -> str:
    """Render the shipped 2-shot ReAct exemplars for this task category as a text demo block."""
    ex = api.load_fewshot(category)
    if not ex:
        return ""
    lines = ["Here are examples of solving similar tasks with Thought/Action reasoning:\n"]
    for m in ex:
        tag = "You" if m["role"] == "assistant" else "Environment"
        lines.append(f"{tag}: {m['content']}")
    lines.append(
        "\nNow solve the new task below the same way: a brief Thought in your content, and in the "
        "same turn call take_action.\n"
    )
    return "\n".join(lines) + "\n"


def run_episode(env, llm, max_steps):
    obs, admissible = env.reset()
    demo = _fewshot_block(api.task_category(getattr(env, "game_relpath", "")))
    messages = [
        {"role": "system", "content": api.SYSTEM_PROMPT + REACT_INSTRUCTION},
        {
            "role": "user",
            "content": demo
            + "Here is your task. "
            + obs
            + api.available_actions_str(admissible),
        },
    ]

    won, status = False, "max_steps"
    for _ in range(max_steps):
        resp = llm.complete(messages, tools=[api.TAKE_ACTION_TOOL], tool_choice="auto")
        if resp is None:
            status = "llm_error"
            break
        am, tool_call_id, raw = api.extract_turn(resp)
        messages.append(am)
        if tool_call_id is None:  # model wrote only a Thought — force the action
            messages.append({"role": "user", "content": FORCE})
            forced = llm.complete(
                messages, tools=[api.TAKE_ACTION_TOOL], tool_choice="required"
            )
            if forced is None:
                status = "llm_error"
                break
            am, tool_call_id, raw = api.extract_turn(forced)
            messages.append(am)
        if tool_call_id is None:
            continue
        action = api.canonicalize(raw, admissible)
        obs, admissible, done, won = env.step(action)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": obs + api.available_actions_str(admissible),
            }
        )
        if done:
            status = "done"
            break

    return {
        "won": bool(won),
        "messages": messages,
        "steps": llm.n_calls,
        "status": status,
    }
