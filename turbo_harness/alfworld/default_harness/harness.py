"""DEFAULT ALFWorld harness — a faithful ReAct baseline matching AgentBench's setup:
zero-shot (system + task prompt only), take_action tool-calling, nearest-admissible
canonicalization, and re-prompt (no env step) when the model fails to call the tool.

This is the FULL agent loop. The meta-harness proposer evolves THIS file: it may change context
management, memory across steps, control flow, retries/repair, stop logic, subgoal tracking, what is
shown each turn, etc. — anything except the frozen student model. Keep the interface:

    run_episode(env, llm, max_steps) -> {"won": bool, "messages": [...], "steps": int, "status": str}
"""

from turbo_harness.alfworld import api


def run_episode(env, llm, max_steps):
    obs, admissible = env.reset()
    messages = [
        {"role": "system", "content": api.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Here is your task. "
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
        assistant_msg, tool_call_id, raw = api.extract_turn(resp)
        messages.append(assistant_msg)

        if (
            tool_call_id is None
        ):  # model didn't call the tool — remind and retry (no env step)
            messages.append(
                {
                    "role": "user",
                    "content": "You MUST call the take_action tool with exactly one action "
                    "from the AVAILABLE ACTIONS.",
                }
            )
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
