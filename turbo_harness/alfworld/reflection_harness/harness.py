"""Reflection / Reflexion baseline (Shinn et al. 2023) for ALFWorld — 2-episode verbal reflection.

Episode 1 runs a ReAct-style tool-calling loop; on failure the student writes a short verbal
reflection on WHY it failed and what to change, which is injected into episode 2's system context.
Reports success@2 (won in ep1 OR ep2). This is a 2-episode protocol (like Harness-R1's 2-rollout) and
is NOT directly comparable to the single-episode rows — report it flagged as such.

    run_episode(env, llm, max_steps) -> {"won", "messages", "steps", "status"}
"""

from turbo_harness.alfworld import api


def _run_once(env, llm, max_steps, reflection=""):
    obs, admissible = env.reset()
    sys = api.SYSTEM_PROMPT
    if reflection:
        sys += (
            "\n\nREFLECTION FROM YOUR PREVIOUS FAILED ATTEMPT (use it to do better now):\n"
            + reflection
        )
    messages = [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": "Here is your task. " + obs + api.available_actions_str(admissible),
        },
    ]
    won, status = False, "max_steps"
    for _ in range(max_steps):
        resp = llm.complete(messages, tools=[api.TAKE_ACTION_TOOL], tool_choice="auto")
        if resp is None:
            status = "llm_error"
            break
        am, tcid, raw = api.extract_turn(resp)
        messages.append(am)
        if tcid is None:
            messages.append(
                {
                    "role": "user",
                    "content": "You MUST call take_action with one action from the AVAILABLE ACTIONS.",
                }
            )
            continue
        action = api.canonicalize(raw, admissible)
        obs, admissible, done, won = env.step(action)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tcid,
                "content": obs + api.available_actions_str(admissible),
            }
        )
        if done:
            status = "done"
            break
    return won, messages, status


def _reflect(critic, messages):
    lines = []
    for m in messages[1:]:  # skip system
        role = m.get("role")
        c = str(m.get("content", ""))[:500]
        if role == "assistant" and m.get("tool_calls"):
            try:
                c = (c + " [action: " + m["tool_calls"][0]["function"]["arguments"] + "]").strip()
            except Exception:
                pass
        lines.append(f"{role}: {c}")
    prompt = [
        {
            "role": "system",
            "content": "You are an agent reflecting on a FAILED household task attempt. In 2-4 "
            "sentences, diagnose WHY it failed and state a concrete, different strategy for the next "
            "attempt. Be specific and actionable.",
        },
        {
            "role": "user",
            "content": "Failed attempt transcript:\n" + "\n".join(lines[-40:]) + "\n\nReflection:",
        },
    ]
    r = critic.complete(prompt, max_tokens=512)
    if r is None:
        return ""
    try:
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def run_episode(env, llm, max_steps):
    critic = api.LLMClient(
        base_url=str(llm.client.base_url), model=llm.model, temperature=0.6, max_tokens=512
    )
    won1, msgs1, status1 = _run_once(env, llm, max_steps)
    if won1:
        return {
            "won": True,
            "messages": msgs1,
            "steps": llm.n_calls,
            "status": "2ep:ep1_" + status1,
        }
    reflection = _reflect(critic, msgs1)
    won2, msgs2, status2 = _run_once(env, llm, max_steps, reflection=reflection)
    return {
        "won": bool(won1 or won2),  # success@2
        "messages": msgs1 + [{"role": "user", "content": "[REFLECTION]\n" + reflection}] + msgs2,
        "steps": llm.n_calls + critic.n_calls,
        "status": f"2ep:{status1}->{status2}",
    }
