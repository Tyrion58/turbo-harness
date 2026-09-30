"""Self-Refine baseline (Madaan et al. 2023) for ALFWorld — per-step propose -> self-critique ->
refine. Each turn the student proposes an action, critiques its own proposal, then re-decides with the
critique in view; the refined action is executed. ~3x LLM calls/step. This is a fixed refine rule
applied uniformly to every step (as in Harness-R1's Table 1) and is often flat/negative vs Default.

    run_episode(env, llm, max_steps) -> {"won", "messages", "steps", "status"}
"""

from turbo_harness.alfworld import api


def _mk_critic(llm):
    # Free-text self-critique at non-greedy temperature (Qwen3.5 greedy free-text can loop).
    return api.LLMClient(
        base_url=str(llm.client.base_url),
        model=llm.model,
        temperature=0.6,
        max_tokens=512,
    )


def _critique(critic, task, last_obs, admissible, candidate):
    prompt = [
        {
            "role": "system",
            "content": "You review an agent's proposed next action in a household task. Give a "
            "1-2 sentence critique: is the candidate a valid available action, does it make progress "
            "toward the goal, or is there a clearly better choice? Be concise and specific.",
        },
        {
            "role": "user",
            "content": f"Task/goal:\n{task}\n\nMost recent observation:\n{last_obs}\n\n"
            f"Available actions:\n{chr(10).join(admissible[:60])}\n\n"
            f"Candidate next action: {candidate}\n\nCritique:",
        },
    ]
    r = critic.complete(prompt, max_tokens=256)
    if r is None:
        return ""
    try:
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def run_episode(env, llm, max_steps):
    obs, admissible = env.reset()
    critic = _mk_critic(llm)
    task = obs
    messages = [
        {"role": "system", "content": api.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Here is your task. " + obs + api.available_actions_str(admissible),
        },
    ]

    won, status = False, "max_steps"
    last_obs = obs
    for _ in range(max_steps):
        # 1. propose a candidate action (ephemeral — not committed to history)
        prop = llm.complete(messages, tools=[api.TAKE_ACTION_TOOL], tool_choice="auto")
        if prop is None:
            status = "llm_error"
            break
        pm, ptcid, praw = api.extract_turn(prop)
        if ptcid is None:  # no tool call — remind and retry
            messages.append(pm)
            messages.append(
                {
                    "role": "user",
                    "content": "You MUST call take_action with one action from the AVAILABLE ACTIONS.",
                }
            )
            continue
        # 2. self-critique the proposal
        fb = _critique(critic, task, last_obs, admissible, praw)
        # 3. refine: re-decide with the critique in view (committed to history)
        refine_user = {
            "role": "user",
            "content": f"You are considering the action: '{praw}'.\nSelf-critique: {fb}\n"
            "Now choose the best single action and call take_action.",
        }
        ref = llm.complete(
            messages + [refine_user], tools=[api.TAKE_ACTION_TOOL], tool_choice="auto"
        )
        if ref is None:
            status = "llm_error"
            break
        am, tcid, raw = api.extract_turn(ref)
        if tcid is not None:
            messages.append(refine_user)
            messages.append(am)
        else:  # refine produced no tool call — commit the original proposal
            messages.append(pm)
            tcid, raw = ptcid, praw
        action = api.canonicalize(raw, admissible)
        obs, admissible, done, won = env.step(action)
        last_obs = obs
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

    return {
        "won": bool(won),
        "messages": messages,
        "steps": llm.n_calls + critic.n_calls,
        "status": status,
    }
