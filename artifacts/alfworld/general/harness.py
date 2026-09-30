"""Tight loop-breaking harness with few-shot for multi-step tasks.

Combines iter5's successful few-shot approach with targeted loop detection
for the 5+ tasks stuck in severe observation loops (35-47 consecutive repeats).

ROOT CAUSE analysis from failed iter5 trajectories:
- Task 5: "look" action looped 47 times, seeing "You are facing the desk 1. Next to it, you see nothing."
- Task 51, 54, 61: Agent alternated "go to drawer 2" (canonicalized to cabinet 2) ↔ "go to drawer 1" 30+ times
- Task 49, 69: Similar tight loops with 35-44 consecutive identical observations

Why iter3's loop detection REGRESSED (-2%):
- Triggered too easily (3+ in last 5 observations) → false positives on valid exploration
- Generic nudge ("try different action") → noise without actionable guidance
- Applied to baseline without few-shot → hurt more than helped

NEW APPROACH for iter6:
- Start from iter5's few-shot (clean/heat/cool) which improved 49% → 62%
- Add MUCH TIGHTER loop detection: only trigger on 5+ CONSECUTIVE identical observations
- Actionable nudge: "You've seen this exact observation 5+ times in a row. Try exploring a completely different location."
- Conservative: limit to 1 nudge per 10 steps to avoid context pollution

Expected impact:
- Fix 5-10 tasks stuck in severe loops (5, 49, 54, 61, 69)
- Maintain 62% on tasks iter5 already solves (few-shot keeps working)
- No regression on easy tasks (tight threshold avoids false positives)

Interface: run_episode(env, llm, max_steps) -> {"won": bool, "messages": [...], "steps": int, "status": str}
"""

from turbo_harness.alfworld import api


def detect_tight_loop(recent_obs):
    """Detect if we're stuck in an extremely tight observation loop.

    Returns True if the last 5 observations are ALL identical.
    This is MUCH stricter than iter3's "3+ in last 5" to avoid false positives.
    """
    if len(recent_obs) < 5:
        return False

    # Check if the last 5 observations are all the same
    last_5 = recent_obs[-5:]
    first = last_5[0].strip().lower()

    return all(obs.strip().lower() == first for obs in last_5)


def run_episode(env, llm, max_steps):
    obs, admissible = env.reset()

    # Detect task category
    category = api.task_category(env.game_relpath)

    # Build initial messages
    messages = [{"role": "system", "content": api.SYSTEM_PROMPT}]

    # Add few-shot examples for multi-step tasks ONLY (from iter5)
    if category in ("clean", "heat", "cool"):
        fewshot = api.load_fewshot(category)
        messages.extend(fewshot)

    # Add the actual task
    messages.append({
        "role": "user",
        "content": "Here is your task. "
        + obs
        + api.available_actions_str(admissible),
    })

    won, status = False, "max_steps"
    recent_observations = []  # Track recent observations for loop detection
    steps_since_last_nudge = 0  # Prevent nudge spam

    for _ in range(max_steps):
        resp = llm.complete(messages, tools=[api.TAKE_ACTION_TOOL], tool_choice="auto")
        if resp is None:
            status = "llm_error"
            break
        assistant_msg, tool_call_id, raw = api.extract_turn(resp)
        messages.append(assistant_msg)

        if tool_call_id is None:
            # Model didn't call the tool — remind and retry (no env step)
            messages.append({
                "role": "user",
                "content": "You MUST call the take_action tool with exactly one action "
                "from the AVAILABLE ACTIONS.",
            })
            continue

        action = api.canonicalize(raw, admissible)
        obs, admissible, done, won = env.step(action)

        # Update observation history BEFORE checking loop
        recent_observations.append(obs)
        if len(recent_observations) > 10:
            recent_observations.pop(0)

        # Check for TIGHT loop (5+ consecutive identical observations)
        in_tight_loop = detect_tight_loop(recent_observations)

        # Build feedback
        feedback = obs + api.available_actions_str(admissible)

        # If stuck in a tight loop AND we haven't nudged recently, add actionable guidance
        if in_tight_loop and steps_since_last_nudge >= 10:
            feedback += "\n\n[Note: You've seen this exact observation 5+ times in a row. You're stuck in a loop. Try exploring a completely different location or trying a fundamentally different action.]"
            steps_since_last_nudge = 0
        else:
            steps_since_last_nudge += 1

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": feedback,
        })

        if done:
            status = "done"
            break

    return {
        "won": bool(won),
        "messages": messages,
        "steps": llm.n_calls,
        "status": status,
    }
