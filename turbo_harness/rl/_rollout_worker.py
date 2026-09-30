"""Isolated rollout worker for the RL env.

Reads a JSON payload on stdin, runs ONE harness+student rollout WITHOUT scoring
(score=False), and prints the resulting patch as JSON. The policy-influenced harness
code therefore executes only in this child process; the trusted parent scores the
returned patch string, so harness code cannot tamper with the reward computation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))


def main() -> None:
    payload = json.loads(sys.stdin.read())
    from turbo_harness import executor as EX

    try:
        r = EX.run_instance(
            payload["gt"],
            payload["student_model"],
            harness=payload["harness"],
            repo=payload["repo"],
            step_limit=int(payload["step_limit"]),
            cost_limit=float(payload["cost_limit"]),
            score=False,
        )
        out = {
            "patch": r.get("patch", ""),
            "steps": r.get("steps", int(payload["step_limit"])),
            "status": r.get("status", ""),
            "agent_cost": r.get("agent_cost", 0.0),
            "instance_id": r.get("instance_id", ""),
        }
    except Exception as e:  # infra failure inside the rollout
        out = {
            "patch": "",
            "steps": int(payload.get("step_limit", 40)),
            "status": f"ERROR:{type(e).__name__}:{e}",
            "agent_cost": 0.0,
            "instance_id": "",
        }

    print("ROLLOUT_JSON:" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
