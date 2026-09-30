"""Score ONE SWE-bench Verified instance in an isolated subprocess.

Called by scoring._compute_score_verified under a hard subprocess timeout so a wedged rootless-docker
op / test can be killed (the evolution/RL/eval run scoring in worker threads and cannot be signalled).
Reads {"instance": <gt dict>, "patch": <str>} as JSON on stdin; prints {"reward","run_id","info"}
as a single JSON line on stdout.
"""
import json
import sys


def main() -> None:
    data = json.load(sys.stdin)
    from turbo_harness.infra.scoring import _verified_score_impl
    reward, run_id, info = _verified_score_impl(data.get("patch", "") or "", data["instance"])
    sys.stdout.write(json.dumps({"reward": reward, "run_id": run_id, "info": info}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
