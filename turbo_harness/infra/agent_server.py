"""Remote agent execution server for SWE-Smith.

This server runs on a VM with Docker access and manages agent sessions
for training/eval jobs running on Mosaic (which don't have Docker access).

Each session maintains a Docker container with the agent state across multiple RL steps.

Endpoints:
    GET /health - Health check
    POST /agent/session/create - Create agent session
    POST /agent/session/{session_id}/step - Execute agent steps
    GET /agent/session/{session_id}/patch - Get git diff
    DELETE /agent/session/{session_id} - Cleanup session

Usage:
    python -m advisor_models.swe_smith.agent_server --port 8081 --host 0.0.0.0

Requirements:
    - Docker must be installed and running
    - Port must be accessible from Mosaic cluster
"""

import argparse
import logging
import uuid
import time
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
import sys
import yaml

# Add mini-swe-agent to path
MINI_SWE_AGENT_PATH = Path(__file__).parent / "mini-swe-agent" / "src"
sys.path.insert(0, str(MINI_SWE_AGENT_PATH))

from minisweagent.agents.default import DefaultAgent, AgentConfig, TerminatingException  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class AgentSession:
    """Manages a single agent session with Docker container."""

    def __init__(
        self,
        session_id: str,
        instance: Dict[str, Any],
        model_name: str = "gpt-4.1-mini",
        max_steps: int = 40,
        cost_limit: float = 3.0,
        log_dir: Optional[str] = None,
    ):
        """Initialize agent session.

        Args:
            session_id: Unique session identifier
            instance: SWE-Smith instance dict
            model_name: LLM model to use
            max_steps: Maximum agent steps
            cost_limit: Maximum cost in dollars
        """
        self.session_id = session_id
        self.instance = instance
        self.model_name = model_name
        self.max_steps = max_steps
        self.cost_limit = cost_limit
        self.log_dir = Path(log_dir) if log_dir else None

        self.created_at = time.time()
        self.last_accessed = time.time()
        self.terminated = False
        self.exit_status = None
        self.total_steps = 0
        self.error = None
        self.step_counter = 0

        self.docker_env: Optional[DockerEnvironment] = None
        self.agent: Optional[DefaultAgent] = None
        self.repo_dir_in_container = "/testbed"
        self.original_commit: Optional[str] = None

        self.rolling_summary = ""
        self.last_summarized_step = 0

        self.log_file = self._setup_log_file() if self.log_dir else None

        self._initialize()

    def _initialize(self):
        """Initialize Docker container and agent."""
        try:
            logger.info(f"Initializing session {self.session_id}")

            repo_name = self.instance.get("repo", "")
            instance_id = self.instance.get("instance_id", "")
            image_name = self.instance.get("image_name", "")
            problem_statement = self.instance.get("problem_statement", "")

            if not image_name:
                raise ValueError(f"No image_name in instance {instance_id}")

            logger.info(
                f"Instance: {instance_id}, Repo: {repo_name}, Image: {image_name}"
            )

            logger.info(f"Starting Docker container with image {image_name}")
            self.docker_env = DockerEnvironment(
                image=image_name,
                cwd=self.repo_dir_in_container,
                timeout=60,
                run_args=["--rm", "--network", "host"],
            )

            logger.info(
                f"Checking out branch {instance_id} in {self.repo_dir_in_container}"
            )

            fetch_result = self.docker_env.execute(
                "git fetch --all",
                cwd=self.repo_dir_in_container,
                timeout=120,
            )
            if fetch_result["returncode"] != 0:
                logger.warning(f"Git fetch warning: {fetch_result['output']}")

            checkout_result = self.docker_env.execute(
                f"git checkout {instance_id}",
                cwd=self.repo_dir_in_container,
            )
            if checkout_result["returncode"] != 0:
                raise RuntimeError(
                    f"Failed to checkout branch {instance_id}: {checkout_result['output']}"
                )

            commit_result = self.docker_env.execute(
                "git rev-parse HEAD",
                cwd=self.repo_dir_in_container,
            )
            if commit_result["returncode"] == 0:
                self.original_commit = commit_result["output"].strip()
                logger.info(
                    f"Checked out branch {instance_id} at commit {self.original_commit[:8]}"
                )
            else:
                logger.warning(f"Could not get commit hash: {commit_result['output']}")
                logger.info(f"Checked out branch {instance_id}")

            config_path = (
                MINI_SWE_AGENT_PATH / "minisweagent" / "config" / "default.yaml"
            )
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            agent_config_data = config_data.get("agent", {})
            agent_config = AgentConfig(
                system_template=agent_config_data.get("system_template", ""),
                instance_template=agent_config_data.get("instance_template", ""),
                action_observation_template=agent_config_data.get(
                    "action_observation_template", ""
                ),
                format_error_template=agent_config_data.get(
                    "format_error_template", ""
                ),
                step_limit=self.max_steps,
                cost_limit=self.cost_limit,
            )

            model = LitellmModel(model_name=self.model_name)
            self.agent = DefaultAgent(
                model,
                self.docker_env,
                config_class=lambda **kwargs: agent_config,
            )

            uname_result = self.docker_env.execute("uname -srvm")
            uname_parts = uname_result["output"].strip().split()
            system_vars = {
                "system": uname_parts[0] if len(uname_parts) > 0 else "Linux",
                "release": uname_parts[1] if len(uname_parts) > 1 else "",
                "version": uname_parts[2] if len(uname_parts) > 2 else "",
                "machine": uname_parts[3] if len(uname_parts) > 3 else "x86_64",
            }

            self.agent.extra_template_vars = {"task": problem_statement, **system_vars}
            self.agent.messages = []
            self.agent.add_message(
                "system", self.agent.render_template(agent_config.system_template)
            )
            self.agent.add_message(
                "user", self.agent.render_template(agent_config.instance_template)
            )

            self._log_initialization(problem_statement)

            logger.info(f"Session {self.session_id} initialized successfully")

        except Exception as e:
            logger.error(
                f"Error initializing session {self.session_id}: {e}", exc_info=True
            )
            self.error = str(e)
            self.terminated = True
            self.cleanup()
            raise

    def execute_step(
        self, advisor_feedback: str, max_steps_per_call: int = 5
    ) -> Dict[str, Any]:
        """Execute agent steps with advisor feedback.

        Args:
            advisor_feedback: Feedback from the advisor model
            max_steps_per_call: Maximum number of agent steps to execute

        Returns:
            Dict with observation, terminated flag, and metadata
        """
        self.last_accessed = time.time()
        self.step_counter += 1

        if self.terminated:
            return {
                "terminated": True,
                "error": self.error or "Session already terminated",
                "total_steps": self.total_steps,
            }

        try:
            self._log_advisor_feedback(advisor_feedback)

            self.agent.add_advice(advisor_feedback)

            agent_actions_start_idx = len(self.agent.messages)

            for i in range(max_steps_per_call):
                if self.agent.model.n_calls >= self.max_steps:
                    logger.info(f"Session {self.session_id} reached max steps")
                    self.terminated = True
                    self.exit_status = "LimitsExceeded"
                    break

                if self.agent.model.cost >= self.cost_limit:
                    logger.info(f"Session {self.session_id} reached cost limit")
                    self.terminated = True
                    self.exit_status = "LimitsExceeded"
                    break

                try:
                    self.total_steps += 1
                    self.agent.step()
                except TerminatingException as e:
                    logger.info(f"Session {self.session_id} terminated by agent: {e}")
                    self.terminated = True
                    self.exit_status = type(e).__name__
                    break
                except Exception as e:
                    import litellm

                    if isinstance(e, litellm.exceptions.RateLimitError):
                        logger.warning(f"Rate limit hit for session {self.session_id}")
                        self.terminated = True
                        self.exit_status = "RateLimitError"
                        break

                    error_msg = self._clean_error_message(e)
                    self.agent.add_message("user", error_msg)

            observation = self._build_observation()

            agent_actions = self.agent.messages[agent_actions_start_idx:]
            self._log_step_completion(agent_actions, observation)

            return {
                "terminated": self.terminated,
                "exit_status": self.exit_status,
                "observation": observation,
                "total_steps": self.total_steps,
                "cost": self.agent.model.cost,
                "rolling_summary": self.rolling_summary,
                "last_summarized_step": self.last_summarized_step,
            }

        except Exception as e:
            logger.error(
                f"Error in execute_step for session {self.session_id}: {e}",
                exc_info=True,
            )
            self.error = str(e)
            self.terminated = True
            return {
                "terminated": True,
                "error": str(e),
                "total_steps": self.total_steps,
            }

    def _build_observation(self) -> str:
        """Build observation for advisor with rolling summary."""
        from .scoring import build_advisor_observation

        (
            observation_text,
            self.rolling_summary,
            self.last_summarized_step,
            summary_cost,
        ) = build_advisor_observation(
            agent_history=self.agent.messages,
            rolling_summary=self.rolling_summary,
            last_summarized_step=self.last_summarized_step,
            current_step=self.total_steps,
            summarizer_model="gpt-4o-mini",
        )

        return observation_text

    def _clean_error_message(self, error: Exception, max_length: int = 300) -> str:
        """Clean and truncate error messages."""
        from itertools import groupby

        error_str = str(error)
        lines = error_str.split("\n")
        cleaned_lines = []
        for line, group in groupby(lines):
            count = sum(1 for _ in group)
            cleaned_lines.append(line)
            if count > 3:
                cleaned_lines.append(f"... (repeated {count} times)")
            elif count > 1:
                cleaned_lines.extend([line] * (count - 1))

        cleaned_str = "\n".join(cleaned_lines)

        if len(cleaned_str) > max_length:
            cleaned_str = cleaned_str[:max_length] + "\n... (truncated)"

        error_type = type(error).__name__
        return f"[{error_type}] {cleaned_str}"

    def get_patch(self) -> str:
        """Get git diff from the session. Only returns patch if agent submitted.

        Returns:
            Git diff as string (empty if agent did not submit)
        """
        self.last_accessed = time.time()

        try:
            if not self.docker_env or not self.original_commit:
                return ""

            if self.exit_status != "Submitted":
                return ""

            self.docker_env.execute(
                "git add -A",
                cwd=self.repo_dir_in_container,
            )
            result = self.docker_env.execute(
                "git diff --cached",
                cwd=self.repo_dir_in_container,
            )

            return result["output"]

        except Exception as e:
            logger.error(f"Error getting patch for session {self.session_id}: {e}")
            return ""

    def cleanup(self):
        """Cleanup Docker container and resources."""
        try:
            if self.docker_env:
                logger.info(f"Cleaning up session {self.session_id}")
                self.docker_env.cleanup()
                self.docker_env = None
        except Exception as e:
            logger.error(f"Error cleaning up session {self.session_id}: {e}")

    def _setup_log_file(self) -> Optional[Path]:
        """Setup log file for this session.

        Returns:
            Path to log file, or None if logging disabled
        """
        if not self.log_dir:
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_id = self.instance.get("instance_id", "unknown")
        instance_id_safe = instance_id.replace("/", "_").replace(":", "_")

        log_filename = f"{timestamp}_{instance_id_safe}_{self.session_id[:8]}.jsonl"
        log_path = self.log_dir / log_filename

        logger.info(f"Session {self.session_id} logging to {log_path}")
        return log_path

    def _log_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the log file.

        Args:
            entry: Dictionary to log as JSON
        """
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to log file {self.log_file}: {e}")

    def _log_initialization(self, problem_statement: str):
        """Log session initialization.

        Args:
            problem_statement: The problem to solve
        """
        entry = {
            "type": "initialization",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "instance_id": self.instance.get("instance_id", "unknown"),
            "repo": self.instance.get("repo", "unknown"),
            "model_name": self.model_name,
            "problem_statement": problem_statement,
            "max_steps": self.max_steps,
            "cost_limit": self.cost_limit,
        }
        self._log_entry(entry)

    def _log_advisor_feedback(self, advisor_feedback: str):
        """Log advisor feedback received.

        Args:
            advisor_feedback: Feedback from advisor model
        """
        entry = {
            "type": "advisor_feedback",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step": self.step_counter,
            "total_agent_steps": self.total_steps,
            "advisor_feedback": advisor_feedback,
        }
        self._log_entry(entry)

    def _log_step_completion(self, agent_actions: list, observation: str):
        """Log completion of a step with agent actions and observation.

        Args:
            agent_actions: List of agent messages during this step
            observation: Observation returned to training loop
        """
        formatted_actions = []
        for msg in agent_actions:
            formatted_actions.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )

        entry = {
            "type": "step_completion",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step": self.step_counter,
            "total_agent_steps": self.total_steps,
            "agent_actions": formatted_actions,
            "observation_returned": observation,
            "terminated": self.terminated,
            "cost": self.agent.model.cost if self.agent else 0.0,
        }
        self._log_entry(entry)

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


class AgentSessionManager:
    """Manages multiple agent sessions."""

    def __init__(
        self,
        max_sessions: int = 128,
        session_timeout: int = 600,
        log_dir: Optional[str] = None,
    ):
        """Initialize session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions
            session_timeout: Session timeout in seconds
            log_dir: Directory for session logs (None to disable logging)
        """
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.log_dir = log_dir
        self.sessions: Dict[str, AgentSession] = {}
        self.lock = threading.Lock()

        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def create_session(
        self, instance: Dict[str, Any], model_name: str = "gpt-4.1-mini"
    ) -> str:
        """Create a new agent session.

        Args:
            instance: SWE-Smith instance dict
            model_name: LLM model to use for the agent

        Returns:
            session_id

        Raises:
            RuntimeError: If max sessions reached
        """
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.max_sessions}) reached. Please try again later."
                )

            session_id = str(uuid.uuid4())

            logger.info(f"Creating session {session_id} with model {model_name}")
            session = AgentSession(
                session_id, instance, model_name=model_name, log_dir=self.log_dir
            )
            self.sessions[session_id] = session

            logger.info(
                f"Session {session_id} created. Total sessions: {len(self.sessions)}"
            )
            return session_id

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            AgentSession or None if not found
        """
        with self.lock:
            return self.sessions.get(session_id)

    def cleanup_session(self, session_id: str):
        """Cleanup and remove a session.

        Args:
            session_id: Session identifier
        """
        with self.lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.cleanup()
                logger.info(
                    f"Session {session_id} removed. Total sessions: {len(self.sessions)}"
                )

    def _cleanup_loop(self):
        """Background thread that cleans up stale sessions."""
        while True:
            time.sleep(60)

            try:
                current_time = time.time()
                to_cleanup = []

                with self.lock:
                    for session_id, session in self.sessions.items():
                        if current_time - session.last_accessed > self.session_timeout:
                            logger.info(
                                f"Session {session_id} timed out (inactive for {self.session_timeout}s)"
                            )
                            to_cleanup.append(session_id)

                for session_id in to_cleanup:
                    self.cleanup_session(session_id)

                if to_cleanup:
                    logger.info(f"Cleaned up {len(to_cleanup)} stale sessions")

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)


session_manager = AgentSessionManager(
    max_sessions=128, session_timeout=600, log_dir=None
)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "active_sessions": len(session_manager.sessions),
            "max_sessions": session_manager.max_sessions,
        }
    )


@app.route("/agent/session/create", methods=["POST"])
def create_session():
    """Create a new agent session.

    Request body:
    {
        "instance": {
            "instance_id": "...",
            "repo": "...",
            "image_name": "...",
            "problem_statement": "...",
            ...
        },
        "model_name": "gpt-4.1-mini"
    }

    Returns:
    {
        "session_id": "uuid",
        "status": "created"
    }
    """
    try:
        data = request.get_json()

        if not data or "instance" not in data:
            return jsonify({"error": "Missing 'instance' in request"}), 400

        instance = data["instance"]
        model_name = data.get("model_name", "gpt-4.1-mini")

        session_id = session_manager.create_session(instance, model_name=model_name)

        return jsonify({"session_id": session_id, "status": "created"})

    except RuntimeError as e:
        logger.warning(f"Session creation failed: {e}")
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.error(f"Error in /agent/session/create: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/agent/session/<session_id>/step", methods=["POST"])
def execute_step(session_id: str):
    """Execute agent steps with advisor feedback.

    Request body:
    {
        "advisor_feedback": "...",
        "max_steps": 5
    }

    Returns:
    {
        "terminated": bool,
        "observation": str (if not terminated),
        "total_steps": int,
        "cost": float,
        "error": str (if error)
    }
    """
    try:
        data = request.get_json()

        if not data or "advisor_feedback" not in data:
            return jsonify({"error": "Missing 'advisor_feedback' in request"}), 400

        advisor_feedback = data["advisor_feedback"]
        max_steps = data.get("max_steps", 5)

        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        result = session.execute_step(advisor_feedback, max_steps_per_call=max_steps)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /agent/session/{session_id}/step: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/agent/session/<session_id>/patch", methods=["GET"])
def get_patch(session_id: str):
    """Get git diff from session.

    Returns:
    {
        "patch": "diff content..."
    }
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        patch = session.get_patch()

        return jsonify({"patch": patch})

    except Exception as e:
        logger.error(f"Error in /agent/session/{session_id}/patch: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/agent/session/<session_id>", methods=["DELETE"])
def cleanup_session(session_id: str):
    """Cleanup and remove a session.

    Returns:
    {
        "status": "cleaned_up"
    }
    """
    try:
        logger.info(f"Received DELETE request for session {session_id}")
        session_manager.cleanup_session(session_id)
        logger.info(f"Successfully cleaned up session {session_id} via DELETE request")
        return jsonify({"status": "cleaned_up"})

    except Exception as e:
        logger.error(f"Error in DELETE /agent/session/{session_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="SWE-Smith Remote Agent Server")
    parser.add_argument("--host", type=str, required=True, help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for session logs (omit to disable logging)",
    )

    args = parser.parse_args()

    global session_manager
    session_manager = AgentSessionManager(
        max_sessions=128, session_timeout=600, log_dir=args.log_dir
    )

    logger.info(f"Starting agent server on {args.host}:{args.port}")
    logger.info(f"Max sessions: {session_manager.max_sessions}")
    logger.info(f"Session timeout: {session_manager.session_timeout}s")
    if args.log_dir:
        logger.info(f"Session logs directory: {args.log_dir}")
    else:
        logger.info("Session logging disabled (no log directory specified)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
