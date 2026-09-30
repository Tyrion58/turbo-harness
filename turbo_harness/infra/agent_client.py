"""Client for remote SWE-Smith agent execution.

This client sends agent execution requests to a remote server running agent_server.py.
Used when Docker is not available locally (e.g., on Mosaic).
"""

import os
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RemoteAgentClient:
    """Client for communicating with remote agent execution server."""

    def __init__(self, server_url: str = None, timeout: int = 600):
        """Initialize the client.

        Args:
            server_url: URL of the agent server (e.g., http://YOUR_VM_IP:8081)
                       If None, reads from AGENT_SERVER_URL environment variable
            timeout: Maximum time to wait for operations (seconds)
        """
        self.server_url = server_url or os.environ.get("AGENT_SERVER_URL")
        if not self.server_url:
            raise ValueError(
                "server_url must be provided or AGENT_SERVER_URL environment variable must be set"
            )

        # Remove trailing slash
        self.server_url = self.server_url.rstrip("/")
        self.timeout = timeout

        # Test connection
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            health_data = response.json()
            logger.info(
                f"Connected to agent server at {self.server_url} "
                f"({health_data.get('active_sessions', 0)}/{health_data.get('max_sessions', 0)} sessions)"
            )
        except Exception as e:
            logger.error(f"Failed to connect to agent server: {e}")
            raise

    def create_session(
        self, instance: Dict[str, Any], model_name: str = "gpt-4.1-mini"
    ) -> str:
        """Create a new agent session.

        Args:
            instance: SWE-Smith instance dict with repo, problem_statement, etc.
            model_name: LLM model to use for the agent (default: gpt-4.1-mini)

        Returns:
            session_id: Unique identifier for this session

        Raises:
            RuntimeError: If session creation fails
        """
        try:
            response = requests.post(
                f"{self.server_url}/agent/session/create",
                json={"instance": instance, "model_name": model_name},
                timeout=120,  # Container startup can take time
            )

            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise RuntimeError(f"Failed to create session: {result['error']}")

            session_id = result["session_id"]
            logger.info(f"Created remote agent session: {session_id}")
            return session_id

        except requests.exceptions.Timeout:
            logger.error("Session creation timeout")
            raise RuntimeError("Session creation timeout after 120s")
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise RuntimeError(f"Failed to create session: {str(e)}")

    def execute_step(
        self, session_id: str, advisor_feedback: str, max_steps: int = 5
    ) -> Dict[str, Any]:
        """Execute agent steps with advisor feedback.

        Args:
            session_id: Session identifier from create_session
            advisor_feedback: Feedback from the advisor model
            max_steps: Maximum number of agent steps to execute

        Returns:
            Dict with:
                - terminated: bool
                - observation: str (if not terminated)
                - total_steps: int
                - cost: float
                - rolling_summary: str
                - last_summarized_step: int
                - error: str (if error occurred)

        Raises:
            RuntimeError: If execution fails
        """
        try:
            response = requests.post(
                f"{self.server_url}/agent/session/{session_id}/step",
                json={
                    "advisor_feedback": advisor_feedback,
                    "max_steps": max_steps,
                },
                timeout=self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            if "error" in result and result.get("terminated"):
                logger.warning(f"Agent step failed: {result['error']}")

            return result

        except requests.exceptions.Timeout:
            logger.error(f"Agent step timeout for session {session_id}")
            return {
                "terminated": True,
                "error": f"Agent step timeout after {self.timeout}s",
                "total_steps": 0,
            }
        except Exception as e:
            logger.error(f"Error executing agent step: {e}")
            return {
                "terminated": True,
                "error": f"Agent step error: {str(e)}",
                "total_steps": 0,
            }

    def get_patch(self, session_id: str) -> str:
        """Get git diff from the session.

        Args:
            session_id: Session identifier

        Returns:
            Git diff as string

        Raises:
            RuntimeError: If getting patch fails
        """
        try:
            response = requests.get(
                f"{self.server_url}/agent/session/{session_id}/patch",
                timeout=30,
            )

            response.raise_for_status()
            result = response.json()

            if "error" in result:
                logger.error(f"Failed to get patch: {result['error']}")
                return ""

            return result.get("patch", "")

        except Exception as e:
            logger.error(f"Error getting patch: {e}")
            return ""

    def cleanup_session(self, session_id: str):
        """Cleanup and remove a session.

        Args:
            session_id: Session identifier
        """
        try:
            response = requests.delete(
                f"{self.server_url}/agent/session/{session_id}",
                timeout=30,
            )

            response.raise_for_status()
            logger.info(f"Cleaned up remote session: {session_id}")

        except Exception as e:
            logger.warning(f"Error cleaning up session {session_id}: {e}")
            # Don't raise - cleanup is best effort
