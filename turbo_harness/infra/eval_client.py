"""Client for remote SWE-Smith evaluation.

This client sends evaluation requests to a remote server running eval_server.py.
Used when Docker is not available locally (e.g., on Mosaic).
"""

import os
import logging
import requests
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class RemoteEvaluationClient:
    """Client for communicating with remote evaluation server."""

    def __init__(self, server_url: str = None, timeout: int = 300):
        """Initialize the client.

        Args:
            server_url: URL of the evaluation server (e.g., http://YOUR_VM_IP:8082)
                       If None, reads from EVAL_SERVER_URL environment variable
            timeout: Maximum time to wait for evaluation (seconds)
        """
        self.server_url = server_url or os.environ.get("EVAL_SERVER_URL")
        if not self.server_url:
            raise ValueError(
                "server_url must be provided or EVAL_SERVER_URL environment variable must be set"
            )

        # Remove trailing slash
        self.server_url = self.server_url.rstrip("/")
        self.timeout = timeout

        # Test connection
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to evaluation server at {self.server_url}")
        except Exception as e:
            logger.error(f"Failed to connect to evaluation server: {e}")
            raise

    def _serialize_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Convert instance to JSON-serializable format.

        Handles numpy arrays and other non-serializable types.
        """
        import numpy as np

        serialized = {}
        for key, value in instance.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_instance(value)
            elif isinstance(value, list):
                serialized[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    def compute_score(
        self, patch: str, instance: Dict[str, Any]
    ) -> Tuple[float, str, str]:
        """Compute score by sending to remote evaluation server.

        Args:
            patch: The generated patch string
            instance: The problem instance dict from SWE-Smith dataset

        Returns:
            Tuple of (reward, run_id, info_string)
        """
        import time as _time

        serialized_instance = self._serialize_instance(instance)
        last_err = None
        # Retry only on transient connection errors (e.g. ephemeral-port exhaustion or a brief
        # server restart). A legitimate resolved=False comes back as a normal 200 and is never
        # retried, so this cannot mask a real failure — it only stops a transient hiccup from
        # zeroing a reward (which would otherwise add noise to RL / batch eval).
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.server_url}/evaluate_sync",
                    json={"patch": patch, "instance": serialized_instance},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                if "error" in result:
                    return 0.0, "", f"Remote evaluation failed: {result['error']}"
                return result["reward"], result["run_id"], result["info"]
            except requests.exceptions.ConnectionError as e:
                last_err = e
                _time.sleep(2 * (attempt + 1))
            except requests.exceptions.Timeout:
                logger.error(
                    f"Evaluation timeout for instance {instance.get('instance_id', 'unknown')}"
                )
                return 0.0, "ERROR", f"Evaluation timeout after {self.timeout}s"
            except Exception as e:
                logger.error(f"Remote evaluation error: {e}")
                return 0.0, "ERROR", f"Remote evaluation error: {str(e)}"
        logger.error(f"Remote evaluation error after 3 attempts: {last_err}")
        return 0.0, "ERROR", f"Remote evaluation error after retries: {last_err}"


_cached_client = None


def compute_score_remote(
    patch: str, instance: Dict[str, Any]
) -> Tuple[float, str, str]:
    """Convenience function that computes score via cached client.

    Uses EVAL_SERVER_URL environment variable for server location.
    """
    global _cached_client
    if _cached_client is None:
        _cached_client = RemoteEvaluationClient()
    return _cached_client.compute_score(patch, instance)
