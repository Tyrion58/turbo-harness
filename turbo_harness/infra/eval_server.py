"""Remote evaluation server for SWE-Smith.

This server runs on a VM with Docker access and handles evaluation requests
from training/eval jobs running on Mosaic (which don't have Docker access).

Endpoints:
    GET /health - Health check
    POST /evaluate_sync - Synchronous evaluation (blocks until complete)

Usage:
    python -m turbo_harness.infra.eval_server --port 5152 --host 0.0.0.0

Requirements:
    - Docker must be installed and running
    - Port must be accessible from Mosaic cluster
"""

import argparse
import logging
from flask import Flask, request, jsonify

from .scoring import compute_score as _compute_score_local

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/evaluate_sync", methods=["POST"])
def evaluate_sync():
    """Synchronous evaluation endpoint (blocks until complete).

    Request body: same as /evaluate

    Returns:
    {
        "reward": float,
        "info": str
    }
    """
    try:
        data = request.get_json()

        if not data or "patch" not in data or "instance" not in data:
            return jsonify({"error": "Missing 'patch' or 'instance' in request"}), 400

        patch = data["patch"]
        instance = data["instance"]

        logger.info(
            f"Synchronous evaluation for instance {instance.get('instance_id', 'unknown')}"
        )

        reward, run_id, info = _compute_score_local(patch, instance)

        return jsonify({"reward": reward, "run_id": run_id, "info": info})

    except Exception as e:
        logger.error(f"Error in /evaluate_sync: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="SWE-Smith Remote Evaluation Server")
    parser.add_argument("--host", type=str, required=True, help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")

    args = parser.parse_args()

    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
