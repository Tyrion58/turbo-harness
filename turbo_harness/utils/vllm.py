"""Utility for starting a local vLLM server."""

import os
import subprocess
import time

import requests


def start_vllm_server(
    model_to_serve_name: str,
    served_model_name: str = "advisor_model",
    max_model_len: int = 32768,
    tensor_parallel_size: int = 4,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> subprocess.Popen:
    import shutil

    vllm_python = shutil.which("python")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in [
        os.path.join(repo_root, "SkyRL", ".venv", "bin", "python"),
        os.path.join(repo_root, "SkyRL", "skyrl-train", ".venv", "bin", "python"),
    ]:
        if os.path.exists(candidate):
            vllm_python = candidate
            break

    cmd = [
        vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_to_serve_name,
        "--served-model-name", served_model_name,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]

    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    print(f"Starting vLLM server: {' '.join(cmd)}")
    log_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm_server.log"), "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    # Wait for server to be ready
    url = f"http://{host}:{port}/health"
    for i in range(300):
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                print(f"vLLM server ready after {i + 1} seconds")
                return process
        except requests.ConnectionError:
            pass
        time.sleep(1)

    raise RuntimeError("vLLM server failed to start within 300 seconds")
