"""Robust image acquisition for SWE-bench Verified (~3GB images, Docker-Hub rate-limited).

Docker Hub throttles ANONYMOUS pulls (~100/6h per IP). A plain `docker run` on a missing image
auto-pulls, gets `toomanyrequests`, and exits 125 -> the instance errors with 0 steps and scores 0
(looks like a terrible student; is actually a bug). `ensure_image()` fixes that by pulling with a
429-aware backoff (wait for the window to reset and retry) BEFORE the container starts, so the
one-time seed of the image set never turns into instance failures.

Two modes:
  * default (VERIFIED_IMG_CACHE unset): pull-with-backoff only. Intended for a docker data-root with
    room to KEEP images (pull-once-keep, no rmi) -> each image pulls once, then is reused live.
  * tar cache (VERIFIED_IMG_CACHE=<dir>): additionally `docker save` each pulled image to a tar on a
    big disk and `docker load` from it next time. Use only when the docker disk is too small to keep
    images resident (bounded working set + VERIFIED_RM_IMAGE=1) so re-use is a local load, not a pull.

Env:
  VERIFIED_IMG_CACHE   tar dir; if unset, tar caching is OFF (pull-once-keep mode)
  VERIFIED_IMG_LOCKS   per-image flock dir (default /tmp/verified_img_locks)
  MSWEA_DOCKER_EXECUTABLE  docker binary (default "docker")
  PULL_BACKOFF_S       sleep between 429 retries (default 1200)
  PULL_MAX_WAIT_S      give up after this much cumulative 429 backoff (default 18h)
"""
from __future__ import annotations

import fcntl
import os
import pathlib
import subprocess
import time

DOCKER = os.environ.get("MSWEA_DOCKER_EXECUTABLE", "docker")
LOCK_DIR = os.environ.get("VERIFIED_IMG_LOCKS", "/tmp/verified_img_locks")
PULL_BACKOFF_S = int(os.environ.get("PULL_BACKOFF_S", "1200"))
PULL_MAX_WAIT_S = int(os.environ.get("PULL_MAX_WAIT_S", str(18 * 3600)))


def _cache_dir() -> str:
    return os.environ.get("VERIFIED_IMG_CACHE", "")


def _run(args, timeout=None):
    return subprocess.run([DOCKER, *args], capture_output=True, text=True, timeout=timeout)


def _tar_path(image: str) -> pathlib.Path:
    safe = image.replace("/", "_").replace(":", "_")
    return pathlib.Path(_cache_dir()) / f"{safe}.tar"


def _present(image: str) -> bool:
    # MUST have a timeout: a bare `docker image inspect` can wedge forever if the rootless
    # (fuse-overlayfs) daemon momentarily stalls under load, and that would deadlock the caller.
    try:
        return _run(["image", "inspect", image], timeout=60).returncode == 0
    except subprocess.TimeoutExpired:
        return False  # treat as not-present; the locked pull path (with its own timeout) handles it


def _pull_with_backoff(image: str) -> None:
    waited = 0
    while True:
        try:
            r = _run(["pull", image], timeout=1800)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"docker pull wedged > 1800s (daemon stall): {image}")
        if r.returncode == 0:
            return
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "toomanyrequests" in err or "rate limit" in err:
            if waited >= PULL_MAX_WAIT_S:
                raise RuntimeError(f"pull rate-limited > {waited}s, giving up: {image}")
            time.sleep(PULL_BACKOFF_S)
            waited += PULL_BACKOFF_S
            continue
        raise RuntimeError(f"docker pull failed for {image}: {err[:300]}")


def ensure_image(image: str, allow_pull: bool = True) -> str:
    """Make `image` available locally, robust to Docker-Hub rate limits.

    Returns present|loaded|pulled|absent. A per-image flock serializes concurrent workers that
    need the same instance image (e.g. GRPO rollout groups) so it is pulled/loaded only once.
    """
    if not image:
        return "absent"
    if _present(image):
        return "present"
    os.makedirs(LOCK_DIR, exist_ok=True)
    lockname = image.replace("/", "_").replace(":", "_") + ".lock"
    with open(os.path.join(LOCK_DIR, lockname), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if _present(image):                 # another worker finished while we waited on the lock
            return "present"
        tar = _tar_path(image) if _cache_dir() else None
        if tar is not None and tar.exists():
            r = _run(["load", "-i", str(tar)], timeout=1800)
            if r.returncode == 0 and _present(image):
                return "loaded"
            try:                            # corrupt/partial tar -> drop and fall through to pull
                tar.unlink()
            except OSError:
                pass
        if not allow_pull:
            return "absent"
        _pull_with_backoff(image)
        if tar is not None:                 # persist to the tar cache (tmp+rename = crash-safe)
            os.makedirs(_cache_dir(), exist_ok=True)
            tmp = str(tar) + ".tmp"
            rs = _run(["save", "-o", tmp, image], timeout=3600)
            if rs.returncode == 0:
                os.replace(tmp, tar)
            else:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return "pulled"
