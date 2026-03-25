"""Shared runtime helpers for OpenMMLab backends."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pose_module.io.cache import tail_text


DEFAULT_OPENMMLAB_ENV_NAME = "openmmlab"


def select_openmmlab_launcher(
    env_name: str,
    *,
    cwd: Path,
    probe_code: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    diagnostics: List[Dict[str, Any]] = []
    for candidate in build_openmmlab_launcher_candidates(env_name):
        probe_command = list(candidate["prefix"]) + ["-c", probe_code]
        completed = subprocess.run(
            probe_command,
            cwd=str(cwd.resolve()),
            text=True,
            capture_output=True,
            check=False,
        )
        diagnostics.append(
            {
                "name": str(candidate["name"]),
                "python": str(candidate["python"]),
                "command": probe_command,
                "returncode": int(completed.returncode),
                "stdout_tail": tail_text(completed.stdout, max_chars=2000) if completed.stdout else "",
                "stderr_tail": tail_text(completed.stderr, max_chars=2000) if completed.stderr else "",
            }
        )
        if completed.returncode == 0:
            return candidate, diagnostics
    return None, diagnostics


def build_openmmlab_launcher_candidates(env_name: str) -> List[Dict[str, Any]]:
    normalized = str(env_name).strip()
    lowered = normalized.lower()
    current_python = {
        "name": "current_python",
        "python": str(Path(sys.executable).resolve()),
        "prefix": [str(Path(sys.executable).resolve())],
    }

    if lowered == "current":
        return [current_python]

    if lowered in {"", "auto"}:
        return _deduplicate_candidates(
            _build_conda_candidates(DEFAULT_OPENMMLAB_ENV_NAME) + [current_python]
        )

    return _deduplicate_candidates(_build_conda_candidates(normalized))


def resolve_conda_env_python(env_name: str) -> Optional[Path]:
    try:
        completed = subprocess.run(
            ["conda", "env", "list", "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if completed.returncode != 0:
        return None

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None

    env_paths = payload.get("envs", [])
    if not isinstance(env_paths, list):
        return None

    normalized = str(env_name).strip()
    for raw_env_path in env_paths:
        env_path = Path(str(raw_env_path))
        if env_path.name != normalized:
            continue
        python_path = env_path / "bin" / "python"
        if python_path.exists():
            return python_path.resolve()
    return None


def _build_conda_candidates(env_name: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    resolved_python = resolve_conda_env_python(env_name)
    if resolved_python is not None:
        candidates.append(
            {
                "name": "conda_env_python",
                "python": str(resolved_python),
                "prefix": [str(resolved_python)],
            }
        )
    candidates.append(
        {
            "name": "conda_env",
            "python": f"conda:{env_name}",
            "prefix": ["conda", "run", "-n", env_name, "python"],
        }
    )
    return candidates


def _deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduplicated: List[Dict[str, Any]] = []
    seen_prefixes = set()
    for candidate in candidates:
        prefix_key = tuple(str(value) for value in candidate["prefix"])
        if prefix_key in seen_prefixes:
            continue
        seen_prefixes.add(prefix_key)
        deduplicated.append(candidate)
    return deduplicated
