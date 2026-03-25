"""Download the exact OpenMMLab checkpoints required by this repository."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pose_module.model_registry import (
    CHECKPOINTS_DIR,
    find_local_checkpoint_path,
    find_local_config_path,
    required_download_specs,
)
from pose_module.openmmlab_runtime import DEFAULT_OPENMMLAB_ENV_NAME


def download_required_models(
    *,
    env_name: str = DEFAULT_OPENMMLAB_ENV_NAME,
    dest_dir: str | Path = CHECKPOINTS_DIR,
    force: bool = False,
) -> Dict[str, Any]:
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)

    downloads: List[Dict[str, Any]] = []
    for spec in required_download_specs():
        existing_config = find_local_config_path(spec, checkpoints_dir=destination)
        existing_checkpoint = find_local_checkpoint_path(spec, checkpoints_dir=destination)
        if existing_config is not None and existing_checkpoint is not None and not force:
            downloads.append(
                {
                    "package_name": spec.package_name,
                    "model_id": spec.model_id,
                    "status": "skipped",
                    "config_path": str(existing_config),
                    "checkpoint_path": str(existing_checkpoint),
                }
            )
            continue

        command = [
            "conda",
            "run",
            "-n",
            str(env_name),
            "python",
            "-m",
            "mim",
            "download",
            spec.package_name,
            "--config",
            spec.model_id,
            "--dest",
            str(destination.resolve()),
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        current_config = find_local_config_path(spec, checkpoints_dir=destination)
        current_checkpoint = find_local_checkpoint_path(spec, checkpoints_dir=destination)
        downloads.append(
            {
                "package_name": spec.package_name,
                "model_id": spec.model_id,
                "status": "ok" if completed.returncode == 0 else "fail",
                "command": command,
                "returncode": int(completed.returncode),
                "config_path": None if current_config is None else str(current_config),
                "checkpoint_path": None if current_checkpoint is None else str(current_checkpoint),
                "stdout_tail": _tail_text(completed.stdout, max_chars=4000),
                "stderr_tail": _tail_text(completed.stderr, max_chars=4000),
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to download {spec.model_id} with env {env_name}. "
                f"See stdout/stderr in the download report."
            )

    return {
        "status": "ok",
        "env_name": str(env_name),
        "destination": str(destination.resolve()),
        "downloads": downloads,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download local MMPose/MMDet checkpoints into pose_module/checkpoints.")
    parser.add_argument(
        "--env-name",
        type=str,
        default=DEFAULT_OPENMMLAB_ENV_NAME,
        help="Conda env that contains Python 3.8 + mim + OpenMMLab packages.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Destination folder for local checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download models even if local files already exist.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = download_required_models(
        env_name=str(args.env_name),
        dest_dir=args.dest_dir,
        force=bool(args.force),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


def _tail_text(raw_text: str, *, max_chars: int) -> str:
    text = str(raw_text or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


if __name__ == "__main__":
    raise SystemExit(main())
