"""ViTPose estimator backend and cross-environment launcher."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from pose_module.model_registry import resolve_local_pose2d_backend_artifacts
from pose_module.openmmlab_runtime import select_openmmlab_launcher
from pose_module.export.debug_video import resolve_debug_overlay_path
from pose_module.interfaces import Pose2DJob, Pose2DResult
from pose_module.io.cache import load_json_file, tail_text, write_json_file
from pose_module.io.video_loader import select_frame_indices


def run_backend_job(
    *,
    job: Pose2DJob,
    env_name: str,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_json_path = output_dir / "backend_job.json"
    result_json_path = output_dir / "backend_result.json"
    write_json_file(job.to_dict(), job_json_path)

    repo_root = Path(__file__).resolve().parents[2]
    launcher, probe_diagnostics = select_openmmlab_launcher(
        str(env_name),
        cwd=repo_root,
        probe_code="import mmpose, mmdet, mmpretrain; print('ok')",
    )
    if launcher is None:
        backend_run = {
            "status": "fail",
            "effective_fps": None,
            "selected_frame_indices": [],
            "artifacts": {
                "raw_prediction_json_path": str(job.raw_prediction_path.resolve()),
                "debug_overlay_path": str(job.debug_overlay_path.resolve()) if job.save_debug else None,
            },
            "quality_report": {},
            "backend": {
                "launcher": None,
                "probe_diagnostics": probe_diagnostics,
            },
            "error": "No Python launcher with mmpose/mmdet/mmpretrain available. "
            "Use --env-name openmmlab and populate pose_module/checkpoints with local model files.",
            "env_name": str(env_name),
            "returncode": 1,
        }
        write_json_file(backend_run, output_dir / "backend_run.json")
        return backend_run

    command = list(launcher["prefix"]) + [
        "-m",
        "pose_module.vitpose.estimator",
        "--job-json",
        str(job_json_path.resolve()),
        "--result-json",
        str(result_json_path.resolve()),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root.resolve())
        if existing_pythonpath == ""
        else str(repo_root.resolve()) + os.pathsep + existing_pythonpath
    )
    completed = subprocess.run(
        command,
        cwd=str(repo_root.resolve()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if result_json_path.exists():
        result = Pose2DResult.from_dict(load_json_file(result_json_path))
    else:
        result = Pose2DResult(
            status="fail",
            effective_fps=None,
            selected_frame_indices=[],
            artifacts={},
            quality_report={},
            backend={},
            error="backend_result_json_missing",
        )

    backend_run = result.to_dict()
    backend_run["env_name"] = str(env_name)
    backend_run["launcher"] = {
        "name": str(launcher["name"]),
        "python": str(launcher["python"]),
    }
    backend_run["probe_diagnostics"] = probe_diagnostics
    backend_run["command"] = command
    backend_run["returncode"] = int(completed.returncode)
    if completed.stdout.strip():
        backend_run["stdout_tail"] = tail_text(completed.stdout, max_chars=8000)
    if completed.stderr.strip():
        backend_run["stderr_tail"] = tail_text(completed.stderr, max_chars=8000)

    if completed.returncode != 0 and backend_run.get("status") == "ok":
        backend_run["status"] = "fail"
        backend_run["error"] = "backend_process_failed"

    write_json_file(backend_run, output_dir / "backend_run.json")
    return backend_run


def run_pose2d_backend(job: Pose2DJob) -> Pose2DResult:
    from mmpose import __version__ as mmpose_version
    from mmpose.apis import MMPoseInferencer
    import torch

    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_prediction_dir = output_dir / "_backend_predictions"
    raw_prediction_dir.mkdir(parents=True, exist_ok=True)

    debug_overlay_path = resolve_debug_overlay_path(
        output_dir,
        filename=str(job.debug_overlay_filename),
        enabled=bool(job.save_debug),
    )
    if debug_overlay_path is not None:
        debug_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(job.device_preference))
    model_artifacts = resolve_local_pose2d_backend_artifacts(str(job.model_alias))
    inferencer = MMPoseInferencer(
        pose2d=str(model_artifacts.pose2d_config_path),
        pose2d_weights=str(model_artifacts.pose2d_checkpoint_path),
        det_model=str(model_artifacts.detector_config_path),
        det_weights=str(model_artifacts.detector_checkpoint_path),
        det_cat_ids=list(job.detector_category_ids),
        device=device,
        show_progress=False,
    )
    generator = inferencer(
        str(Path(job.video_path).resolve()),
        pred_out_dir=str(raw_prediction_dir.resolve()),
        vis_out_dir="" if debug_overlay_path is None else str(debug_overlay_path.resolve()),
    )
    for _ in generator:
        pass

    generated_raw_prediction_path = raw_prediction_dir / (Path(job.video_path).stem + ".json")
    if not generated_raw_prediction_path.exists():
        raise FileNotFoundError(
            "MMPose backend did not create the raw prediction JSON file at "
            + str(generated_raw_prediction_path)
        )

    final_raw_prediction_path = job.raw_prediction_path
    final_raw_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated_raw_prediction_path), str(final_raw_prediction_path))
    if raw_prediction_dir.exists():
        shutil.rmtree(raw_prediction_dir, ignore_errors=True)

    raw_predictions = json.loads(final_raw_prediction_path.read_text(encoding="utf-8"))
    frames_total = int(len(raw_predictions))
    frames_with_detections = int(
        sum(1 for frame_payload in raw_predictions if len(frame_payload.get("instances", [])) > 0)
    )

    frame_count_for_sampling = job.video_num_frames if job.video_num_frames is not None else frames_total
    selected_frame_indices, effective_fps, _ = select_frame_indices(
        int(frame_count_for_sampling),
        job.video_fps,
        int(job.fps_target),
    )
    warnings = []
    if job.video_num_frames is not None and int(job.video_num_frames) != frames_total:
        warnings.append("video_num_frames_differs_from_backend_predictions")

    return Pose2DResult(
        status="ok",
        effective_fps=effective_fps,
        selected_frame_indices=selected_frame_indices.astype(int).tolist(),
        artifacts={
            "raw_prediction_json_path": str(final_raw_prediction_path.resolve()),
            "debug_overlay_path": (
                None
                if debug_overlay_path is None or not debug_overlay_path.exists()
                else str(debug_overlay_path.resolve())
            ),
        },
        quality_report={
            "fps_original": job.video_fps,
            "effective_fps": effective_fps,
            "frames_total": frames_total,
            "frames_selected": int(len(selected_frame_indices)),
            "frames_with_detections": frames_with_detections,
            "warnings": warnings,
        },
        backend={
            "model_alias": str(job.model_alias),
            "pose2d_model_id": str(model_artifacts.pose2d_model_id),
            "pose2d_config_path": str(model_artifacts.pose2d_config_path),
            "pose2d_checkpoint_path": str(model_artifacts.pose2d_checkpoint_path),
            "detector_model_id": str(model_artifacts.detector_model_id),
            "detector_config_path": str(model_artifacts.detector_config_path),
            "detector_checkpoint_path": str(model_artifacts.detector_checkpoint_path),
            "device": str(device),
            "device_preference": str(job.device_preference),
            "detector_category_ids": [int(value) for value in job.detector_category_ids],
            "mmpose_version": str(mmpose_version),
            "torch_cuda_available": bool(torch.cuda.is_available()),
        },
        error=None,
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run ViTPose pose2d backend in openmmlab.")
    parser.add_argument("--job-json", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.job_json.read_text(encoding="utf-8"))
    job = Pose2DJob.from_dict(payload)

    try:
        result = run_pose2d_backend(job)
    except Exception as exc:
        result = Pose2DResult(
            status="fail",
            effective_fps=None,
            selected_frame_indices=[],
            artifacts={
                "raw_prediction_json_path": str(job.raw_prediction_path.resolve()),
                "debug_overlay_path": str(job.debug_overlay_path.resolve()) if job.save_debug else None,
            },
            quality_report={
                "fps_original": job.video_fps,
                "effective_fps": None,
                "frames_total": 0,
                "frames_selected": 0,
                "frames_with_detections": 0,
                "warnings": [],
            },
            backend={
                "model_alias": str(job.model_alias),
                "device_preference": str(job.device_preference),
            },
            error=str(exc),
        )

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result.status == "ok" else 1


def _resolve_device(device_preference: str) -> str:
    import torch

    preference = str(device_preference).strip().lower()
    if preference not in {"", "auto"}:
        return str(device_preference)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
