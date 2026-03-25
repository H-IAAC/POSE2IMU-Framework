"""Stage 5.5: lift MotionBERT-ready 2D skeletons into a temporal 3D sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from pose_module.interfaces import (
    MOTIONBERT_17_JOINT_NAMES,
    MOTIONBERT_17_PARENT_INDICES,
    MotionBERTJob,
    PoseSequence2D,
    PoseSequence3D,
)
from pose_module.io.cache import load_json_file, tail_text, write_json_file
from pose_module.model_registry import resolve_local_motionbert_artifacts
from pose_module.motionbert.adapter import (
    build_motionbert_window_batch,
    canonicalize_motionbert_output,
    merge_motionbert_window_predictions,
    write_pose_sequence3d_npz,
)
from pose_module.openmmlab_runtime import select_openmmlab_launcher
from pose_module.vitpose.adapter import write_pose_sequence_npz


DEFAULT_WINDOW_SIZE = 81
DEFAULT_WINDOW_OVERLAP = 0.5
DEFAULT_INCLUDE_CONFIDENCE = True
DEFAULT_BACKEND_NAME = "mmpose_motionbert"
DEFAULT_FALLBACK_BACKEND_NAME = "motionbert_heuristic_baseline"
DEFAULT_SEQUENCE_BATCH_SIZE = 32

MotionBERTPredictor = Callable[[np.ndarray], Union[np.ndarray, Mapping[str, Any]]]

_DEPTH_PRIORS = {
    "pelvis": 0.00,
    "left_hip": 0.00,
    "right_hip": 0.00,
    "spine": 0.04,
    "left_knee": 0.03,
    "right_knee": 0.03,
    "thorax": 0.08,
    "left_ankle": 0.05,
    "right_ankle": 0.05,
    "neck": 0.10,
    "head": 0.14,
    "left_shoulder": 0.06,
    "right_shoulder": 0.06,
    "left_elbow": 0.08,
    "right_elbow": 0.08,
    "left_wrist": 0.10,
    "right_wrist": 0.10,
}


def run_motionbert_lifter(
    sequence: PoseSequence2D,
    *,
    output_dir: str | Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_overlap: float = DEFAULT_WINDOW_OVERLAP,
    include_confidence: bool = DEFAULT_INCLUDE_CONFIDENCE,
    predictor: Optional[MotionBERTPredictor] = None,
    backend_name: Optional[str] = None,
    checkpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "auto",
    env_name: str = "openmmlab",
    allow_fallback_backend: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_backend_name = str(
        backend_name or ("motionbert_callable_backend" if predictor is not None else DEFAULT_BACKEND_NAME)
    )
    resolved_artifacts = (
        None
        if predictor is not None
        else resolve_local_motionbert_artifacts(
            config_path=config_path,
            checkpoint_path=checkpoint,
        )
    )

    job = MotionBERTJob(
        clip_id=str(sequence.clip_id),
        output_dir=str(output_dir.resolve()),
        window_size=int(window_size),
        window_overlap=float(window_overlap),
        include_confidence=bool(include_confidence),
        backend_name=requested_backend_name,
        checkpoint=None if resolved_artifacts is None else str(resolved_artifacts.checkpoint_path),
        config_path=None if resolved_artifacts is None else str(resolved_artifacts.config_path),
        device=str(device),
        pose2d_source=str(sequence.source),
    )

    write_pose_sequence_npz(sequence, job.input_pose2d_path)

    if predictor is not None:
        result = _run_motionbert_callable_backend(
            sequence,
            job=job,
            predictor=predictor,
            backend_name=requested_backend_name,
        )
        write_json_file(result["run_report"], job.run_report_path)
        return result

    backend_run = run_motionbert_backend_job(
        job=job,
        env_name=str(env_name),
        output_dir=output_dir,
    )
    if backend_run.get("status") == "ok":
        return {
            "status": str(backend_run["status"]),
            "pose_sequence": _load_pose_sequence3d_npz(job.pose3d_npz_path),
            "quality_report": dict(backend_run["quality_report"]),
            "artifacts": dict(backend_run["artifacts"]),
            "run_report": dict(backend_run),
        }

    if not bool(allow_fallback_backend):
        raise RuntimeError(str(backend_run.get("error", "motionbert_backend_failed")))

    result = _run_motionbert_callable_backend(
        sequence,
        job=job,
        predictor=None,
        backend_name=DEFAULT_FALLBACK_BACKEND_NAME,
    )
    fallback_notes = list(result["quality_report"].get("notes", []))
    fallback_notes.append(
        "real_motionbert_backend_failed:" + str(backend_run.get("error", "unknown_backend_error"))
    )
    result["quality_report"]["notes"] = list(dict.fromkeys(fallback_notes))
    result["quality_report"]["status"] = "warning"
    result["run_report"] = {
        **result["run_report"],
        "status": "warning",
        "requested_backend": requested_backend_name,
        "fallback_trigger": str(backend_run.get("error", "unknown_backend_error")),
        "backend_attempt": dict(backend_run),
        "quality_report": dict(result["quality_report"]),
    }
    write_json_file(result["run_report"], job.run_report_path)
    return result


def run_motionbert_backend_job(
    *,
    job: MotionBERTJob,
    env_name: str,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_json_path = output_dir / "motionbert_backend_job.json"
    result_json_path = output_dir / "motionbert_backend_result.json"
    write_json_file(job.to_dict(), job_json_path)

    repo_root = Path(__file__).resolve().parents[2]
    launcher, probe_diagnostics = select_openmmlab_launcher(
        str(env_name),
        cwd=repo_root,
        probe_code=(
            "import mmpose, torch; from mmpose.apis import init_model; "
            "from mmengine.dataset import Compose; print('ok')"
        ),
    )
    if launcher is None:
        backend_run = {
            "status": "fail",
            "quality_report": {},
            "artifacts": {
                "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
                "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
                "motionbert_run_json_path": str(job.run_report_path.resolve()),
            },
            "error": "No Python launcher with mmpose pose-lifter support available.",
            "env_name": str(env_name),
            "backend": {
                "launcher": None,
                "probe_diagnostics": probe_diagnostics,
            },
            "returncode": 1,
        }
        write_json_file(backend_run, job.run_report_path)
        return backend_run

    command = list(launcher["prefix"]) + [
        "-m",
        "pose_module.motionbert.lifter",
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
        backend_run = load_json_file(result_json_path)
    else:
        backend_run = {
            "status": "fail",
            "quality_report": {},
            "artifacts": {
                "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
                "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
                "motionbert_run_json_path": str(job.run_report_path.resolve()),
            },
            "error": "motionbert_backend_result_json_missing",
        }

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
        backend_run["error"] = "motionbert_backend_process_failed"

    write_json_file(backend_run, job.run_report_path)
    return backend_run


def run_motionbert_backend(job: MotionBERTJob) -> Dict[str, Any]:
    import torch
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmpose.apis import init_model

    if job.config_path in (None, "") or job.checkpoint in (None, ""):
        raise FileNotFoundError("MotionBERT config/checkpoint could not be resolved.")

    sequence = _load_pose_sequence2d_npz(job.input_pose2d_path)
    model = init_model(
        config=str(job.config_path),
        checkpoint=str(job.checkpoint),
        device=_resolve_device(str(job.device)),
    )

    init_default_scope(model.cfg.get("default_scope", "mmpose"))
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    dataset_meta = dict(model.dataset_meta or {})
    effective_window_size = _resolve_model_sequence_length(model, fallback=int(job.window_size))
    window_batch = build_motionbert_window_batch(
        sequence,
        window_size=int(effective_window_size),
        window_overlap=float(job.window_overlap),
        include_confidence=bool(job.include_confidence),
    )

    predictions_3d = []
    for start_index in range(0, window_batch.num_windows, DEFAULT_SEQUENCE_BATCH_SIZE):
        end_index = min(start_index + DEFAULT_SEQUENCE_BATCH_SIZE, window_batch.num_windows)
        data_list = []
        batch_inputs = window_batch.inputs[start_index:end_index]
        for window_input in batch_inputs:
            window_keypoints = np.asarray(window_input[..., :2], dtype=np.float32)
            window_visibility = (
                np.asarray(window_input[..., 2], dtype=np.float32)
                if window_input.shape[-1] > 2
                else np.ones(window_input.shape[:2], dtype=np.float32)
            )
            window_keypoints = window_keypoints + np.asarray([1.0, 1.0], dtype=np.float32)
            data_info = {
                "keypoints": window_keypoints,
                "keypoints_visible": window_visibility,
                "lifting_target": np.zeros(
                    (window_keypoints.shape[0], window_keypoints.shape[1], 3),
                    dtype=np.float32,
                ),
                "factor": np.zeros((window_keypoints.shape[0],), dtype=np.float32),
                "lifting_target_visible": np.ones(
                    (window_keypoints.shape[0], window_keypoints.shape[1], 1),
                    dtype=np.float32,
                ),
                "camera_param": {"w": 2.0, "h": 2.0},
            }
            data_info.update(dataset_meta)
            data_list.append(pipeline(data_info))

        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
        predictions_3d.extend(
            _extract_backend_window_predictions(
                results,
                expected_window_size=int(effective_window_size),
            )
        )

    if len(predictions_3d) != int(window_batch.num_windows):
        raise RuntimeError(
            f"MotionBERT backend produced {len(predictions_3d)} windows, expected {window_batch.num_windows}"
        )

    joint_positions_xyz = merge_motionbert_window_predictions(
        np.asarray(predictions_3d, dtype=np.float32),
        window_batch,
        num_frames=int(sequence.num_frames),
    )
    pose_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(MOTIONBERT_17_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.asarray(sequence.confidence, dtype=np.float32),
        skeleton_parents=list(MOTIONBERT_17_PARENT_INDICES),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_{job.backend_name}",
        coordinate_space="camera",
    )
    np.save(job.raw_keypoints_3d_path, np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32))
    write_pose_sequence3d_npz(pose_sequence, job.pose3d_npz_path)

    notes = []
    if int(effective_window_size) != int(job.window_size):
        notes.append(f"window_size_adjusted_to_model_seq_len:{effective_window_size}")

    quality_report = _build_motionbert_quality_report(
        pose_sequence=pose_sequence,
        backend_name=str(job.backend_name),
        include_confidence=bool(job.include_confidence),
        fallback_backend_used=False,
        requested_window_overlap=float(job.window_overlap),
        effective_window_size=int(effective_window_size),
        num_windows=int(window_batch.num_windows),
        input_channels=3 if job.include_confidence else 2,
        notes=notes,
    )

    return {
        "status": str(quality_report["status"]),
        "quality_report": quality_report,
        "artifacts": {
            "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
            "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
            "motionbert_run_json_path": str(job.run_report_path.resolve()),
        },
        "backend": {
            "name": str(job.backend_name),
            "config_path": str(job.config_path),
            "checkpoint_path": str(job.checkpoint),
            "device": _resolve_device(str(job.device)),
            "mode": "mmpose_pose_lifter",
            "effective_window_size": int(effective_window_size),
        },
        "error": None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run MotionBERT pose-lifter backend.")
    parser.add_argument("--job-json", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args(argv)

    job = MotionBERTJob.from_dict(load_json_file(args.job_json))
    try:
        result = run_motionbert_backend(job)
    except Exception as exc:
        result = {
            "status": "fail",
            "quality_report": {},
            "artifacts": {
                "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
                "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
                "motionbert_run_json_path": str(job.run_report_path.resolve()),
            },
            "backend": {
                "name": str(job.backend_name),
                "config_path": job.config_path,
                "checkpoint_path": job.checkpoint,
                "device": str(job.device),
            },
            "error": str(exc),
        }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0 if result.get("status") == "ok" else 1


def _run_motionbert_callable_backend(
    sequence: PoseSequence2D,
    *,
    job: MotionBERTJob,
    predictor: Optional[MotionBERTPredictor],
    backend_name: str,
) -> Dict[str, Any]:
    batch = build_motionbert_window_batch(
        sequence,
        window_size=int(job.window_size),
        window_overlap=float(job.window_overlap),
        include_confidence=bool(job.include_confidence),
    )
    raw_backend_output = (
        _heuristic_motionbert_predict(batch.inputs) if predictor is None else predictor(batch.inputs)
    )
    canonical_predictions = canonicalize_motionbert_output(
        raw_backend_output,
        expected_batch_size=batch.num_windows,
        expected_window_size=batch.window_size,
    )
    fused_predictions = merge_motionbert_window_predictions(
        canonical_predictions,
        batch,
        num_frames=int(sequence.num_frames),
    )

    pose_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(MOTIONBERT_17_JOINT_NAMES),
        joint_positions_xyz=fused_predictions.astype(np.float32, copy=False),
        joint_confidence=np.asarray(sequence.confidence, dtype=np.float32),
        skeleton_parents=list(MOTIONBERT_17_PARENT_INDICES),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_{backend_name}",
        coordinate_space="camera",
    )

    np.save(job.raw_keypoints_3d_path, np.asarray(pose_sequence.joint_positions_xyz, dtype=np.float32))
    write_pose_sequence3d_npz(pose_sequence, job.pose3d_npz_path)

    quality_report = _build_motionbert_quality_report(
        pose_sequence=pose_sequence,
        backend_name=str(backend_name),
        include_confidence=bool(job.include_confidence),
        fallback_backend_used=predictor is None,
        requested_window_overlap=float(job.window_overlap),
        effective_window_size=int(job.window_size),
        num_windows=int(batch.num_windows),
        input_channels=int(batch.inputs.shape[-1]),
        notes=[],
    )
    artifacts = {
        "pose3d_npz_path": str(job.pose3d_npz_path.resolve()),
        "pose3d_raw_keypoints_path": str(job.raw_keypoints_3d_path.resolve()),
        "motionbert_run_json_path": str(job.run_report_path.resolve()),
    }
    run_report = {
        "status": str(quality_report["status"]),
        "quality_report": quality_report,
        "artifacts": dict(artifacts),
        "backend": {
            "name": str(backend_name),
            "mode": "callable" if predictor is not None else "heuristic_baseline",
            "num_windows": int(batch.num_windows),
            "window_size": int(batch.window_size),
        },
        "error": None,
    }
    return {
        "status": str(quality_report["status"]),
        "pose_sequence": pose_sequence,
        "quality_report": quality_report,
        "artifacts": artifacts,
        "run_report": run_report,
    }


def _extract_backend_window_predictions(
    results: Sequence[Any],
    *,
    expected_window_size: int,
) -> List[np.ndarray]:
    predictions: List[np.ndarray] = []
    for result in results:
        pred_instances = getattr(result, "pred_instances", None)
        if pred_instances is None:
            raise RuntimeError("MotionBERT result missing pred_instances.")
        keypoints = None
        if hasattr(pred_instances, "keypoints"):
            keypoints = pred_instances.keypoints
        elif hasattr(pred_instances, "keypoints_3d"):
            keypoints = pred_instances.keypoints_3d
        if keypoints is None:
            raise RuntimeError("MotionBERT result missing keypoints/keypoints_3d.")

        array = np.asarray(keypoints, dtype=np.float32)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[1:] != (len(MOTIONBERT_17_JOINT_NAMES), 3):
            raise RuntimeError(f"Unexpected MotionBERT keypoint shape: {array.shape}")
        if int(array.shape[0]) != int(expected_window_size):
            raise RuntimeError(
                f"Unexpected MotionBERT temporal window shape: {array.shape[0]} != {expected_window_size}"
            )
        predictions.append(array.astype(np.float32, copy=False))
    return predictions


def _resolve_model_sequence_length(model: Any, *, fallback: int) -> int:
    backbone = getattr(model, "backbone", None)
    seq_len = getattr(backbone, "seq_len", None)
    if seq_len is None:
        seq_len = model.cfg.get("model", {}).get("backbone", {}).get("seq_len")
    if seq_len in (None, 0):
        return int(fallback)
    return int(seq_len)


def _resolve_device(device_preference: str) -> str:
    import torch

    preference = str(device_preference).strip().lower()
    if preference not in {"", "auto"}:
        return str(device_preference)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _load_pose_sequence2d_npz(path: str | Path) -> PoseSequence2D:
    with np.load(Path(path), allow_pickle=False) as payload:
        return PoseSequence2D.from_npz_payload({key: payload[key] for key in payload.files})


def _load_pose_sequence3d_npz(path: str | Path) -> PoseSequence3D:
    with np.load(Path(path), allow_pickle=False) as payload:
        return PoseSequence3D.from_npz_payload({key: payload[key] for key in payload.files})


def _heuristic_motionbert_predict(window_inputs: np.ndarray) -> np.ndarray:
    xy = np.asarray(window_inputs[..., :2], dtype=np.float32)
    confidence = (
        np.asarray(window_inputs[..., 2], dtype=np.float32)
        if window_inputs.shape[-1] > 2
        else np.ones(window_inputs.shape[:-1], dtype=np.float32)
    )

    output = np.zeros((xy.shape[0], xy.shape[1], xy.shape[2], 3), dtype=np.float32)
    output[..., 0] = xy[..., 0]
    output[..., 1] = -xy[..., 1]

    temporal_delta = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[1] > 1:
        temporal_delta[:, 1:] = xy[:, 1:] - xy[:, :-1]
    temporal_speed = np.linalg.norm(temporal_delta, axis=3)

    for joint_index, joint_name in enumerate(MOTIONBERT_17_JOINT_NAMES):
        base_depth = float(_DEPTH_PRIORS[joint_name])
        lateral_bias = 0.15 * np.abs(xy[..., joint_index, 0])
        vertical_bias = 0.10 * np.maximum(-xy[..., joint_index, 1], 0.0)
        temporal_bias = 0.25 * temporal_speed[..., joint_index]
        joint_depth = (base_depth + lateral_bias + vertical_bias + temporal_bias) * confidence[..., joint_index]
        output[..., joint_index, 2] = joint_depth.astype(np.float32, copy=False)

    return output


def _build_motionbert_quality_report(
    *,
    pose_sequence: PoseSequence3D,
    backend_name: str,
    include_confidence: bool,
    fallback_backend_used: bool,
    requested_window_overlap: float,
    effective_window_size: int,
    num_windows: int,
    input_channels: int,
    notes: Sequence[str],
) -> Dict[str, Any]:
    joint_confidence = np.asarray(pose_sequence.joint_confidence, dtype=np.float32)
    visible_joint_ratio = (
        float(np.count_nonzero(joint_confidence > 0.0) / float(joint_confidence.size))
        if joint_confidence.size > 0
        else 0.0
    )
    valid_confidence = joint_confidence[joint_confidence > 0.0]
    mean_confidence = float(np.mean(valid_confidence)) if valid_confidence.size > 0 else 0.0
    depth_values = np.asarray(pose_sequence.joint_positions_xyz[..., 2], dtype=np.float32)
    depth_variation = float(np.std(depth_values)) if depth_values.size > 0 else 0.0
    window_coverage_ratio = 1.0 if pose_sequence.num_frames > 0 else 0.0

    report_notes = list(str(value) for value in notes)
    if fallback_backend_used:
        report_notes.append("fallback_backend_without_motionbert_weights")
    if depth_variation <= 1e-4:
        report_notes.append("depth_variation_too_low")
    if mean_confidence < 0.5:
        report_notes.append("mean_confidence_below_threshold")

    status = "ok"
    if depth_values.shape[0] != pose_sequence.num_frames:
        status = "fail"
    elif fallback_backend_used or mean_confidence < 0.5 or depth_variation <= 1e-4:
        status = "warning"

    return {
        "clip_id": str(pose_sequence.clip_id),
        "status": str(status),
        "fps": None if pose_sequence.fps is None else float(pose_sequence.fps),
        "fps_original": None if pose_sequence.fps_original is None else float(pose_sequence.fps_original),
        "num_frames": int(pose_sequence.num_frames),
        "num_joints": int(pose_sequence.num_joints),
        "input_joint_format": list(MOTIONBERT_17_JOINT_NAMES),
        "output_joint_format": list(MOTIONBERT_17_JOINT_NAMES),
        "coordinate_space": str(pose_sequence.coordinate_space),
        "backend_name": str(backend_name),
        "window_size": int(effective_window_size),
        "window_overlap": float(requested_window_overlap),
        "num_windows": int(num_windows),
        "input_channels": int(input_channels),
        "include_confidence_channel": bool(include_confidence),
        "visible_joint_ratio": float(visible_joint_ratio),
        "mean_confidence": float(mean_confidence),
        "depth_variation": float(depth_variation),
        "window_coverage_ratio": float(window_coverage_ratio),
        "notes": list(dict.fromkeys(report_notes)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
