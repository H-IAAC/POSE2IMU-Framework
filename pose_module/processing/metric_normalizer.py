"""Stage 5.7: normalize mapped 3D poses into a local metric body frame."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, PoseSequence3D
from pose_module.processing.temporal_filters import savgol_smooth

DEFAULT_TARGET_FEMUR_LENGTH_M = 0.45
DEFAULT_SAVGOL_WINDOW_LENGTH = 9
DEFAULT_SAVGOL_POLYORDER = 2
BODY_METRIC_LOCAL_COORDINATE_SPACE = "body_metric_local"
_EPSILON = 1e-6

_IMUGPT22_INDEX = {name: index for index, name in enumerate(IMUGPT_22_JOINT_NAMES)}


def run_metric_normalizer(
    sequence: PoseSequence3D,
    *,
    target_femur_length_m: float = DEFAULT_TARGET_FEMUR_LENGTH_M,
    smoothing_window_length: int = DEFAULT_SAVGOL_WINDOW_LENGTH,
    smoothing_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> Dict[str, Any]:
    """Convert an IMUGPT22 pose into a smoothed local metric body-frame pose."""

    _validate_imugpt22_sequence(sequence)

    joint_positions_3d_norm = np.asarray(sequence.joint_positions_xyz, dtype=np.float32).copy()
    pelvis_index = _IMUGPT22_INDEX["Pelvis"]
    centered_positions = joint_positions_3d_norm - joint_positions_3d_norm[:, pelvis_index : pelvis_index + 1, :]

    body_frame_rotations, body_frame_fallback_mask = _build_body_frame_rotation_matrices(
        joint_positions_3d_norm
    )
    joint_positions_body_frame = np.einsum(
        "tki,tij->tkj",
        centered_positions,
        body_frame_rotations,
        dtype=np.float32,
    ).astype(np.float32, copy=False)

    scale_factor, observed_femur_length_model_units, used_identity_scale = _estimate_scale_factor(
        joint_positions_body_frame,
        target_femur_length_m=float(target_femur_length_m),
    )
    joint_positions_metric_local = (
        joint_positions_body_frame * np.float32(scale_factor)
    ).astype(np.float32, copy=False)
    joint_positions_smoothed = _smooth_metric_local_pose(
        joint_positions_metric_local,
        window_length=int(smoothing_window_length),
        polyorder=int(smoothing_polyorder),
    )

    metric_pose_sequence = PoseSequence3D(
        clip_id=str(sequence.clip_id),
        fps=None if sequence.fps is None else float(sequence.fps),
        fps_original=None if sequence.fps_original is None else float(sequence.fps_original),
        joint_names_3d=list(sequence.joint_names_3d),
        joint_positions_xyz=joint_positions_smoothed.astype(np.float32, copy=False),
        joint_confidence=np.asarray(sequence.joint_confidence, dtype=np.float32),
        skeleton_parents=list(sequence.skeleton_parents),
        frame_indices=np.asarray(sequence.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence.timestamps_sec, dtype=np.float32),
        source=f"{sequence.source}_metric",
        coordinate_space=BODY_METRIC_LOCAL_COORDINATE_SPACE,
    )

    normalization_result = {
        "joint_positions_3d_norm": joint_positions_3d_norm.astype(np.float32, copy=False),
        "joint_positions_body_frame": joint_positions_body_frame.astype(np.float32, copy=False),
        "joint_positions_metric_local": joint_positions_metric_local.astype(np.float32, copy=False),
        "joint_positions_smoothed": joint_positions_smoothed.astype(np.float32, copy=False),
        "scale_factor": float(scale_factor),
    }
    quality_report = _build_metric_normalizer_quality_report(
        sequence=sequence,
        metric_pose_sequence=metric_pose_sequence,
        scale_factor=float(scale_factor),
        target_femur_length_m=float(target_femur_length_m),
        observed_femur_length_model_units=observed_femur_length_model_units,
        body_frame_fallback_mask=body_frame_fallback_mask,
        smoothing_window_length=int(smoothing_window_length),
        smoothing_polyorder=int(smoothing_polyorder),
        used_identity_scale=bool(used_identity_scale),
    )
    artifacts = {
        "body_frame_fallback_mask": body_frame_fallback_mask.astype(bool, copy=False),
        "body_frame_rotation_matrices": body_frame_rotations.astype(np.float32, copy=False),
    }
    return {
        "pose_sequence": metric_pose_sequence,
        "normalization_result": normalization_result,
        "quality_report": quality_report,
        "artifacts": artifacts,
    }


def _build_metric_normalizer_quality_report(
    *,
    sequence: PoseSequence3D,
    metric_pose_sequence: PoseSequence3D,
    scale_factor: float,
    target_femur_length_m: float,
    observed_femur_length_model_units: float | None,
    body_frame_fallback_mask: np.ndarray,
    smoothing_window_length: int,
    smoothing_polyorder: int,
    used_identity_scale: bool,
) -> Dict[str, Any]:
    notes = []
    body_frame_fallback_frames = int(np.count_nonzero(body_frame_fallback_mask))
    if body_frame_fallback_frames > 0:
        notes.append(f"body_frame_fallback_frames:{body_frame_fallback_frames}")
    if used_identity_scale:
        notes.append("scale_factor_fallback_to_identity")

    finite_positions = np.isfinite(np.asarray(metric_pose_sequence.joint_positions_xyz, dtype=np.float32)).all()
    contract_ok = list(metric_pose_sequence.joint_names_3d) == list(IMUGPT_22_JOINT_NAMES)
    metric_pose_ok = bool(finite_positions and contract_ok and scale_factor > 0.0)
    if not finite_positions:
        notes.append("metric_pose_contains_nan")
    if not contract_ok:
        notes.append("metric_joint_contract_mismatch")
    if not (scale_factor > 0.0):
        notes.append("invalid_scale_factor")

    status = "ok"
    if not metric_pose_ok:
        status = "fail"
    elif body_frame_fallback_frames > 0 or used_identity_scale:
        status = "warning"

    return {
        "clip_id": str(metric_pose_sequence.clip_id),
        "status": str(status),
        "fps": None if metric_pose_sequence.fps is None else float(metric_pose_sequence.fps),
        "fps_original": (
            None if metric_pose_sequence.fps_original is None else float(metric_pose_sequence.fps_original)
        ),
        "num_frames": int(metric_pose_sequence.num_frames),
        "num_joints": int(metric_pose_sequence.num_joints),
        "input_joint_format": list(sequence.joint_names_3d),
        "output_joint_format": list(metric_pose_sequence.joint_names_3d),
        "input_coordinate_space": str(sequence.coordinate_space),
        "coordinate_space": str(metric_pose_sequence.coordinate_space),
        "metric_pose_ok": bool(metric_pose_ok),
        "scale_factor": float(scale_factor),
        "target_femur_length_m": float(target_femur_length_m),
        "observed_femur_length_model_units": (
            None
            if observed_femur_length_model_units is None
            else float(observed_femur_length_model_units)
        ),
        "scale_reference": "median_femur_length",
        "body_frame_fallback_frames": int(body_frame_fallback_frames),
        "body_frame_fallback_ratio": (
            float(body_frame_fallback_frames) / float(metric_pose_sequence.num_frames)
            if metric_pose_sequence.num_frames > 0
            else 0.0
        ),
        "smoothing_window_length": int(smoothing_window_length),
        "smoothing_polyorder": int(smoothing_polyorder),
        "assumptions": [
            "body_frame_from_hips_and_neck",
            "single_subject_per_clip",
            "anthropometric_scale_prior",
        ],
        "limitations": [
            "not_global_pose",
            "not_absolute_depth",
            "heuristic_metric_scale",
        ],
        "notes": list(dict.fromkeys(notes)),
    }


def _validate_imugpt22_sequence(sequence: PoseSequence3D) -> None:
    joint_names = [str(name) for name in sequence.joint_names_3d]
    if joint_names != list(IMUGPT_22_JOINT_NAMES):
        raise ValueError("Metric normalizer expects stage-5.6 output ordered as IMUGPT_22_JOINT_NAMES.")
    points = np.asarray(sequence.joint_positions_xyz, dtype=np.float32)
    confidence = np.asarray(sequence.joint_confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (len(IMUGPT_22_JOINT_NAMES), 3):
        raise ValueError("Metric normalizer expects joint_positions_xyz with shape [T, 22, 3].")
    if confidence.shape != points.shape[:2]:
        raise ValueError("Metric normalizer expects joint_confidence with shape [T, 22].")
    if not np.isfinite(points).all():
        raise ValueError("Metric normalizer expects finite joint_positions_xyz without NaN.")


def _build_body_frame_rotation_matrices(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(positions, dtype=np.float32)
    num_frames = int(points.shape[0])
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], num_frames, axis=0)
    fallback_mask = np.zeros((num_frames,), dtype=bool)
    previous_valid_rotation = None

    for frame_index in range(num_frames):
        pelvis = points[frame_index, _IMUGPT22_INDEX["Pelvis"]]
        neck = points[frame_index, _IMUGPT22_INDEX["Neck"]]
        left_hip = points[frame_index, _IMUGPT22_INDEX["Left_hip"]]
        right_hip = points[frame_index, _IMUGPT22_INDEX["Right_hip"]]

        lateral = right_hip - left_hip
        vertical = neck - pelvis
        lateral_norm = np.linalg.norm(lateral)
        vertical_norm = np.linalg.norm(vertical)
        if lateral_norm > _EPSILON and vertical_norm > _EPSILON:
            x_axis = lateral / lateral_norm
            y_seed = vertical / vertical_norm
            z_axis = np.cross(x_axis, y_seed)
            z_norm = np.linalg.norm(z_axis)
            if z_norm > _EPSILON:
                z_axis = z_axis / z_norm
                y_axis = np.cross(z_axis, x_axis)
                y_norm = np.linalg.norm(y_axis)
                if y_norm > _EPSILON:
                    y_axis = y_axis / y_norm
                    rotations[frame_index] = np.stack([x_axis, y_axis, z_axis], axis=1).astype(
                        np.float32,
                        copy=False,
                    )
                    previous_valid_rotation = rotations[frame_index].copy()
                    continue

        fallback_mask[frame_index] = True
        if previous_valid_rotation is not None:
            rotations[frame_index] = previous_valid_rotation

    return rotations, fallback_mask


def _estimate_scale_factor(
    body_positions: np.ndarray,
    *,
    target_femur_length_m: float,
) -> tuple[float, float | None, bool]:
    points = np.asarray(body_positions, dtype=np.float32)
    observed_segments = []
    for hip_name, knee_name in (("Left_hip", "Left_knee"), ("Right_hip", "Right_knee")):
        segment = points[:, _IMUGPT22_INDEX[knee_name]] - points[:, _IMUGPT22_INDEX[hip_name]]
        lengths = np.linalg.norm(segment, axis=1)
        valid_mask = np.isfinite(lengths) & (lengths > _EPSILON)
        if np.any(valid_mask):
            observed_segments.append(lengths[valid_mask])

    if len(observed_segments) == 0:
        return 1.0, None, True

    observed_femur_length = float(np.median(np.concatenate(observed_segments, axis=0)))
    if observed_femur_length <= _EPSILON:
        return 1.0, observed_femur_length, True
    return float(target_femur_length_m) / observed_femur_length, observed_femur_length, False


def _smooth_metric_local_pose(
    metric_local_positions: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    points = np.asarray(metric_local_positions, dtype=np.float32).copy()
    for joint_index in range(points.shape[1]):
        points[:, joint_index] = savgol_smooth(
            points[:, joint_index],
            window_length=int(window_length),
            polyorder=int(polyorder),
        )
    return points.astype(np.float32, copy=False)
