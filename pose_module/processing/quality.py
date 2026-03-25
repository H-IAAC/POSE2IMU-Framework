"""Quality-report helpers for the pose pipeline."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def merge_stage53_quality_reports(
    *,
    clip_id: str,
    backend_quality: Mapping[str, Any],
    track_report: Mapping[str, Any],
    pose_quality: Mapping[str, Any],
    cleaner_quality: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cleaner_quality = {} if cleaner_quality is None else dict(cleaner_quality)
    notes = []
    notes.extend([str(value) for value in backend_quality.get("warnings", [])])
    notes.extend([str(value) for value in track_report.get("warnings", [])])
    notes.extend([str(value) for value in pose_quality.get("notes", [])])
    notes.extend([str(value) for value in cleaner_quality.get("notes", [])])

    status = "ok"
    if (
        track_report.get("status") == "fail"
        or pose_quality.get("status") == "fail"
        or cleaner_quality.get("status") == "fail"
    ):
        status = "fail"
    elif (
        track_report.get("status") == "warning"
        or pose_quality.get("status") == "warning"
        or cleaner_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    return {
        "clip_id": str(clip_id),
        "status": str(status),
        "fps_original": backend_quality.get("fps_original", pose_quality.get("fps_original")),
        "fps": backend_quality.get(
            "effective_fps",
            cleaner_quality.get("fps", pose_quality.get("fps")),
        ),
        "frames_total": backend_quality.get("frames_total"),
        "frames_selected": backend_quality.get("frames_selected", pose_quality.get("num_selected_frames")),
        "frames_with_detections": track_report.get("frames_with_detections"),
        "frames_with_selected_track": pose_quality.get("frames_with_selected_track"),
        "selected_track_id": track_report.get("selected_track_id"),
        "selected_track_stability": track_report.get("selected_track_stability"),
        "visible_joint_ratio": cleaner_quality.get("visible_joint_ratio", pose_quality.get("visible_joint_ratio")),
        "mean_confidence": cleaner_quality.get("mean_confidence", pose_quality.get("mean_confidence")),
        "temporal_jitter_score": cleaner_quality.get("temporal_jitter_score"),
        "outlier_ratio": cleaner_quality.get("outlier_ratio"),
        "frames_interpolated": cleaner_quality.get("frames_interpolated"),
        "interpolated_joint_ratio": cleaner_quality.get("interpolated_joint_ratio"),
        "frames_over_missing_joint_threshold": cleaner_quality.get("frames_over_missing_joint_threshold"),
        "normalization_mode": cleaner_quality.get("normalization_mode"),
        "notes": list(dict.fromkeys(notes)),
    }


def merge_stage55_quality_reports(
    *,
    pose2d_quality: Mapping[str, Any],
    lifter_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    pose2d_quality = dict(pose2d_quality)
    lifter_quality = dict(lifter_quality)

    notes = []
    notes.extend([str(value) for value in pose2d_quality.get("notes", [])])
    notes.extend([str(value) for value in lifter_quality.get("notes", [])])

    status = "ok"
    if pose2d_quality.get("status") == "fail" or lifter_quality.get("status") == "fail":
        status = "fail"
    elif (
        pose2d_quality.get("status") == "warning"
        or lifter_quality.get("status") == "warning"
        or len(notes) > 0
    ):
        status = "warning"

    merged = dict(pose2d_quality)
    merged.update(
        {
            "status": str(status),
            "pose3d_num_frames": lifter_quality.get("num_frames"),
            "pose3d_num_joints": lifter_quality.get("num_joints"),
            "pose3d_coordinate_space": lifter_quality.get("coordinate_space"),
            "pose3d_joint_format": lifter_quality.get("output_joint_format"),
            "motionbert_backend_name": lifter_quality.get("backend_name"),
            "motionbert_window_size": lifter_quality.get("window_size"),
            "motionbert_window_overlap": lifter_quality.get("window_overlap"),
            "motionbert_num_windows": lifter_quality.get("num_windows"),
            "motionbert_input_channels": lifter_quality.get("input_channels"),
            "depth_variation": lifter_quality.get("depth_variation"),
            "window_coverage_ratio": lifter_quality.get("window_coverage_ratio"),
            "notes": list(dict.fromkeys(notes)),
        }
    )
    return merged
