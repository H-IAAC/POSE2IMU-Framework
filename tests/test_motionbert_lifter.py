import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import MOTIONBERT_17_JOINT_NAMES, PoseSequence2D
from pose_module.motionbert.adapter import (
    build_motionbert_window_batch,
    merge_motionbert_window_predictions,
)
from pose_module.motionbert.lifter import run_motionbert_lifter
from pose_module.pipeline import run_pose3d_pipeline


_MB17_BASE_POINTS = {
    "pelvis": (0.00, 0.00),
    "left_hip": (-0.10, 0.02),
    "right_hip": (0.10, 0.02),
    "spine": (0.00, -0.10),
    "left_knee": (-0.12, 0.26),
    "right_knee": (0.12, 0.26),
    "thorax": (0.00, -0.22),
    "left_ankle": (-0.12, 0.52),
    "right_ankle": (0.12, 0.52),
    "neck": (0.00, -0.32),
    "head": (0.00, -0.44),
    "left_shoulder": (-0.18, -0.22),
    "right_shoulder": (0.18, -0.22),
    "left_elbow": (-0.24, -0.14),
    "right_elbow": (0.24, -0.14),
    "left_wrist": (-0.30, -0.04),
    "right_wrist": (0.30, -0.04),
}


def _make_motionbert_sequence(num_frames: int) -> PoseSequence2D:
    keypoints_xy = np.zeros((num_frames, len(MOTIONBERT_17_JOINT_NAMES), 2), dtype=np.float32)
    confidence = np.full((num_frames, len(MOTIONBERT_17_JOINT_NAMES)), 0.95, dtype=np.float32)
    bbox_xywh = np.tile(np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32), (num_frames, 1))

    for frame_index in range(num_frames):
        phase = float(frame_index) / 10.0
        for joint_index, joint_name in enumerate(MOTIONBERT_17_JOINT_NAMES):
            base_x, base_y = _MB17_BASE_POINTS[joint_name]
            x = base_x + (0.02 * np.sin(phase))
            y = base_y + (0.01 * np.cos(phase))
            keypoints_xy[frame_index, joint_index] = np.asarray([x, y], dtype=np.float32)

    return PoseSequence2D(
        clip_id="clip_motionbert",
        fps=20.0,
        fps_original=30.0,
        joint_names_2d=list(MOTIONBERT_17_JOINT_NAMES),
        keypoints_xy=keypoints_xy,
        confidence=confidence,
        bbox_xywh=bbox_xywh,
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / 20.0,
        source="vitpose-b_motionbert17_clean",
    )


def _predictor(batch_inputs: np.ndarray):
    output = np.zeros((batch_inputs.shape[0], batch_inputs.shape[1], batch_inputs.shape[2], 3), dtype=np.float32)
    output[..., 0] = batch_inputs[..., 0]
    output[..., 1] = -batch_inputs[..., 1]
    output[..., 2] = np.arange(batch_inputs.shape[2], dtype=np.float32)[None, None, :]
    return {
        "keypoints_3d": output,
        "joint_names": list(MOTIONBERT_17_JOINT_NAMES),
    }


class MotionBERTAdapterTests(unittest.TestCase):
    def test_build_motionbert_window_batch_includes_confidence_channel(self) -> None:
        sequence = _make_motionbert_sequence(120)

        batch = build_motionbert_window_batch(
            sequence,
            window_size=50,
            window_overlap=0.5,
            include_confidence=True,
        )

        self.assertEqual(batch.inputs.shape[1:], (50, 17, 3))
        self.assertGreater(batch.num_windows, 1)
        self.assertEqual(int(batch.frame_index_map[0, 0]), 0)
        last_valid_position = int(np.flatnonzero(batch.valid_mask[-1])[-1])
        self.assertEqual(int(batch.frame_index_map[-1, last_valid_position]), 119)

    def test_merge_motionbert_window_predictions_blends_overlaps(self) -> None:
        sequence = _make_motionbert_sequence(120)
        batch = build_motionbert_window_batch(
            sequence,
            window_size=50,
            window_overlap=0.5,
            include_confidence=False,
        )
        predictions = np.zeros((batch.num_windows, batch.window_size, 17, 3), dtype=np.float32)
        for window_index in range(batch.num_windows):
            predictions[window_index] = float(window_index)

        fused = merge_motionbert_window_predictions(
            predictions,
            batch,
            num_frames=sequence.num_frames,
        )

        self.assertEqual(fused.shape, (120, 17, 3))
        self.assertAlmostEqual(float(fused[0, 0, 0]), 0.0, places=5)
        self.assertGreater(float(fused[30, 0, 0]), 0.0)
        self.assertLess(float(fused[30, 0, 0]), 1.0)
        self.assertGreater(float(fused[60, 0, 0]), 1.0)
        self.assertLess(float(fused[60, 0, 0]), 2.0)


class MotionBERTLifterTests(unittest.TestCase):
    def test_run_motionbert_lifter_exports_pose3d_artifacts(self) -> None:
        sequence = _make_motionbert_sequence(90)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_motionbert_lifter(
                sequence,
                output_dir=tmp_dir,
                window_size=50,
                window_overlap=0.5,
                predictor=_predictor,
                backend_name="unit_test_motionbert",
            )

            pose_sequence = result["pose_sequence"]
            self.assertEqual(pose_sequence.joint_names_3d, list(MOTIONBERT_17_JOINT_NAMES))
            self.assertEqual(pose_sequence.joint_positions_xyz.shape, (90, 17, 3))
            self.assertTrue(Path(result["artifacts"]["pose3d_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["pose3d_raw_keypoints_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["motionbert_run_json_path"]).exists())
            self.assertEqual(result["quality_report"]["backend_name"], "unit_test_motionbert")
            self.assertEqual(result["quality_report"]["status"], "ok")
            self.assertAlmostEqual(float(pose_sequence.joint_positions_xyz[0, 4, 2]), 4.0, places=5)


class Pose3DPipelineTests(unittest.TestCase):
    def test_run_pose3d_pipeline_merges_motionbert_stage(self) -> None:
        sequence = _make_motionbert_sequence(60)
        pose2d_result = {
            "clip_id": "clip_pipeline3d",
            "pose_sequence": sequence,
            "raw_pose_sequence": sequence,
            "quality_report": {
                "clip_id": "clip_pipeline3d",
                "status": "ok",
                "visible_joint_ratio": 1.0,
                "mean_confidence": 0.95,
                "notes": [],
            },
            "track_report": {"status": "ok", "warnings": []},
            "backend_run": {"status": "ok"},
            "artifacts": {
                "quality_report_json_path": None,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.pipeline.run_pose2d_pipeline", return_value=pose2d_result):
                result = run_pose3d_pipeline(
                    clip_id="clip_pipeline3d",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=tmp_dir,
                    fps_target=20,
                    save_debug=False,
                    env_name="current",
                    motionbert_window_size=40,
                    motionbert_window_overlap=0.5,
                    motionbert_predictor=_predictor,
                    motionbert_backend_name="unit_test_motionbert",
                )

            self.assertEqual(result["quality_report"]["motionbert_backend_name"], "unit_test_motionbert")
            self.assertTrue(Path(result["artifacts"]["pose3d_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["motionbert_run_json_path"]).exists())
            self.assertEqual(result["pose_sequence"].joint_names_3d, list(MOTIONBERT_17_JOINT_NAMES))

    def test_run_pose3d_pipeline_exports_side_by_side_raw_3d_debug_video(self) -> None:
        sequence = _make_motionbert_sequence(24)
        pose2d_result = {
            "clip_id": "clip_pipeline3d_debug",
            "pose_sequence": sequence,
            "raw_pose_sequence": sequence,
            "quality_report": {
                "clip_id": "clip_pipeline3d_debug",
                "status": "ok",
                "visible_joint_ratio": 1.0,
                "mean_confidence": 0.95,
                "notes": [],
            },
            "track_report": {"status": "ok", "warnings": []},
            "backend_run": {"status": "ok"},
            "cleaner_artifacts": {
                "normalization_centers_xy": np.zeros((sequence.num_frames, 2), dtype=np.float32),
                "normalization_scales": np.ones((sequence.num_frames,), dtype=np.float32),
            },
            "artifacts": {
                "quality_report_json_path": None,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            rendered_outputs = []

            def _fake_render_pose3d_side_by_side_video(*, output_path, **kwargs):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake mp4")
                rendered_outputs.append(output_path.name)
                return output_path.resolve()

            with patch("pose_module.pipeline.run_pose2d_pipeline", return_value=pose2d_result), patch(
                "pose_module.pipeline.render_pose3d_side_by_side_video",
                side_effect=_fake_render_pose3d_side_by_side_video,
            ):
                result = run_pose3d_pipeline(
                    clip_id="clip_pipeline3d_debug",
                    video_path=str(Path(tmp_dir) / "video.mp4"),
                    output_dir=tmp_dir,
                    fps_target=20,
                    save_debug=True,
                    env_name="current",
                    motionbert_window_size=24,
                    motionbert_window_overlap=0.5,
                    motionbert_predictor=_predictor,
                    motionbert_backend_name="unit_test_motionbert",
                )

            self.assertTrue(Path(result["artifacts"]["debug_overlay_pose3d_raw_path"]).exists())
            self.assertEqual(rendered_outputs, ["debug_overlay_pose3d_raw.mp4"])


if __name__ == "__main__":
    unittest.main()
