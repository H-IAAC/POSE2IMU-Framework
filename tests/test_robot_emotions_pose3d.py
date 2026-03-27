import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D
from pose_module.robot_emotions.cli import main as robot_emotions_cli_main
from pose_module.robot_emotions.extractor import RobotEmotionsClipRecord
from pose_module.robot_emotions.pose3d import run_robot_emotions_pose3d
from tests.test_motionbert_lifter import _make_motionbert_sequence


def _make_record(clip_id: str) -> RobotEmotionsClipRecord:
    return RobotEmotionsClipRecord(
        clip_id=clip_id,
        domain="10ms",
        user_id=2,
        tag_number=5,
        tag_dir=Path("data/RobotEmotions/10ms/User2/Tag5"),
        imu_csv_path=Path("data/RobotEmotions/10ms/User2/Tag5/ESP_2_5.csv"),
        video_path=Path("data/RobotEmotions/10ms/User2/Tag5/TAG_2_5.mp4"),
        source_rel_dir="10ms/User2/Tag5",
        take_id=None,
        participant={"name": "Test User", "age": 20, "gender": "F"},
        protocol={"emotion": "Joy"},
    )


def _make_pose3d_sequence() -> PoseSequence3D:
    sequence_2d = _make_motionbert_sequence(12)
    joint_positions_xyz = np.zeros((sequence_2d.num_frames, len(IMUGPT_22_JOINT_NAMES), 3), dtype=np.float32)
    joint_positions_xyz[..., 0] = np.linspace(
        -0.4,
        0.4,
        len(IMUGPT_22_JOINT_NAMES),
        dtype=np.float32,
    )[None, :]
    joint_positions_xyz[..., 1] = np.linspace(
        0.8,
        -0.8,
        len(IMUGPT_22_JOINT_NAMES),
        dtype=np.float32,
    )[None, :]
    joint_positions_xyz[..., 2] = np.linspace(
        0.0,
        1.0,
        sequence_2d.num_frames * len(IMUGPT_22_JOINT_NAMES),
        dtype=np.float32,
    ).reshape(sequence_2d.num_frames, len(IMUGPT_22_JOINT_NAMES))
    return PoseSequence3D(
        clip_id=str(sequence_2d.clip_id),
        fps=sequence_2d.fps,
        fps_original=sequence_2d.fps_original,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.full(
            (sequence_2d.num_frames, len(IMUGPT_22_JOINT_NAMES)),
            0.95,
            dtype=np.float32,
        ),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.asarray(sequence_2d.frame_indices, dtype=np.int32),
        timestamps_sec=np.asarray(sequence_2d.timestamps_sec, dtype=np.float32),
        source="vitpose-b_motionbert17_clean_mmpose_motionbert_imugpt22",
        coordinate_space="camera",
    )


class _FakeExtractor:
    def __init__(self, dataset_root: str, *, domains: tuple[str, ...]) -> None:
        self.dataset_root = dataset_root
        self.domains = domains

    def select_records(self, *, clip_ids=None):
        record = _make_record("robot_emotions_10ms_u02_tag05")
        if clip_ids is None:
            return [record]
        if record.clip_id in set(clip_ids):
            return [record]
        return []

    def ensure_exported_clip(self, record: RobotEmotionsClipRecord, *, output_root: str | Path):
        return {
            "labels": {"emotion": "Joy"},
            "source": {"video_path": str(record.video_path)},
            "video": {"fps": 30.0, "num_frames": 120, "duration_sec": 4.0},
            "artifacts": {
                "imu_npz_path": str(Path(output_root) / "imu.npz"),
                "metadata_json_path": str(Path(output_root) / "metadata.json"),
            },
        }


class RobotEmotionsPose3DTests(unittest.TestCase):
    def test_run_robot_emotions_pose3d_writes_manifest_and_summary(self) -> None:
        record = _make_record("robot_emotions_10ms_u02_tag05")
        pose2d_sequence = _make_motionbert_sequence(12)
        pose3d_sequence = _make_pose3d_sequence()

        fake_pipeline_result = {
            "clip_id": record.clip_id,
            "pose_sequence": pose3d_sequence,
            "pose_sequence_2d": pose2d_sequence,
            "raw_pose_sequence_2d": pose2d_sequence,
            "quality_report": {"clip_id": record.clip_id, "status": "ok", "notes": []},
            "pose2d_quality_report": {"clip_id": record.clip_id, "status": "ok"},
            "motionbert_quality_report": {"clip_id": record.clip_id, "status": "ok"},
            "skeleton_mapper_quality_report": {"clip_id": record.clip_id, "status": "ok", "skeleton_mapping_ok": True},
            "metric_normalization_quality_report": {"clip_id": record.clip_id, "status": "ok", "metric_pose_ok": True},
            "root_trajectory_quality_report": {"clip_id": record.clip_id, "status": "ok", "root_translation_ok": True},
            "track_report": {"status": "ok"},
            "backend_run": {"status": "ok"},
            "motionbert_run": {"status": "ok", "backend": {"name": "mmpose_motionbert"}},
            "artifacts": {
                "pose2d_npz_path": "/tmp/fake_pose2d.npz",
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "pose3d_metric_local_npz_path": "/tmp/fake_pose3d_metric_local.npz",
                "pose3d_bvh_path": "/tmp/fake_pose3d.bvh",
                "pose3d_metric_keypoints_path": "/tmp/fake_3d_keypoints_metric.npy",
                "root_translation_npy_path": "/tmp/fake_root_translation.npy",
                "motionbert_run_json_path": "/tmp/fake_motionbert_run.json",
                "debug_overlay_pose3d_raw_path": "/tmp/fake_debug_overlay_pose3d_raw.mp4",
                "debug_overlay_pose3d_imugpt22_path": "/tmp/fake_debug_overlay_pose3d_imugpt22.mp4",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.robot_emotions.pose3d.RobotEmotionsExtractor", _FakeExtractor):
                with patch(
                    "pose_module.robot_emotions.pose3d.run_pose3d_pipeline",
                    return_value=fake_pipeline_result,
                ) as mocked_pipeline:
                    summary = run_robot_emotions_pose3d(
                        dataset_root="data/RobotEmotions",
                        output_dir=tmp_dir,
                        clip_ids=[record.clip_id],
                        env_name="openmmlab",
                    )

            self.assertEqual(summary["num_ok"], 1)
            self.assertEqual(summary["num_warning"], 0)
            self.assertEqual(summary["num_fail"], 0)
            manifest_path = Path(summary["pose3d_manifest_path"])
            self.assertTrue(manifest_path.exists())
            self.assertTrue((Path(tmp_dir) / "pose3d_summary.json").exists())

            manifest_entries = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip() != ""
            ]
            self.assertEqual(len(manifest_entries), 1)
            self.assertEqual(manifest_entries[0]["clip_id"], record.clip_id)
            self.assertEqual(manifest_entries[0]["pose3d"]["num_joints"], len(IMUGPT_22_JOINT_NAMES))
            self.assertEqual(manifest_entries[0]["artifacts"]["motionbert_run_json_path"], "/tmp/fake_motionbert_run.json")
            self.assertEqual(
                manifest_entries[0]["artifacts"]["pose3d_metric_keypoints_path"],
                "/tmp/fake_3d_keypoints_metric.npy",
            )
            self.assertEqual(
                manifest_entries[0]["artifacts"]["root_translation_npy_path"],
                "/tmp/fake_root_translation.npy",
            )
            self.assertEqual(manifest_entries[0]["artifacts"]["pose3d_bvh_path"], "/tmp/fake_pose3d.bvh")
            self.assertEqual(
                manifest_entries[0]["artifacts"]["debug_overlay_pose3d_raw_path"],
                "/tmp/fake_debug_overlay_pose3d_raw.mp4",
            )
            self.assertEqual(
                manifest_entries[0]["artifacts"]["debug_overlay_pose3d_imugpt22_path"],
                "/tmp/fake_debug_overlay_pose3d_imugpt22.mp4",
            )
            self.assertTrue(manifest_entries[0]["skeleton_mapper_quality_report"]["skeleton_mapping_ok"])
            self.assertTrue(manifest_entries[0]["metric_normalizer_quality_report"]["metric_pose_ok"])
            self.assertTrue(manifest_entries[0]["root_trajectory_quality_report"]["root_translation_ok"])
            mocked_pipeline.assert_called_once()

    def test_run_robot_emotions_pose3d_counts_warning_without_marking_failure(self) -> None:
        record = _make_record("robot_emotions_10ms_u02_tag05")
        pose2d_sequence = _make_motionbert_sequence(12)
        pose3d_sequence = _make_pose3d_sequence()

        fake_pipeline_result = {
            "clip_id": record.clip_id,
            "pose_sequence": pose3d_sequence,
            "pose_sequence_2d": pose2d_sequence,
            "raw_pose_sequence_2d": pose2d_sequence,
            "quality_report": {"clip_id": record.clip_id, "status": "warning", "notes": ["example_warning"]},
            "pose2d_quality_report": {"clip_id": record.clip_id, "status": "ok"},
            "motionbert_quality_report": {"clip_id": record.clip_id, "status": "ok"},
            "skeleton_mapper_quality_report": {"clip_id": record.clip_id, "status": "ok", "skeleton_mapping_ok": True},
            "metric_normalization_quality_report": {"clip_id": record.clip_id, "status": "ok", "metric_pose_ok": True},
            "root_trajectory_quality_report": {"clip_id": record.clip_id, "status": "warning", "root_translation_ok": True},
            "track_report": {"status": "ok"},
            "backend_run": {"status": "ok"},
            "motionbert_run": {"status": "ok", "backend": {"name": "mmpose_motionbert"}},
            "artifacts": {
                "pose2d_npz_path": "/tmp/fake_pose2d.npz",
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "pose3d_metric_local_npz_path": "/tmp/fake_pose3d_metric_local.npz",
                "pose3d_bvh_path": "/tmp/fake_pose3d.bvh",
                "pose3d_metric_keypoints_path": "/tmp/fake_3d_keypoints_metric.npy",
                "root_translation_npy_path": "/tmp/fake_root_translation.npy",
                "motionbert_run_json_path": "/tmp/fake_motionbert_run.json",
                "debug_overlay_pose3d_raw_path": "/tmp/fake_debug_overlay_pose3d_raw.mp4",
                "debug_overlay_pose3d_imugpt22_path": "/tmp/fake_debug_overlay_pose3d_imugpt22.mp4",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.robot_emotions.pose3d.RobotEmotionsExtractor", _FakeExtractor):
                with patch(
                    "pose_module.robot_emotions.pose3d.run_pose3d_pipeline",
                    return_value=fake_pipeline_result,
                ):
                    summary = run_robot_emotions_pose3d(
                        dataset_root="data/RobotEmotions",
                        output_dir=tmp_dir,
                        clip_ids=[record.clip_id],
                        env_name="openmmlab",
                    )

        self.assertEqual(summary["num_ok"], 0)
        self.assertEqual(summary["num_warning"], 1)
        self.assertEqual(summary["num_fail"], 0)

    def test_cli_export_pose3d_dispatches_wrapper(self) -> None:
        with patch(
            "pose_module.robot_emotions.cli.run_robot_emotions_pose3d",
            return_value={"status": "ok"},
        ) as mocked_wrapper:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_cli_main(
                    [
                        "export-pose3d",
                        "--output-dir",
                        "/tmp/robot_emotions_pose3d",
                        "--clip-id",
                        "robot_emotions_10ms_u02_tag05",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mocked_wrapper.assert_called_once()
        self.assertEqual(mocked_wrapper.call_args.kwargs["output_dir"], "/tmp/robot_emotions_pose3d")
        self.assertEqual(mocked_wrapper.call_args.kwargs["clip_ids"], ["robot_emotions_10ms_u02_tag05"])
        self.assertEqual(mocked_wrapper.call_args.kwargs["motionbert_window_size"], 81)
        self.assertEqual(mocked_wrapper.call_args.kwargs["allow_motionbert_fallback_backend"], False)
        mocked_print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
