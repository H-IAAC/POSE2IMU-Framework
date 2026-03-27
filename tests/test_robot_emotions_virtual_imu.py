import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.interfaces import (
    IKSequence,
    IMUGPT_22_JOINT_NAMES,
    IMUGPT_22_PARENT_INDICES,
    PoseSequence3D,
    VirtualIMUSequence,
)
from pose_module.robot_emotions.cli import main as robot_emotions_cli_main
from pose_module.robot_emotions.extractor import RobotEmotionsClipRecord
from pose_module.robot_emotions.virtual_imu import run_robot_emotions_virtual_imu


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
    num_frames = 6
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    root_translation = np.zeros((num_frames, 3), dtype=np.float32)
    joint_positions_xyz = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    for frame_index in range(num_frames):
        root_translation[frame_index] = np.asarray([0.02 * frame_index, 0.0, 0.0], dtype=np.float32)
        joint_positions_xyz[frame_index, :, 0] = root_translation[frame_index, 0] + np.linspace(
            -0.4,
            0.4,
            num_joints,
            dtype=np.float32,
        )
        joint_positions_xyz[frame_index, :, 1] = np.linspace(0.8, -0.8, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, :, 2] = np.linspace(0.0, 0.3, num_joints, dtype=np.float32)
        joint_positions_xyz[frame_index, 0] = root_translation[frame_index]
    return PoseSequence3D(
        clip_id="clip_virtual_imu",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.full((num_frames, num_joints), 0.95, dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_pose3d_root",
        coordinate_space="pseudo_global_metric",
        root_translation_m=root_translation,
    )


def _make_ik_sequence() -> IKSequence:
    num_frames = 6
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    local_joint_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    local_joint_rotations[..., 0] = 1.0
    return IKSequence(
        clip_id="clip_virtual_imu",
        fps=20.0,
        fps_original=30.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        local_joint_rotations=local_joint_rotations,
        root_translation_m=np.zeros((num_frames, 3), dtype=np.float32),
        joint_offsets_m=np.zeros((num_joints, 3), dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_ik",
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


class RobotEmotionsVirtualIMUTests(unittest.TestCase):
    def test_run_robot_emotions_virtual_imu_writes_manifest_and_summary(self) -> None:
        record = _make_record("robot_emotions_10ms_u02_tag05")
        fake_pipeline_result = {
            "clip_id": record.clip_id,
            "pose_sequence": _make_pose3d_sequence(),
            "virtual_imu_sequence": VirtualIMUSequence(
                clip_id=record.clip_id,
                fps=20.0,
                sensor_names=["waist", "head", "right_forearm", "left_forearm"],
                acc=np.zeros((6, 4, 3), dtype=np.float32),
                gyro=np.zeros((6, 4, 3), dtype=np.float32),
                timestamps_sec=np.arange(6, dtype=np.float32) / np.float32(20.0),
                source="unit_test_virtual_imu",
            ),
            "ik_sequence": _make_ik_sequence(),
            "quality_report": {"clip_id": record.clip_id, "status": "ok", "notes": []},
            "pose3d_quality_report": {"clip_id": record.clip_id, "status": "ok"},
            "ik_quality_report": {"clip_id": record.clip_id, "status": "ok", "ik_ok": True},
            "virtual_imu_quality_report": {
                "clip_id": record.clip_id,
                "status": "ok",
                "virtual_imu_ok": True,
            },
            "artifacts": {
                "pose3d_npz_path": "/tmp/fake_pose3d.npz",
                "ik_sequence_npz_path": "/tmp/fake_ik_sequence.npz",
                "ik_bvh_path": "/tmp/fake_pose3d_ik.bvh",
                "virtual_imu_npz_path": "/tmp/fake_virtual_imu.npz",
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pose_module.robot_emotions.virtual_imu.RobotEmotionsExtractor", _FakeExtractor):
                with patch(
                    "pose_module.robot_emotions.virtual_imu.run_virtual_imu_pipeline",
                    return_value=fake_pipeline_result,
                ) as mocked_pipeline:
                    summary = run_robot_emotions_virtual_imu(
                        dataset_root="data/RobotEmotions",
                        output_dir=tmp_dir,
                        clip_ids=[record.clip_id],
                        env_name="openmmlab",
                    )

            self.assertEqual(summary["num_ok"], 1)
            manifest_path = Path(summary["virtual_imu_manifest_path"])
            self.assertTrue(manifest_path.exists())
            manifest_entries = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip() != ""
            ]
            self.assertEqual(manifest_entries[0]["artifacts"]["virtual_imu_npz_path"], "/tmp/fake_virtual_imu.npz")
            self.assertTrue(manifest_entries[0]["ik_quality_report"]["ik_ok"])
            self.assertTrue(manifest_entries[0]["virtual_imu_quality_report"]["virtual_imu_ok"])
            mocked_pipeline.assert_called_once()

    def test_cli_export_virtual_imu_dispatches_wrapper(self) -> None:
        with patch(
            "pose_module.robot_emotions.cli.run_robot_emotions_virtual_imu",
            return_value={"status": "ok"},
        ) as mocked_wrapper:
            with patch("builtins.print") as mocked_print:
                exit_code = robot_emotions_cli_main(
                    [
                        "export-virtual-imu",
                        "--output-dir",
                        "/tmp/robot_emotions_virtual_imu",
                        "--clip-id",
                        "robot_emotions_10ms_u02_tag05",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mocked_wrapper.assert_called_once()
        self.assertEqual(mocked_wrapper.call_args.kwargs["output_dir"], "/tmp/robot_emotions_virtual_imu")
        mocked_print.assert_called_once()
