import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose_module.export.bvh import export_pose_sequence3d_to_bvh
from pose_module.export.bvh import main as bvh_cli_main
from pose_module.interfaces import IMUGPT_22_JOINT_NAMES, IMUGPT_22_PARENT_INDICES, PoseSequence3D


def _make_pose3d_sequence(*, fps: float | None = 20.0) -> PoseSequence3D:
    num_frames = 8
    num_joints = len(IMUGPT_22_JOINT_NAMES)
    joint_positions_xyz = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    base_x = np.linspace(-0.3, 0.3, num_joints, dtype=np.float32)
    base_y = np.linspace(0.8, -0.9, num_joints, dtype=np.float32)
    for frame_index in range(num_frames):
        joint_positions_xyz[frame_index, :, 0] = base_x
        joint_positions_xyz[frame_index, :, 1] = base_y
        joint_positions_xyz[frame_index, :, 2] = np.float32(frame_index) * 0.02
    return PoseSequence3D(
        clip_id="clip_bvh",
        fps=fps,
        fps_original=30.0,
        joint_names_3d=list(IMUGPT_22_JOINT_NAMES),
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.full((num_frames, num_joints), 0.9, dtype=np.float32),
        skeleton_parents=list(IMUGPT_22_PARENT_INDICES),
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_metric_pose",
        coordinate_space="body_metric_local",
    )


def _make_generic_pose3d_sequence_nonzero_root() -> PoseSequence3D:
    num_frames = 6
    joint_names = ["HipChild", "Root", "Arm", "Hand"]
    parents = [1, -1, 1, 2]
    joint_positions_xyz = np.zeros((num_frames, len(joint_names), 3), dtype=np.float32)
    for frame_index in range(num_frames):
        root = np.asarray([0.05 * frame_index, 1.0, 0.02 * frame_index], dtype=np.float32)
        joint_positions_xyz[frame_index, 1] = root
        joint_positions_xyz[frame_index, 0] = root + np.asarray([-0.20, -0.40, 0.00], dtype=np.float32)
        joint_positions_xyz[frame_index, 2] = root + np.asarray([0.15, 0.25, 0.10], dtype=np.float32)
        joint_positions_xyz[frame_index, 3] = joint_positions_xyz[frame_index, 2] + np.asarray(
            [0.12, 0.18, -0.05], dtype=np.float32
        )

    return PoseSequence3D(
        clip_id="clip_generic_bvh",
        fps=20.0,
        fps_original=20.0,
        joint_names_3d=joint_names,
        joint_positions_xyz=joint_positions_xyz,
        joint_confidence=np.full((num_frames, len(joint_names)), 0.95, dtype=np.float32),
        skeleton_parents=parents,
        frame_indices=np.arange(num_frames, dtype=np.int32),
        timestamps_sec=np.arange(num_frames, dtype=np.float32) / np.float32(20.0),
        source="unit_test_generic_pose",
        coordinate_space="generic_space",
    )


class BVHExportTests(unittest.TestCase):
    def test_export_pose_sequence3d_to_bvh_writes_readable_bvh(self) -> None:
        sequence = _make_pose3d_sequence()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pose3d.bvh"

            artifacts = export_pose_sequence3d_to_bvh(sequence, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(artifacts["pose3d_bvh_path"], str(output_path.resolve()))
            self.assertEqual(artifacts["joint_format"], list(IMUGPT_22_JOINT_NAMES))
            contents = output_path.read_text(encoding="utf-8")
            self.assertIn("HIERARCHY\n", contents)
            self.assertIn("ROOT Pelvis\n", contents)
            self.assertIn("JOINT Left_hip\n", contents)
            self.assertIn("Frames: 8\n", contents)
            self.assertIn("Frame Time: 0.05000000\n", contents)
            self.assertIn("End Site\n", contents)
            motion_lines = [line for line in contents.splitlines() if line.strip() != ""]
            self.assertTrue(any(line.startswith("-0.300000 1.700000 0.000000") for line in motion_lines))

    def test_export_pose_sequence3d_to_bvh_uses_timestamp_fps_when_sequence_fps_missing(self) -> None:
        sequence = _make_pose3d_sequence(fps=None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pose3d.bvh"

            artifacts = export_pose_sequence3d_to_bvh(sequence, output_path)

            self.assertAlmostEqual(float(artifacts["bvh_fps"]), 20.0, places=5)

    def test_export_pose_sequence3d_to_bvh_accepts_generic_skeleton(self) -> None:
        sequence = _make_generic_pose3d_sequence_nonzero_root()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "generic_pose3d.bvh"

            artifacts = export_pose_sequence3d_to_bvh(sequence, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(artifacts["joint_format"], list(sequence.joint_names_3d))
            self.assertEqual(artifacts["coordinate_space"], "generic_space")
            contents = output_path.read_text(encoding="utf-8")
            self.assertIn("ROOT Root\n", contents)
            self.assertIn("JOINT HipChild\n", contents)
            self.assertIn("JOINT Arm\n", contents)
            self.assertIn("Frames: 6\n", contents)

    def test_cli_exports_pose3d_npz_to_requested_output_path(self) -> None:
        sequence = _make_pose3d_sequence()

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_npz_path = Path(tmp_dir) / "pose3d.npz"
            output_bvh_path = Path(tmp_dir) / "custom" / "my_clip_export.bvh"
            np.savez_compressed(input_npz_path, **sequence.to_npz_payload())

            with patch("builtins.print") as mocked_print:
                exit_code = bvh_cli_main(
                    [
                        "--pose3d-npz",
                        str(input_npz_path),
                        "--output-bvh",
                        str(output_bvh_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_bvh_path.exists())
            mocked_print.assert_called_once()
            summary = json.loads(mocked_print.call_args.args[0])
            self.assertEqual(summary["pose3d_bvh_path"], str(output_bvh_path.resolve()))
