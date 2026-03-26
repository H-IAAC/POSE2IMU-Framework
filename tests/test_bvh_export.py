import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_module.export.bvh import export_pose_sequence3d_to_bvh
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
