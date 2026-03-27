import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_module.export.ik_adapter import forward_kinematics_from_ik_sequence, run_ik
from pose_module.processing.metric_normalizer import run_metric_normalizer
from pose_module.processing.root_estimator import PSEUDO_GLOBAL_METRIC_COORDINATE_SPACE, run_root_trajectory_estimator
from tests.test_metric_normalizer import _make_imugpt22_sequence


def _make_pseudo_global_pose_sequence():
    mapped_sequence = _make_imugpt22_sequence(num_frames=10, yaw_rad=0.25)
    metric_result = run_metric_normalizer(
        mapped_sequence,
        target_femur_length_m=0.45,
        smoothing_window_length=5,
        smoothing_polyorder=2,
    )
    root_result = run_root_trajectory_estimator(
        metric_result["pose_sequence"],
        normalization_result=metric_result["normalization_result"],
        smoothing_window_length=5,
        smoothing_polyorder=2,
    )
    return root_result["pose_sequence"]


class IKAdapterTests(unittest.TestCase):
    def test_run_ik_exports_sequence_and_bvh(self) -> None:
        pose_sequence = _make_pseudo_global_pose_sequence()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_ik(pose_sequence, output_dir=tmp_dir)

            self.assertEqual(pose_sequence.coordinate_space, PSEUDO_GLOBAL_METRIC_COORDINATE_SPACE)
            self.assertEqual(result["ik_sequence"].local_joint_rotations.shape, (10, 22, 4))
            self.assertEqual(result["root_translation_m"].shape, (10, 3))
            self.assertTrue(Path(result["artifacts"]["ik_sequence_npz_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["ik_report_json_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["ik_bvh_path"]).exists())
            self.assertEqual(result["quality_report"]["status"], "ok")
            self.assertTrue(result["quality_report"]["ik_ok"])

            reconstructed = forward_kinematics_from_ik_sequence(result["ik_sequence"])
            reconstruction_error = np.linalg.norm(
                reconstructed["joint_positions_global_m"] - pose_sequence.joint_positions_xyz,
                axis=2,
            )
            self.assertLess(float(np.mean(reconstruction_error)), 0.05)

