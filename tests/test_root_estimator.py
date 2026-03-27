import unittest

import numpy as np

from pose_module.interfaces import IMUGPT_22_JOINT_NAMES
from pose_module.processing.metric_normalizer import (
    BODY_METRIC_LOCAL_COORDINATE_SPACE,
    run_metric_normalizer,
)
from pose_module.processing.root_estimator import (
    PSEUDO_GLOBAL_METRIC_COORDINATE_SPACE,
    run_root_trajectory_estimator,
)
from tests.test_metric_normalizer import _make_imugpt22_sequence


class RootTrajectoryEstimatorTests(unittest.TestCase):
    def test_run_root_trajectory_estimator_builds_pseudo_global_pose(self) -> None:
        mapped_sequence = _make_imugpt22_sequence()
        metric_result = run_metric_normalizer(
            mapped_sequence,
            target_femur_length_m=0.45,
            smoothing_window_length=5,
            smoothing_polyorder=2,
        )

        result = run_root_trajectory_estimator(
            metric_result["pose_sequence"],
            normalization_result=metric_result["normalization_result"],
            smoothing_window_length=5,
            smoothing_polyorder=2,
        )

        pose_sequence = result["pose_sequence"]
        root_translation = np.asarray(result["root_translation_m"], dtype=np.float32)
        pelvis_index = IMUGPT_22_JOINT_NAMES.index("Pelvis")

        self.assertEqual(metric_result["pose_sequence"].coordinate_space, BODY_METRIC_LOCAL_COORDINATE_SPACE)
        self.assertEqual(pose_sequence.coordinate_space, PSEUDO_GLOBAL_METRIC_COORDINATE_SPACE)
        self.assertEqual(result["quality_report"]["status"], "ok")
        self.assertTrue(result["quality_report"]["root_translation_ok"])
        np.testing.assert_allclose(pose_sequence.root_translation_m, root_translation, atol=1e-6)
        np.testing.assert_allclose(
            pose_sequence.joint_positions_xyz[:, pelvis_index, :],
            root_translation,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            pose_sequence.joint_positions_xyz - root_translation[:, None, :],
            metric_result["pose_sequence"].joint_positions_xyz,
            atol=1e-5,
        )

        expected_root = (
            metric_result["normalization_result"]["joint_positions_3d_norm"][:, pelvis_index, :]
            * np.float32(metric_result["normalization_result"]["scale_factor"])
        )
        np.testing.assert_allclose(root_translation, expected_root, atol=5e-3)

    def test_run_root_trajectory_estimator_can_planarize_vertical_drift(self) -> None:
        mapped_sequence = _make_imugpt22_sequence()
        metric_result = run_metric_normalizer(
            mapped_sequence,
            target_femur_length_m=0.45,
            smoothing_window_length=5,
            smoothing_polyorder=2,
        )

        result = run_root_trajectory_estimator(
            metric_result["pose_sequence"],
            normalization_result=metric_result["normalization_result"],
            smoothing_window_length=5,
            smoothing_polyorder=2,
            planarize_vertical=True,
        )

        root_translation = np.asarray(result["root_translation_m"], dtype=np.float32)
        self.assertAlmostEqual(float(np.ptp(root_translation[:, 1])), 0.0, places=6)
        self.assertTrue(result["quality_report"]["planarize_vertical"])
        self.assertIn("vertical_root_planarized_to_clip_median", result["quality_report"]["notes"])
