import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_module.imu_alignment import (
    AlignmentConfig,
    AlignmentFittingError,
    IMUSequence,
    apply_sensor_subject_transform,
    estimate_rotation_procrustes,
    estimate_time_lag,
    fit_sensor_subject_transforms,
    run_geometric_alignment,
)
from pose_module.interfaces import VirtualIMUSequence


def _apply_rotation(values_xyz: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(values_xyz, dtype=np.float32) @ np.asarray(rotation_matrix, dtype=np.float32).T


def _shift_signal(values_xyz: np.ndarray, lag_samples: int) -> np.ndarray:
    values = np.asarray(values_xyz, dtype=np.float32)
    shifted = np.zeros_like(values)
    if lag_samples > 0:
        shifted[lag_samples:] = values[:-lag_samples]
    elif lag_samples < 0:
        lag = abs(lag_samples)
        shifted[:-lag] = values[lag:]
    else:
        shifted[:] = values
    return shifted


def _sensor_waveforms(timestamps: np.ndarray, *, sensor_offset: float, capture_offset: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(timestamps, dtype=np.float32)
    gyro = np.stack(
        [
            0.65 * np.sin(1.4 * t + sensor_offset) + 0.18 * np.cos(0.4 * t + capture_offset),
            0.52 * np.cos(1.0 * t + 0.5 * sensor_offset) + 0.12 * np.sin(1.7 * t + capture_offset),
            0.48 * np.sin(1.8 * t + 0.3 * sensor_offset + capture_offset),
        ],
        axis=1,
    ).astype(np.float32)
    acc = np.stack(
        [
            0.34 * np.sin(0.8 * t + sensor_offset) + 0.07 * np.cos(0.2 * t + capture_offset),
            9.81 + 0.21 * np.cos(0.6 * t + capture_offset) + 0.06 * np.sin(1.2 * t + sensor_offset),
            0.28 * np.sin(0.55 * t + 0.4 * sensor_offset) + 0.09 * np.cos(0.9 * t + capture_offset),
        ],
        axis=1,
    ).astype(np.float32)
    return acc, gyro


def _make_pair(
    *,
    subject_id: str,
    capture_id: str,
    sensor_names: list[str],
    rotations: dict[str, np.ndarray],
    lags: dict[str, int],
    num_frames: int = 240,
    fps: float = 20.0,
    capture_offset: float = 0.0,
) -> tuple[IMUSequence, IMUSequence]:
    timestamps = np.arange(num_frames, dtype=np.float32) / np.float32(fps)
    virtual_acc = np.zeros((num_frames, len(sensor_names), 3), dtype=np.float32)
    virtual_gyro = np.zeros_like(virtual_acc)
    real_acc = np.zeros_like(virtual_acc)
    real_gyro = np.zeros_like(virtual_acc)
    for sensor_index, sensor_name in enumerate(sensor_names):
        acc, gyro = _sensor_waveforms(
            timestamps,
            sensor_offset=0.4 * (sensor_index + 1),
            capture_offset=float(capture_offset),
        )
        virtual_acc[:, sensor_index, :] = acc
        virtual_gyro[:, sensor_index, :] = gyro
        real_acc[:, sensor_index, :] = _shift_signal(
            _apply_rotation(acc, rotations[sensor_name]),
            lags[sensor_name],
        )
        real_gyro[:, sensor_index, :] = _shift_signal(
            _apply_rotation(gyro, rotations[sensor_name]),
            lags[sensor_name],
        )

    real_sequence = IMUSequence(
        subject_id=subject_id,
        capture_id=capture_id,
        sensor_names=list(sensor_names),
        fps=fps,
        timestamps=timestamps,
        acc=real_acc,
        gyro=real_gyro,
    )
    virtual_sequence = IMUSequence(
        subject_id=subject_id,
        capture_id=capture_id,
        sensor_names=list(sensor_names),
        fps=fps,
        timestamps=timestamps,
        acc=virtual_acc,
        gyro=virtual_gyro,
    )
    return real_sequence, virtual_sequence


class IMUAlignmentTests(unittest.TestCase):
    def test_estimate_rotation_procrustes_recovers_known_rotation(self) -> None:
        rng = np.random.default_rng(42)
        virtual = rng.normal(size=(200, 3)).astype(np.float32)
        rotation_matrix = Rotation.from_euler("xyz", [18.0, -11.0, 32.0], degrees=True).as_matrix().astype(np.float32)
        real = _apply_rotation(virtual, rotation_matrix)

        estimated = estimate_rotation_procrustes(virtual, real)
        delta = Rotation.from_matrix(estimated) * Rotation.from_matrix(rotation_matrix).inv()

        self.assertLess(np.degrees(delta.magnitude()), 0.5)
        self.assertGreater(float(np.linalg.det(estimated)), 0.999)

    def test_estimate_rotation_procrustes_enforces_positive_determinant(self) -> None:
        rng = np.random.default_rng(123)
        virtual = rng.normal(size=(120, 3)).astype(np.float32)
        reflected = virtual.copy()
        reflected[:, 2] *= -1.0

        estimated = estimate_rotation_procrustes(virtual, reflected)

        self.assertGreater(float(np.linalg.det(estimated)), 0.999)

    def test_estimate_time_lag_recovers_known_shift(self) -> None:
        timestamps = np.arange(200, dtype=np.float32) / np.float32(20.0)
        _, gyro = _sensor_waveforms(timestamps, sensor_offset=0.5, capture_offset=0.25)
        real = _shift_signal(gyro, 4)

        lag = estimate_time_lag(real, gyro, max_lag=10, mode="gyro_norm")

        self.assertEqual(lag, 4)

    def test_fit_sensor_subject_transforms_recovers_subject_sensor_rotations(self) -> None:
        sensor_names = ["left_forearm", "right_forearm"]
        rotations = {
            "left_forearm": Rotation.from_euler("xyz", [-20.0, 11.0, 26.0], degrees=True).as_matrix().astype(np.float32),
            "right_forearm": Rotation.from_euler("xyz", [16.0, -14.0, 31.0], degrees=True).as_matrix().astype(np.float32),
        }
        lags_a = {"left_forearm": 3, "right_forearm": -2}
        lags_b = {"left_forearm": -1, "right_forearm": 4}
        real_a, virtual_a = _make_pair(
            subject_id="user_01",
            capture_id="capture_a",
            sensor_names=sensor_names,
            rotations=rotations,
            lags=lags_a,
            capture_offset=0.1,
        )
        real_b, virtual_b = _make_pair(
            subject_id="user_01",
            capture_id="capture_b",
            sensor_names=sensor_names,
            rotations=rotations,
            lags=lags_b,
            capture_offset=0.7,
        )

        transforms = fit_sensor_subject_transforms(
            [real_a, real_b],
            [virtual_a, virtual_b],
            AlignmentConfig(),
        )

        self.assertEqual(sorted(transforms.keys()), [("user_01", "left_forearm"), ("user_01", "right_forearm")])
        for sensor_name in sensor_names:
            estimated = transforms[("user_01", sensor_name)].rotation
            delta = Rotation.from_matrix(estimated) * Rotation.from_matrix(rotations[sensor_name]).inv()
            self.assertLess(np.degrees(delta.magnitude()), 4.0)
            self.assertGreater(float(np.linalg.det(estimated)), 0.999)

    def test_apply_sensor_subject_transform_improves_rmse_and_correlation(self) -> None:
        sensor_names = ["left_forearm", "right_forearm"]
        rotations = {
            "left_forearm": Rotation.from_euler("xyz", [-18.0, 13.0, 24.0], degrees=True).as_matrix().astype(np.float32),
            "right_forearm": Rotation.from_euler("xyz", [14.0, -9.0, 29.0], degrees=True).as_matrix().astype(np.float32),
        }
        train_real, train_virtual = _make_pair(
            subject_id="user_02",
            capture_id="capture_train",
            sensor_names=sensor_names,
            rotations=rotations,
            lags={"left_forearm": 2, "right_forearm": -3},
            capture_offset=0.2,
        )
        test_real, test_virtual = _make_pair(
            subject_id="user_02",
            capture_id="capture_test",
            sensor_names=sensor_names,
            rotations=rotations,
            lags={"left_forearm": -2, "right_forearm": 1},
            capture_offset=0.9,
        )
        transforms = fit_sensor_subject_transforms([train_real], [train_virtual], AlignmentConfig())

        results = apply_sensor_subject_transform(test_real, test_virtual, transforms, AlignmentConfig())

        self.assertEqual(len(results), 2)
        for result in results:
            before_acc_rmse = sum(result.metrics_before["modalities"]["acc"]["rmse_per_axis"].values())
            after_acc_rmse = sum(result.metrics_after["modalities"]["acc"]["rmse_per_axis"].values())
            before_gyro_rmse = sum(result.metrics_before["modalities"]["gyro"]["rmse_per_axis"].values())
            after_gyro_rmse = sum(result.metrics_after["modalities"]["gyro"]["rmse_per_axis"].values())
            before_acc_corr = np.mean(
                [
                    value
                    for value in result.metrics_before["modalities"]["acc"]["corr_per_axis"].values()
                    if value is not None
                ]
            )
            after_acc_corr = np.mean(
                [
                    value
                    for value in result.metrics_after["modalities"]["acc"]["corr_per_axis"].values()
                    if value is not None
                ]
            )
            self.assertLess(after_acc_rmse, before_acc_rmse)
            self.assertLess(after_gyro_rmse, before_gyro_rmse)
            self.assertGreater(after_acc_corr, before_acc_corr)

    def test_fit_sensor_subject_transforms_fails_on_missing_pairs_or_sensors(self) -> None:
        real_sequence, virtual_sequence = _make_pair(
            subject_id="user_03",
            capture_id="capture_ok",
            sensor_names=["left_forearm", "right_forearm"],
            rotations={
                "left_forearm": np.eye(3, dtype=np.float32),
                "right_forearm": np.eye(3, dtype=np.float32),
            },
            lags={"left_forearm": 0, "right_forearm": 0},
        )
        mismatched_virtual = IMUSequence(
            subject_id="user_03",
            capture_id="capture_other",
            sensor_names=["left_forearm"],
            fps=virtual_sequence.fps,
            timestamps=virtual_sequence.timestamps,
            acc=virtual_sequence.acc[:, :1, :],
            gyro=virtual_sequence.gyro[:, :1, :],
        )

        with self.assertRaises(AlignmentFittingError):
            fit_sensor_subject_transforms([real_sequence], [mismatched_virtual], AlignmentConfig())

        mismatched_sensor_virtual = IMUSequence(
            subject_id="user_03",
            capture_id="capture_ok",
            sensor_names=["left_forearm"],
            fps=virtual_sequence.fps,
            timestamps=virtual_sequence.timestamps,
            acc=virtual_sequence.acc[:, :1, :],
            gyro=virtual_sequence.gyro[:, :1, :],
        )
        with self.assertRaises(AlignmentFittingError):
            fit_sensor_subject_transforms([real_sequence], [mismatched_sensor_virtual], AlignmentConfig())

    def test_run_geometric_alignment_is_passthrough_when_disabled(self) -> None:
        timestamps = np.arange(32, dtype=np.float32) / np.float32(20.0)
        acc, gyro = _sensor_waveforms(timestamps, sensor_offset=0.3, capture_offset=0.6)
        virtual_sequence = VirtualIMUSequence(
            clip_id="clip_alignment_disabled",
            fps=20.0,
            sensor_names=["left_forearm"],
            acc=acc[:, None, :],
            gyro=gyro[:, None, :],
            timestamps_sec=timestamps,
            source="unit_test_virtual",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_geometric_alignment(virtual_sequence, output_dir=tmp_dir)

            self.assertEqual(result["status"], "not_enabled")
            np.testing.assert_allclose(
                result["aligned_virtual_imu_sequence"].acc,
                virtual_sequence.acc,
                atol=1e-6,
            )
            self.assertIsNone(result["artifacts"]["virtual_imu_geometric_aligned_npz_path"])


if __name__ == "__main__":
    unittest.main()
