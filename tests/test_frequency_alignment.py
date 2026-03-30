import unittest

import numpy as np

from pose_module.processing.frequency_alignment import (
    estimate_sampling_frequency_hz,
    prepare_real_signal_for_virtual_comparison,
    undersample_signal_to_reference,
)


class FrequencyAlignmentTests(unittest.TestCase):
    def test_undersample_signal_to_reference_uses_nearest_real_frames(self) -> None:
        real_timestamps_sec = np.arange(10, dtype=np.float32) * np.float32(0.01)
        real_values = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
        virtual_timestamps_sec = np.asarray([0.011, 0.029, 0.061, 0.089], dtype=np.float32)

        aligned = undersample_signal_to_reference(
            source_timestamps_sec=real_timestamps_sec,
            source_values=real_values,
            reference_timestamps_sec=virtual_timestamps_sec,
        )

        np.testing.assert_allclose(aligned["timestamps_sec"], virtual_timestamps_sec)
        np.testing.assert_array_equal(aligned["source_indices"], np.asarray([1, 3, 6, 9], dtype=np.int32))
        np.testing.assert_allclose(
            aligned["values"][:, 0, 0],
            np.asarray([1.0, 3.0, 6.0, 9.0], dtype=np.float32),
        )
        self.assertLess(float(np.max(np.abs(aligned["time_error_sec"]))), 0.011)

    def test_prepare_real_signal_for_virtual_comparison_reports_adjusted_frequency(self) -> None:
        real_timestamps_sec = np.arange(0.0, 1.0, 0.01, dtype=np.float32)
        virtual_timestamps_sec = np.arange(0.0, 1.0, 0.05, dtype=np.float32)
        real_values = np.stack(
            [
                np.sin(2.0 * np.pi * real_timestamps_sec),
                np.cos(2.0 * np.pi * real_timestamps_sec),
            ],
            axis=1,
        ).reshape(real_timestamps_sec.shape[0], 1, 2)

        aligned = prepare_real_signal_for_virtual_comparison(
            real_timestamps_sec=real_timestamps_sec,
            real_values=real_values,
            virtual_timestamps_sec=virtual_timestamps_sec,
        )

        summary = aligned["summary"]

        self.assertEqual(summary["real_original_frames"], 100)
        self.assertEqual(summary["real_plot_frames"], 20)
        self.assertEqual(summary["virtual_frames"], 20)
        self.assertAlmostEqual(summary["real_original_frequency_hz"], 100.0, places=3)
        self.assertAlmostEqual(summary["real_plot_frequency_hz"], 20.0, places=3)
        self.assertAlmostEqual(summary["virtual_frequency_hz"], 20.0, places=3)
        self.assertLess(summary["max_time_error_ms"], 6.0)
        self.assertAlmostEqual(estimate_sampling_frequency_hz(aligned["timestamps_sec"]), 20.0, places=3)
