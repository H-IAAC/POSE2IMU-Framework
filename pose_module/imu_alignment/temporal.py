"""Temporal alignment helpers for IMU signals."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_time_lag(
    real_xyz: np.ndarray,
    virt_xyz: np.ndarray,
    max_lag: int,
    mode: str,
) -> int:
    """Estimate the sample lag that best aligns real and virtual signal norms.

    The returned lag follows the convention used by :func:`align_streams_with_lag`.
    Positive lag means the real stream is delayed relative to the virtual stream,
    so the aligned overlap is ``real[lag:]`` against ``virt[:-lag]``.
    """

    if mode not in {"gyro_norm", "acc_norm"}:
        raise ValueError("estimate_time_lag mode must be 'gyro_norm' or 'acc_norm'.")

    real = _validate_xyz_signal(real_xyz, "real_xyz")
    virt = _validate_xyz_signal(virt_xyz, "virt_xyz")
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("estimate_time_lag max_lag must be non-negative.")

    real_norm = np.linalg.norm(real, axis=1)
    virt_norm = np.linalg.norm(virt, axis=1)

    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        aligned_real, aligned_virt = align_streams_with_lag(real_norm, virt_norm, lag)
        finite_mask = np.isfinite(aligned_real) & np.isfinite(aligned_virt)
        if int(np.count_nonzero(finite_mask)) < 3:
            continue
        score = _centered_correlation(aligned_real[finite_mask], aligned_virt[finite_mask])
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def align_streams_with_lag(
    real_values: np.ndarray,
    virt_values: np.ndarray,
    lag_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the overlapping slices implied by a discrete lag."""

    real = np.asarray(real_values)
    virt = np.asarray(virt_values)
    lag_samples = int(lag_samples)
    if real.shape[0] == 0 or virt.shape[0] == 0:
        return real[:0], virt[:0]
    if lag_samples > 0:
        if lag_samples >= min(real.shape[0], virt.shape[0]):
            return real[:0], virt[:0]
        return real[lag_samples:], virt[:-lag_samples]
    if lag_samples < 0:
        lag = abs(lag_samples)
        if lag >= min(real.shape[0], virt.shape[0]):
            return real[:0], virt[:0]
        return real[:-lag], virt[lag:]
    return real, virt


def _validate_xyz_signal(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [T, 3].")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return array


def _centered_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    y_centered = np.asarray(y, dtype=np.float64) - float(np.mean(y))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0.0:
        return -np.inf
    return float(np.dot(x_centered, y_centered) / denominator)
