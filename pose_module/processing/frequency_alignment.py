from __future__ import annotations

from typing import Any

import numpy as np


def _validate_timestamps(name: str, timestamps_sec: np.ndarray | list[float]) -> np.ndarray:
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError(f"{name} deve ser um array 1D de timestamps.")
    if timestamps.size == 0:
        raise ValueError(f"{name} nao pode ser vazio.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{name} contem timestamps nao finitos.")
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"{name} deve estar ordenado em ordem crescente.")
    return timestamps


def estimate_sampling_frequency_hz(timestamps_sec: np.ndarray | list[float]) -> float:
    timestamps = _validate_timestamps("timestamps_sec", timestamps_sec)
    if timestamps.size < 2:
        raise ValueError("Nao e possivel estimar frequencia com menos de 2 amostras.")
    deltas_sec = np.diff(timestamps)
    positive_deltas_sec = deltas_sec[deltas_sec > 0.0]
    if positive_deltas_sec.size == 0:
        raise ValueError("Nao e possivel estimar frequencia quando todos os timestamps sao iguais.")
    return float(1.0 / np.median(positive_deltas_sec))


def _nearest_source_indices(source_timestamps_sec: np.ndarray, reference_timestamps_sec: np.ndarray) -> np.ndarray:
    right_indices = np.searchsorted(source_timestamps_sec, reference_timestamps_sec, side="left")
    right_indices = np.clip(right_indices, 0, source_timestamps_sec.shape[0] - 1)
    left_indices = np.clip(right_indices - 1, 0, source_timestamps_sec.shape[0] - 1)

    right_distance = np.abs(source_timestamps_sec[right_indices] - reference_timestamps_sec)
    left_distance = np.abs(source_timestamps_sec[left_indices] - reference_timestamps_sec)
    use_right = right_distance < left_distance
    return np.where(use_right, right_indices, left_indices)


def undersample_signal_to_reference(
    source_timestamps_sec: np.ndarray | list[float],
    source_values: np.ndarray,
    reference_timestamps_sec: np.ndarray | list[float],
) -> dict[str, Any]:
    source_timestamps = _validate_timestamps("source_timestamps_sec", source_timestamps_sec)
    reference_timestamps = _validate_timestamps("reference_timestamps_sec", reference_timestamps_sec)
    values = np.asarray(source_values)
    if values.ndim == 0:
        raise ValueError("source_values precisa ter pelo menos um eixo temporal.")
    if values.shape[0] != source_timestamps.shape[0]:
        raise ValueError("source_values e source_timestamps_sec precisam ter o mesmo numero de frames.")

    source_indices = _nearest_source_indices(source_timestamps, reference_timestamps)
    matched_source_timestamps = source_timestamps[source_indices]
    matched_values = np.asarray(values[source_indices], dtype=values.dtype)
    time_error_sec = matched_source_timestamps - reference_timestamps

    return {
        "timestamps_sec": np.asarray(reference_timestamps, dtype=np.float32),
        "values": matched_values,
        "source_indices": np.asarray(source_indices, dtype=np.int32),
        "matched_source_timestamps_sec": np.asarray(matched_source_timestamps, dtype=np.float32),
        "time_error_sec": np.asarray(time_error_sec, dtype=np.float32),
    }


def prepare_real_signal_for_virtual_comparison(
    real_timestamps_sec: np.ndarray | list[float],
    real_values: np.ndarray,
    virtual_timestamps_sec: np.ndarray | list[float],
) -> dict[str, Any]:
    aligned = undersample_signal_to_reference(
        source_timestamps_sec=real_timestamps_sec,
        source_values=real_values,
        reference_timestamps_sec=virtual_timestamps_sec,
    )
    time_error_sec = np.abs(np.asarray(aligned["time_error_sec"], dtype=np.float64))

    aligned["summary"] = {
        "real_original_frames": int(np.asarray(real_timestamps_sec).shape[0]),
        "real_plot_frames": int(aligned["values"].shape[0]),
        "virtual_frames": int(np.asarray(virtual_timestamps_sec).shape[0]),
        "real_original_frequency_hz": estimate_sampling_frequency_hz(real_timestamps_sec),
        "real_plot_frequency_hz": estimate_sampling_frequency_hz(aligned["timestamps_sec"]),
        "virtual_frequency_hz": estimate_sampling_frequency_hz(virtual_timestamps_sec),
        "mean_time_error_ms": float(time_error_sec.mean() * 1000.0),
        "max_time_error_ms": float(time_error_sec.max() * 1000.0),
    }
    return aligned
