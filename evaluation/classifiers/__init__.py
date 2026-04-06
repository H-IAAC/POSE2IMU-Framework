from __future__ import annotations

from .alignment import align_target_to_reference, estimate_lag_cross_correlation, resample_values_to_reference
from .data import (
    ALL_CAPTURE_BLACKLIST,
    WindowedDatasetConfig,
    apply_capture_blacklist,
    build_classifier_capture_table,
    build_windowed_multimodal_dataset,
    load_capture_modalities,
    normalize_capture_blacklist,
    prepare_capture_windows,
)
from .metrics import compute_domain_gap_summary, compute_multitask_metrics, plot_confusion_matrices

__all__ = [
    "WindowedDatasetConfig",
    "ALL_CAPTURE_BLACKLIST",
    "align_target_to_reference",
    "apply_capture_blacklist",
    "build_classifier_capture_table",
    "build_windowed_multimodal_dataset",
    "compute_domain_gap_summary",
    "compute_multitask_metrics",
    "estimate_lag_cross_correlation",
    "load_capture_modalities",
    "normalize_capture_blacklist",
    "plot_confusion_matrices",
    "prepare_capture_windows",
    "resample_values_to_reference",
]

try:  # pragma: no cover - depends on optional torch dependency
    from .experiments import EXPERIMENT_SPECS, SplitConfig, build_subject_group_splits, run_experiment_suite, run_single_experiment
    from .training import ModelConfig, TrainingConfig

    __all__.extend(
        [
            "EXPERIMENT_SPECS",
            "ModelConfig",
            "SplitConfig",
            "TrainingConfig",
            "build_subject_group_splits",
            "run_experiment_suite",
            "run_single_experiment",
        ]
    )
except ImportError:
    pass
