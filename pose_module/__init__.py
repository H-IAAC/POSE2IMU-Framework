"""Pose pipeline package."""

from __future__ import annotations

__all__ = ["run_pose2d_pipeline", "run_pose3d_pipeline", "download_required_models"]


def __getattr__(name: str):
    if name == "run_pose2d_pipeline":
        from .pipeline import run_pose2d_pipeline

        return run_pose2d_pipeline
    if name == "run_pose3d_pipeline":
        from .pipeline import run_pose3d_pipeline

        return run_pose3d_pipeline
    if name == "download_required_models":
        from .download_models import download_required_models

        return download_required_models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
