"""RobotEmotions extraction and pose utilities."""

from __future__ import annotations

__all__ = [
    "CHANNEL_AXIS_ORDER",
    "RobotEmotionsClipRecord",
    "RobotEmotionsExtractedClip",
    "RobotEmotionsExtractor",
    "run_robot_emotions_pose2d",
    "run_robot_emotions_pose3d",
]


def __getattr__(name: str):
    if name in {
        "CHANNEL_AXIS_ORDER",
        "RobotEmotionsClipRecord",
        "RobotEmotionsExtractedClip",
        "RobotEmotionsExtractor",
    }:
        from .extractor import (
            CHANNEL_AXIS_ORDER,
            RobotEmotionsClipRecord,
            RobotEmotionsExtractedClip,
            RobotEmotionsExtractor,
        )

        exports = {
            "CHANNEL_AXIS_ORDER": CHANNEL_AXIS_ORDER,
            "RobotEmotionsClipRecord": RobotEmotionsClipRecord,
            "RobotEmotionsExtractedClip": RobotEmotionsExtractedClip,
            "RobotEmotionsExtractor": RobotEmotionsExtractor,
        }
        return exports[name]

    if name == "run_robot_emotions_pose2d":
        from .pose2d import run_robot_emotions_pose2d

        return run_robot_emotions_pose2d
    if name == "run_robot_emotions_pose3d":
        from .pose3d import run_robot_emotions_pose3d

        return run_robot_emotions_pose3d

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
