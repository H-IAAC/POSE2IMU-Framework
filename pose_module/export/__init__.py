"""Export helpers for debug artifacts."""

from typing import Any

from .debug_video import (
    render_pose_overlay_video,
    render_pose3d_side_by_side_video,
    resolve_debug_overlay_path,
    resolve_debug_overlay_variant_path,
)


def export_pose_sequence3d_to_bvh(*args: Any, **kwargs: Any) -> Any:
    from .bvh import export_pose_sequence3d_to_bvh as _export_pose_sequence3d_to_bvh

    return _export_pose_sequence3d_to_bvh(*args, **kwargs)

__all__ = [
    "export_pose_sequence3d_to_bvh",
    "render_pose_overlay_video",
    "render_pose3d_side_by_side_video",
    "resolve_debug_overlay_path",
    "resolve_debug_overlay_variant_path",
]
