"""Retarget IMUGPT22 joint positions to SMPLX22-compatible space for Kimodo.

IMUGPT22 and SMPLX22 share identical 22-joint topology and parent indices.
Two corrections are needed before passing positions to Kimodo:

1. Bone-length rescaling: MotionBERT produces positions scaled to the subject's
   actual body; Kimodo's SMPLX22 neutral skeleton has fixed canonical lengths.

2. Hip bone direction inversion: in IMUGPT22 the pelvis→hip vector points +Y
   (hips above pelvis in the pipeline restpose); in SMPLX22 it points −Y.
   Passing raw IMUGPT hip positions to _estimate_global_rotations_from_positions
   causes a ~180° rotation error on both hips, making the person appear seated.

``positions_imugpt22_to_smplx22_space`` applies both corrections in one pass.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_SMPLX22_ASSETS_PATH = (
    Path(__file__).resolve().parents[1]
    / "kimodo" / "kimodo" / "assets" / "skeletons" / "smplx22" / "joints.p"
)

# Canonical parent indices for both IMUGPT22 and SMPLX22 (identical topology).
_PARENTS: tuple[int, ...] = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

# Joints whose bone direction is inverted between IMUGPT22 and SMPLX22.
# pelvis→hip points +Y in IMUGPT22 and −Y in SMPLX22.
_INVERTED_BONE_JOINTS: frozenset[int] = frozenset({1, 2})  # L_Hip, R_Hip


@lru_cache(maxsize=1)
def _load_smplx22_neutral() -> np.ndarray:
    """Load and cache SMPLX22 neutral joint positions, shape (22, 3), pelvis-centred."""
    if not _SMPLX22_ASSETS_PATH.exists():
        raise FileNotFoundError(
            f"SMPLX22 neutral joints not found at {_SMPLX22_ASSETS_PATH}. "
            "Ensure the kimodo submodule is initialised."
        )
    import torch
    neutral = torch.load(str(_SMPLX22_ASSETS_PATH), map_location="cpu", weights_only=False)
    neutral = np.asarray(neutral, dtype=np.float64)
    neutral -= neutral[0]
    return neutral.astype(np.float32, copy=False)


@lru_cache(maxsize=1)
def _smplx22_bone_lengths() -> np.ndarray:
    """Return canonical SMPLX22 bone lengths, shape (22,). Root bone length is 0."""
    neutral = _load_smplx22_neutral()
    lengths = np.zeros(22, dtype=np.float32)
    for i, p in enumerate(_PARENTS):
        if p >= 0:
            lengths[i] = float(np.linalg.norm(neutral[i] - neutral[p]))
    return lengths


def positions_imugpt22_to_smplx22_space(
    positions: np.ndarray,
    *,
    pelvis_index: int = 0,
) -> np.ndarray:
    """Convert IMUGPT22 joint positions to SMPLX22-compatible space.

    Applies two corrections in a single topological pass:
    - Rescales each bone to the SMPLX22 canonical length.
    - Flips the bone vector for L_Hip and R_Hip to match SMPLX22 orientation.

    After this transform, Kimodo's _estimate_global_rotations_from_positions
    produces correct rotations for all joints including the hips.

    Args:
        positions: float32 (T, 22, 3) in IMUGPT pseudo-global space.
        pelvis_index: pelvis joint index (0 for both skeletons).

    Returns:
        float32 (T, 22, 3) — SMPLX22 bone lengths and orientations,
        same pelvis trajectory as input.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 2:
        positions = positions[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    pelvis = positions[:, pelvis_index : pelvis_index + 1, :].copy()  # (T, 1, 3)
    centered = positions - pelvis

    target_lengths = _smplx22_bone_lengths()
    neutral = _load_smplx22_neutral()
    out = centered.copy()

    for i, p in enumerate(_PARENTS):
        if p < 0:
            continue

        target_len = float(target_lengths[i])
        if target_len < 1e-6:
            continue

        bone_vec = centered[:, i, :] - centered[:, p, :]  # (T, 3)
        bone_norms = np.linalg.norm(bone_vec, axis=1, keepdims=True)  # (T, 1)
        degenerate = bone_norms[:, 0] < 1e-6

        rest_dir = (neutral[i] - neutral[p]).astype(np.float32)
        rest_norm = float(np.linalg.norm(rest_dir))
        rest_dir = rest_dir / rest_norm if rest_norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

        bone_dirs = np.where(
            degenerate[:, np.newaxis],
            rest_dir[np.newaxis],
            bone_vec / np.maximum(bone_norms, 1e-6),
        )

        if i in _INVERTED_BONE_JOINTS:
            bone_dirs = -bone_dirs

        out[:, i, :] = out[:, p, :] + bone_dirs * target_len

    result = (out + pelvis).astype(np.float32, copy=False)
    return result[0] if squeeze else result


def _ensure_kimodo_on_syspath() -> None:
    import sys
    kimodo_repo = Path(__file__).resolve().parents[1] / "kimodo"
    kimodo_str = str(kimodo_repo)
    if kimodo_repo.exists() and kimodo_str not in sys.path:
        sys.path.insert(0, kimodo_str)
