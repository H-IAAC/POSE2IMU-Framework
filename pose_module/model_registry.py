"""Local model registry for pose-module checkpoints stored inside this repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
MMPOSE_MODELS_FILE = Path(__file__).resolve().parent / "mmpose_models.txt"

DEFAULT_VITPOSE_ALIAS = "vitpose-b"
DEFAULT_VITPOSE_MODEL_ID = "td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192"
DEFAULT_MOTIONBERT_MODEL_ID = "motionbert_dstformer-ft-243frm_8xb32-120e_h36m"
DEFAULT_PERSON_DETECTOR_MODEL_ID = "rtmdet_m_8xb32-300e_coco"


@dataclass(frozen=True)
class ModelAssetSpec:
    package_name: str
    model_id: str
    config_filename: str
    checkpoint_patterns: tuple[str, ...]


@dataclass(frozen=True)
class Pose2DBackendArtifacts:
    pose2d_model_id: str
    pose2d_config_path: Path
    pose2d_checkpoint_path: Path
    detector_model_id: str
    detector_config_path: Path
    detector_checkpoint_path: Path


@dataclass(frozen=True)
class MotionBERTArtifacts:
    model_id: str
    config_path: Path
    checkpoint_path: Path


VITPOSE_B_SPEC = ModelAssetSpec(
    package_name="mmpose",
    model_id=DEFAULT_VITPOSE_MODEL_ID,
    config_filename=f"{DEFAULT_VITPOSE_MODEL_ID}.py",
    checkpoint_patterns=(f"{DEFAULT_VITPOSE_MODEL_ID}*.pth",),
)

MOTIONBERT_SPEC = ModelAssetSpec(
    package_name="mmpose",
    model_id=DEFAULT_MOTIONBERT_MODEL_ID,
    config_filename=f"{DEFAULT_MOTIONBERT_MODEL_ID}.py",
    checkpoint_patterns=(
        "motionbert_ft_h36m-d80af323_20230531.pth",
        "motionbert_ft_h36m*.pth",
    ),
)

PERSON_DETECTOR_SPEC = ModelAssetSpec(
    package_name="mmdet",
    model_id=DEFAULT_PERSON_DETECTOR_MODEL_ID,
    config_filename=f"{DEFAULT_PERSON_DETECTOR_MODEL_ID}.py",
    checkpoint_patterns=(f"{DEFAULT_PERSON_DETECTOR_MODEL_ID}*.pth",),
)

_VITPOSE_ALIAS_TO_SPEC = {
    "vitpose": VITPOSE_B_SPEC,
    "vitpose-b": VITPOSE_B_SPEC,
    DEFAULT_VITPOSE_MODEL_ID.lower(): VITPOSE_B_SPEC,
}


def resolve_local_pose2d_backend_artifacts(model_alias: str) -> Pose2DBackendArtifacts:
    normalized_alias = str(model_alias).strip().lower()
    spec = _VITPOSE_ALIAS_TO_SPEC.get(normalized_alias)
    if spec is None:
        raise FileNotFoundError(
            f"Unsupported local pose2d model alias {model_alias!r}. "
            f"Supported aliases: {sorted(_VITPOSE_ALIAS_TO_SPEC)}"
        )

    return Pose2DBackendArtifacts(
        pose2d_model_id=spec.model_id,
        pose2d_config_path=require_local_config_path(spec),
        pose2d_checkpoint_path=require_local_checkpoint_path(spec),
        detector_model_id=PERSON_DETECTOR_SPEC.model_id,
        detector_config_path=require_local_config_path(PERSON_DETECTOR_SPEC),
        detector_checkpoint_path=require_local_checkpoint_path(PERSON_DETECTOR_SPEC),
    )


def resolve_local_motionbert_artifacts(
    *,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
) -> MotionBERTArtifacts:
    resolved_config_path = (
        Path(str(config_path)).expanduser().resolve()
        if config_path not in (None, "")
        else require_local_config_path(MOTIONBERT_SPEC)
    )
    resolved_checkpoint_path = (
        Path(str(checkpoint_path)).expanduser().resolve()
        if checkpoint_path not in (None, "")
        else require_local_checkpoint_path(MOTIONBERT_SPEC)
    )
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"MotionBERT config not found: {resolved_config_path}")
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"MotionBERT checkpoint not found: {resolved_checkpoint_path}")
    return MotionBERTArtifacts(
        model_id=MOTIONBERT_SPEC.model_id,
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
    )


def read_mmpose_manifest_models() -> List[str]:
    if not MMPOSE_MODELS_FILE.exists():
        return []
    return [
        line.strip()
        for line in MMPOSE_MODELS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() != ""
    ]


def required_download_specs() -> List[ModelAssetSpec]:
    manifest_models = set(read_mmpose_manifest_models())
    required_pose_models = [VITPOSE_B_SPEC.model_id, MOTIONBERT_SPEC.model_id]
    missing_from_manifest = [model_id for model_id in required_pose_models if model_id not in manifest_models]
    if missing_from_manifest:
        raise RuntimeError(
            "Required mmpose models missing from pose_module/mmpose_models.txt: "
            + ", ".join(missing_from_manifest)
        )
    return [VITPOSE_B_SPEC, MOTIONBERT_SPEC, PERSON_DETECTOR_SPEC]


def required_download_targets() -> List[tuple[str, str]]:
    return [(spec.package_name, spec.model_id) for spec in required_download_specs()]


def iter_local_checkpoint_files() -> Iterable[Path]:
    if not CHECKPOINTS_DIR.exists():
        return ()
    return tuple(sorted(CHECKPOINTS_DIR.iterdir()))


def find_local_config_path(
    spec: ModelAssetSpec,
    *,
    checkpoints_dir: Optional[Path] = None,
) -> Optional[Path]:
    base_dir = CHECKPOINTS_DIR if checkpoints_dir is None else Path(checkpoints_dir)
    config_path = base_dir / spec.config_filename
    return config_path.resolve() if config_path.exists() else None


def find_local_checkpoint_path(
    spec: ModelAssetSpec,
    *,
    checkpoints_dir: Optional[Path] = None,
) -> Optional[Path]:
    base_dir = CHECKPOINTS_DIR if checkpoints_dir is None else Path(checkpoints_dir)
    for pattern in spec.checkpoint_patterns:
        candidates = sorted(base_dir.glob(pattern))
        if candidates:
            return candidates[0].resolve()
    return None


def require_local_config_path(spec: ModelAssetSpec) -> Path:
    config_path = find_local_config_path(spec)
    if config_path is not None:
        return config_path
    raise FileNotFoundError(
        f"Missing local config for {spec.model_id!r} in {CHECKPOINTS_DIR}. "
        "Run `.venv/bin/python -m pose_module.download_models --env-name openmmlab`."
    )


def require_local_checkpoint_path(spec: ModelAssetSpec) -> Path:
    checkpoint_path = find_local_checkpoint_path(spec)
    if checkpoint_path is not None:
        return checkpoint_path
    raise FileNotFoundError(
        f"Missing local checkpoint for {spec.model_id!r} in {CHECKPOINTS_DIR}. "
        "Run `.venv/bin/python -m pose_module.download_models --env-name openmmlab`."
    )
