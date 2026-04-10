"""Dataset discovery and video metadata helpers for RobotEmotions VLM export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .metadata import canonicalize_labels, get_protocol_info, get_user_profile

USER_DIR_RE = re.compile(r"(?i)^user(\d+)$")
TAG_DIR_RE = re.compile(r"(?i)^tag(\d+)$")


@dataclass(frozen=True)
class RobotEmotionsVideoClip:
    """Single RobotEmotions capture used as input to the VLM."""

    clip_id: str
    domain: str
    user_id: int
    tag_number: int
    take_id: str | None
    tag_dir: Path
    video_path: Path
    imu_csv_path: Path | None
    source_rel_dir: str
    participant: dict[str, Any]
    protocol: dict[str, Any] | None
    labels: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        """Serialize clip metadata for debug artifacts."""

        return {
            "clip_id": self.clip_id,
            "dataset": "RobotEmotions",
            "domain": self.domain,
            "user_id": int(self.user_id),
            "tag_number": int(self.tag_number),
            "take_id": self.take_id,
            "labels": dict(self.labels),
            "participant": dict(self.participant),
            "protocol": None if self.protocol is None else dict(self.protocol),
            "source": {
                "tag_dir": str(self.tag_dir.resolve()),
                "source_rel_dir": self.source_rel_dir,
                "video_path": str(self.video_path.resolve()),
                "imu_csv_path": None if self.imu_csv_path is None else str(self.imu_csv_path.resolve()),
            },
        }


class RobotEmotionsDataset:
    """Discover RobotEmotions captures without depending on pose_module."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        domains: tuple[str, ...] = ("10ms", "30ms"),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.domains = tuple(str(domain) for domain in domains)

    def scan(self) -> list[RobotEmotionsVideoClip]:
        """Scan the dataset tree and return one entry per video capture."""

        records: list[RobotEmotionsVideoClip] = []
        clip_id_counts: dict[str, int] = {}
        for domain in self.domains:
            domain_root = self.dataset_root / domain
            if not domain_root.exists():
                continue
            for tag_dir in sorted(domain_root.rglob("*")):
                if not tag_dir.is_dir():
                    continue
                tag_number = _parse_tag_number(tag_dir.name)
                if tag_number is None:
                    continue
                user_id = _find_last_user_id(tag_dir.relative_to(domain_root).parts)
                if user_id is None:
                    continue

                for imu_csv_path, video_path in _match_capture_pairs(
                    tag_dir=tag_dir,
                    user_id=user_id,
                    tag_number=tag_number,
                ):
                    take_id = _infer_take_id(
                        video_path=video_path,
                        user_id=user_id,
                        tag_number=tag_number,
                        imu_csv_path=imu_csv_path,
                    )
                    clip_id = _build_clip_id(
                        domain=domain,
                        user_id=user_id,
                        tag_number=tag_number,
                        take_id=take_id,
                        clip_id_counts=clip_id_counts,
                    )
                    protocol = get_protocol_info(domain, tag_number)
                    records.append(
                        RobotEmotionsVideoClip(
                            clip_id=clip_id,
                            domain=domain,
                            user_id=int(user_id),
                            tag_number=int(tag_number),
                            take_id=take_id,
                            tag_dir=tag_dir,
                            video_path=video_path,
                            imu_csv_path=imu_csv_path,
                            source_rel_dir=str(tag_dir.relative_to(self.dataset_root)),
                            participant=get_user_profile(domain, user_id),
                            protocol=protocol,
                            labels=canonicalize_labels(protocol),
                        )
                    )

        records.sort(
            key=lambda item: (
                item.domain,
                item.user_id,
                item.tag_number,
                item.source_rel_dir,
                "" if item.take_id is None else item.take_id,
            )
        )
        return records

    def select_records(
        self,
        *,
        clip_ids: list[str] | tuple[str, ...] | None = None,
    ) -> list[RobotEmotionsVideoClip]:
        """Return either the full scan or a filtered subset by clip id."""

        records = self.scan()
        if clip_ids is None:
            return records

        requested = {str(clip_id) for clip_id in clip_ids}
        records = [record for record in records if record.clip_id in requested]
        found = {record.clip_id for record in records}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Unknown clip_id values requested: {missing}")
        return records


def resolve_clip_output_dir(output_root: str | Path, record: RobotEmotionsVideoClip) -> Path:
    """Use the same stable per-clip layout for all exported artifacts."""

    return Path(output_root) / record.domain / f"user_{record.user_id:02d}" / record.clip_id


def read_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Read lightweight video metadata from ``av`` inside the kimodo environment."""

    path = Path(video_path)
    try:
        import av
    except ImportError:
        return {
            "available": False,
            "video_path": str(path.resolve()),
            "reason": "pyav_not_installed",
        }

    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return {
                    "available": False,
                    "video_path": str(path.resolve()),
                    "reason": "no_video_stream",
                }

            fps = None if stream.average_rate is None else float(stream.average_rate)
            num_frames = int(stream.frames) if stream.frames else None
            width = int(stream.width) if stream.width else None
            height = int(stream.height) if stream.height else None

            duration_sec = None
            if stream.duration is not None and stream.time_base is not None:
                duration_sec = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration_sec = float(container.duration / av.time_base)

            if num_frames is None and fps is not None and duration_sec is not None:
                num_frames = int(round(fps * duration_sec))

            return {
                "available": True,
                "video_path": str(path.resolve()),
                "fps": fps,
                "num_frames": num_frames,
                "duration_sec": duration_sec,
                "width": width,
                "height": height,
            }
    except Exception as exc:  # pragma: no cover - exact backend exceptions vary with pyav
        return {
            "available": False,
            "video_path": str(path.resolve()),
            "reason": "pyav_open_failed",
            "error": str(exc),
        }


def _parse_tag_number(name: str) -> int | None:
    match = TAG_DIR_RE.match(str(name).strip())
    if match is None:
        return None
    return int(match.group(1))


def _find_last_user_id(parts: tuple[str, ...] | list[str]) -> int | None:
    user_id: int | None = None
    for part in parts:
        match = USER_DIR_RE.match(str(part).strip())
        if match is not None:
            user_id = int(match.group(1))
    return user_id


def _match_capture_pairs(
    *,
    tag_dir: Path,
    user_id: int,
    tag_number: int,
) -> list[tuple[Path | None, Path]]:
    csv_candidates = sorted(tag_dir.glob("*.csv"))
    video_candidates = sorted(tag_dir.glob("*.mp4"))
    if len(video_candidates) == 0:
        return []

    csv_by_key = _group_candidates_by_capture_key(
        candidates=csv_candidates,
        prefixes=[f"esp_{user_id}_{tag_number}"],
    )
    video_by_key = _group_candidates_by_capture_key(
        candidates=video_candidates,
        prefixes=[
            f"tag_{user_id}_{tag_number}",
            f"tag_{tag_number}",
            f"tag{tag_number}",
        ],
    )

    pairs: list[tuple[Path | None, Path]] = []
    for capture_key in sorted(video_by_key, key=_capture_sort_key):
        video_paths = sorted(video_by_key[capture_key])
        csv_paths = sorted(csv_by_key.get(capture_key, []))
        if len(csv_paths) == 0:
            for video_path in video_paths:
                pairs.append((None, video_path))
            continue
        for index, video_path in enumerate(video_paths):
            pairs.append((csv_paths[min(index, len(csv_paths) - 1)], video_path))
    return pairs


def _group_candidates_by_capture_key(
    *,
    candidates: list[Path],
    prefixes: list[str],
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for candidate in candidates:
        capture_key = _extract_capture_key(candidate.stem, prefixes)
        grouped.setdefault(capture_key, []).append(candidate)
    return grouped


def _extract_capture_key(stem: str, prefixes: list[str]) -> str:
    normalized_stem = stem.strip().lower()
    normalized_prefixes = sorted((prefix.lower() for prefix in prefixes), key=len, reverse=True)
    for prefix in normalized_prefixes:
        if normalized_stem.startswith(prefix):
            return normalized_stem[len(prefix) :].strip("_- ")
    return normalized_stem


def _capture_sort_key(capture_key: str) -> tuple[int, int | str]:
    if capture_key == "":
        return (0, 0)
    if capture_key.isdigit():
        return (1, int(capture_key))
    return (2, capture_key)


def _infer_take_id(
    *,
    video_path: Path,
    user_id: int,
    tag_number: int,
    imu_csv_path: Path | None,
) -> str | None:
    prefixes = [
        f"esp_{user_id}_{tag_number}",
        f"tag_{user_id}_{tag_number}",
        f"tag_{tag_number}",
        f"tag{tag_number}",
    ]
    extras: list[str] = []
    stems = [video_path.stem.lower()]
    if imu_csv_path is not None:
        stems.append(imu_csv_path.stem.lower())

    for stem in stems:
        candidate = stem
        for prefix in prefixes:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        candidate = candidate.strip("_- ")
        if candidate:
            extras.append(candidate)

    if len(extras) == 0:
        return None
    merged = "_".join(dict.fromkeys(extras))
    return re.sub(r"[^a-z0-9]+", "_", merged).strip("_") or None


def _build_clip_id(
    *,
    domain: str,
    user_id: int,
    tag_number: int,
    take_id: str | None,
    clip_id_counts: dict[str, int],
) -> str:
    base = f"robot_emotions_{domain}_u{user_id:02d}_tag{tag_number:02d}"
    if take_id is not None:
        base = f"{base}_{take_id}"

    count = clip_id_counts.get(base, 0) + 1
    clip_id_counts[base] = count
    if count == 1:
        return base
    return f"{base}_dup{count}"
