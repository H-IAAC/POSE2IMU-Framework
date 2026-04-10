"""Artifact and catalog writing for RobotEmotions VLM runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import RobotEmotionsVideoClip, resolve_clip_output_dir
from .schemas import VideoDescription


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON file with stable ASCII output."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a JSONL file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_text(path: str | Path, text: str) -> None:
    """Write a plain-text artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(text), encoding="utf-8")


def write_clip_artifacts(
    *,
    output_root: str | Path,
    record: RobotEmotionsVideoClip,
    description_artifact: dict[str, Any],
    raw_response: str,
    prompt_context: dict[str, Any],
    quality_report: dict[str, Any],
) -> dict[str, str]:
    """Write the per-capture artifact bundle required by the specification."""

    clip_dir = resolve_clip_output_dir(output_root, record)
    clip_dir.mkdir(parents=True, exist_ok=True)

    description_path = clip_dir / "description.json"
    raw_response_path = clip_dir / "raw_response.txt"
    prompt_context_path = clip_dir / "prompt_context.json"
    quality_report_path = clip_dir / "quality_report.json"

    write_json(description_path, description_artifact)
    write_text(raw_response_path, raw_response)
    write_json(prompt_context_path, prompt_context)
    write_json(quality_report_path, quality_report)

    return {
        "description_json_path": str(description_path.resolve()),
        "raw_response_txt_path": str(raw_response_path.resolve()),
        "prompt_context_json_path": str(prompt_context_path.resolve()),
        "quality_report_json_path": str(quality_report_path.resolve()),
    }


def build_manifest_entry(
    *,
    record: RobotEmotionsVideoClip,
    video_metadata: dict[str, Any],
    status: str,
    model_id: str,
    generation_config: dict[str, Any],
    description: VideoDescription | None,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    """Build one manifest row for the batch-level JSONL."""

    return {
        "clip_id": record.clip_id,
        "dataset": "RobotEmotions",
        "domain": record.domain,
        "user_id": int(record.user_id),
        "tag_number": int(record.tag_number),
        "take_id": record.take_id,
        "labels": dict(record.labels),
        "source": {
            "video_path": str(record.video_path.resolve()),
        },
        "video": dict(video_metadata),
        "status": status,
        "model_id": model_id,
        "generation_config": dict(generation_config),
        "description": None if description is None else description.to_dict(),
        "artifacts": dict(artifacts),
    }


def build_kimodo_catalog_entry(
    *,
    record: RobotEmotionsVideoClip,
    description: VideoDescription,
    model_id: str,
    seed: int,
    num_samples: int,
) -> dict[str, Any]:
    """Build the compact prompt catalog entry consumed by kimodo."""

    return {
        "prompt_id": record.clip_id,
        "prompt_text": description.prompt_text,
        "labels": dict(record.labels),
        "seed": int(seed),
        "num_samples": int(num_samples),
        "reference_clip_id": record.clip_id,
        "source_metadata": {
            "dataset": "RobotEmotions",
            "model_id": model_id,
            "body_parts": description.body_parts.to_dict(),
        },
    }


def write_root_outputs(
    *,
    output_dir: str | Path,
    manifest_entries: list[dict[str, Any]],
    catalog_entries: list[dict[str, Any]],
    summary: dict[str, Any],
    catalog_output_path: str | Path | None = None,
) -> dict[str, str]:
    """Write the batch-level manifest, summary, and kimodo catalog."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "video_description_manifest.jsonl"
    summary_path = output_root / "video_description_summary.json"
    if catalog_output_path is None:
        catalog_path = output_root / "kimodo_prompt_catalog.jsonl"
    else:
        catalog_path = Path(catalog_output_path)

    write_jsonl(manifest_path, manifest_entries)
    write_jsonl(catalog_path, catalog_entries)
    write_json(summary_path, summary)

    return {
        "manifest_path": str(manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "catalog_path": str(catalog_path.resolve()),
    }
