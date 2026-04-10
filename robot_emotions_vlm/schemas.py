"""Normalization and validation for Qwen JSON responses."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


class DescriptionValidationError(ValueError):
    """Raised when the model response cannot be normalized to the required schema."""


@dataclass(frozen=True)
class DescriptionBodyParts:
    """Detailed movement descriptions for each required body region."""

    arms: str
    trunk: str
    head: str
    legs: str

    def to_dict(self) -> dict[str, str]:
        return {
            "arms": self.arms,
            "trunk": self.trunk,
            "head": self.head,
            "legs": self.legs,
        }


@dataclass(frozen=True)
class VideoDescription:
    """Normalized description returned by the VLM module."""

    prompt_text: str
    dominant_behaviors: tuple[str, ...]
    body_parts: DescriptionBodyParts
    clip_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "dominant_behaviors": list(self.dominant_behaviors),
            "body_parts": self.body_parts.to_dict(),
            "clip_notes": self.clip_notes,
        }


@dataclass(frozen=True)
class ParsedDescription:
    """Parsed response plus normalization diagnostics."""

    description: VideoDescription
    raw_payload: dict[str, Any]
    warnings: tuple[str, ...]


def parse_model_response(raw_response: str) -> ParsedDescription:
    """Parse a raw model response into the strict export contract."""

    payload = _extract_json_payload(raw_response)
    warnings: list[str] = []

    prompt_text, prompt_warnings = _normalize_prompt_text(payload.get("prompt_text"))
    warnings.extend(prompt_warnings)

    dominant_behaviors, dominant_warnings = _normalize_dominant_behaviors(payload.get("dominant_behaviors"))
    warnings.extend(dominant_warnings)

    body_parts_payload = payload.get("body_parts")
    if not isinstance(body_parts_payload, dict):
        raise DescriptionValidationError("body_parts must be a JSON object")

    body_parts = DescriptionBodyParts(
        arms=_normalize_required_text(body_parts_payload.get("arms"), "body_parts.arms"),
        trunk=_normalize_required_text(body_parts_payload.get("trunk"), "body_parts.trunk"),
        head=_normalize_required_text(body_parts_payload.get("head"), "body_parts.head"),
        legs=_normalize_required_text(body_parts_payload.get("legs"), "body_parts.legs"),
    )
    clip_notes = _normalize_optional_text(payload.get("clip_notes"))

    _validate_prompt_text(prompt_text)
    description = VideoDescription(
        prompt_text=prompt_text,
        dominant_behaviors=tuple(dominant_behaviors),
        body_parts=body_parts,
        clip_notes=clip_notes,
    )
    return ParsedDescription(
        description=description,
        raw_payload=payload,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _extract_json_payload(raw_response: str) -> dict[str, Any]:
    if not str(raw_response).strip():
        raise DescriptionValidationError("raw model response is empty")

    for fenced_payload in re.findall(r"```(?:json)?\s*(.*?)```", raw_response, flags=re.IGNORECASE | re.DOTALL):
        try:
            parsed = json.loads(fenced_payload.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    for start_index, character in enumerate(raw_response):
        if character != "{":
            continue
        candidate = _extract_balanced_braces(raw_response, start_index)
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise DescriptionValidationError("could not extract a JSON object from the model response")


def _extract_balanced_braces(text: str, start_index: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    return None


def _normalize_prompt_text(raw_value: Any) -> tuple[str, list[str]]:
    text = _normalize_optional_text(raw_value)
    warnings: list[str] = []
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
        warnings.append("prompt_text_wrapping_quotes_removed")
    if text.endswith("."):
        text = text.rstrip(". ").strip()
        warnings.append("prompt_text_trailing_period_removed")
    if text.lower().startswith("a person") and not text.startswith("A person"):
        text = "A person" + text[len("A person") :]
        warnings.append("prompt_text_prefix_normalized")
    return text, warnings


def _normalize_dominant_behaviors(raw_value: Any) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if raw_value is None:
        return [], warnings
    if isinstance(raw_value, str):
        raw_items = [raw_value]
        warnings.append("dominant_behaviors_coerced_from_string")
    elif isinstance(raw_value, list):
        raw_items = raw_value
    else:
        raise DescriptionValidationError("dominant_behaviors must be a list of strings or a string")

    normalized: list[str] = []
    for item in raw_items:
        text = _normalize_optional_text(item)
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized)), warnings


def _normalize_required_text(raw_value: Any, field_name: str) -> str:
    text = _normalize_optional_text(raw_value)
    if not text:
        raise DescriptionValidationError(f"{field_name} cannot be empty")
    return text


def _normalize_optional_text(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    return re.sub(r"\s+", " ", str(raw_value)).strip()


def _validate_prompt_text(prompt_text: str) -> None:
    errors: list[str] = []
    if not prompt_text:
        errors.append("prompt_text cannot be empty")
    if not prompt_text.startswith("A person"):
        errors.append("prompt_text must start with 'A person'")
    if any(mark in prompt_text for mark in ".!?"):
        errors.append("prompt_text must contain exactly one sentence without internal sentence punctuation")
    if errors:
        raise DescriptionValidationError("; ".join(errors))
