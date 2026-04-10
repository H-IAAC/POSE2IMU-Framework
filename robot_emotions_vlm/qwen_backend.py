"""Qwen3-VL backend for direct video description inside the kimodo environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QwenGenerationConfig:
    """Runtime settings for one batch execution of the Qwen backend."""

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    local_files_only: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str = "sdpa"
    num_video_frames: int = 32
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "local_files_only": bool(self.local_files_only),
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "attn_implementation": self.attn_implementation,
            "num_video_frames": int(self.num_video_frames),
            "max_new_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }


class QwenVideoBackend:
    """Load Qwen3-VL once and reuse it across all clips."""

    def __init__(self, config: QwenGenerationConfig | None = None) -> None:
        self.config = QwenGenerationConfig() if config is None else config
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load the processor and model lazily."""

        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - exercised through environment validation, not unit tests
            raise RuntimeError(
                "Qwen3-VL runtime dependencies are missing in the kimodo environment. "
                "Install transformers, torch, accelerate, safetensors, huggingface_hub, and av."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            local_files_only=self.config.local_files_only,
        )
        # Transformers 5.1.0 ships this processor with a default fps=2, which
        # conflicts with explicit num_frames sampling.
        if getattr(self._processor, "video_processor", None) is not None:
            setattr(self._processor.video_processor, "fps", None)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            local_files_only=self.config.local_files_only,
            device_map=self.config.device_map,
            torch_dtype=_resolve_torch_dtype(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
        )

    def describe_video(
        self,
        video_path: str | Path,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Run Qwen3-VL on a local video path and return the decoded text response."""

        self.load()
        assert self._model is not None
        assert self._processor is not None

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(Path(video_path).resolve())},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=int(self.config.num_video_frames),
            fps=None,
        )
        inputs.pop("token_type_ids", None)
        inputs = _move_batch_to_model_device(inputs, self._model)

        do_sample = float(self.config.temperature) > 0.0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(self.config.temperature)
            generate_kwargs["top_p"] = float(self.config.top_p)

        generated_ids = self._model.generate(**inputs, **generate_kwargs)
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids, strict=False)
        ]
        decoded = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return "" if len(decoded) == 0 else str(decoded[0]).strip()


def _resolve_torch_dtype(torch_module: Any, raw_value: str) -> Any:
    normalized = str(raw_value).strip().lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {raw_value}")
    return mapping[normalized]


def _move_batch_to_model_device(inputs: Any, model: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(model.device)
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(model.device)
        else:
            moved[key] = value
    return moved
