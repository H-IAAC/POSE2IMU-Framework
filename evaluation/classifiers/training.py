from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .metrics import compute_multitask_metrics
from .model import MultitaskFusionClassifier, ensure_torch_available

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as exc:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    WeightedRandomSampler = object  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


HEAD_TO_LOGIT_KEY = {
    "emotion": "emotion_logits",
    "modality": "modality_logits",
    "stimulus": "stimulus_logits",
    "flat_tag": "flat_tag_logits",
}


def ensure_training_dependencies() -> None:
    ensure_torch_available()
    if torch is None:
        raise ImportError(
            "evaluation.classifiers.training requires PyTorch."
        ) from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    trunk_blocks: int = 2
    graph_layout: str = "imugpt22"
    use_flat_tag_head: bool = True
    use_domain_head: bool = True
    quality_dim: int = 8
    modality_dropout_p: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    use_cb_focal: bool = False
    focal_gamma: float = 2.0
    device: str = "auto"
    domain_loss_weight: float = 0.1
    flat_tag_loss_weight: float = 0.1
    emotion_loss_weight: float = 1.0
    modality_loss_weight: float = 1.0
    stimulus_loss_weight: float = 1.0
    sampler_power: float = 1.0
    num_workers: int = 0


class WindowTensorDataset(Dataset):
    def __init__(self, arrays: Mapping[str, Any]) -> None:
        self.pose = np.asarray(arrays["pose"], dtype=np.float32)
        self.imu = np.asarray(arrays["imu"], dtype=np.float32)
        self.quality = np.asarray(arrays["quality"], dtype=np.float32)
        self.domain = np.asarray(arrays["domain"], dtype=np.int64)
        self.classification_mask = np.asarray(arrays["classification_mask"], dtype=np.float32)
        self.targets = {
            head_name: np.asarray(values, dtype=np.int64)
            for head_name, values in dict(arrays["targets"]).items()
        }

    def __len__(self) -> int:
        return int(self.pose.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ensure_training_dependencies()
        item = {
            "pose": torch.from_numpy(self.pose[index]),
            "imu": torch.from_numpy(self.imu[index]),
            "quality": torch.from_numpy(self.quality[index]),
            "domain": torch.tensor(int(self.domain[index]), dtype=torch.long),
            "classification_mask": torch.tensor(float(self.classification_mask[index]), dtype=torch.float32),
        }
        for head_name, values in self.targets.items():
            item[f"{head_name}_target"] = torch.tensor(int(values[index]), dtype=torch.long)
        return item


def _resolve_device(device: str) -> torch.device:
    ensure_training_dependencies()
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _normalize_pose_batch(pose_batch: torch.Tensor) -> torch.Tensor:
    mean = pose_batch.mean(dim=(1, 2), keepdim=True)
    std = pose_batch.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (pose_batch - mean) / std


def _normalize_imu_batch(imu_batch: torch.Tensor) -> torch.Tensor:
    mean = imu_batch.mean(dim=1, keepdim=True)
    std = imu_batch.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (imu_batch - mean) / std


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    ensure_training_dependencies()
    present_classes = np.unique(np.asarray(labels, dtype=np.int64))
    if present_classes.size == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    weights_present = compute_class_weight(class_weight="balanced", classes=present_classes, y=labels)
    weights = np.ones(int(num_classes), dtype=np.float32)
    weights[present_classes] = weights_present.astype(np.float32, copy=False)
    return torch.tensor(weights, dtype=torch.float32)


def _balanced_sample_weights(flat_labels: np.ndarray, classification_mask: np.ndarray, *, power: float = 1.0) -> np.ndarray:
    labels = np.asarray(flat_labels, dtype=np.int64)
    supervised_mask = np.asarray(classification_mask, dtype=np.float32) > 0.0
    counts = pd.Series(labels[supervised_mask]).value_counts().to_dict() if int(np.count_nonzero(supervised_mask)) > 0 else {}
    weights = np.ones(labels.shape[0], dtype=np.float32)
    for sample_index, label in enumerate(labels.tolist()):
        if not supervised_mask[sample_index]:
            continue
        label_count = max(1, int(counts.get(int(label), 1)))
        weights[sample_index] = float((1.0 / label_count) ** float(power))
    return weights


def _weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    num_classes = logits.shape[1]
    log_probs = torch.log_softmax(logits, dim=1)
    with torch.no_grad():
        target_distribution = torch.full_like(log_probs, fill_value=float(label_smoothing) / max(1, num_classes - 1))
        target_distribution.scatter_(1, targets[:, None], 1.0 - float(label_smoothing))
        if num_classes == 1:
            target_distribution.fill_(1.0)
    losses = -(target_distribution * log_probs).sum(dim=1)
    if class_weights is not None:
        losses = losses * class_weights.to(logits.device)[targets]
    return losses.mean()


def _class_balanced_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    ce_loss = F.cross_entropy(logits, targets, weight=None, reduction="none")
    pt = torch.exp(-ce_loss)
    focal = torch.pow(1.0 - pt, float(gamma)) * ce_loss
    if class_weights is not None:
        focal = focal * class_weights.to(logits.device)[targets]
    return focal.mean()


def _head_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    use_cb_focal: bool,
    focal_gamma: float,
) -> torch.Tensor:
    if use_cb_focal:
        return _class_balanced_focal_loss(
            logits,
            targets,
            class_weights=class_weights,
            gamma=focal_gamma,
        )
    return _weighted_cross_entropy(
        logits,
        targets,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )


def infer_model_shapes(arrays: Mapping[str, Any]) -> dict[str, int]:
    pose = np.asarray(arrays["pose"], dtype=np.float32)
    imu = np.asarray(arrays["imu"], dtype=np.float32)
    return {
        "pose_in_channels": int(pose.shape[-1]),
        "imu_in_channels": int(imu.shape[2] * imu.shape[3]),
        "quality_dim": int(np.asarray(arrays["quality"], dtype=np.float32).shape[1]),
    }


def build_model(
    *,
    arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    model_config: ModelConfig,
    use_pose_branch: bool,
    use_imu_branch: bool,
    use_domain_head: bool,
) -> MultitaskFusionClassifier:
    ensure_training_dependencies()
    shapes = infer_model_shapes(arrays)
    return MultitaskFusionClassifier(
        pose_in_channels=shapes["pose_in_channels"],
        imu_in_channels=shapes["imu_in_channels"],
        num_emotions=len(label_encoders["emotion"]["class_names"]),
        num_modalities=len(label_encoders["modality"]["class_names"]),
        num_stimuli=len(label_encoders["stimulus"]["class_names"]),
        num_flat_tags=(len(label_encoders["flat_tag"]["class_names"]) if model_config.use_flat_tag_head else None),
        use_pose_branch=bool(use_pose_branch),
        use_imu_branch=bool(use_imu_branch),
        use_domain_head=bool(use_domain_head and model_config.use_domain_head),
        graph_layout=str(model_config.graph_layout),
        hidden_dim=int(model_config.hidden_dim),
        trunk_blocks=int(model_config.trunk_blocks),
        dropout=float(model_config.dropout),
        quality_dim=int(shapes["quality_dim"] if model_config.quality_dim > 0 else 0),
        modality_dropout_p=float(model_config.modality_dropout_p),
    )


def _build_sampler(arrays: Mapping[str, Any], training_config: TrainingConfig) -> WeightedRandomSampler:
    ensure_training_dependencies()
    weights = _balanced_sample_weights(
        np.asarray(arrays["targets"]["flat_tag"], dtype=np.int64),
        np.asarray(arrays["classification_mask"], dtype=np.float32),
        power=float(training_config.sampler_power),
    )
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def _prepare_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    prepared = {key: value.to(device) for key, value in batch.items()}
    prepared["pose"] = _normalize_pose_batch(prepared["pose"])
    prepared["imu"] = _normalize_imu_batch(prepared["imu"])
    return prepared


def _gather_predictions(
    model: MultitaskFusionClassifier,
    arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    ensure_training_dependencies()
    dataset = WindowTensorDataset(arrays)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    predictions: dict[str, list[np.ndarray]] = {head_name: [] for head_name in label_encoders.keys()}
    probabilities: dict[str, list[np.ndarray]] = {head_name: [] for head_name in label_encoders.keys()}
    targets: dict[str, np.ndarray] = {
        head_name: np.asarray(arrays["targets"][head_name], dtype=np.int64)
        for head_name in label_encoders.keys()
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            prepared = _prepare_batch(batch, device)
            outputs = model(
                pose_inputs=prepared["pose"],
                imu_inputs=prepared["imu"],
                quality_inputs=prepared["quality"],
                domain_lambda=1.0,
            )
            for head_name in label_encoders.keys():
                logit_key = HEAD_TO_LOGIT_KEY[head_name]
                if logit_key not in outputs:
                    continue
                logits = outputs[logit_key]
                probs = torch.softmax(logits, dim=1)
                predictions[head_name].append(torch.argmax(probs, dim=1).cpu().numpy())
                probabilities[head_name].append(probs.cpu().numpy())

    merged_predictions = {
        head_name: np.concatenate(head_blocks, axis=0).astype(np.int64)
        for head_name, head_blocks in predictions.items()
        if len(head_blocks) > 0
    }
    merged_probabilities = {
        head_name: np.concatenate(head_blocks, axis=0).astype(np.float32)
        for head_name, head_blocks in probabilities.items()
        if len(head_blocks) > 0
    }
    return {
        "targets": targets,
        "predictions": merged_predictions,
        "probabilities": merged_probabilities,
        "metrics": compute_multitask_metrics(
            y_true=targets,
            y_pred=merged_predictions,
            probabilities=merged_probabilities,
            label_encoders=label_encoders,
        ),
    }


def evaluate_multitask_model(
    model: MultitaskFusionClassifier,
    arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    ensure_training_dependencies()
    return _gather_predictions(
        model,
        arrays,
        label_encoders,
        batch_size=batch_size,
        device=device,
    )


def train_multitask_model(
    *,
    train_arrays: Mapping[str, Any],
    val_arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    use_pose_branch: bool,
    use_imu_branch: bool,
    use_domain_head: bool,
) -> dict[str, Any]:
    ensure_training_dependencies()
    device = _resolve_device(training_config.device)
    model = build_model(
        arrays=train_arrays,
        label_encoders=label_encoders,
        model_config=model_config,
        use_pose_branch=use_pose_branch,
        use_imu_branch=use_imu_branch,
        use_domain_head=use_domain_head,
    ).to(device)

    train_dataset = WindowTensorDataset(train_arrays)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config.batch_size),
        sampler=_build_sampler(train_arrays, training_config),
        num_workers=int(training_config.num_workers),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.learning_rate),
        weight_decay=float(training_config.weight_decay),
    )

    supervised_mask = np.asarray(train_arrays["classification_mask"], dtype=np.float32) > 0.0
    class_weights = {
        head_name: _compute_class_weights(
            np.asarray(train_arrays["targets"][head_name], dtype=np.int64)[supervised_mask],
            len(label_encoders[head_name]["class_names"]),
        ).to(device)
        for head_name in ("emotion", "modality", "stimulus", "flat_tag")
        if head_name in train_arrays["targets"]
    }
    domain_class_weights = _compute_class_weights(
        np.asarray(train_arrays["domain"], dtype=np.int64),
        2,
    ).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -np.inf
    history: list[dict[str, Any]] = []

    for epoch_index in range(int(training_config.max_epochs)):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            prepared = _prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pose_inputs=prepared["pose"],
                imu_inputs=prepared["imu"],
                quality_inputs=prepared["quality"],
                domain_lambda=float(training_config.domain_loss_weight),
            )

            classification_mask = prepared["classification_mask"] > 0.0
            loss = torch.zeros((), device=device)
            if bool(torch.any(classification_mask)):
                for head_name, head_weight in (
                    ("emotion", training_config.emotion_loss_weight),
                    ("modality", training_config.modality_loss_weight),
                    ("stimulus", training_config.stimulus_loss_weight),
                ):
                    masked_logits = outputs[HEAD_TO_LOGIT_KEY[head_name]][classification_mask]
                    masked_targets = prepared[f"{head_name}_target"][classification_mask]
                    loss = loss + float(head_weight) * _head_loss(
                        masked_logits,
                        masked_targets,
                        class_weights=class_weights[head_name],
                        label_smoothing=float(training_config.label_smoothing),
                        use_cb_focal=bool(training_config.use_cb_focal),
                        focal_gamma=float(training_config.focal_gamma),
                    )

                if "flat_tag_logits" in outputs and "flat_tag" in class_weights:
                    masked_logits = outputs["flat_tag_logits"][classification_mask]
                    masked_targets = prepared["flat_tag_target"][classification_mask]
                    loss = loss + float(training_config.flat_tag_loss_weight) * _head_loss(
                        masked_logits,
                        masked_targets,
                        class_weights=class_weights["flat_tag"],
                        label_smoothing=float(training_config.label_smoothing),
                        use_cb_focal=bool(training_config.use_cb_focal),
                        focal_gamma=float(training_config.focal_gamma),
                    )

            if bool(use_domain_head) and "domain_logits" in outputs:
                loss = loss + float(training_config.domain_loss_weight) * _head_loss(
                    outputs["domain_logits"],
                    prepared["domain"],
                    class_weights=domain_class_weights,
                    label_smoothing=0.0,
                    use_cb_focal=False,
                    focal_gamma=float(training_config.focal_gamma),
                )

            loss.backward()
            if float(training_config.grad_clip_norm) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(training_config.grad_clip_norm))
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        val_report = _gather_predictions(
            model,
            val_arrays,
            label_encoders,
            batch_size=int(training_config.batch_size),
            device=device,
        )
        val_metric = val_report["metrics"].get("global_score_macro_f1_mean")
        val_metric_value = -np.inf if val_metric is None else float(val_metric)
        if val_metric_value > best_metric:
            best_metric = val_metric_value
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": int(epoch_index + 1),
                "train_loss": float(running_loss / max(1, num_batches)),
                "val_global_score_macro_f1_mean": None if val_metric is None else float(val_metric),
            }
        )

    model.load_state_dict(best_state)
    final_train_report = _gather_predictions(
        model,
        train_arrays,
        label_encoders,
        batch_size=int(training_config.batch_size),
        device=device,
    )
    final_val_report = _gather_predictions(
        model,
        val_arrays,
        label_encoders,
        batch_size=int(training_config.batch_size),
        device=device,
    )
    return {
        "model": model,
        "history": history,
        "train_report": final_train_report,
        "val_report": final_val_report,
        "best_val_global_score_macro_f1_mean": None if best_metric == -np.inf else float(best_metric),
        "device": str(device),
    }
