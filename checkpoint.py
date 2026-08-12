from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from config import Config


@dataclass(frozen=True)
class ResumeState:
    step: int
    best_val_loss: float
    extra_state: dict


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    step: int,
    checkpoint_path: Path,
    *,
    scaler: torch.amp.GradScaler | None = None,
    best_val_loss: float = float("inf"),
    extra_state: dict | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    checkpoint_path: Path,
    *,
    scaler: torch.amp.GradScaler | None = None,
) -> ResumeState:
    checkpoint_data = load_checkpoint_file(config.device, checkpoint_path)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint_data:
        scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
    model.to(config.device)
    return ResumeState(
        step=int(checkpoint_data.get("step", 0)),
        best_val_loss=float(checkpoint_data.get("best_val_loss", float("inf"))),
        extra_state=dict(checkpoint_data.get("extra_state", {})),
    )


def load_model_checkpoint(model: torch.nn.Module, config: Config, checkpoint_path: Path) -> dict:
    checkpoint_data = load_checkpoint_file(config.device, checkpoint_path)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(config.device)
    return checkpoint_data


def load_checkpoint_file(device: str | torch.device, checkpoint_path: Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint file not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)
