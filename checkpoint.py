import os
import torch
from config import Config
from pathlib import Path

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Config, step: int, checkpoint_path: Path):
    to_save = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "step": step
    }
    #checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    torch.save(to_save, checkpoint_path)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Config, checkpoint_path: Path):
    # checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise ValueError("checkpoint file not found")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    # config = checkpoint['config']
    model_state = checkpoint['model_state_dict']
    optimizer_state = checkpoint['optimizer_state_dict']
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    model.to(config.device)
    return checkpoint["step"] + 1

def load_model_checkpoint(model, config: Config, checkpoint_path: Path):
    # checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise ValueError("checkpoint file not found")

    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    model_state = checkpoint['model_state_dict']
    model.load_state_dict(model_state)
    model.to(config.device)
    return checkpoint

def load_checkpoint_file(device: str | torch.device, checkpoint_path: Path):
    #checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise ValueError("checkpoint file not found")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return checkpoint
