# Save and load training state

"""
saves:
model weights, optimizer state, current training step, config, maybe latest train/ val loss
"""

import os
import torch
from config import Config

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Config, step: int):
    to_save = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "step": step
    }
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    torch.save(to_save, checkpoint_path)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Config):
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise ValueError("checkpoint file not found")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    # config = checkpoint['config']
    model_state = checkpoint['model_state_dict']
    optimizer_state = checkpoint['optimizer_state_dict']
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    model.to(config.device)
    return checkpoint["step"] + 1
