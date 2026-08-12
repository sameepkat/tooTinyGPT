from pathlib import Path

import numpy as np
import torch

from config import Config


class Data:
    """Random-window batches over pre-tokenized little-endian uint16 files."""

    def __init__(self, train_file: Path, val_file: Path, config: Config):
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.config = config
        self.train_data = np.memmap(self.train_file, dtype="<u2", mode="r")
        self.val_data = np.memmap(self.val_file, dtype="<u2", mode="r")

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.config.block_size
        if max_start <= 0:
            raise ValueError(f"{split} data is too short for block_size={self.config.block_size}")

        starts = torch.randint(max_start, (self.config.batch_size,)).tolist()
        # np.stack creates a small writable batch copy, avoiding PyTorch's non-writable
        # memmap warning without materializing the corpus in RAM.
        x_np = np.stack([data[i : i + self.config.block_size] for i in starts])
        y_np = np.stack([data[i + 1 : i + self.config.block_size + 1] for i in starts])
        x = torch.from_numpy(x_np).to(self.config.device, dtype=torch.long)
        y = torch.from_numpy(y_np).to(self.config.device, dtype=torch.long)
        return x, y
