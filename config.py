from __future__ import annotations

import inspect

import torch


class Config:
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.1,
        batch_size: int = 32,
        lr: float = 3e-4,
        warmup_iters: int = 500,
        decay_iters: int = 20000,
        min_lr: float = 3e-5,
        max_lr: float = 3e-4,
        max_steps: int = 5000,
        device: str | torch.device = "cpu",
        eval_interval: int = 500,
        eval_steps: int = 20,
        grad_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        train_split: float = 0.9,
        checkpoint_interval: int = 500,
        weight_decay: float = 0.0,
        seed: int = 123,
        resume: bool = False,
        use_amp: bool = False,
    ):
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_split = train_split
        self.checkpoint_interval = checkpoint_interval
        self.weight_decay = weight_decay
        self.resume = resume
        self.seed = seed
        self.use_amp = use_amp

    def to_dict(self) -> dict:
        values = dict(self.__dict__)
        values["device"] = str(self.device)
        return values

    @classmethod
    def from_dict(cls, values: dict, *, device: str | torch.device | None = None) -> "Config":
        """Load old checkpoint configs while ignoring fields from future versions."""
        allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
        kwargs = {key: value for key, value in values.items() if key in allowed}
        if device is not None:
            kwargs["device"] = device
        return cls(**kwargs)
