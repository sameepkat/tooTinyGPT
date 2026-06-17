import torch


class Config:
    def __init__(
        self,
        vocab_size,
        block_size=128,  # T - How many tokens per chunk
        n_embd=128,  # C - Vector size after embedding
        n_head=4,
        n_layer=4,
        dropout=0.1,
        batch_size=32,  # B - How many chunks
        lr=3e-4,
        max_steps=5000,
        device="cpu",
        eval_interval=200,
        eval_steps=20,
        grad_clip=1.0,
        train_split=0.9,
        checkpoint_interval=500,
        weight_decay=0,
        seed=123,
        resume=False
    ):
        if n_embd % n_head != 0:
            raise ValueError("incorrect value for n_embd or n_head")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.grad_clip = grad_clip
        self.train_split = train_split
        self.checkpoint_interval = checkpoint_interval
        self.weight_decay = weight_decay
        self.resume = resume
        self.seed = seed
