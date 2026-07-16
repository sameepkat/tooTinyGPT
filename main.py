import argparse
from pathlib import Path

import tiktoken
import torch

from config import Config
from sample import sample_with_prompt
from train import train


def handle_train(args: argparse.Namespace) -> None:
    # Setup
    enc = tiktoken.get_encoding("gpt2")

    vocab_size = enc.n_vocab
    block_size = 256
    batch_size = 8
    gradient_accumulation_steps = 4
    n_embd = 256
    n_head = 4
    n_layer = 6
    dropout = 0.1
    lr = 3e-4
    min_lr = 3e-5
    warmup_iters = 200
    decay_iters = 10000
    max_steps = 10000
    eval_interval = 250
    eval_steps = 20
    checkpoint_interval = 500
    grad_clip = 1.0
    weight_decay = 0.1
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    conf = Config(
        batch_size=batch_size,
        vocab_size=vocab_size,
        block_size=block_size,
        max_steps=max_steps,
        n_embd=n_embd,
        dropout=dropout,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        n_head=n_head,
        n_layer=n_layer,
        lr=lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        decay_iters=decay_iters,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        checkpoint_interval=checkpoint_interval,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
    )

    train_file: Path = args.train_file
    val_file: Path = args.val_file
    resume_mode: str = args.resume_mode
    checkpoint_path: Path = args.checkpoint

    train(conf, train_file, val_file, resume_mode, checkpoint_path)


def handle_sample(args: argparse.Namespace) -> None:
    sample_with_prompt(args.checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Arguments for training the model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--train-file", type=Path, required=True)
    train_parser.add_argument("--val-file", type=Path, required=True)
    train_parser.add_argument(
        "--resume-mode", choices=["none", "full", "model"], default="none"
    )
    train_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    # train_parser.add_argument("--lr", type=float, default=3e-4)
    # train_parser.add_argument("--max-steps", type=int, default=10000)
    train_parser.set_defaults(handler=handle_train)

    sample_parser = subparsers.add_parser("sample", help="Sample from a checkpoint")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.set_defaults(handler=handle_sample)

    args = parser.parse_args()

    # if args.command == "train":
    #     if args.resume_mode != "none" and args.checkpoint is None:
    #         train_parser.error("--checkpoint is required when --resume-mode is 'full' or 'model'")

    #     if args.resume_mode == "none" and args.checkpoint is not None:
    #         train_parser.error("--checkpoint cannot be used when --resume-mode is 'none'")

    args.handler(args)


if __name__ == "__main__":
    main()
