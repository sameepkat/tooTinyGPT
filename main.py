import argparse
from pathlib import Path

import tiktoken
import torch

from config import Config
from sample import sample_with_prompt
from train import train


def handle_train(args: argparse.Namespace) -> None:
    enc = tiktoken.get_encoding(args.enc)
    conf = Config(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size if args.vocab_size is not None else enc.n_vocab,
        block_size=args.block_size,
        max_steps=args.max_steps,
        n_embd=args.n_embd,
        dropout=args.dropout,
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_head=args.n_head,
        n_layer=args.n_layer,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        decay_iters=args.decay_iters,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        checkpoint_interval=args.checkpoint_interval,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        max_lr=args.max_lr,
        seed=args.seed,
        train_split=args.train_split,
    )

    train_file: Path = args.train_file
    val_file: Path = args.val_file
    resume_mode: str = args.resume_mode
    checkpoint_path: Path = args.checkpoint

    train(conf, train_file, val_file, resume_mode, checkpoint_path)


def handle_sample(args: argparse.Namespace) -> None:
    sample_with_prompt(args.checkpoint, args.enc)


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

    train_parser.add_argument("--enc", type=str, default="gpt2")
    train_parser.add_argument("--vocab-size", type=int, default=None)
    train_parser.add_argument("--block-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_parser.add_argument("--n-embd", type=int, default=256)
    train_parser.add_argument("--n-head", type=int, default=4)
    train_parser.add_argument("--n-layer", type=int, default=6)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--min-lr", type=float, default=3e-5)
    train_parser.add_argument("--warmup-iters", type=int, default=200)
    train_parser.add_argument("--decay-iters", type=int, default=10000)
    train_parser.add_argument("--max-steps", type=int, default=10000)
    train_parser.add_argument("--eval-interval", type=int, default=250)
    train_parser.add_argument("--eval-steps", type=int, default=20)
    train_parser.add_argument("--checkpoint-interval", type=int, default=500)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--max-lr", type=float, default=3e-4)
    train_parser.add_argument("--train-split", type=float, default=0.9)
    train_parser.add_argument("--seed", type=int, default=123)
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_parser.set_defaults(handler=handle_train)

    sample_parser = subparsers.add_parser("sample", help="Sample from a checkpoint")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--enc", type=str, default="gpt2")
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
