import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import Config
from sample import sample_with_prompt
from train import train


def _vocab_size(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    tokenizer_vocab_size = None
    if args.tokenizer is not None:
        tokenizer_vocab_size = Tokenizer.from_file(str(args.tokenizer)).get_vocab_size()
    if args.vocab_size is None and tokenizer_vocab_size is None:
        parser.error("one of --vocab-size or --tokenizer is required")
    if args.vocab_size is not None and tokenizer_vocab_size is not None and args.vocab_size != tokenizer_vocab_size:
        parser.error(f"--vocab-size ({args.vocab_size}) does not match tokenizer vocab size ({tokenizer_vocab_size})")
    return args.vocab_size if args.vocab_size is not None else tokenizer_vocab_size


def handle_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.resume_mode == "model" and args.init_checkpoint is None:
        parser.error("--init-checkpoint is required with --resume-mode model")
    if args.resume_mode != "model" and args.init_checkpoint is not None:
        parser.error("--init-checkpoint is only valid with --resume-mode model")
    if args.resume_mode == "model" and args.checkpoint.resolve() == args.init_checkpoint.resolve():
        parser.error("--checkpoint must differ from --init-checkpoint with --resume-mode model")
    conf = Config(
        vocab_size=_vocab_size(args, parser), block_size=args.block_size, n_embd=args.n_embd,
        n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, max_steps=args.max_steps,
        lr=args.lr, min_lr=args.min_lr, warmup_iters=args.warmup_iters, decay_iters=args.decay_iters,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, eval_interval=args.eval_interval,
        eval_steps=args.eval_steps, checkpoint_interval=args.checkpoint_interval, device=args.device,
        use_amp=args.amp, seed=args.seed,
    )
    train(conf, args.train_file, args.val_file, resume_mode=args.resume_mode,
          checkpoint_path=args.checkpoint, init_checkpoint_path=args.init_checkpoint)


def handle_sample(args: argparse.Namespace) -> None:
    sample_with_prompt(args.checkpoint, args.tokenizer, args.prompt, device=args.device,
                       max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="tooTinyGPT training and sampling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    train_parser = subparsers.add_parser("train", help="Train a model from random, full, or model-only initialization")
    train_parser.add_argument("--train-file", type=Path, required=True)
    train_parser.add_argument("--val-file", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True, help="Current run output; full-resume input when --resume-mode full")
    train_parser.add_argument("--init-checkpoint", type=Path, help="Model-only input; required for --resume-mode model")
    train_parser.add_argument("--resume-mode", choices=["none", "full", "model"], default="none")
    train_parser.add_argument("--tokenizer", type=Path)
    train_parser.add_argument("--vocab-size", type=int)
    train_parser.add_argument("--block-size", type=int, default=256)
    train_parser.add_argument("--n-embd", type=int, default=256)
    train_parser.add_argument("--n-head", type=int, default=4)
    train_parser.add_argument("--n-layer", type=int, default=6)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--max-steps", type=int, default=10000)
    train_parser.add_argument("--warmup-iters", type=int, default=200)
    train_parser.add_argument("--decay-iters", type=int, default=10000)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--min-lr", type=float, default=3e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--eval-interval", type=int, default=250)
    train_parser.add_argument("--eval-steps", type=int, default=20)
    train_parser.add_argument("--checkpoint-interval", type=int, default=500)
    train_parser.add_argument("--seed", type=int, default=123)
    train_parser.add_argument("--device", default=default_device)
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                              help="Enable CUDA FP16 AMP (ignored on CPU/MPS)")

    sample_parser = subparsers.add_parser("sample", help="Sample with a frozen custom tokenizer")
    sample_parser.add_argument("--checkpoint", type=Path, required=True)
    sample_parser.add_argument("--tokenizer", type=Path, required=True)
    sample_parser.add_argument("--prompt", required=True)
    sample_parser.add_argument("--max-new-tokens", type=int, default=256)
    sample_parser.add_argument("--temperature", type=float, default=0.8)
    sample_parser.add_argument("--top-k", type=int, default=None)
    sample_parser.add_argument("--device", default=default_device)

    args = parser.parse_args()
    if args.command == "train":
        handle_train(args, train_parser)
    else:
        handle_sample(args)


if __name__ == "__main__":
    main()
