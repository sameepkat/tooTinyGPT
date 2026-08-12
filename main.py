import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import Config
from inference import infer_instruction
from sample import sample_with_prompt
from sft_train import SFTTrainOptions, train_sft
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


def handle_sft(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.resume_mode == "model" and args.init_checkpoint is None:
        parser.error("--init-checkpoint is required with --resume-mode model")
    if args.resume_mode == "full" and args.init_checkpoint is not None:
        parser.error("--init-checkpoint is only valid with --resume-mode model")
    if args.resume_mode == "model" and args.checkpoint.resolve() == args.init_checkpoint.resolve():
        parser.error("--checkpoint must differ from --init-checkpoint with --resume-mode model")
    conf = Config(
        vocab_size=_vocab_size(args, parser),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        use_amp=args.amp,
        seed=args.seed,
    )
    options = SFTTrainOptions(
        epochs=args.epochs,
        warmup_updates=args.warmup_updates,
        max_prompt_tokens=args.max_prompt_tokens,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        resume_mode=args.resume_mode,
    )
    train_sft(
        conf,
        args.train_file,
        args.val_file,
        args.tokenizer,
        args.checkpoint,
        args.init_checkpoint,
        options,
    )


def handle_instruct(args: argparse.Namespace) -> None:
    temperature = 0.0 if args.greedy else args.temperature
    response = infer_instruction(
        args.prompt,
        args.checkpoint,
        args.tokenizer,
        args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_k=args.top_k,
    )
    print(response)


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

    sft_parser = subparsers.add_parser("sft", help="Supervised fine-tune instruction-to-Lum examples")
    sft_parser.add_argument("--train-file", type=Path, required=True)
    sft_parser.add_argument("--val-file", type=Path, required=True)
    sft_parser.add_argument("--init-checkpoint", type=Path, help="Stage-B model checkpoint for --resume-mode model")
    sft_parser.add_argument("--checkpoint", type=Path, required=True, help="SFT output checkpoint; full-resume input when --resume-mode full")
    sft_parser.add_argument("--resume-mode", choices=["model", "full"], default="model")
    sft_parser.add_argument("--tokenizer", type=Path, required=True)
    sft_parser.add_argument("--vocab-size", type=int)
    sft_parser.add_argument("--block-size", type=int, default=512)
    sft_parser.add_argument("--n-embd", type=int, default=512)
    sft_parser.add_argument("--n-head", type=int, default=8)
    sft_parser.add_argument("--n-layer", type=int, default=12)
    sft_parser.add_argument("--dropout", type=float, default=0.1)
    sft_parser.add_argument("--epochs", type=int, default=5)
    sft_parser.add_argument("--batch-size", type=int, default=16)
    sft_parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    sft_parser.add_argument("--lr", type=float, default=5e-5)
    sft_parser.add_argument("--min-lr", type=float, default=5e-6)
    sft_parser.add_argument("--weight-decay", type=float, default=0.01)
    sft_parser.add_argument("--grad-clip", type=float, default=1.0)
    sft_parser.add_argument("--warmup-updates", type=int, default=10)
    sft_parser.add_argument("--max-prompt-tokens", type=int, default=128)
    sft_parser.add_argument("--checkpoint-interval", type=int, default=50)
    sft_parser.add_argument("--log-interval", type=int, default=10)
    sft_parser.add_argument("--seed", type=int, default=123)
    sft_parser.add_argument("--device", default=default_device)
    sft_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                            help="Enable CUDA FP16 AMP (ignored on CPU/MPS)")

    instruct_parser = subparsers.add_parser("instruct", help="Generate Lum code from an English instruction")
    instruct_parser.add_argument("--checkpoint", type=Path, required=True)
    instruct_parser.add_argument("--tokenizer", type=Path, required=True)
    instruct_parser.add_argument("--prompt", required=True)
    instruct_parser.add_argument("--max-new-tokens", type=int, default=256)
    instruct_parser.add_argument("--temperature", type=float, default=0.2)
    instruct_parser.add_argument("--top-k", type=int, default=20)
    instruct_parser.add_argument("--greedy", action="store_true", help="Use deterministic argmax decoding")
    instruct_parser.add_argument("--device", default=default_device)

    args = parser.parse_args()
    if args.command == "train":
        handle_train(args, train_parser)
    elif args.command == "sft":
        handle_sft(args, sft_parser)
    elif args.command == "instruct":
        handle_instruct(args)
    else:
        handle_sample(args)


if __name__ == "__main__":
    main()
