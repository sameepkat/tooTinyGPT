from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import checkpoint
from config import Config
from model import GPT
from sft_data import SFTDataset, collate_sft_batch, print_sft_stats


EARLY_RESPONSE_TOKENS = 32
MID_RESPONSE_TOKENS = 64
EARLY_RESPONSE_WEIGHT = 4.0
MID_RESPONSE_WEIGHT = 2.0
LATE_RESPONSE_WEIGHT = 1.0


@dataclass(frozen=True)
class SFTTrainOptions:
    epochs: int
    warmup_updates: int
    max_prompt_tokens: int
    checkpoint_interval: int
    log_interval: int
    resume_mode: str = "model"


def _autocast_context(amp_enabled: bool):
    if amp_enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _lr_scheduler(update_index: int, total_updates: int, conf: Config, warmup_updates: int) -> float:
    if warmup_updates > 0 and update_index < warmup_updates:
        return conf.lr * (update_index + 1) / warmup_updates
    if update_index >= total_updates:
        return conf.min_lr
    if total_updates <= warmup_updates:
        return conf.min_lr
    progress = (update_index - warmup_updates) / (total_updates - warmup_updates)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return conf.min_lr + cosine_decay * (conf.lr - conf.min_lr)


def _ordered_indices(dataset_size: int, epoch: int, seed: int, shuffle: bool) -> list[int]:
    indices = list(range(dataset_size))
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
    return indices


def _batch_indices(dataset_size: int, batch_size: int, epoch: int, seed: int, shuffle: bool) -> list[list[int]]:
    indices = _ordered_indices(dataset_size, epoch, seed, shuffle)
    return [indices[start : start + batch_size] for start in range(0, dataset_size, batch_size)]


def _weighted_response_loss(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, vocab_size = logits.shape
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(y)

    supervised_mask = y != -100
    response_position = supervised_mask.long().cumsum(dim=1)
    weights = torch.zeros_like(token_loss)
    weights = torch.where(
        supervised_mask & (response_position <= EARLY_RESPONSE_TOKENS),
        torch.full_like(weights, EARLY_RESPONSE_WEIGHT),
        weights,
    )
    weights = torch.where(
        supervised_mask
        & (response_position > EARLY_RESPONSE_TOKENS)
        & (response_position <= MID_RESPONSE_TOKENS),
        torch.full_like(weights, MID_RESPONSE_WEIGHT),
        weights,
    )
    weights = torch.where(
        supervised_mask & (response_position > MID_RESPONSE_TOKENS),
        torch.full_like(weights, LATE_RESPONSE_WEIGHT),
        weights,
    )

    weight_sum = weights.sum(dim=1)
    valid_examples = weight_sum > 0
    if not valid_examples.any():
        raise ValueError("SFT batch produced no supervised response tokens")
    per_example_loss = (token_loss * weights).sum(dim=1) / weight_sum.clamp_min(1.0)
    loss = per_example_loss[valid_examples].mean()
    supervised_tokens = int(supervised_mask.sum().item())
    return loss, int(valid_examples.sum().item()), supervised_tokens


def _weighted_sft_loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    logits, _ = model(x)
    return _weighted_response_loss(logits, y)


def _evaluate_sft(model: GPT, dataset: SFTDataset, conf: Config, amp_enabled: bool) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    batches = _batch_indices(len(dataset), conf.batch_size, epoch=0, seed=conf.seed, shuffle=False)
    with torch.no_grad():
        for batch in batches:
            examples = [dataset[index] for index in batch]
            x, y = collate_sft_batch(examples, dataset.special_tokens.pad_id, conf.device)
            with _autocast_context(amp_enabled):
                loss, valid_examples, _ = _weighted_sft_loss(model, x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite SFT validation loss: {loss.item()}")
            total_loss += loss.item() * valid_examples
            total_examples += valid_examples
    if was_training:
        model.train()
    if total_examples == 0:
        raise ValueError("SFT validation set produced no supervised response tokens")
    return total_loss / total_examples


def train_sft(
    conf: Config,
    train_file: Path,
    val_file: Path,
    tokenizer_path: Path,
    checkpoint_path: Path,
    init_checkpoint_path: Path | None,
    options: SFTTrainOptions,
) -> None:
    if options.resume_mode not in {"model", "full"}:
        raise ValueError("SFT resume_mode must be one of: model, full")
    checkpoint_path = Path(checkpoint_path)
    if options.resume_mode == "model":
        if init_checkpoint_path is None:
            raise ValueError("--init-checkpoint is required for SFT model initialization")
        if checkpoint_path.resolve() == Path(init_checkpoint_path).resolve():
            raise ValueError("--checkpoint must differ from --init-checkpoint for SFT")
    elif init_checkpoint_path is not None:
        raise ValueError("--init-checkpoint is only valid with --resume-mode model")

    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
    amp_enabled = conf.use_amp and conf.device.type == "cuda"
    if conf.use_amp and not amp_enabled:
        print(f"AMP requested but disabled on device {conf.device}")
    scaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda") if amp_enabled else None
    print(f"Using device: {conf.device}; FP16 AMP: {amp_enabled}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != conf.vocab_size:
        raise ValueError(f"config vocab_size {conf.vocab_size} does not match tokenizer vocab size {tokenizer_vocab_size}")

    train_dataset = SFTDataset(train_file, tokenizer, conf.block_size, options.max_prompt_tokens)
    val_dataset = SFTDataset(val_file, tokenizer, conf.block_size, options.max_prompt_tokens)
    print_sft_stats("train", train_dataset.stats)
    print_sft_stats("validation", val_dataset.stats)

    train_batches_per_epoch = math.ceil(len(train_dataset) / conf.batch_size)
    updates_per_epoch = math.ceil(train_batches_per_epoch / conf.gradient_accumulation_steps)
    total_updates = options.epochs * updates_per_epoch
    conf.max_steps = total_updates
    conf.warmup_iters = options.warmup_updates
    conf.decay_iters = total_updates

    model = GPT(conf).to(conf.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    start_epoch = 0
    start_batch_index = 0
    completed_updates = 0
    best_eval_loss = float("inf")

    if options.resume_mode == "full":
        state = checkpoint.load_checkpoint(model, optimizer, conf, checkpoint_path, scaler=scaler)
        completed_updates = state.step
        best_eval_loss = state.best_val_loss
        start_epoch = int(state.extra_state.get("epoch", 0))
        start_batch_index = int(state.extra_state.get("next_batch_index", 0))
        print(f"Resumed SFT from epoch {start_epoch + 1}, batch {start_batch_index}, update {completed_updates}")
    else:
        checkpoint.load_model_checkpoint(model, conf, Path(init_checkpoint_path))
        print(f"Initialized SFT model weights from {init_checkpoint_path}")

    total_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")
    started_at = time.perf_counter()

    for epoch in range(start_epoch, options.epochs):
        batches = _batch_indices(len(train_dataset), conf.batch_size, epoch, conf.seed, shuffle=True)
        batch_cursor = start_batch_index if epoch == start_epoch else 0
        while batch_cursor < len(batches):
            group = batches[batch_cursor : batch_cursor + conf.gradient_accumulation_steps]
            if not group:
                break
            lr = _lr_scheduler(completed_updates, total_updates, conf, options.warmup_updates)
            for optimizer_group in optimizer.param_groups:
                optimizer_group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            batch_tensors: list[tuple[torch.Tensor, torch.Tensor, int]] = []
            supervised_tokens = 0
            total_examples = 0

            for batch in group:
                examples = [train_dataset[index] for index in batch]
                x, y = collate_sft_batch(examples, train_dataset.special_tokens.pad_id, conf.device)
                supervised_tokens += int((y != -100).sum().item())
                valid_examples = int((y != -100).any(dim=1).sum().item())
                total_examples += valid_examples
                batch_tensors.append((x, y, valid_examples))

            if supervised_tokens == 0:
                raise ValueError("SFT training batch produced no supervised response tokens")
            if total_examples == 0:
                raise ValueError("SFT training batch produced no valid supervised examples")

            weighted_loss_sum = 0.0
            for x, y, valid_examples in batch_tensors:
                with _autocast_context(amp_enabled):
                    loss, valid_examples, _ = _weighted_sft_loss(model, x, y)
                    backward_loss = loss * (valid_examples / total_examples)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite SFT training loss at update {completed_updates + 1}: {loss.item()}")
                weighted_loss_sum += loss.item() * valid_examples
                if scaler is not None:
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()

            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            completed_updates += 1
            batch_cursor += len(group)
            train_loss = weighted_loss_sum / total_examples

            if completed_updates % options.log_interval == 0 or completed_updates == 1:
                elapsed = time.perf_counter() - started_at
                print(
                    f"epoch {epoch + 1}/{options.epochs} | update {completed_updates}/{total_updates} | "
                    f"weighted response loss {train_loss:.4f} | lr {lr:.6g} | supervised tokens {supervised_tokens:,} | "
                    f"elapsed {elapsed:.1f}s | grad norm {float(grad_norm):.3f}"
                )

            if completed_updates % options.checkpoint_interval == 0:
                checkpoint.save_checkpoint(
                    model,
                    optimizer,
                    conf,
                    completed_updates,
                    checkpoint_path,
                    scaler=scaler,
                    best_val_loss=best_eval_loss,
                    extra_state={"epoch": epoch, "next_batch_index": batch_cursor, "epochs": options.epochs},
                )
                print(f"Saved SFT checkpoint: {checkpoint_path}")

        start_batch_index = 0
        eval_loss = _evaluate_sft(model, val_dataset, conf, amp_enabled)
        print(f"epoch {epoch + 1}/{options.epochs} validation weighted response loss {eval_loss:.4f} | best {best_eval_loss:.4f}")
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            checkpoint.save_checkpoint(
                model,
                optimizer,
                conf,
                completed_updates,
                best_checkpoint_path,
                scaler=scaler,
                best_val_loss=best_eval_loss,
                extra_state={"epoch": epoch + 1, "next_batch_index": 0, "epochs": options.epochs},
            )
            print(f"Saved best SFT checkpoint: {best_checkpoint_path}")
        checkpoint.save_checkpoint(
            model,
            optimizer,
            conf,
            completed_updates,
            checkpoint_path,
            scaler=scaler,
            best_val_loss=best_eval_loss,
            extra_state={"epoch": epoch + 1, "next_batch_index": 0, "epochs": options.epochs},
        )
        print(f"Saved SFT checkpoint: {checkpoint_path}")

    print(f"Finished SFT at update {completed_updates}; best validation weighted response loss {best_eval_loss:.4f}")
