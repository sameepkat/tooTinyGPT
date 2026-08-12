from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path

import numpy as np
import torch

import checkpoint
from config import Config
from data import Data
from model import GPT


def lr_scheduler(it: int, conf: Config) -> float:
    if conf.warmup_iters > 0 and it < conf.warmup_iters:
        return conf.lr * (it + 1) / conf.warmup_iters
    if it >= conf.decay_iters:
        return conf.min_lr
    if conf.decay_iters <= conf.warmup_iters:
        return conf.min_lr
    progress = (it - conf.warmup_iters) / (conf.decay_iters - conf.warmup_iters)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return conf.min_lr + cosine_decay * (conf.lr - conf.min_lr)


def _autocast_context(amp_enabled: bool):
    if amp_enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _evaluate(model: GPT, data_obj: Data, conf: Config, amp_enabled: bool) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(conf.eval_steps):
            x, y = data_obj.get_batch("val")
            with _autocast_context(amp_enabled):
                _, loss = model(x, y)
            assert loss is not None
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite validation loss: {loss.item()}")
            losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def train(
    conf: Config,
    train_file: Path,
    val_file: Path,
    *,
    resume_mode: str = "none",
    checkpoint_path: Path,
    init_checkpoint_path: Path | None = None,
) -> None:
    if resume_mode not in {"none", "full", "model"}:
        raise ValueError("resume_mode must be one of: none, full, model")
    checkpoint_path = Path(checkpoint_path)
    if resume_mode == "model":
        if init_checkpoint_path is None:
            raise ValueError("--init-checkpoint is required with --resume-mode model")
        if checkpoint_path.resolve() == Path(init_checkpoint_path).resolve():
            raise ValueError("--checkpoint must differ from --init-checkpoint for model initialization")
    elif init_checkpoint_path is not None:
        raise ValueError("--init-checkpoint is only valid with --resume-mode model")

    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
    amp_enabled = conf.use_amp and conf.device.type == "cuda"
    if conf.use_amp and not amp_enabled:
        print(f"AMP requested but disabled on device {conf.device}")
    scaler: torch.amp.GradScaler | None = None
    if amp_enabled:
        scaler = torch.amp.GradScaler("cuda")
    print(f"Using device: {conf.device}; FP16 AMP: {amp_enabled}")

    data_obj = Data(train_file, val_file, conf)
    model = GPT(conf).to(conf.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    start_step = 0
    best_eval_loss = float("inf")
    if resume_mode == "full":
        state = checkpoint.load_checkpoint(model, optimizer, conf, checkpoint_path, scaler=scaler)
        start_step, best_eval_loss = state.step, state.best_val_loss
        print(f"Resumed full training from update {start_step}")
    elif resume_mode == "model":
        checkpoint.load_model_checkpoint(model, conf, Path(init_checkpoint_path))
        print(f"Initialized model weights from {init_checkpoint_path}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")
    started_at = time.perf_counter()

    for update_index in range(start_step, conf.max_steps):
        lr = lr_scheduler(update_index, conf)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(conf.gradient_accumulation_steps):
            x, y = data_obj.get_batch("train")
            with _autocast_context(amp_enabled):
                _, loss = model(x, y)
                assert loss is not None
                backward_loss = loss / conf.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at update {update_index + 1}: {loss.item()}"
                )
            micro_losses.append(loss.item())
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

        completed_step = update_index + 1
        train_loss = float(np.mean(micro_losses))
        tokens_processed = completed_step * conf.batch_size * conf.block_size * conf.gradient_accumulation_steps

        if completed_step % conf.checkpoint_interval == 0:
            checkpoint.save_checkpoint(model, optimizer, conf, completed_step, checkpoint_path, scaler=scaler, best_val_loss=best_eval_loss)
            print(f"Saved checkpoint: {checkpoint_path}")

        if completed_step % conf.eval_interval == 0:
            elapsed = time.perf_counter() - started_at
            tokens_per_second = tokens_processed / elapsed if elapsed else 0.0
            print(
                f"update {completed_step}/{conf.max_steps} | train loss {train_loss:.4f} | "
                f"lr {lr:.6g} | elapsed {elapsed:.1f}s | tokens {tokens_processed:,} | "
                f"tokens/s {tokens_per_second:,.0f} | grad norm {float(grad_norm):.3f}"
            )
            eval_loss = _evaluate(model, data_obj, conf, amp_enabled)
            print(f"validation loss {eval_loss:.4f} | best {best_eval_loss:.4f}")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                checkpoint.save_checkpoint(model, optimizer, conf, completed_step, best_checkpoint_path, scaler=scaler, best_val_loss=best_eval_loss)
                print(f"Saved best checkpoint: {best_checkpoint_path}")
            if conf.device.type == "cuda":
                print(
                    f"CUDA peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | "
                    f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
                )

    final_eval_loss = _evaluate(model, data_obj, conf, amp_enabled)
    final_step = conf.max_steps
    if final_eval_loss < best_eval_loss:
        best_eval_loss = final_eval_loss
        checkpoint.save_checkpoint(model, optimizer, conf, final_step, best_checkpoint_path, scaler=scaler, best_val_loss=best_eval_loss)
        print(f"Saved best checkpoint: {best_checkpoint_path}")
    checkpoint.save_checkpoint(model, optimizer, conf, final_step, checkpoint_path, scaler=scaler, best_val_loss=best_eval_loss)
    print(f"Final validation loss {final_eval_loss:.4f}; best validation loss {best_eval_loss:.4f}")
