import math
from pathlib import Path

import numpy as np
import torch

import checkpoint
from config import Config
from data import Data
from model import GPT


def lr_scheduler(it: int, conf: Config):
    if conf.warmup_iters > 0 and it < conf.warmup_iters:
        return conf.lr * (it + 1) / conf.warmup_iters
    if it >= conf.decay_iters:
        return conf.min_lr

    progress = (it - conf.warmup_iters) / (conf.decay_iters - conf.warmup_iters)

    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return conf.min_lr + cosine_decay * (conf.lr - conf.min_lr)


def train(
    conf: Config,
    train_file: Path,
    val_file: Path,
    resume_mode: str = "none",
    checkpoint_path: Path = Path("checkpoint.pt"),
) -> None:
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)

    print(f"Using device: {conf.device}")

    # Data
    data_obj = Data(train_file, val_file, conf)
    x, y = data_obj.get_batch("train")

    # Model
    model = GPT(conf).to(conf.device)
    start_step = 0

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay
    )

    match resume_mode:
        case "full":
            start_step = checkpoint.load_checkpoint(
                model, optimizer, conf, checkpoint_path
            )

        case "model":
            checkpoint.load_model_checkpoint(model, conf, checkpoint_path)
            start_step = 0
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay
            )

        case _:
            start_step = 0

    logits, loss = model(x, y)
    total_params = sum([p.numel() for p in model.parameters()])
    # Loss
    train_losses = []
    print(f"Initial loss: {loss}")
    print(f"Total parameters: {total_params:,}")

    best_eval_loss = float("inf")
    best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")

    last_step = start_step
    # Training loop
    optimizer.zero_grad()
    update_step = start_step
    N = conf.gradient_accumulation_steps

    for each_loop in range(start_step * N, conf.max_steps * N):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_scheduler(update_step, conf)

        x, y = data_obj.get_batch("train")
        logits, loss = model(x, y)
        assert loss is not None
        train_losses.append(loss.item())
        loss = loss / N  # for gradient accumulation
        loss.backward()

        if (each_loop + 1) % N == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            update_step += 1

            if update_step > 0 and update_step % conf.checkpoint_interval == 0:
                checkpoint.save_checkpoint(
                    model, optimizer, conf, update_step, checkpoint_path
                )
            if update_step > 0 and update_step % conf.eval_interval == 0:
                model.eval()
                train_loss = np.mean(train_losses[-conf.eval_interval * N :])
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Train loss at update step {update_step} and lr {lr}: {train_loss}"
                )
                eval_losses = []
                with torch.no_grad():
                    for _ in range(conf.eval_steps):
                        x, y = data_obj.get_batch("val")
                        logits, loss = model(x, y)
                        eval_losses.append(loss.item())

                eval_loss = np.mean(eval_losses)
                print(f"Eval loss at update step {update_step}: {eval_loss}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    checkpoint.save_checkpoint(model, optimizer, conf, update_step, best_checkpoint_path)
                    print(f"Saved best checkpoint to {best_checkpoint_path} with eval loss {best_eval_loss}")

                model.train()
            last_step = update_step

    model.eval()
    # Loss Evaluation
    eval_losses = []

    with torch.no_grad():
        for _ in range(20):
            x, y = data_obj.get_batch("val")
            logits, loss = model(x, y)
            eval_losses.append(loss.item())

    if len(train_losses) == 0:
        print("No training steps run")
    else:
        print(f"Average training loss: {np.mean(train_losses)}")
        checkpoint.save_checkpoint(model, optimizer, conf, last_step, checkpoint_path)

    estimated_loss = np.mean(eval_losses)

    if estimated_loss < best_eval_loss:
        checkpoint.save_checkpoint(model, optimizer, conf, last_step, best_checkpoint_path)
        print(f"Saved best checkpoint to {best_checkpoint_path} with eval loss {best_eval_loss}")
    print(f"Estimated loss: {estimated_loss}")
