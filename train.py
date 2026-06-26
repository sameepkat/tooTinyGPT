import math
import os
import sys

import numpy as np
import tiktoken
import torch

import checkpoint
from config import Config
from data import Data
from model import GPT

# Setup
enc = tiktoken.get_encoding("gpt2")

vocab_size = enc.n_vocab
batch_size = 4
max_steps = 1000
block_size = 64
n_embd = 384
# max_new_tokens = 256
# temperature = 0.8
# top_k = 50
dropout = 0.1
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
)

if len(sys.argv) < 3:
    raise RuntimeError(
        "Can't find arguments for train.bin and val.bin\n"
        "Usage: python train.py train.bin val.bin"
    )


def lr_scheduler(it: int):
    if conf.warmup_iters > 0 and it < conf.warmup_iters:
        return conf.lr * (it + 1) / conf.warmup_iters
    if it >= conf.decay_iters:
        return conf.min_lr

    progress = (it - conf.warmup_iters) / (conf.decay_iters - conf.warmup_iters)

    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return conf.min_lr + cosine_decay * (conf.lr - conf.min_lr)


train_file = os.path.join(
    os.path.dirname(__file__), sys.argv[1]
)  # argv[1] is training file name
val_file = os.path.join(os.path.dirname(__file__), sys.argv[2])
# with open(input_file, 'r', encoding='utf-8') as f:
#     text = f.read()

print(f"Using device: {device}")

# Data
data_obj = Data(train_file, val_file, conf)
x, y = data_obj.get_batch("train")

# Model
model = GPT(conf).to(device)
start_step = 0

optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay
)

if conf.resume:
    start_step = checkpoint.load_checkpoint(model, optimizer, conf)

logits, loss = model(x, y)
total_params = sum([p.numel() for p in model.parameters()])
# Loss
train_losses = []
print(f"Initial loss: {loss}")
print(f"Total parameters: {total_params:,}")

last_step = 0
# Training loop
for each_loop in range(start_step, conf.max_steps):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_scheduler(each_loop)

    x, y = data_obj.get_batch("train")
    logits, loss = model(x, y)
    assert loss is not None
    train_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if each_loop > 0 and each_loop % conf.checkpoint_interval == 0:
        checkpoint.save_checkpoint(model, optimizer, conf, each_loop)
    if each_loop > 0 and each_loop % conf.eval_interval == 0:
        model.eval()
        train_loss = np.mean(train_losses[-conf.eval_interval :])
        lr = optimizer.param_groups[0]["lr"]
        print(f"Train loss at step {each_loop} and lr {lr}: {train_loss}")
        eval_losses = []
        with torch.no_grad():
            for _ in range(conf.eval_steps):
                x, y = data_obj.get_batch("val")
                logits, loss = model(x, y)
                eval_losses.append(loss.item())

        eval_loss = np.mean(eval_losses)
        print(f"Eval loss at {each_loop}: {eval_loss}")

        model.train()
    last_step = each_loop

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
    checkpoint.save_checkpoint(model, optimizer, conf, last_step)


estimated_loss = np.mean(eval_losses)
print(f"Estimated loss: {estimated_loss}")

"""
# prompt = "To be"
# encoded = enc.encode(prompt)
encoded = enc.encode(enc.decode(x[:100].tolist()))
prompt_tensor = torch.tensor(encoded, dtype=torch.long)
prompt_tensor = prompt_tensor.view(1, -1) # add a batch dimension -> (1, T)
prompt_tensor = prompt_tensor.to(device)


# Generation
with torch.no_grad():
    token_ids = model.generate(prompt_tensor, max_new_tokens, temperature=temperature, top_k=top_k)[0]
decoded_str = enc.decode(token_ids.cpu().tolist())
print(f"Original str: {text[:200]}")
print(f"Prompt str: {prompt}")
print(f"Decoded str: {decoded_str}")


"""
