# orchestrate

"""
1. create Config from config.py
2. choose tokenizer from data.py
3. load raw text from sample file
4. create Data object from data.py
5. create model from model.py
6. create optimizer (probably here from pytorch.py)

Then, repepatedly
- get batch -> run model -> compute loss -> backpropagate -> update weights -> occasionally evaluate -> ocasionally save checkpoint
"""

import torch
from data import Data
from config import Config
import tiktoken
from model import GPT
import numpy as np
import os
import checkpoint

# Setup 
enc = tiktoken.get_encoding("gpt2")

vocab_size = enc.n_vocab
batch_size = 4
max_steps = 1000
block_size = 64
n_embd = 384
max_new_tokens = 256
temperature = 0.8
top_k = 50
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
conf = Config(batch_size=batch_size, vocab_size=vocab_size, block_size=block_size, max_steps=max_steps, n_embd=n_embd, dropout=dropout, device=device)

input_file = os.path.join(os.path.dirname(__file__), 'data' ,  'shakespeare.txt')
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()
    
print(f"Using device: {device}")

# Data
data_obj = Data(text, conf)
x, y = data_obj.get_batch("train")

# Model
model = GPT(conf).to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
start_step = 0

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
    x, y = data_obj.get_batch("train")
    logits, loss = model(x, y)
    assert loss is not None
    train_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if each_loop % 10 == 0:
        print(f"Loop: {each_loop}\tLoss: {loss.item()}")
    if each_loop > 0 and each_loop % conf.checkpoint_interval == 0:
        checkpoint.save_checkpoint(model, optimizer, conf, each_loop)
    last_step = each_loop

prompt = "To be"
encoded = enc.encode(prompt)
prompt_tensor = torch.tensor(encoded, dtype=torch.long)
prompt_tensor = prompt_tensor.view(1, -1) # add a batch dimension -> (1, T)
prompt_tensor = prompt_tensor.to(device)

model.eval()
# Loss Evaluation
eval_losses = []

with torch.no_grad():
    for _ in range(20):
        x, y = data_obj.get_batch("train")
        logits, loss = model(x, y)
        eval_losses.append(loss.item())
    

# Generation
with torch.no_grad():
    token_ids = model.generate(prompt_tensor, max_new_tokens, temperature=temperature, top_k=top_k)[0]
decoded_str = enc.decode(token_ids.cpu().tolist())
print(f"Original str: {text[:200]}")
print(f"Prompt str: {prompt}")
print(f"Decoded str: {decoded_str}")

if len(train_losses) == 0:
    print("No training steps run")
else:
    print(f"Average training loss: {np.mean(train_losses)}")
    checkpoint.save_checkpoint(model, optimizer, conf, last_step)

estimated_loss = np.mean(eval_losses)
print(f"Estimated loss: {estimated_loss}")
