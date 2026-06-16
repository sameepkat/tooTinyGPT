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
from model import GPT, FeedForward, Block
import numpy as np


text = "The quick brown fox jumps over the lazy dog."
enc = tiktoken.get_encoding("gpt2")

vocab_size = enc.n_vocab
batch_size = 4
max_steps = 200
block_size = 4
n_embd = 384
conf = Config(batch_size=batch_size, vocab_size=vocab_size, block_size=block_size, max_steps=max_steps, n_embd=n_embd)

data_obj = Data(text, conf)
x, y = data_obj.get_batch("train")

#print(f"x.shape: {x.shape}")
#print(f"y.shape: {y.shape}")
#print(f"Encoded x: {x}")
#print(f"Encoded y: {y}")

# decoded_text = enc.decode(x[0].tolist())
decoded_text  = [enc.decode(xi) for xi in x.tolist()]
decoded_y = [enc.decode(yi) for yi in y.tolist()]

#print(f"Decoded x: {decoded_text}")
#print(f"Decoded y: {decoded_y}")

model = GPT(conf)
logits, loss = model.forward(x, y)

print(f"Initial loss: {loss}")

optimizer = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

losses = []

for each_loop in range(conf.max_steps):
    x, y = data_obj.get_batch("train")
    logits, loss = model.forward(x, y)
    assert loss is not None
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

prompt = " brown"
encoded = enc.encode(prompt)
prompt_tensor = torch.tensor(encoded, dtype=torch.long)
prompt_tensor = prompt_tensor.view(1, -1) # add a batch dimension -> (1, T)


token_ids = model.generate(prompt_tensor, 4)[0]
decoded_str = enc.decode(token_ids.tolist())
print(f"Original str: {text}")
print(f"Given str: {prompt}")
print(f"Decoded str: {decoded_str}")

logits, loss = model(x, y)

print(f"Average loss: {np.mean(losses)}")
print(f"Final loss: {loss}")

