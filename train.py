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
from torch import dtype
from data import Data
from config import Config
import tiktoken
from model import GPT


text = "Hello World. Is this a text?"
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
conf = Config(vocab_size=vocab_size, block_size=4)

data_obj = Data(text, conf)

x, y = data_obj.get_batch("train")

print(f"x.shape: {x.shape}")
print(f"y.shape: {y.shape}")
print(f"Encoded x: {x[0]}")
print(f"Encoded y: {y[0]}")

decoded_text = enc.decode(x[0].tolist())
decoded_y = enc.decode(y[0].tolist())

print(f"Decoded x: {decoded_text}")
print(f"Decoded y: {decoded_y}")

prompt = "Hello"
encoded_prompt = enc.encode(prompt)
prompt_tensor = torch.tensor(encoded_prompt, dtype=torch.long)
prompt_tensor = prompt_tensor.view(1, 1)

model = GPT(conf)
token_ids = model.generate(prompt_tensor, 4)[0]
decoded_str = enc.decode(token_ids.tolist())
print(f"Token Ids: {token_ids}")
print(f"Decoded str: {decoded_str}")
