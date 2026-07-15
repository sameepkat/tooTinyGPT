import os
import sys
import tiktoken
import numpy as np
import json


if len(sys.argv) < 3:
    raise RuntimeError(
        "Can't find argument for raw file\n"
        "Usage: python prepare_data.py raw_file output_dir"
    )

enc = tiktoken.get_encoding("gpt2")
raw_file = os.path.join(os.path.dirname(__file__), sys.argv[1])
output_dir = os.path.join(os.path.dirname(__file__), sys.argv[2])

with open(raw_file, encoding="utf-8") as rf:
    content = rf.read()

encoded = enc.encode(content, allowed_special={"<|endoftext|>"})
size = len(encoded)

train_size = int(0.95 * size) 

train, val = encoded[:train_size], encoded[train_size:]
print(f"Length of train ids: {train_size} and length of val ids: {size - train_size}")

meta = {
    "tokenizer": "gpt2",
    "vocab_size": enc.n_vocab,
    "train_tokens": train_size,
    "val_tokens": size - train_size,
    "dtype": "uint16"
}

os.makedirs(output_dir, exist_ok=True)

train_output = str(os.path.join(output_dir, "train.bin"))
val_output = str(os.path.join(output_dir, "val.bin"))
meta_output = str(os.path.join(output_dir, "meta.json"))

np.array(train, dtype=np.uint16).tofile(train_output)
print(f"{train_output}: OK")
np.array(val, dtype=np.uint16).tofile(val_output)
print(f"{val_output}: OK")


with open(meta_output, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
    
print(f"{meta_output}: OK")
