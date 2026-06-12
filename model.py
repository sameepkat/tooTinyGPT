# convert x of size (Batch_size, Sequence_length) to logits whose shape is (Batch_size, Sequence_length, Vocab_size)
# each position gets score for all tokens in vocab_size

"""
Implement
1. Token Embedding
token ID -> vector

2. Positional Embedding
position number -> vector

3. Transformer Blocks
attention + feed-forward + layernorm

4. Final linear head
hidden vector -> vocabulary score
"""

import torch
import torch.nn as nn
from data import Data
from config import Config

class GPT:
    def __init__(self, config: Config):
        vocab_size = config.vocab_size
        embedding_dimension = config.n_embd


        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimension)


    def forward(self, x):
        tok_emb_x = self.token_embedding_table(x);

        return tok_emb_x


class Block:
    def __init__(self):
        pass

    def forward(self):
        pass

class CausalSelfAttention:
    def __init__(self):
        pass

    def forward(self):
        pass

class MultiHeadAttention:
    def __init__(self):
        pass

    def forward(self):
        pass

class FeedForward:
    def __init__(self):
        pass

    def forward(self):
        pass
