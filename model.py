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
from config import Config
from typing import Optional

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        vocab_size = config.vocab_size
        embedding_dimension = config.n_embd
        self.block_size = config.block_size
        n_embd = config.n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimension)
        self.position_embedding_table = nn.Embedding(self.block_size, embedding_dimension)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)


    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]  = None): # x.shape = (B, T), y(B, T) = Correct next tokens
        """
        Used for training , evaluation & generation
        """
        B, T, = x.shape
        tok_emb_x = self.token_embedding_table(x) # (B, T, C)

        pos_ids = torch.arange(T) # ( T, )
        pos_emb_x = self.position_embedding_table(pos_ids) # (T, C)

        x_emb = tok_emb_x+ pos_emb_x  # (B, T, C) + (T, C) = (B, T, C)
        # TODO: Support GPU
        logits = self.lm_head(x_emb) # (B, T, vocab_size)

        B, T, vocab_size = logits.shape

        loss_fn = nn.CrossEntropyLoss()
        N = B*T

        if y is None:
            loss = None
        else:
            y = y.reshape(N)
            loss = loss_fn(logits.reshape(N, vocab_size), y) # scalar average error
        
        return logits, loss

    def generate(self, token_ids: torch.Tensor, max_new_tokens: int): # - token IDs generated so far - shape = (B, T)
        """
        Used for generation/ sampling
        """
        i = max_new_tokens
        while i> 0:
            B, T = token_ids.shape
            ctx = token_ids
            if T > self.block_size:
                ctx = token_ids[:, -self.block_size:]

            logits, _ = self.forward(ctx)
            latest = logits[:, -1, :] # (B, vocab_size)

            probabilities = nn.functional.softmax(latest, -1) # softmax over vocab dimension to get probabilities over vocabulary
            prediction = torch.argmax(input=probabilities, dim=-1) # greedy first for now # has shape Of (B,)
            prediction = prediction.reshape(-1, 1) # (B) -> (B, 1) for cat
            # TODO: Use sampling instead of greedy approach

            token_ids = torch.cat([token_ids, prediction], dim=1) # stack along T dimension
            i -= 1
        return token_ids
        


class Block(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class CausalSelfAttention(nn.Module):
    def __init__(self,  config: Config):  # x.shape = (B, T, C)
        super().__init__()

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.head_size = self.n_embd // self.n_head
        self.W_q = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)
        self.W_k = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)
        self.W_v = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)
        self.W_o = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n_head = self.n_head
        head_size = C // n_head

        q = self.W_q(x) # (batch_size, seq_len, d_model)
        k = self.W_k(x)
        v = self.W_v(x)

        Q = q.reshape(B, -1, n_head, head_size) # Split d_model into n_head * head_size = C -> n_head * head_size
        K = k.reshape(B, -1, n_head, head_size)
        V = v.reshape(B, -1, n_head, head_size)

        Q = Q.permute(0, 2, 1, 3) # bring n_head to front: (batch_size, n_head, seq_len , head_size)
        K = K.permute(0, 2, 1, 3) # turns out permute is similar to transpose that allows multiple_dimensions like we are doing here
        V = V.permute(0, 2, 1, 3)

        K_T = K.permute(0, 1, 3, 2)# from the dot attention formula: We need QK^T: Swapping last two dimensions

        scores = Q @ K_T
        mask = torch.tril(torch.ones_like(scores))
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        

        scaled = masked_scores / (self.head_size ** 0.5)
        attention = nn.functional.softmax(scaled, dim=-1) @ V # attention = (QK^T)/sqrt(d_k) * V # softmax over last dimension

        output = attention.permute(0, 2, 1, 3) # transpose back to (batch_size, seq_len, n_head, head_size)
        reshaped_output = output.reshape(B, T, C) # now i guess head should be (batch_size, seq_len, )
        final = self.W_o(reshaped_output) # mix information acorss heads
        
        return final
    

class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class FeedForward(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
