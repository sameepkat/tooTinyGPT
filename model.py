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
        self.device = config.device
        n_layer = config.n_layer
        n_embd = config.n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimension)
        self.position_embedding_table = nn.Embedding(self.block_size, embedding_dimension)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)
        self.blocks =  nn.ModuleList(Block(config) for _ in range(n_layer))
        self.layer_norm = nn.LayerNorm(n_embd, bias=False)


    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]  = None): # x.shape = (B, T), y(B, T) = Correct next tokens
        """
        Used for training , evaluation & generation
        """
        B, T, = x.shape
        tok_emb_x = self.token_embedding_table(x) # (B, T, C)

        pos_ids = torch.arange(T).to(x.device) # ( T, )
        pos_emb_x = self.position_embedding_table(pos_ids) # (T, C)

        x_emb = tok_emb_x + pos_emb_x  # (B, T, C) + (T, C) = (B, T, C)
        for block in self.blocks:
            x_emb = block(x_emb)

        x_emb = self.layer_norm(x_emb)
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

    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature=0.8, top_k=None): # - token IDs generated so far - shape = (B, T)
        """
        Used for generation/ sampling
        """
        i = max_new_tokens

        if temperature <= 0:
            raise ValueError(f"Invalid temperature: {temperature}")
        while i> 0:
            B, T = token_ids.shape
            ctx = token_ids
            if T > self.block_size:
                ctx = token_ids[:, -self.block_size:]

            logits, _ = self.forward(ctx)
            latest = logits[:, -1, :] # (B, vocab_size)
            latest = latest / temperature
            if top_k is not None and top_k <=0:
                raise ValueError(f"top_k can't be 0 or negative: {top_k}")
            if top_k is not None:
                top_k = min(top_k, latest.shape[-1]) # clamping 
                values, indices = torch.topk(latest, k=top_k , dim=-1)
                threshold = values[:, [-1]]
                latest = torch.masked_fill(latest, latest < threshold, float("-inf"))


            probabilities = nn.functional.softmax(latest, -1) # softmax over vocab dimension to get probabilities over vocabulary
            prediction = torch.multinomial(probabilities, num_samples=1)

            token_ids = torch.cat([token_ids, prediction], dim=1) # stack along T dimension
            i -= 1
        return token_ids
        


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # TODO: Implement my own LayerNorm
        self.first_layernorm = nn.LayerNorm(config.n_embd, bias=False) # normalize for attention
        self.attention = CausalSelfAttention(config)
        self.second_layernorm = nn.LayerNorm(config.n_embd, bias=False) # attention block changes the representation, so 
        self.feedforward = FeedForward(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.first_layernorm(x))
        x = x + self.feedforward(self.second_layernorm(x))
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self,  config: Config):  # x.shape = (B, T, C)
        super().__init__()

        block_size = config.block_size
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.head_size = self.n_embd // self.n_head
        self.W_q = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)
        self.W_k = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)
        self.W_v = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)
        self.W_o = nn.Linear(in_features=self.n_embd, out_features=self.n_embd, bias=False)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.mask: torch.Tensor
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))) # So, it can be moved to device

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
        mask = self.mask[:T, :T]
        

        scaled = scores / (self.head_size ** 0.5)
        masked_scores = scaled.masked_fill(~mask, float("-inf"))
        attention = nn.functional.softmax(masked_scores, dim=-1) # attention = (QK^T)/sqrt(d_k) * V # softmax over last dimension
        attention = self.attention_dropout(attention) @ V

        output = attention.permute(0, 2, 1, 3) # transpose back to (batch_size, seq_len, n_head, head_size)
        reshaped_output = output.reshape(B, T, C) # now i guess head should be (batch_size, seq_len, )
        final = self.W_o(reshaped_output) # mix information acorss heads
        final = self.residual_dropout(final)
        
        return final

class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        in_features = config.n_embd 
        hidden_size = 4 * config.n_embd  # to allow larger hidden layer more capacity to transform each token vector

        self.ff_expand_layer = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.gelu = nn.GELU()
        
        out_features = config.n_embd

        self.out_layer = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.feedforward_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor): # x: (B, T, C)
        x = self.ff_expand_layer(x)
        x = self.gelu(x)
        x = self.out_layer(x)
        x = self.feedforward_dropout(x)

        return x
        # return (B, T, C)
