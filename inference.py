from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from tokenizers import Tokenizer

from checkpoint import load_checkpoint_file
from config import Config
from model import GPT


class InferenceEngine:
    def __init__(self, checkpoint_path: Path, tokenizer_path: Path, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.eos_token_id = self.tokenizer.token_to_id("<EOS>")
        if self.eos_token_id is None:
            raise ValueError("tokenizer does not define <EOS>")
        checkpoint_data = load_checkpoint_file(self.device, checkpoint_path)
        self.config = Config.from_dict(checkpoint_data["config"], device=self.device)
        tokenizer_vocab_size = self.tokenizer.get_vocab_size()
        if self.config.vocab_size != tokenizer_vocab_size:
            raise ValueError(
                f"checkpoint vocab_size ({self.config.vocab_size}) does not match "
                f"tokenizer vocab size ({tokenizer_vocab_size})"
            )
        self.model = GPT(self.config).to(self.device)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int | None = None,
    ) -> str:
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if not encoded:
            raise ValueError("prompt must encode to at least one token")
        prompt_length = len(encoded)
        prompt_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            token_ids = self.model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token_id=self.eos_token_id,
            )[0]
        return self.tokenizer.decode(token_ids[prompt_length:].tolist(), skip_special_tokens=False)


@lru_cache(maxsize=4)
def get_inference_engine(checkpoint_path: str, tokenizer_path: str, device: str = "cpu") -> InferenceEngine:
    """Cache heavyweight model/tokenizer loading for repeated API requests."""
    return InferenceEngine(Path(checkpoint_path), Path(tokenizer_path), device)


def infer(
    prompt: str,
    checkpoint_file: Path,
    tokenizer_file: Path,
    device: str | torch.device = "cpu",
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = None,
) -> str:
    return get_inference_engine(str(checkpoint_file), str(tokenizer_file), str(device)).generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
