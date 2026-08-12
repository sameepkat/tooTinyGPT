from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer


IGNORE_INDEX = -100


@dataclass(frozen=True)
class SFTSpecialTokens:
    pad_id: int
    eos_id: int
    user_id: int
    assistant_id: int


@dataclass(frozen=True)
class SFTExample:
    x: list[int]
    y: list[int]
    instruction_tokens: int
    response_tokens: int
    truncated: bool
    response_truncated: bool


@dataclass(frozen=True)
class SFTDatasetStats:
    total_examples: int
    avg_instruction_tokens: float
    avg_response_tokens: float
    max_instruction_tokens: int
    max_response_tokens: int
    fitting_examples: int
    truncated_examples: int

    @property
    def truncated_percent(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return 100.0 * self.truncated_examples / self.total_examples


def load_sft_special_tokens(tokenizer: Tokenizer) -> SFTSpecialTokens:
    required = {
        "pad_id": "<PAD>",
        "eos_id": "<EOS>",
        "user_id": "<USER>",
        "assistant_id": "<ASSISTANT>",
    }
    ids: dict[str, int] = {}
    for name, token in required.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer does not define {token}")
        ids[name] = token_id
    return SFTSpecialTokens(**ids)


class SFTDataset:
    def __init__(
        self,
        jsonl_file: Path,
        tokenizer: Tokenizer,
        block_size: int,
        max_prompt_tokens: int = 128,
    ):
        self.jsonl_file = Path(jsonl_file)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_prompt_tokens = max_prompt_tokens
        self.special_tokens = load_sft_special_tokens(tokenizer)
        self.examples: list[SFTExample] = []
        self._load()
        self.stats = self._build_stats()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTExample:
        return self.examples[index]

    def _load(self) -> None:
        if not self.jsonl_file.is_file():
            raise ValueError(f"SFT data file not found: {self.jsonl_file}")
        with self.jsonl_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "input" not in row or "output" not in row:
                    raise ValueError(f"{self.jsonl_file}:{line_number} must contain input and output")
                self.examples.append(self._encode_example(str(row["input"]), str(row["output"])))
        if not self.examples:
            raise ValueError(f"SFT data file contains no examples: {self.jsonl_file}")

    def _encode_example(self, instruction: str, response: str) -> SFTExample:
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=False).ids
        response_ids = self.tokenizer.encode(response, add_special_tokens=False).ids
        max_sequence_tokens = self.block_size + 1
        complete_length = 1 + len(instruction_ids) + 1 + len(response_ids) + 1

        if complete_length <= max_sequence_tokens:
            kept_instruction = instruction_ids
            kept_response = response_ids
            include_eos = True
            truncated = False
            response_truncated = False
        else:
            min_response_tokens = 1 if response_ids else 0
            prompt_budget = max(0, max_sequence_tokens - 2 - min_response_tokens)
            kept_instruction_count = min(len(instruction_ids), self.max_prompt_tokens, prompt_budget)
            kept_instruction = instruction_ids[:kept_instruction_count]
            response_budget = max_sequence_tokens - (1 + len(kept_instruction) + 1)
            kept_response = response_ids[: max(0, response_budget)]
            response_truncated = len(kept_response) < len(response_ids)
            include_eos = not response_truncated and (1 + len(kept_instruction) + 1 + len(kept_response) + 1 <= max_sequence_tokens)
            truncated = True

        sequence = (
            [self.special_tokens.user_id]
            + kept_instruction
            + [self.special_tokens.assistant_id]
            + kept_response
        )
        if include_eos:
            sequence.append(self.special_tokens.eos_id)
        if len(sequence) < 2:
            raise ValueError("SFT example produced fewer than two tokens")

        x = sequence[:-1]
        y = sequence[1:]
        assistant_index = 1 + len(kept_instruction)
        labels = [IGNORE_INDEX if i < assistant_index else token_id for i, token_id in enumerate(y)]
        return SFTExample(
            x=x,
            y=labels,
            instruction_tokens=len(instruction_ids),
            response_tokens=len(response_ids),
            truncated=truncated,
            response_truncated=response_truncated,
        )

    def _build_stats(self) -> SFTDatasetStats:
        total = len(self.examples)
        instruction_lengths = [example.instruction_tokens for example in self.examples]
        response_lengths = [example.response_tokens for example in self.examples]
        truncated = sum(1 for example in self.examples if example.truncated)
        return SFTDatasetStats(
            total_examples=total,
            avg_instruction_tokens=sum(instruction_lengths) / total,
            avg_response_tokens=sum(response_lengths) / total,
            max_instruction_tokens=max(instruction_lengths),
            max_response_tokens=max(response_lengths),
            fitting_examples=total - truncated,
            truncated_examples=truncated,
        )


def collate_sft_batch(examples: list[SFTExample], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(example.x) for example in examples)
    x_rows: list[list[int]] = []
    y_rows: list[list[int]] = []
    for example in examples:
        pad_count = max_len - len(example.x)
        x_rows.append(example.x + [pad_id] * pad_count)
        y_rows.append(example.y + [IGNORE_INDEX] * pad_count)
    x = torch.tensor(x_rows, dtype=torch.long, device=device)
    y = torch.tensor(y_rows, dtype=torch.long, device=device)
    return x, y


def print_sft_stats(name: str, stats: SFTDatasetStats) -> None:
    print(
        f"{name}: examples {stats.total_examples} | "
        f"avg instruction tokens {stats.avg_instruction_tokens:.1f} | "
        f"avg response tokens {stats.avg_response_tokens:.1f} | "
        f"max instruction tokens {stats.max_instruction_tokens} | "
        f"max response tokens {stats.max_response_tokens} | "
        f"fits {stats.fitting_examples} | truncated {stats.truncated_examples} "
        f"({stats.truncated_percent:.1f}%)"
    )
