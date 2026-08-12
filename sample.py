from pathlib import Path

from inference import infer


def sample_with_prompt(
    checkpoint_file: Path,
    tokenizer_file: Path,
    prompt: str,
    *,
    device: str = "cpu",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = None,
) -> str:
    continuation = infer(
        prompt,
        checkpoint_file,
        tokenizer_file,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    print("INPUT PROMPT")
    print(prompt)
    print("GENERATED CONTINUATION")
    print(continuation)
    return continuation
