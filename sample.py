import inference
from pathlib import Path

def sample_with_prompt(checkpoint_file: Path = Path("checkpoint.pt"), encoding: str = "gpt2"):
    lines = []

    print("Enter prompt. Type END when finished: ")

    while True:
        line = input()

        if line == "END":
            break
        lines.append(line)

    text = "\n".join(lines)
    generated_text = inference.infer(text, checkpoint_file, encoding)

    print(generated_text)

if __name__ == "__main__":
    sample_with_prompt(Path("checkpoint.pt"))

