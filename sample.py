from checkpoint import  load_checkpoint_file
from config import Config
import torch
import tiktoken
from pathlib import Path

from model import GPT


def sample(prompt: str, checkpoint_file: Path):
    enc = tiktoken.get_encoding("gpt2")
    encoded = enc.encode(prompt)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


    prompt_tensor = torch.tensor(encoded, dtype=torch.long)
    prompt_tensor = prompt_tensor.view(1, -1) # add a batch dimension -> (1, T)
    prompt_tensor = prompt_tensor.to(device)

    checkpoint = load_checkpoint_file(device, checkpoint_file)

    config_dict = dict(checkpoint["config"])
    config_dict["device"] = device
    config = Config(**config_dict)

    max_new_tokens = 256 # static or dynamic
    # TODO: Pass flag for max new token

    model = GPT(config).to(config.device)
    model_state = checkpoint['model_state_dict']
    model.load_state_dict(model_state)

    model.eval()

    with torch.no_grad():
        token_ids = model.generate(prompt_tensor, max_new_tokens)[0]

    generated_str = enc.decode(token_ids.cpu().tolist())
    print(f"Prompt str: {prompt}")
    print(f"Generated str: {generated_str}")

    
def sample_with_prompt(checkpoint_path: Path):
    prompt = input("Enter the prompt: ")
    sample(prompt, checkpoint_path)

if __name__ == "__main__": 
    sample_with_prompt(Path("checkpoint.pt"))
