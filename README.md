# TooTinyGPT

A decoder-only Transformer trained from random initialization. Training data is pre-tokenized with the frozen byte-level BPE tokenizer loaded through the lightweight `tokenizers` package; `transformers` and GPT-2 token IDs are not used.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Pretokenized Training

`train.bin` and `val.bin` are little-endian `uint16` token streams. They are read with `numpy.memmap`, so the corpus is not loaded into RAM. The train command requires an explicit output checkpoint; its parent directory is created automatically. The best validation checkpoint is written alongside it with `_best` before the suffix.

Random initialization (`--resume-mode none` is the default):

```bash
python main.py train --train-file data/lum/stage_a_50m/train.bin --val-file data/lum/stage_a_50m/val.bin --checkpoint checkpoints/pilot42_stage_a.pt --tokenizer data/lum/tokenizer/tokenizer.json --vocab-size 8192 --block-size 512 --n-embd 512 --n-head 8 --n-layer 12 --dropout 0.1 --batch-size 32 --gradient-accumulation-steps 1 --max-steps 3052 --warmup-iters 100 --decay-iters 3052 --lr 3e-4 --min-lr 3e-5 --weight-decay 0.1 --grad-clip 1.0 --eval-interval 100 --eval-steps 20 --checkpoint-interval 250 --device cuda --amp
```

CUDA AMP uses FP16 `torch.autocast` and `torch.amp.GradScaler`. CPU and MPS remain functional without CUDA AMP. Full resume restores model, optimizer, scaler when present, completed update step, and best validation loss:

```bash
python main.py train --resume-mode full --train-file data/lum/stage_a_50m/train.bin --val-file data/lum/stage_a_50m/val.bin --checkpoint checkpoints/pilot42_stage_a.pt --tokenizer data/lum/tokenizer/tokenizer.json --vocab-size 8192 --block-size 512 --n-embd 512 --n-head 8 --n-layer 12 --dropout 0.1 --batch-size 32 --gradient-accumulation-steps 1 --max-steps 3052 --warmup-iters 100 --decay-iters 3052 --lr 3e-4 --min-lr 3e-5 --weight-decay 0.1 --grad-clip 1.0 --eval-interval 100 --eval-steps 20 --checkpoint-interval 250 --device cuda --amp
```

For a later stage, model-only initialization loads weights but starts a new optimizer, scaler, and step counter. The output checkpoint must be different from the input checkpoint:

```bash
python main.py train --resume-mode model --init-checkpoint checkpoints/stage_a.pt --checkpoint checkpoints/stage_b.pt --train-file PATH/train.bin --val-file PATH/val.bin --tokenizer PATH/tokenizer.json --vocab-size 8192
```

## Sampling

Sampling uses the same frozen tokenizer and prints the prompt and continuation separately:

```bash
python main.py sample --checkpoint checkpoints/pilot42_stage_a_best.pt --tokenizer data/lum/tokenizer/tokenizer.json --prompt "fn fibonacci(n) {" --max-new-tokens 128 --temperature 0.8 --top-k 50
```

## Stage-C SFT

Supervised fine-tuning reads JSONL rows with `input` and `output` fields. Each row remains one example and is formatted as token IDs:

```text
[<USER>] + encode(input) + [<ASSISTANT>] + encode(output) + [<EOS>]
```

Labels use response-only loss. Targets for `<USER>`, the instruction, `<ASSISTANT>`, and padding are set to `-100`; response tokens and the real EOS target are supervised. If an example exceeds context, the instruction is capped by `--max-prompt-tokens`, response tokens are kept from the beginning, and EOS is not appended when the response is truncated.

Stage C should usually start from the Stage-B model with a fresh optimizer, scaler, and LR schedule:

```bash
python main.py sft --train-file data/lum/train.jsonl --val-file data/lum/validation.jsonl --init-checkpoint checkpoints/pilot42_stage_b_best.pt --checkpoint checkpoints/pilot42_sft.pt --tokenizer data/lum/tokenizer/tokenizer.json --vocab-size 8192 --block-size 512 --n-embd 512 --n-head 8 --n-layer 12 --dropout 0.1 --epochs 5 --batch-size 16 --gradient-accumulation-steps 2 --lr 5e-5 --min-lr 5e-6 --weight-decay 0.01 --grad-clip 1.0 --warmup-updates 10 --max-prompt-tokens 128 --device cuda --amp
```

Full resume restores the SFT checkpoint state:

```bash
python main.py sft --resume-mode full --train-file data/lum/train.jsonl --val-file data/lum/validation.jsonl --checkpoint checkpoints/pilot42_sft.pt --tokenizer data/lum/tokenizer/tokenizer.json --vocab-size 8192 --block-size 512 --n-embd 512 --n-head 8 --n-layer 12 --dropout 0.1 --epochs 5 --batch-size 16 --gradient-accumulation-steps 2 --lr 5e-5 --min-lr 5e-6 --weight-decay 0.01 --grad-clip 1.0 --warmup-updates 10 --max-prompt-tokens 128 --device cuda --amp
```

Instruction inference uses the same training format, without decoding the prompt:

```bash
python main.py instruct --checkpoint checkpoints/pilot42_sft_best.pt --tokenizer data/lum/tokenizer/tokenizer.json --prompt "Write a function that returns the nth Fibonacci number." --max-new-tokens 256 --temperature 0.2 --top-k 20
```

Use `--greedy` for deterministic argmax decoding.

## API

The API paths are configured at process startup, not by requests:

```bash
TOOTINYGPT_CHECKPOINT=checkpoints/pilot42_stage_a_best.pt TOOTINYGPT_TOKENIZER=data/lum/tokenizer/tokenizer.json TOOTINYGPT_DEVICE=cpu uvicorn api:app --host 0.0.0.0 --port 8000
```

`POST /generate` accepts `prompt`, `max_new_tokens`, `temperature`, and optional `top_k`.
