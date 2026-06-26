# TooTinyGPT
A basic educational GPT-Style decoder-only Transformer implemented in Pytorch.

## Dependencies
To install dependencies just run:
```python
python -m venv .venv # create virtual environment
pip install -r requirements.txt
```

# How to Train
Run:
```bash
python train.py train.bin val.bin
```
where `train.bin` is the training file and `val.bin` is the validation file.

# How to Sample
```bash
python sample.py
# Enter prompt: 
```
Enter the prompt to be used as the base text to sample from the trained model.

# Project Directory
```bash
.
├── checkpoint.py	# Handles saving and loading checkpoint 
├── config.py		# Handles config initalization for the model
├── data.py		# Prepares train.bin and val.bin for model training
├── model.py		# Implements GPT, Block, CausalSelfAttention and FeedForward Layer
├── pyrightconfig.json
├── README.md
├── requirements.txt	# Requirements file
├── sample.py		# Takes user input -> encodes -> pass to model -> decode 
└── train.py		# Contains training loop for actual model training
```

# Contributing
Any contributions are welcome. ^-^
Fork the repo and create a pull request.