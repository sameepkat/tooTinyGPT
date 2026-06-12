# Use a trained model to generate text
"""
if config says resume
    load model
else

1. create model
2. load trained weights
3. encode a prompt
4. repeatedly predict next token
5. decode token IDs back to text using data.py
"""
