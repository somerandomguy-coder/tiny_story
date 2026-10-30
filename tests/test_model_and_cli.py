import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

def test_model_build_and_generation():
    from train import build_model, generate_story
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = build_model(vocab_size=len(tokenizer))
    
    total_params = sum(p.numel() for p in set(model.parameters()) if p.requires_grad)
    assert 45_000_000 < total_params < 52_000_000, f"Expected ~48.9M params, got {total_params}"
    
    output = generate_story(model, tokenizer, prompt="Once upon a time", max_new_tokens=10)
    assert isinstance(output, str)
    assert len(output) > len("Once upon a time")

if __name__ == "__main__":
    test_model_build_and_generation()
    print("Model build and generation tests passed successfully!")
