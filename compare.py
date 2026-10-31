import os
import sys
import argparse
import torch

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
from transformers.trainer_utils import get_last_checkpoint
from train import generate_story, get_device

DEFAULT_PROMPTS = [
    "Once upon a time, there was a little dog named Spot who loved to",
    "Lily and her brother Tom decided to explore the dark cave.",
    "One sunny morning, a tiny bird wanted to fly high into the sky,",
]

def resolve_checkpoint(checkpoint_dir: str = None) -> str:
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        return os.path.abspath(checkpoint_dir)
    
    # Priority check for final models
    for candidate in ["./tinystories-1m-final", "./tinystories-final", "./tinystories-3m-final"]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
            
    # Check latest checkpoints in potential checkpoint folders
    for folder in ["./tinystories-1m-checkpoints", "./tinystories-checkpoints", "./tinystories-3m-checkpoints"]:
        if os.path.exists(folder):
            latest = get_last_checkpoint(folder)
            if latest and os.path.exists(latest):
                return os.path.abspath(latest)
        
    return None

def run_comparison(checkpoint_dir: str = None, max_new_tokens: int = 80, custom_prompt: str = None):
    device = get_device()
    resolved_path = resolve_checkpoint(checkpoint_dir)
    
    if not resolved_path:
        print("Error: No trained checkpoint found. Please provide a path using --checkpoint.")
        return
        
    print(f"Loading tokenizer & trained model from: {resolved_path}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    trained_model = LlamaForCausalLM.from_pretrained(resolved_path).to(device)
    
    print("Building untrained baseline model (exact architecture, random scratch initialization)...")
    config = AutoConfig.from_pretrained(resolved_path)
    untrained_model = LlamaForCausalLM(config).to(device)
    
    trained_params = sum(p.numel() for p in trained_model.parameters())
    ckpt_name = os.path.basename(resolved_path)
    print(f"Model parameters: {trained_params / 1e6:.2f}M ({trained_params:,} params) | Device: {device} | Checkpoint: {ckpt_name}\n")
    print("=" * 80)
    print(" " * 15 + f"UNTRAINED (SCRATCH) vs TRAINED ({ckpt_name}) COMPARISON")
    print("=" * 80)
    
    prompts = [custom_prompt] if custom_prompt else DEFAULT_PROMPTS
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[TEST PROMPT {i}]: \"{prompt}\"")
        print("-" * 80)
        
        # 1. Untrained model generation
        untrained_output = generate_story(
            untrained_model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=40,
        )
        print(">>> BEFORE TRAINING (Scratch / Random Initialization):")
        print(untrained_output)
        print("-" * 40)
        
        # 2. Trained model generation
        trained_output = generate_story(
            trained_model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=40,
        )
        print(f">>> AFTER TRAINING ({ckpt_name}):")
        print(trained_output)
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Compare Untrained vs Trained TinyStories model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (auto-detects if omitted)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for comparison test")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens to generate")
    args = parser.parse_args()
    
    run_comparison(checkpoint_dir=args.checkpoint, max_new_tokens=args.max_tokens, custom_prompt=args.prompt)

if __name__ == "__main__":
    main()
