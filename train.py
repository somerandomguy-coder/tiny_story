import os
import glob
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)

def get_device():
    """Detect available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def build_model(vocab_size: int = 50257) -> LlamaForCausalLM:
    """Construct lightweight ~29.4M parameter LLaMA model from scratch."""
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1376,       # SwiGLU ~8/3 * hidden_size
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,        # Grouped-Query Attention
        max_position_embeddings=512,
        tie_word_embeddings=True,     # Avoids duplicating embedding matrix for output projection
        pad_token_id=50256,           # GPT-2 eos/pad token id
        bos_token_id=50256,
        eos_token_id=50256,
    )
    model = LlamaForCausalLM(config)
    return model

def get_dataset(
    data_dir: str = "./data",
    test_mode: bool = False,
    max_train_samples: int = 200_000,
    max_val_samples: int = 5_000,
):
    """
    Load dataset from local parquet files if present, else download from Hugging Face.
    """
    train_pattern = os.path.join(data_dir, "train-*.parquet")
    val_pattern = os.path.join(data_dir, "validation-*.parquet")
    
    train_files = sorted(glob.glob(train_pattern))
    val_files = sorted(glob.glob(val_pattern))
    
    if train_files and val_files:
        train_files_to_load = train_files[:1] if test_mode else train_files
        print(f"Loading local parquet dataset from {data_dir} ({len(train_files_to_load)} train, {len(val_files)} val files)...")
        dataset = load_dataset(
            "parquet",
            data_files={"train": train_files_to_load, "validation": val_files},
        )
        train_split = dataset["train"]
        val_split = dataset["validation"]
    else:
        print("Local parquet files not found; loading dataset from Hugging Face hub (roneneldan/TinyStories)...")
        train_split = load_dataset("roneneldan/TinyStories", split="train")
        val_split = load_dataset("roneneldan/TinyStories", split="validation")
        
    if test_mode:
        train_count = min(500, len(train_split))
        val_count = min(100, len(val_split))
    else:
        train_count = min(max_train_samples, len(train_split))
        val_count = min(max_val_samples, len(val_split))
        
    print(f"Selecting {train_count:,} train and {val_count:,} validation samples.")
    train_subset = train_split.select(range(train_count))
    val_subset = val_split.select(range(val_count))
    
    return train_subset, val_subset

def tokenize_dataset(
    train_dataset,
    val_dataset,
    tokenizer,
    max_length: int = 512,
    num_proc: int = 0,
):
    """Tokenize train and validation datasets with padding/truncation."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    print(f"Tokenizing datasets (max_length={max_length}, num_proc={num_proc})...")
    tokenized_train = train_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc if num_proc > 0 else None,
        remove_columns=["text"],
    )
    tokenized_val = val_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc if num_proc > 0 else None,
        remove_columns=["text"],
    )
    return tokenized_train, tokenized_val

def generate_story(
    model,
    tokenizer,
    prompt: str = "Once upon a time, there was a little dog named Spot who loved to",
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate story text from prompt using model and tokenizer."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def run_training(
    output_dir: str = "./tinystories-checkpoints",
    final_dir: str = "./tinystories-final",
    test_run: bool = False,
    max_steps: int = 10_000,
    batch_size: int = 32,
    grad_accum: int = 2,
    learning_rate: float = 5e-4,
    data_dir: str = "./data",
):
    """Main training routine."""
    torch.manual_seed(42)
    device = get_device()
    use_cuda = (device == "cuda")
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    
    print(f"Hardware Setup: device={device}, use_bf16={use_bf16}")
    if use_cuda:
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Model
    model = build_model(vocab_size=len(tokenizer))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized LLaMA model from scratch with {total_params / 1e6:.2f}M trainable parameters.")
    
    # 3. Data
    if test_run:
        output_dir = "./tinystories-test-checkpoints"
        final_dir = "./tinystories-test-final"
        steps = 50
        eval_steps = 25
        save_steps = 50
        logging_steps = 10
        train_ds, val_ds = get_dataset(data_dir=data_dir, test_mode=True)
    else:
        steps = max_steps
        eval_steps = 1_000
        save_steps = 2_500
        logging_steps = 100
        train_ds, val_ds = get_dataset(data_dir=data_dir, test_mode=False)
        
    tokenized_train, tokenized_val = tokenize_dataset(
        train_ds,
        val_ds,
        tokenizer,
        max_length=512,
        num_proc=0,  # Safe on Windows without multiprocess fork issues
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=min(500, max(5, steps // 10)),
        max_steps=steps,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16 and use_cuda,
        dataloader_num_workers=0,  # Direct and safe for Windows
        weight_decay=0.01,
        logging_first_step=True,
        report_to="none",
    )
    
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    print(f"\nBeginning training ({steps} steps)...")
    trainer.train()
    
    # 6. Save Model
    print(f"\nSaving final model and tokenizer to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved successfully to {final_dir}")
    
    # 7. Generation Sanity Check
    print("\n--- Model Output Generation Test ---")
    sample_prompt = "Once upon a time, there was a little dog named Spot who loved to"
    story = generate_story(model, tokenizer, prompt=sample_prompt, max_new_tokens=100)
    print(f"Generated Story:\n{story}\n")
    return final_dir

def main():
    parser = argparse.ArgumentParser(description="TinyStories Training and Generation Pipeline")
    parser.add_argument("--test-run", action="store_true", help="Run quick 50-step end-to-end verification")
    parser.add_argument("--train", action="store_true", help="Run full training")
    parser.add_argument("--generate", action="store_true", help="Generate story from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="./tinystories-final", help="Path to checkpoint directory for generation")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little dog named Spot who loved to", help="Prompt for generation")
    parser.add_argument("--max-steps", type=int, default=10_000, help="Max steps for training")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing parquet files")
    
    args = parser.parse_args()
    
    if args.generate:
        if not os.path.exists(args.checkpoint_dir):
            print(f"Error: Checkpoint directory '{args.checkpoint_dir}' does not exist.")
            return
        print(f"Loading model and tokenizer from '{args.checkpoint_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
        model = LlamaForCausalLM.from_pretrained(args.checkpoint_dir)
        device = get_device()
        model.to(device)
        print(f"\nPrompt: {args.prompt}")
        story = generate_story(model, tokenizer, prompt=args.prompt, max_new_tokens=120)
        print(f"\nResult:\n{story}\n")
    elif args.test_run:
        run_training(test_run=True, batch_size=args.batch_size, data_dir=args.data_dir)
    elif args.train:
        run_training(
            test_run=False,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
        )
    else:
        # Default behavior: run test-run if no arguments provided
        print("No specific mode specified. Running quick --test-run (50 steps) to verify pipeline...")
        run_training(test_run=True, batch_size=args.batch_size, data_dir=args.data_dir)

if __name__ == "__main__":
    main()