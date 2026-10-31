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
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

def get_device():
    """Detect available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_or_train_tokenizer(model_size: str = "49m", data_dir: str = "./data"):
    """
    Get standard GPT-2 tokenizer for 49M model, or train/load a domain-specific
    compact BPE tokenizer for 1M (2k vocab) or 3M (4k vocab) models.
    """
    if model_size == "49m":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    vocab_size = 2048 if model_size == "1m" else 4096
    tok_dir = f"./tokenizer-{vocab_size // 1024 if vocab_size >= 1024 else vocab_size}k"
    if model_size == "1m":
        tok_dir = "./tokenizer-2k"
    elif model_size == "3m":
        tok_dir = "./tokenizer-4k"

    # Check if existing tokenizer is already present
    if os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        print(f"Loading existing custom tokenizer from '{tok_dir}' (vocab size ~{vocab_size})...")
        return AutoTokenizer.from_pretrained(tok_dir)

    print(f"Building custom domain BPE tokenizer ({vocab_size} tokens) for {model_size.upper()} model...")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    if train_files:
        dataset = load_dataset("parquet", data_files={"train": train_files[:1]})["train"]
    else:
        dataset = load_dataset("roneneldan/TinyStories", split="train")

    sample_count = min(30000, len(dataset))
    def batch_iterator():
        for i in range(0, sample_count, 1000):
            yield dataset[i : i + 1000]["text"]

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        batch_iterator(),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<pad>"],
    )

    os.makedirs(tok_dir, exist_ok=True)
    tok.save(os.path.join(tok_dir, "tokenizer.json"))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tok_dir, "tokenizer.json"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
    )
    hf_tok.save_pretrained(tok_dir)
    print(f"Saved custom {vocab_size}-token tokenizer to '{tok_dir}'.")
    return hf_tok

def build_model(
    vocab_size: int = 50257,
    model_size: str = "49m",
    pad_token_id: int = None,
    bos_token_id: int = None,
    eos_token_id: int = None,
) -> LlamaForCausalLM:
    """
    Construct lightweight LLaMA model from scratch.
    Configurations:
      - 1m: 4 layers, 128 hidden, 344 intermediate, 4 heads (~0.99M params)
      - 3m: 6 layers, 192 hidden, 512 intermediate, 6 heads (~3.23M params)
      - 49m: 8 layers, 512 hidden, 1376 intermediate, 8 heads (~48.94M params)
    """
    if model_size == "1m":
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=344,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            tie_word_embeddings=True,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    elif model_size == "3m":
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=192,
            intermediate_size=512,
            num_hidden_layers=6,
            num_attention_heads=6,
            num_key_value_heads=3,
            max_position_embeddings=256,
            tie_word_embeddings=True,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    else:  # "49m" or default
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=512,
            intermediate_size=1376,       # SwiGLU ~8/3 * hidden_size
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,        # Grouped-Query Attention
            max_position_embeddings=512,
            tie_word_embeddings=True,     # Avoids duplicating embedding matrix for output projection
            pad_token_id=pad_token_id or 50256,
            bos_token_id=bos_token_id or 50256,
            eos_token_id=eos_token_id or 50256,
        )
    model = LlamaForCausalLM(config)
    return model

def get_dataset(
    data_dir: str = "./data",
    test_mode: bool = False,
    max_train_samples: int = 200_000,
    max_val_samples: int = 500,
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
    max_length: int = 256,
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
    
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=pad_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

class SaveTokenizerAndSampleCallback(TrainerCallback):
    """Callback to save tokenizer into each checkpoint folder and display live sample."""
    def __init__(self, tokenizer, sample_prompt: str = "Once upon a time, there was a little"):
        self.tokenizer = tokenizer
        self.sample_prompt = sample_prompt

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_folder):
            self.tokenizer.save_pretrained(checkpoint_folder)
        if model is not None:
            print(f"\n--- [Checkpoint at Step {state.global_step}] Live Story Sample ---")
            story = generate_story(model, self.tokenizer, prompt=self.sample_prompt, max_new_tokens=60)
            print(f"{story}\n---------------------------------------------------------------\n")

def run_training(
    model_size: str = "49m",
    output_dir: str = None,
    final_dir: str = None,
    test_run: bool = False,
    max_steps: int = 1_000,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 20,
    save_total_limit: int = 3,
    batch_size: int = 32,
    grad_accum: int = 2,
    max_length: int = 256,
    learning_rate: float = None,
    data_dir: str = "./data",
    resume: bool = False,
):
    """Main training routine."""
    torch.manual_seed(42)
    device = get_device()
    use_cuda = (device == "cuda")
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    
    # Defaults directories based on model size
    if output_dir is None:
        output_dir = "./tinystories-checkpoints" if model_size == "49m" else f"./tinystories-{model_size}-checkpoints"
    if final_dir is None:
        final_dir = "./tinystories-final" if model_size == "49m" else f"./tinystories-{model_size}-final"

    # Default learning rate based on model size (smaller models train better with slightly higher LR)
    if learning_rate is None:
        learning_rate = 1e-3 if model_size in ("1m", "3m") else 5e-4
    
    print(f"Hardware Setup: device={device}, use_bf16={use_bf16}")
    if use_cuda:
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training Model Size: {model_size.upper()} | Learning Rate: {learning_rate}")
    
    # 1. Tokenizer
    tokenizer = get_or_train_tokenizer(model_size=model_size, data_dir=data_dir)
    
    # 2. Model
    model = build_model(
        vocab_size=len(tokenizer),
        model_size=model_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    total_params = sum(p.numel() for p in set(model.parameters()) if p.requires_grad)
    print(f"Initialized {model_size.upper()} LLaMA model from scratch with {total_params / 1e6:.2f}M ({total_params:,}) trainable parameters.")
    
    # 3. Data configuration
    if test_run:
        output_dir = f"./tinystories-{model_size}-test-checkpoints"
        final_dir = f"./tinystories-{model_size}-test-final"
        steps = 50
        eval_steps_val = 25
        save_steps_val = 50
        logging_steps_val = 10
        train_ds, val_ds = get_dataset(data_dir=data_dir, test_mode=True)
    else:
        steps = max_steps
        eval_steps_val = eval_steps
        save_steps_val = save_steps
        logging_steps_val = logging_steps
        train_ds, val_ds = get_dataset(data_dir=data_dir, test_mode=False, max_val_samples=500)
        
    tokenized_train, tokenized_val = tokenize_dataset(
        train_ds,
        val_ds,
        tokenizer,
        max_length=max_length,
        num_proc=0,  # Direct and safe for Windows
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
        logging_steps=logging_steps_val,
        eval_strategy="steps",
        eval_steps=eval_steps_val,
        save_steps=save_steps_val,
        save_total_limit=save_total_limit,
        bf16=use_bf16,
        fp16=not use_bf16 and use_cuda,
        dataloader_num_workers=0,  # Safe for Windows
        weight_decay=0.01,
        logging_first_step=True,
        report_to="none",
    )
    
    # 5. Trainer with Auto Tokenizer Saving & Sampling Callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[SaveTokenizerAndSampleCallback(tokenizer)],
    )
    
    # 6. Checkpoint Resume Handling
    resume_checkpoint = None
    if resume:
        if isinstance(resume, str) and os.path.exists(resume):
            print(f"Resuming training from specified checkpoint: {resume}")
            resume_checkpoint = resume
        else:
            last_ckpt = get_last_checkpoint(output_dir)
            if last_ckpt is not None:
                print(f"Resuming training from latest checkpoint: {last_ckpt}")
                resume_checkpoint = last_ckpt
            else:
                print(f"No previous checkpoint found in '{output_dir}'. Starting fresh training.")

    if resume_checkpoint is not None:
        try:
            ckpt_step = int(os.path.basename(os.path.normpath(resume_checkpoint)).split("-")[-1])
            if steps <= ckpt_step:
                print(f"\n[ATTENTION] Checkpoint '{resume_checkpoint}' is already at step {ckpt_step}!")
                print(f"Current --max-steps is {steps}. To train further beyond step {ckpt_step}, set --max-steps to a higher number (e.g. --max-steps {ckpt_step + 500} or {ckpt_step + 1000}).\n")
        except Exception:
            pass
    
    print(f"\nBeginning training ({steps} steps, checkpoint every {save_steps_val} steps)...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 7. Save Final Model
    print(f"\nSaving final model and tokenizer to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved successfully to {final_dir}")
    
    # 8. Final Generation Test
    print("\n--- Final Model Output Generation Test ---")
    sample_prompt = "Once upon a time, there was a little dog named Spot who loved to"
    story = generate_story(model, tokenizer, prompt=sample_prompt, max_new_tokens=120)
    print(f"Generated Story:\n{story}\n")
    return final_dir

def main():
    parser = argparse.ArgumentParser(description="TinyStories Training and Generation Pipeline")
    parser.add_argument("--model-size", type=str, default="49m", choices=["1m", "3m", "49m"], help="Model architecture size: 1m (~0.99M params), 3m (~3.2M params), or 49m (~48.9M params, default)")
    parser.add_argument("--test-run", action="store_true", help="Run quick 50-step end-to-end verification")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume training (auto-detects latest checkpoint, or specify path)")
    parser.add_argument("--generate", action="store_true", help="Generate story from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to checkpoint directory for generation")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints (defaults based on model-size)")
    parser.add_argument("--final-dir", type=str, default=None, help="Directory to save final model (defaults based on model-size)")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little dog named Spot who loved to", help="Prompt for generation")
    parser.add_argument("--max-steps", type=int, default=1_000, help="Max steps for training (default: 1000)")
    parser.add_argument("--save-steps", type=int, default=100, help="Save a checkpoint every X steps (default: 100)")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluate every X steps (default: 100)")
    parser.add_argument("--logging-steps", type=int, default=20, help="Log loss every X steps (default: 20)")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Max checkpoints to keep on disk (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (default: 1e-3 for 1m/3m, 5e-4 for 49m)")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length (default: 256 for fast training)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory containing parquet files")
    
    args = parser.parse_args()
    
    if args.generate:
        checkpoint_dir = args.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = "./tinystories-final" if args.model_size == "49m" else f"./tinystories-{args.model_size}-final"
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory '{checkpoint_dir}' does not exist.")
            return
        print(f"Loading model and tokenizer from '{checkpoint_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = LlamaForCausalLM.from_pretrained(checkpoint_dir)
        device = get_device()
        model.to(device)
        print(f"\nPrompt: {args.prompt}")
        story = generate_story(model, tokenizer, prompt=args.prompt, max_new_tokens=120)
        print(f"\nResult:\n{story}\n")
    elif args.train:
        run_training(
            model_size=args.model_size,
            test_run=False,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            output_dir=args.output_dir,
            final_dir=args.final_dir,
            data_dir=args.data_dir,
            resume=args.resume,
        )
    elif args.test_run:
        run_training(model_size=args.model_size, test_run=True, batch_size=args.batch_size, data_dir=args.data_dir)
    else:
        # Default behavior: run test-run if no arguments provided
        print(f"No specific mode specified. Running quick --test-run (50 steps) for {args.model_size.upper()} model...")
        run_training(model_size=args.model_size, test_run=True, batch_size=args.batch_size, data_dir=args.data_dir)

if __name__ == "__main__":
    main()
