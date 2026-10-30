import os
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

# 1. Hardware checks & Seed
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# 2. Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. Model Architecture Config (~29.4M parameters)
config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=512,
    intermediate_size=1376,  # SwiGLU 8/3 * hidden_size rounded
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,   # Grouped-Query Attention
    max_position_embeddings=512,
    tie_word_embeddings=True,  # Crucial: avoids doubling parameter size for embedding table
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = LlamaForCausalLM(config)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Initialized Llama model from scratch with {total_params / 1e6:.2f}M trainable parameters.")

# 4. Dataset Loading & Preprocessing
print("Streaming dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")

# Subsample for fast iteration (adjust or remove slicing to train on full 2.1M dataset)
train_subset = dataset.select(range(200_000))
eval_subset = eval_dataset.select(range(5_000))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )

print("Tokenizing datasets...")
tokenized_train = train_subset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"],
)
tokenized_eval = eval_subset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal Language Modeling
)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./tinystories-28m-checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effective batch size = 64
    learning_rate=5e-4,             # Higher LR is standard for small transformers from scratch
    min_lr_ratio=0.1,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    max_steps=10_000,               # ~640,000 samples seen
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=1_000,
    save_steps=2_500,
    save_total_limit=2,
    bf16=use_bf16,
    fp16=not use_bf16 and torch.cuda.is_available(),
    dataloader_num_workers=4,
    weight_decay=0.01,
    logging_first_step=True,
    report_to="none",               # Set to "wandb" or "tensorboard" if configured
)

# 6. Trainer Execution
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

print("Beginning training...")
trainer.train()

# 7. Save Final Checkpoint
final_dir = "./tinystories-28m-final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Model saved to {final_dir}")

# 8. Generation Sanity Check
print("\n--- Model Output Test ---")
prompt = "Once upon a time, there was a little dog named Spot who loved to"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.1,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))