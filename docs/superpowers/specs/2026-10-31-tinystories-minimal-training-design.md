# TinyStories Minimal Architecture & Training Pipeline Design

**Date:** 2026-10-31  
**Status:** Approved  
**Author:** Pair Programming Agent & User  

---

## 1. Overview & Goals

The goal of this project is to create a reliable, streamlined training and generation pipeline for the **TinyStories** dataset, making the model as small and efficient as possible while speaking fluent, coherent English.

### Key Objectives:
1. **Fix Critical Execution Bugs**:
   - Install missing `datasets` package.
   - Upgrade PyTorch to CUDA 12.6 to utilize the **NVIDIA GeForce RTX 5060 Ti (16GB VRAM)**, replacing CPU execution with GPU acceleration (50–100x speedup).
   - Add proper Windows entry-point guards (`if __name__ == '__main__':`) to prevent recursive process-spawning crashes in Windows multiprocessing.
2. **Eliminate Redundant Network Downloads**:
   - Utilize local parquet files in `data/` (`train-*.parquet` and `validation-*.parquet`) rather than re-downloading ~1GB from Hugging Face Hub on every run.
   - Fall back to Hugging Face download automatically if local files are absent.
3. **Model Architecture (~29.4M parameters)**:
   - Modern LLaMA Causal LM backbone: 8 transformer layers, 512 hidden dimension, 8 query heads, 4 key-value heads (Grouped-Query Attention), SwiGLU intermediate dimension of 1,376, tied word embeddings, GPT-2 BPE vocabulary (50,257 tokens).
4. **All-in-One CLI Workflow**:
   - Provide `--test-run` for instant sanity testing (100 steps, < 30 seconds).
   - Provide standard training with evaluation and checkpoint saving.
   - Provide `--generate` to interactively sample stories from any saved model checkpoint.

---

## 2. Hardware & Environment Specifications

- **OS:** Windows 11
- **GPU:** NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **CUDA Runtime:** CUDA 12.6 (`torch-2.14.1+cu126`)
- **Precision:** `bf16` (supported on RTX 50-series Blackwell) with fallback to `fp16`
- **Key Libraries:** `torch`, `transformers`, `datasets`, `pyarrow`

---

## 3. Data Pipeline

1. **Detection:**
   - Script checks for `data/train-*.parquet` and `data/validation-*.parquet`.
   - If present, loads directly via:
     ```python
     load_dataset("parquet", data_files={"train": train_files, "validation": val_files})
     ```
   - If absent, loads via Hugging Face `roneneldan/TinyStories`.
2. **Preprocessing:**
   - GPT-2 tokenizer with `pad_token = eos_token`.
   - Max sequence length: 512 tokens.
   - Subsetting:
     - Full training: up to 200,000 samples (or configurable via CLI).
     - Test run mode: 500 samples.
   - Windows-safe tokenization map:
     - Wrapped inside `main()` under `if __name__ == '__main__':`.
     - `num_proc` set adaptively (up to 4, or 0 if single-process test).

---

## 4. Model Architecture & Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Model Type | `LlamaForCausalLM` | Modern architecture with RoPE, RMSNorm, SwiGLU, and GQA |
| Hidden Size | 512 | Balances capacity with fast training |
| Layers | 8 | Sufficient depth for causal narrative structure and grammar |
| Attention Heads | 8 | Standard multi-head attention |
| KV Heads | 4 | Grouped-Query Attention (GQA) for memory & throughput efficiency |
| Intermediate Size | 1,376 | SwiGLU standard ~8/3 * hidden_size |
| Vocabulary Size | 50,257 | Standard GPT-2 tokenizer |
| Tie Word Embeddings | True | Saves 25.7M parameters by reusing input embedding table as output projection |
| Context Length | 512 | Fits typical TinyStories length (average ~150–250 tokens) |
| Total Parameters | ~29.4M | Optimal for grammatically fluent English with simple plot structures |

---

## 5. Training Configuration

- **Optimizer & Scheduler:** AdamW (`weight_decay=0.01`), cosine learning rate scheduler with 500 warmup steps.
- **Learning Rate:** `5e-4` (optimal for small transformers trained from scratch).
- **Batch Size:**
  - Per-device batch size: 32
  - Gradient accumulation steps: 2 (effective batch size: 64)
- **Precision:** `bf16=True`
- **Checkpoints:**
  - Intermediate checkpoints saved to `./tinystories-28m-checkpoints`
  - Final model and tokenizer saved to `./tinystories-28m-final`

---

## 6. CLI & Execution Interface

```bash
# Sanity test run (50-100 steps, < 30 seconds to verify GPU, data loader, loss drop, and generation)
python train.py --test-run

# Standard training run
python train.py --train --max-steps 10000 --batch-size 32

# Interactive story generation from final model or checkpoint
python train.py --generate --prompt "Once upon a time, there was a little dog named Spot"
```

---

## 7. Verification & Success Criteria

1. `torch.cuda.is_available()` returns `True`, identifying the RTX 5060 Ti.
2. Dataset loads directly from local `data/` parquet without network delay.
3. Windows multiprocessing runs cleanly without process fork bombs or crashes.
4. `--test-run` executes 50 steps, logs decreasing loss, saves a test checkpoint, and outputs a generated text sample.
5. Generated output adheres to basic English syntax and vocabulary.
