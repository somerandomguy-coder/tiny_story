# TinyStories Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean, reliable, Windows-safe, GPU-accelerated training and generation pipeline for TinyStories with an all-in-one CLI interface.

**Architecture:** Use PyTorch with CUDA 12.6 on RTX 5060 Ti, Hugging Face `transformers` (LLaMA architecture ~29.4M parameters), `datasets` loading directly from local `data/*.parquet` files, and an all-in-one CLI in `train.py` supporting `--test-run`, `--train`, and `--generate`.

**Tech Stack:** Python 3.14, PyTorch (CUDA 12.6), Transformers, Datasets, PyArrow, argparse.

## Global Constraints
- Target GPU: NVIDIA GeForce RTX 5060 Ti (16GB VRAM) using CUDA 12.6 and `bf16`/`fp16` mixed precision.
- Data Source: Auto-detect local `data/train-*.parquet` and `data/validation-*.parquet` files first; fallback to HF `roneneldan/TinyStories` if absent.
- Multiprocessing safety: Must include `if __name__ == '__main__':` on Windows to avoid process-spawning crashes.
- Model architecture: LLaMA Causal LM (~29.4M params: 8 layers, hidden_size 512, 8 attention heads, 4 KV heads GQA, tied embeddings, GPT-2 tokenizer).

---

### Task 1: Environment & CUDA PyTorch Setup

**Files:**
- Test/Verification: `tests/test_env.py`
- Modify: Python environment (install `torch-2.14.1+cu126` and `datasets`)

**Interfaces:**
- Produces: Working PyTorch environment with `torch.cuda.is_available() == True` and `datasets` importable.

- [ ] **Step 1: Write verification script `tests/test_env.py`**

```python
import torch
import datasets
import transformers

def test_cuda_and_libraries():
    assert torch.cuda.is_available(), "CUDA must be available"
    device_name = torch.cuda.get_device_name(0)
    assert "RTX" in device_name or "GeForce" in device_name, f"Expected RTX GPU, got {device_name}"
    print(f"CUDA verified: {device_name}, Torch: {torch.__version__}, Transformers: {transformers.__version__}, Datasets: {datasets.__version__}")

if __name__ == "__main__":
    test_cuda_and_libraries()
```

- [ ] **Step 2: Run verification script to confirm it fails on current CPU environment**

Run: `python tests/test_env.py`  
Expected: FAIL with `AssertionError: CUDA must be available` or `ModuleNotFoundError: No module named 'datasets'`

- [ ] **Step 3: Install CUDA-enabled PyTorch and datasets**

Run:
```powershell
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install datasets
```

- [ ] **Step 4: Run verification script to confirm it passes on GPU**

Run: `python tests/test_env.py`  
Expected: PASS with "CUDA verified: NVIDIA GeForce RTX 5060 Ti"

- [ ] **Step 5: Commit**

```bash
git add tests/test_env.py
git commit -m "chore: setup and verify PyTorch CUDA environment"
```

---

### Task 2: Dataset Loading & Preprocessing Unit

**Files:**
- Test: `tests/test_data_loader.py`
- Modify: `train.py` (implement `get_dataset` and `tokenize_dataset`)

**Interfaces:**
- Produces: `get_dataset(data_dir, test_mode)` returning train and eval splits, and `tokenize_dataset(dataset, tokenizer, max_length)` returning tokenized dataset.

- [ ] **Step 1: Write test for dataset loading and tokenization**

```python
import os
from train import get_dataset, tokenize_dataset
from transformers import AutoTokenizer

def test_local_data_loader():
    train_ds, val_ds = get_dataset(data_dir="./data", test_mode=True)
    assert len(train_ds) > 0, "Train dataset should have samples"
    assert len(val_ds) > 0, "Validation dataset should have samples"
    assert "text" in train_ds.column_names, "Dataset must contain 'text' column"
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_train, tokenized_val = tokenize_dataset(train_ds, val_ds, tokenizer, max_length=128, num_proc=0)
    assert "input_ids" in tokenized_train.column_names
    assert "attention_mask" in tokenized_train.column_names

if __name__ == "__main__":
    test_local_data_loader()
    print("Dataset loader tests passed successfully!")
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `python tests/test_data_loader.py`  
Expected: FAIL with `ImportError: cannot import name 'get_dataset'`

- [ ] **Step 3: Implement `get_dataset` and `tokenize_dataset` in `train.py`**

Implement local parquet globbing with fallback to Hugging Face Hub, and batched tokenization.

- [ ] **Step 4: Run test to confirm it passes**

Run: `python tests/test_data_loader.py`  
Expected: PASS with "Dataset loader tests passed successfully!"

- [ ] **Step 5: Commit**

```bash
git add tests/test_data_loader.py train.py
git commit -m "feat: implement local parquet dataset loader and tokenizer"
```

---

### Task 3: Model Architecture, CLI Interface, & Generation Routine

**Files:**
- Modify: `train.py`
- Test: `tests/test_model_and_cli.py`

**Interfaces:**
- Consumes: `get_dataset`, `tokenize_dataset`
- Produces: `build_model(vocab_size)`, `generate_story(model, tokenizer, prompt, max_new_tokens)`, and CLI arguments (`--test-run`, `--train`, `--generate`).

- [ ] **Step 1: Write test for model creation and generation routine**

```python
import torch
from transformers import AutoTokenizer
from train import build_model, generate_story

def test_model_build_and_generation():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = build_model(vocab_size=len(tokenizer))
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Check that model is around 29.4M parameters
    assert 28_000_000 < total_params < 31_000_000, f"Expected ~29.4M params, got {total_params}"
    
    # Test generation function with untransformed model
    output = generate_story(model, tokenizer, prompt="Once upon a time", max_new_tokens=10)
    assert isinstance(output, str)
    assert len(output) > len("Once upon a time")

if __name__ == "__main__":
    test_model_build_and_generation()
    print("Model build and generation tests passed successfully!")
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `python tests/test_model_and_cli.py`  
Expected: FAIL with `ImportError: cannot import name 'build_model'`

- [ ] **Step 3: Complete `train.py` implementation**

Implement `build_model`, `generate_story`, `main()` with `argparse`, training loop with HuggingFace `Trainer`, and proper `if __name__ == '__main__':` block.

- [ ] **Step 4: Run test to confirm it passes**

Run: `python tests/test_model_and_cli.py`  
Expected: PASS with "Model build and generation tests passed successfully!"

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_and_cli.py train.py
git commit -m "feat: complete model config, generation, and CLI interface"
```

---

### Task 4: End-to-End Test Run Verification

**Files:**
- Run: `train.py` with `--test-run`

**Interfaces:**
- Produces: Executed sanity training run (50 steps), checkpoint directory, and printed generated story sample.

- [ ] **Step 1: Execute `--test-run`**

Run: `python train.py --test-run`  
Expected:
- Detects local parquet files from `data/`.
- Initializes 29.4M parameter LLaMA model on CUDA device (`NVIDIA GeForce RTX 5060 Ti`).
- Trains for 50 steps with decreasing loss logs.
- Saves model and tokenizer to `./tinystories-test-final`.
- Generates a sample story to stdout.

- [ ] **Step 2: Verify interactive `--generate` flag**

Run: `python train.py --generate --checkpoint-dir ./tinystories-test-final --prompt "Once upon a time, there was a little cat named Mia"`  
Expected: Generates a story continuation starting from the prompt.

- [ ] **Step 3: Commit and clean up test artifacts**

```bash
git add train.py
git commit -m "test: verify end-to-end training and generation on RTX 5060 Ti"
```
