# TinyStories Implementation & Comparison

A lightweight implementation of the TinyStories language model architecture from scratch using PyTorch and Hugging Face Transformers. Designed to train small models (1M, 3M, and 49M parameters) that speak coherent, grammatical English.

## Features

- **Architectures**:
  - **1M Model (`--model-size 1m`)**: 4 layers, 128 hidden dim, 344 intermediate dim, 4 attention heads (~0.99M parameters). Uses a custom 2,048-token domain BPE tokenizer. Trains ultra-fast (~8-14 steps/s on modern GPUs).
  - **3M Model (`--model-size 3m`)**: 6 layers, 192 hidden dim, 512 intermediate dim, 6 attention heads (~3.23M parameters). Uses a custom 4,096-token domain BPE tokenizer.
  - **49M Model (`--model-size 49m`)**: 8 layers, 512 hidden dim, 1376 intermediate dim, 8 attention heads (~48.94M parameters). Uses standard GPT-2 tokenizer.
- **Before vs. After Comparison Tool (`compare.py`)**: Runs side-by-side prompt comparisons between an untrained scratch model (random initialization) and a trained model checkpoint.
- **Checkpoint Resumption**: Save checkpoints every N steps (`--save-steps 100`) and resume seamlessly with `--resume`.
- **Hardware Acceleration**: Automatic CUDA / BF16 detection with support for modern GPUs (including RTX 50-series Blackwell).

---

## Quickstart

### 1. Compare Current Model (Before vs. After)
Compare an untrained scratch baseline with your trained checkpoint:
```bash
python compare.py
```
Or specify a particular checkpoint:
```bash
python compare.py --checkpoint ./tinystories-checkpoints/checkpoint-2900
```

### 2. Train a 1M Model
Train the 1M parameter model from scratch:
```bash
python train.py --train --model-size 1m --max-steps 1000 --save-steps 100
```
Resume training an existing 1M run:
```bash
python train.py --train --model-size 1m --resume --max-steps 2000
```

### 3. Generate Stories
Generate a story from a trained model:
```bash
python train.py --generate --checkpoint-dir ./tinystories-final --prompt "Once upon a time, there was a little puppy named Max"
```

---

## Comparison Sample

```
[Prompt]: "Once upon a time, there was a little dog named Spot who loved to"

>>> BEFORE TRAINING (Scratch / Random Initialization):
Once upon a time, there was a little dog named Spot who loved to rookiesHonicult Ralph give 
talents clouds Adventure147 Mik262 Dud SmartARDSTes Calebomething empirerawling emotions...

>>> AFTER TRAINING (tinystories-final / Step 2900):
Once upon a time, there was a little dog named Spot who loved to play outside. He would run 
and jump in the park every day for hours. One sunny day, while playing, Spot heard a loud 
noise coming from the park...
```
