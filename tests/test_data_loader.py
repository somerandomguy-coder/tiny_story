import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

def test_local_data_loader():
    from train import get_dataset, tokenize_dataset
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
