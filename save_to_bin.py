import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("ag_news")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# Preprocess function


def preprocess(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": tokens["input_ids"].squeeze(0).numpy(),
        "attention_mask": tokens["attention_mask"].squeeze(0).numpy(),
        "label": np.array(example["label"], dtype=np.int64),
    }


# Apply preprocessing
dataset = dataset.map(preprocess)

# Save as raw .bin files
for split in ["train", "test"]:
    input_ids = np.array([x["input_ids"]
                         for x in dataset[split]], dtype=np.int32)
    attention_masks = np.array([x["attention_mask"]
                               for x in dataset[split]], dtype=np.int32)
    labels = np.array([x["label"] for x in dataset[split]], dtype=np.int64)

    # Save each component separately
    input_ids.tofile(f"ag_news_{split}_input_ids.bin")
    attention_masks.tofile(f"ag_news_{split}_attention_masks.bin")
    labels.tofile(f"ag_news_{split}_labels.bin")

print("Saved dataset as raw .bin files")

