#!/usr/bin/env python3
"""
Task 3 — Select & Load a Model from HuggingFace Hub
=====================================================
Loads the tokenizer and DistilBERT classification head using the
label mapping in data/id2label.json.

Run standalone to verify the model loads correctly:
    python src/load_model.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_MODEL = "distilbert-base-uncased"
ID2LABEL_PATH = Path(__file__).resolve().parent.parent / "data" / "id2label.json"


def load_id2label(path: Path = ID2LABEL_PATH) -> dict[str, str]:
    """Return {str_int → label_str} mapping from id2label.json."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/prepare_data.py` first."
        )
    with open(path) as f:
        return json.load(f)


def load_tokenizer(model_name: str = BASE_MODEL):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(
    model_name: str = BASE_MODEL,
    id2label: dict[str, str] | None = None,
):
    """Load DistilBERT with a classification head sized to num_labels."""
    if id2label is None:
        id2label = load_id2label()

    int_id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=int_id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


if __name__ == "__main__":
    id2label = load_id2label()
    tokenizer = load_tokenizer()
    model = load_model(id2label=id2label)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model            : {BASE_MODEL}")
    print(f"Num labels       : {model.config.num_labels}")
    print(f"Labels           : {list(id2label.values())}")
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    # Quick tokenisation smoke-test
    sample = "[debit] AMAZON MKTPL*K8R2M5VN7"
    tokens = tokenizer(sample, return_tensors="pt", truncation=True, max_length=64)
    print(f"\nSample input     : {sample!r}")
    print(f"Token IDs shape  : {tokens['input_ids'].shape}")
    print("load_model.py passed all checks.")
