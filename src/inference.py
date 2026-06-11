#!/usr/bin/env python3
"""
Inference Script
================
Loads the fine-tuned transaction categoriser from HuggingFace Hub
and classifies a bank transaction description.

Used by:
  - Docker container  (docker run ... -e INPUT_TEXT="...")
  - GitHub Actions    (.github/workflows/inference.yml)
  - Local testing     (python src/inference.py)

Environment variables:
  HF_MODEL_REPO  — HuggingFace model repo  (default: fahadkamraan/transaction-categorizer)
  INPUT_TEXT     — Transaction string to classify (required)
  HF_TOKEN       — HuggingFace token for private repos (optional for public model)
"""

import os
import sys
import textwrap

from huggingface_hub import login
from transformers import pipeline


HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO", "fahadkamraan/transaction-categorizer"
)
INPUT_TEXT = os.environ.get("INPUT_TEXT", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
TOP_K = int(os.environ.get("TOP_K", "3"))


def authenticate() -> None:
    if HF_TOKEN:
        login(token=HF_TOKEN)


def validate_input(text: str) -> None:
    if not text:
        print(
            "Error: INPUT_TEXT environment variable is not set or empty.\n"
            "Usage: INPUT_TEXT='[debit] STARBUCKS STORE 12345' python src/inference.py",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(text) > 512:
        print(
            "Error: INPUT_TEXT exceeds 512 characters — truncate before passing.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_inference(text: str) -> list[dict]:
    print(f"Loading model : {HF_MODEL_REPO}")
    classifier = pipeline(
        task="text-classification",
        model=HF_MODEL_REPO,
        tokenizer=HF_MODEL_REPO,
        device=-1,       # CPU — avoids GPU dependency in CI
        top_k=TOP_K,
    )
    result = classifier(text)
    # pipeline with top_k returns [[dict, ...]] for a single string input
    return result[0] if result and isinstance(result[0], list) else result


def display(text: str, predictions: list[dict]) -> None:
    border = "─" * 52
    print(f"\n{border}")
    print(f"  Transaction : {textwrap.shorten(text, width=42)}")
    print(border)
    for rank, pred in enumerate(predictions, start=1):
        bar_len = int(pred['score'] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {rank}. {pred['label']:<22s}  {bar}  {pred['score']:.4f}")
    print(border)


def main() -> None:
    authenticate()
    validate_input(INPUT_TEXT)
    predictions = run_inference(INPUT_TEXT)
    display(INPUT_TEXT, predictions)


if __name__ == "__main__":
    main()
