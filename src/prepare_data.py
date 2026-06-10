#!/usr/bin/env python3
"""
Task 2 — Data Preparation & Normalisation
==========================================
Downloads DoDataThings/us-bank-transaction-categories-v2 from HuggingFace Hub,
inspects the raw data, cleans it, stratified-samples to 51,000 rows
(3,000 per class × 17 classes), encodes labels, and saves:
  - data/id2label.json     (committed to git)
  - data/processed/        (excluded from git via .gitignore)
"""

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "DoDataThings/us-bank-transaction-categories-v2"
SAMPLES_PER_CLASS = 3_000
RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ID2LABEL_PATH = DATA_DIR / "id2label.json"

# Regex patterns for noise removal
_REF_CODE_RE = re.compile(r'\b[A-Z0-9]{6,}\b')
_SPACES_RE = re.compile(r'\s{2,}')
_SPECIAL_RE = re.compile(r'[*#@&]+')


def clean_description(text: str) -> str:
    """
    Normalise a raw bank transaction description string.
    Steps:
      1. Lowercase
      2. Strip trailing reference codes (e.g. K8R2M5VN7, 00042381)
      3. Collapse special characters (*, #, @)
      4. Collapse multiple spaces
    """
    text = str(text).lower()
    text = _REF_CODE_RE.sub('', text)
    text = _SPECIAL_RE.sub(' ', text)
    text = _SPACES_RE.sub(' ', text).strip()
    return text


def inspect(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("RAW DATA INSPECTION")
    print("=" * 50)
    print(f"Shape            : {df.shape}")
    print(f"Columns          : {df.columns.tolist()}")
    print(f"Missing values   :\n{df.isnull().sum()}")
    print(f"Duplicate rows   : {df.duplicated().sum()}")
    print("\nClass distribution (raw):")
    dist = df['category'].value_counts()
    for cat, cnt in dist.items():
        print(f"  {cat:<30s}  {cnt:>5d}")
    print("\nDescription length stats (chars):")
    lengths = df['description'].astype(str).str.len()
    print(f"  min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, median={lengths.median():.0f}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ──────────────────────────────────────────────────────────
    print(f"Loading '{DATASET_ID}' from HuggingFace Hub...")
    raw_ds = load_dataset(DATASET_ID)

    # The dataset has a single 'train' split; work in pandas for EDA
    df = raw_ds['train'].to_pandas()

    # ── 2. Inspect ───────────────────────────────────────────────────────────
    inspect(df)

    # ── 3. Clean ─────────────────────────────────────────────────────────────
    print("\nCleaning descriptions...")
    df = df.drop_duplicates()
    df = df.dropna(subset=['description', 'category'])
    df['description'] = df['description'].apply(clean_description)
    # Drop rows where cleaning left an empty or very short string
    df = df[df['description'].str.len() >= 3].reset_index(drop=True)

    print(f"Shape after cleaning: {df.shape}")

    # ── 4. Stratified subsample → 3,000 per class = 51,000 total ─────────────
    print(f"\nStratified sampling: {SAMPLES_PER_CLASS} per class...")
    df_sampled = (
        df.groupby('category', group_keys=False)
          .apply(lambda g: g.sample(n=min(SAMPLES_PER_CLASS, len(g)), random_state=RANDOM_SEED))
          .reset_index(drop=True)
    )
    print(f"Sampled shape: {df_sampled.shape}")
    print("\nClass distribution after sampling:")
    for cat, cnt in df_sampled['category'].value_counts().items():
        print(f"  {cat:<30s}  {cnt:>5d}")

    # ── 5. Encode labels ──────────────────────────────────────────────────────
    labels = sorted(df_sampled['category'].unique().tolist())
    id2label = {str(i): lbl for i, lbl in enumerate(labels)}
    label2id = {lbl: i for i, lbl in enumerate(labels)}

    df_sampled['label'] = df_sampled['category'].map(label2id)

    print(f"\nNumber of classes : {len(id2label)}")
    print(f"Labels            : {list(id2label.values())}")

    # ── 6. Save id2label.json (committed to git) ──────────────────────────────
    with open(ID2LABEL_PATH, 'w') as f:
        json.dump(id2label, f, indent=2)
    print(f"\nSaved: {ID2LABEL_PATH}  ← commit this file")

    # ── 7. Train / val / test split (80 / 10 / 10) ───────────────────────────
    cols = ['description', 'label', 'category']
    train_df, temp_df = train_test_split(
        df_sampled[cols],
        test_size=0.20,
        stratify=df_sampled['label'],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df['label'],
        random_state=RANDOM_SEED,
    )

    print(f"\nSplit sizes — "
          f"train: {len(train_df):,}  |  "
          f"val: {len(val_df):,}  |  "
          f"test: {len(test_df):,}")

    # ── 8. Save processed CSVs (not committed to git) ─────────────────────────
    train_df.to_csv(PROCESSED_DIR / 'train.csv', index=False)
    val_df.to_csv(PROCESSED_DIR / 'val.csv', index=False)
    test_df.to_csv(PROCESSED_DIR / 'test.csv', index=False)

    print(f"Saved processed splits to {PROCESSED_DIR}/")
    print("\nDone. Only commit data/id2label.json — never commit CSV files.")


if __name__ == '__main__':
    main()
