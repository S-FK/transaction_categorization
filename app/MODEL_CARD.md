---
language: en
license: mit
tags:
  - text-classification
  - finance
  - banking
  - distilbert
  - mlops
datasets:
  - DoDataThings/us-bank-transaction-categories-v2
metrics:
  - accuracy
  - f1
base_model: distilbert-base-uncased
pipeline_tag: text-classification
---

# Transaction Categoriser

Fine-tuned **DistilBERT** model that classifies raw bank transaction descriptions
into **17 merchant spending categories** with 99.88% test accuracy.

**Try it live:** [🤗 Spaces Demo](https://huggingface.co/spaces/fahadkamraan/transaction-categorizer-demo)

---

## Model Details

| Property | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Multi-class text classification (17 classes) |
| Parameters | 66M |
| Training platform | Kaggle (GPU T4 x2) |
| Framework | HuggingFace Transformers 4.44.2 |
| Inference | CPU-compatible |

## Labels

```python
{
  0: "Education",       1: "Entertainment",   2: "Fees",
  3: "Groceries",       4: "Healthcare",       5: "Income",
  6: "Insurance",       7: "Mortgage",         8: "Personal Care",
  9: "Rent",           10: "Restaurants",     11: "Shopping",
 12: "Subscription",   13: "Transfer",        14: "Transportation",
 15: "Travel",         16: "Utilities"
}
```

## Performance

Evaluated on a held-out test set of 4,298 samples (10% of 42,975 total).

| Metric | V1 (baseline) | **V2 (this model)** |
|---|---|---|
| Accuracy | 99.70% | **99.88%** |
| F1 (weighted) | 99.70% | **99.88%** |
| F1 (macro) | 99.71% | **99.89%** |

All experiments tracked on [W&B](https://wandb.ai/fahadkamraan_sfk/mlops-transaction-classifier).

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="fahadkamraan/transaction-categorizer",
    device=-1,   # CPU
    top_k=3,
)

results = classifier("starbucks coffee purchase")
# [{'label': 'Restaurants', 'score': 0.9999}]
```

### With HuggingFace Hub

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("fahadkamraan/transaction-categorizer")
model = AutoModelForSequenceClassification.from_pretrained("fahadkamraan/transaction-categorizer")

inputs = tokenizer("netflix monthly subscription", return_tensors="pt", truncation=True, max_length=64)
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class = logits.argmax(-1).item()
print(model.config.id2label[predicted_class])  # → Subscription
```

### Docker

```bash
docker run --rm \
  -e HF_TOKEN=<token> \
  -e INPUT_TEXT="delta airlines flight booking" \
  fahadkamraan/mlops-transaction-classifier:latest
```

## Training Data

- **Dataset:** [`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)
- **Licence:** MIT
- **Raw size:** 68,000 rows · 17 categories
- **Used:** 42,975 rows (up to 3,000 per class after deduplication)
- **Split:** 80% train / 10% val / 10% test

## Training Procedure

Two versions were trained and compared on Kaggle (GPU T4 x2):

| Hyperparameter | V1 | **V2 (this model)** |
|---|---|---|
| Learning rate | 2e-5 | **5e-5** |
| Epochs | 3 | **5** |
| Weight decay | 0.0 | **0.01** |
| Preprocessing | Light | **Extra normalisation** |
| Batch size | 32 | 32 |

**V2 preprocessing** additionally strips `[debit]`/`[credit]` prefix tags and standardises
store number suffixes, giving the model cleaner input.

## Limitations

- Supports only the **17 categories** present in the training dataset. Transactions from
  categories like "Investments" or "Taxes" will be mapped to the nearest available label.
- Trained on US-style bank transaction descriptions. May generalise poorly to non-English
  or very different regional bank formats.
- Maximum input length: **64 tokens** (sufficient for typical transaction strings).

## MLOps Pipeline

This model is part of a complete MLOps assignment pipeline:

- **GitHub:** [S-FK/transaction_categorization](https://github.com/S-FK/transaction_categorization)
- **Docker:** [fahadkamraan/mlops-transaction-classifier](https://hub.docker.com/r/fahadkamraan/mlops-transaction-classifier)
- **W&B:** [mlops-transaction-classifier](https://wandb.ai/fahadkamraan_sfk/mlops-transaction-classifier)
- **Spaces Demo:** [transaction-categorizer-demo](https://huggingface.co/spaces/fahadkamraan/transaction-categorizer-demo)

## Citation

```bibtex
@misc{kamraan2025transactioncategorizer,
  author    = {S Fahad Kamraan and Dhruvi Patel and Mahesh V and Himanshu Choubey},
  title     = {Bank Transaction Merchant Categoriser},
  year      = {2025},
  publisher = {HuggingFace Hub},
  url       = {https://huggingface.co/fahadkamraan/transaction-categorizer}
}
```

*IIT Jodhpur — ML Ops Assignment (CSL 7040)*
