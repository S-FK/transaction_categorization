# MLOps Group Project Report
## Bank Transaction Merchant Categorisation

**Institution:** IIT Jodhpur — M.Tech Artificial Intelligence
**Course:** ML Ops
**Team:**

| Name | Roll No. | GitHub |
|---|---|---|
| S Fahad Kamraan | G25AIT2091 | @s-fk |
| Dhruvi Patel | — | @dhruvi9660 |
| Mahesh | — | @Mahesh2058 |
| Himanshu Chouhan | — | — |

---

## 1. Problem Statement

Bank transaction descriptions are raw, noisy strings (e.g., `[debit] AMAZON MKTPL*K8R2M5VN7`).
The goal is to automatically classify each transaction into one of 17 merchant spending categories
to help users understand their spending patterns.

**Categories:**
Auto & Transport, Bills & Utilities, Business Services, Education, Entertainment,
Fees & Charges, Food & Dining, Gifts & Donations, Health & Fitness, Home,
Income, Investments, Personal Care, Shopping, Taxes, Transfer, Travel

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | [`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) |
| Licence | MIT |
| Raw size | ~68,000 rows |
| Used subset | 51,000 rows — stratified 3,000 per class × 17 classes |
| Columns | `description` (str), `category` (str) |
| Split | 80% train / 10% val / 10% test |

### Preprocessing

Two preprocessing strategies were compared:

**V1 — Light clean** (`prepare_data.py` / `train_v1.ipynb`):
1. Lowercase
2. Strip trailing alphanumeric reference codes (`\b[A-Z0-9]{6,}\b`)
3. Collapse special characters and extra whitespace

**V2 — Extra normalisation** (`train_v2.ipynb`):
All V1 steps plus:
1. Strip `[debit]` / `[credit]` prefix tags
2. Standardise store number suffixes (e.g., `STORE 12345` → `STORE`)

---

## 3. Model

| Property | Value |
|---|---|
| Base model | [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) |
| Parameters | ~66 million (66 MB) |
| Architecture | 6-layer DistilBERT + linear classification head (768 → 17) |
| Tokenisation | WordPiece, max_length=64 |
| Task | Sequence classification (17-class) |

DistilBERT was selected because it is 40% smaller and 60% faster than BERT-base while
retaining 97% of BERT's performance on GLUE, making it ideal for CPU inference.

---

## 4. Training Experiments

Both experiments were run on **Kaggle** (free GPU tier — T4) with the HuggingFace `Trainer` API.
Metrics were logged to **Weights & Biases** (`mlops-transaction-classifier` project).

| Hyperparameter | V1 | V2 |
|---|---|---|
| Learning rate | 2e-5 | 5e-5 |
| Epochs | 3 | 5 |
| Weight decay | 0.0 | 0.01 |
| Batch size | 32 | 32 |
| Preprocessing | Light clean | Extra normalisation |
| Early stopping | patience=2 | patience=2 |
| fp16 | Yes (GPU) | Yes (GPU) |

### Results

<!-- Fill in after running training on Kaggle -->

| Metric | V1 | V2 |
|---|---|---|
| Val Accuracy | _TBD_ | _TBD_ |
| Val F1 (macro) | _TBD_ | _TBD_ |
| Test Accuracy | _TBD_ | _TBD_ |
| Test F1 (macro) | _TBD_ | _TBD_ |
| Training time | _TBD_ | _TBD_ |

**W&B Dashboard:** _(link to be added after training)_

### Observations

<!-- Fill in after comparing V1 vs V2 runs in W&B -->

---

## 5. MLOps Pipeline

```
GitHub Repository (develop ← feature/* PRs, CI on every push)
    │
    ├── Task 2: src/prepare_data.py    → data/processed/{train,val,test}.csv
    │
    ├── Task 3: src/load_model.py      → verify model + tokenizer load
    │
    ├── Task 4: notebooks/train_v1.ipynb  ─┐
    │           notebooks/train_v2.ipynb  ─┴→ Kaggle GPU → W&B → HuggingFace Hub
    │
    ├── Task 6: Dockerfile             → Docker Hub image
    │           src/inference.py       ← reads HF_MODEL_REPO env var
    │
    └── Task 7: .github/workflows/
                ├── ci.yml             → flake8 lint on push to develop
                └── inference.yml      → manual dispatch inference test
```

---

## 6. Inference

The final model is served via:

**Docker:**
```bash
docker build --build-arg HF_MODEL_NAME=fahadkamraan/transaction-categorizer \
  -t fahadkamraan/mlops-transaction-classifier:latest .

docker run --rm \
  -e HF_TOKEN=<token> \
  -e INPUT_TEXT="[debit] STARBUCKS STORE 12345" \
  fahadkamraan/mlops-transaction-classifier:latest
```

**GitHub Actions:** Actions → Inference → Run workflow → enter transaction string.

**HuggingFace Model:** [fahadkamraan/transaction-categorizer](https://huggingface.co/fahadkamraan/transaction-categorizer) _(published after V2 training)_

---

## 7. Repository Structure

| Path | Responsibility |
|---|---|
| `.github/workflows/ci.yml` | CI — flake8 lint |
| `.github/workflows/inference.yml` | Manual inference dispatch |
| `src/prepare_data.py` | Task 2 — data pipeline (Dhruvi) |
| `src/load_model.py` | Task 3 — model loading (Dhruvi) |
| `src/inference.py` | Task 6/7 — inference (Fahad) |
| `notebooks/train_v1.ipynb` | Task 4 V1 (Mahesh) |
| `notebooks/train_v2.ipynb` | Task 4 V2 (Mahesh) |
| `Dockerfile` | Task 6 — container (Fahad) |
| `docs/report.md` | This report (Himanshu) |

---

## 8. References

- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers
- Weights & Biases documentation: https://docs.wandb.ai
- Dataset: DoDataThings/us-bank-transaction-categories-v2 (HuggingFace Hub)
