# Transaction Categorizer

[![CI](https://github.com/s-fk/transaction_categorization/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/s-fk/transaction_categorization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

End-to-end MLOps pipeline that fine-tunes **DistilBERT** on real bank transaction
descriptions to classify them into 17 merchant spending categories.

> MLOps Group Assignment

---

## Project Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/s-fk/transaction_categorization |
| Kaggle Notebook — V1 | https://www.kaggle.com/code/mahesh2058/mlops-assignment-v1-v2-code |
| Kaggle Notebook — V2 | https://www.kaggle.com/code/mahesh2058/mlops-assignment-v1-v2-code |
| HuggingFace Model | https://huggingface.co/Mahesh2058/mlops-finance-classifier |
| Docker Image | _(To Do)_ |
| W&B Dashboard | https://wandb.ai/maheshv2058-iitj-ac-in/mlops-assignment3 |

---

## Dataset & Model

| | |
|---|---|
| **Dataset** | [`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) · MIT licence |
| **Task** | 17-class bank transaction merchant categorisation |
| **Subset** | 51,000 samples — stratified 3,000 per class |
| **Model** | [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) · 66 MB |

---

## Team

| Name | Roll No. | GitHub |
|---|---|---|
| S Fahad Kamraan | G25AIT2091 | @s-fk |
| Dhruvi Patel | G25AIT2030 | @dhruvi9660 |
| Mahesh V | G25AIT2058 | @maheshv2058-iitj |
| Himanshu Choubey | G25AIT2039 | @g25ait2039-uid |

---

## Repository Structure

```
transaction_categorization/
├── .github/
│   ├── ISSUE_TEMPLATE/          # Bug report & task templates
│   ├── workflows/
│   │   ├── ci.yml               # Lint on push to develop
│   │   └── inference.yml        # Manual inference trigger
│   ├── CODEOWNERS               # Auto-assigns reviewers
│   └── PULL_REQUEST_TEMPLATE.md
├── data/
│   └── id2label.json            # 17-class label map (only file committed here)
├── docs/                        # Report and supporting assets
├── notebooks/
│   ├── train_v1.ipynb           # Kaggle training — V1 hyperparameters
│   └── train_v2.ipynb           # Kaggle training — V2 hyperparameters
├── src/
│   ├── prepare_data.py          # Task 2: download, clean, sample, save
│   ├── load_model.py            # Task 3: tokenizer + model init
│   └── inference.py             # Inference: used by Docker + Actions
├── .editorconfig
├── .gitignore
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── requirements.txt             # Full training deps (pinned)
└── requirements-inference.txt   # Slim inference-only deps (Docker/CI)
```

---

## Quick Start

```bash
git clone https://github.com/s-fk/transaction_categorization.git
cd transaction_categorization
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Task 2 — Prepare data
```bash
python src/prepare_data.py
# outputs: data/id2label.json  +  data/processed/{train,val,test}.csv
```

### Task 3 — Verify model load
```bash
python src/load_model.py
```

### Task 4 — Train on Kaggle
Upload `notebooks/train_v1.ipynb` and `notebooks/train_v2.ipynb` to Kaggle.
Add `WANDB_API_KEY` and `HF_TOKEN` as Kaggle Secrets before running.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full Kaggle setup steps.

### Task 6 — Docker
```bash
docker build \
  --build-arg HF_MODEL_NAME=fahadkamraan/transaction-categorizer \
  -t fahadkamraan/mlops-transaction-classifier:latest .

docker run --rm \
  -e HF_TOKEN=<token> \
  -e INPUT_TEXT="[debit] STARBUCKS STORE 12345" \
  fahadkamraan/mlops-transaction-classifier:latest
```

### Task 7 — GitHub Actions Inference
Go to **Actions → Inference → Run workflow** and enter a transaction string.

---

## Branching Strategy

```
main      ← protected · submission state · PR from develop only
  └── develop  ← protected · default branch · CI runs here
        └── feature/<description>  ← where all work happens
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, commit format, and PR rules.

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Branch naming convention (`feature/`, `fix/`, `docs/`, `chore/`)
- Commit message format (Conventional Commits)
- PR checklist and review process
- Code style and lint rules
- Who owns what and who to contact when blocked

---

## GitHub Secrets Required

Add these at **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Pull model from HuggingFace Hub in Actions |
| `WANDB_API_KEY` | Log metrics to W&B (set in Kaggle Secrets too) |
