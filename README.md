# Transaction Categorizer

[![CI](https://github.com/S-FK/transaction_categorization/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/S-FK/transaction_categorization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/fahadkamraan/transaction-categorizer)
[![Spaces](https://img.shields.io/badge/HuggingFace-Demo-blue)](https://huggingface.co/spaces/fahadkamraan/transaction-categorizer-demo)

End-to-end MLOps pipeline that fine-tunes **DistilBERT** on real bank transaction
descriptions to classify them into **17 merchant spending categories** — achieving **99.88% test accuracy**.

> IIT Jodhpur · M.Tech AI · ML Ops — CSL 7040

---

## Overview

This repository implements a complete, production-style MLOps workflow: data is prepared and
versioned, a compact transformer is fine-tuned on Kaggle GPUs, every experiment is tracked on
Weights & Biases, the best model is published to HuggingFace Hub, and inference is shipped three
ways — a Docker container, a GitHub Actions workflow, and a live Gradio web app.

| | |
|---|---|
| **Dataset** | [`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) · MIT licence |
| **Task** | 17-class bank transaction merchant categorisation |
| **Subset** | 42,975 samples — up to 3,000 per class × 17 classes |
| **Model** | [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) · 66 MB |
| **Best result** | **99.88%** test accuracy (V2) |

---

## Live Demo

The fastest way to see it in action is the hosted web app — nothing to install:

🌐 **[huggingface.co/spaces/fahadkamraan/transaction-categorizer-demo](https://huggingface.co/spaces/fahadkamraan/transaction-categorizer-demo)**

It supports two modes:

- **Single transaction** — type or paste a description (e.g. `STARBUCKS STORE 12345`) and get the
  predicted category with a confidence breakdown.
- **Bank statement upload** — drop in a CSV and every row is categorised at once. A ready-made
  sample, [`test_transactions.csv`](app/test_transactions.csv) with 49 realistic transactions, is
  included to try immediately.

---

## How It Works — Pipeline Overview

The project wires together seven stages into one reproducible pipeline:

```mermaid
flowchart TD
    A["📦 RAW DATA\nDoDataThings/us-bank-transaction-categories-v2\n68,000 rows · 17 categories · HuggingFace Hub"]
    B["🔧 DATA PREPARATION\nsrc/prepare_data.py\nClean → Sample 42,975 rows → Encode labels\nSave data/id2label.json"]
    C["🤖 MODEL SELECTION\ndistilbert-base-uncased · 66 MB\nAdd 17-class head · Load from HuggingFace Hub"]
    D["🏋️ TRAINING on Kaggle\ntrain_v1.ipynb  V1: lr=2e-5 · 3 epochs → 99.70%\ntrain_v2.ipynb  V2: lr=5e-5 · 5 epochs → 99.88%"]
    E["📊 W&B TRACKING\nAll metrics logged in real time\neval/loss · accuracy · F1 · hyperparameters"]
    F["🚀 HUGGINGFACE HUB\nfahadkamraan/transaction-categorizer\nBest model pushed · publicly accessible"]
    G["🐳 DOCKER CONTAINER\nfahadkamraan/mlops-transaction-classifier\npython:3.11-slim · CPU torch · non-root user"]
    H["⚙️ GITHUB ACTIONS\nci.yml: flake8 lint on every push\ninference.yml: manual inference trigger"]
    I["🌐 GRADIO WEB APP\nHuggingFace Spaces — free 24/7 hosting\nSingle transaction + CSV batch upload"]

    A --> B --> C --> D
    D --> E
    D --> F
    F --> G
    F --> H
    F --> I

    style A fill:#4F46E5,color:#fff,stroke:#4338CA
    style B fill:#3B82F6,color:#fff,stroke:#2563EB
    style C fill:#F59E0B,color:#fff,stroke:#D97706
    style D fill:#4338CA,color:#fff,stroke:#3730A3
    style E fill:#EF4444,color:#fff,stroke:#DC2626
    style F fill:#10B981,color:#fff,stroke:#059669
    style G fill:#334155,color:#fff,stroke:#1E293B
    style H fill:#1E293B,color:#fff,stroke:#0F172A
    style I fill:#7C3AED,color:#fff,stroke:#6D28D9
```

Everything downstream of training is loosely coupled through **HuggingFace Hub as the central model
registry** — the Docker container, the Actions workflow, and the Gradio app all load the model at
runtime rather than baking it in.

### What happens when data changes

The pipeline has a clear **cut point** at the HuggingFace Hub push:

```
DATA CHANGES
     │
     │  ← MANUAL (you do these)
     ▼
  1. Edit src/prepare_data.py  →  re-run  →  new CSVs
  2. Re-run notebooks on Kaggle (V1 and V2)
  3. New model pushed to HuggingFace Hub
     │
     │  ← AUTOMATIC (happens on its own)
     ▼
  Gradio Space  ─┐
  Docker run    ─┼─ all load the model from HF Hub at runtime
  GitHub Actions─┘  → pick up the new model with zero redeployment
```

Because inference loads the model at runtime, updating the model on HF Hub automatically updates all
three consumers. To retrain with new data: re-run the notebooks → push new weights → done.

### MLOps maturity

This project sits at **Level 1** on the MLOps maturity scale:

```
Level 0        Level 1 ← WE ARE HERE     Level 2              Level 3
──────────     ────────────────────────   ──────────────────   ──────────────────────
Jupyter        Scripts + Git + W&B        CI/CD triggers        Feature store +
notebooks      HF Hub + Docker +          retraining on         drift monitoring +
only           GitHub Actions +           data change +         auto-rollback +
               Gradio demo               model registry        A/B testing
```

---

## Quick Start

```bash
git clone https://github.com/S-FK/transaction_categorization.git
cd transaction_categorization
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Prepare data**
```bash
python src/prepare_data.py
# outputs: data/id2label.json  +  data/processed/{train,val,test}.csv
```

**Verify the model loads**
```bash
python src/load_model.py
```

**Train on Kaggle** — upload `notebooks/train_v1.ipynb` and `notebooks/train_v2.ipynb` to Kaggle,
add `WANDB_API_KEY` and `HF_TOKEN` as Kaggle Secrets, then run.

**Docker inference**
```bash
docker build \
  --build-arg HF_MODEL_NAME=fahadkamraan/transaction-categorizer \
  -t fahadkamraan/mlops-transaction-classifier:latest .

docker run --rm \
  -e HF_TOKEN=<token> \
  -e INPUT_TEXT="starbucks coffee purchase" \
  fahadkamraan/mlops-transaction-classifier:latest
```

**GitHub Actions inference** — go to **Actions → Inference → Run workflow** and enter any
transaction string.

> **GitHub Secrets required** (Settings → Secrets and variables → Actions):
> `HF_TOKEN` to pull the model from HuggingFace Hub, and `WANDB_API_KEY` for metric logging.

---

## Repository Structure

```
transaction_categorization/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml               # Lint on push to develop
│   │   └── inference.yml        # Manual inference trigger
│   ├── CODEOWNERS               # Auto-assigns reviewers
│   └── PULL_REQUEST_TEMPLATE.md
├── app/
│   ├── app.py                   # Gradio web app (deployed on HF Spaces)
│   ├── requirements.txt         # Spaces deps
│   ├── test_transactions.csv    # 49 sample transactions for demo
│   └── sample_bank_statement.csv# Same 49 with ground truth labels
├── data/
│   └── id2label.json            # 17-class label map (only committed file)
├── docs/                        # Report, screenshots, diagrams
├── notebooks/
│   ├── train_v1.ipynb           # Kaggle training — V1 hyperparameters
│   └── train_v2.ipynb           # Kaggle training — V2 hyperparameters
├── src/
│   ├── prepare_data.py          # Data pipeline: download, clean, sample, save
│   ├── load_model.py            # Load tokenizer + model from HF Hub
│   └── inference.py             # Inference: used by Docker + Actions + Gradio
├── Dockerfile                   # CPU-only inference container
├── requirements.txt             # Full training deps (pinned)
└── requirements-inference.txt   # Slim inference-only deps
```

---

## Project Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/S-FK/transaction_categorization |
| Kaggle Notebook — V1 | https://www.kaggle.com/code/fahadkamraaniitj/train-v1 |
| Kaggle Notebook — V2 | https://www.kaggle.com/code/fahadkamraaniitj/train-v2 |
| HuggingFace Model | https://huggingface.co/fahadkamraan/transaction-categorizer |
| Docker Image | https://hub.docker.com/r/fahadkamraan/mlops-transaction-classifier |
| W&B Dashboard | https://wandb.ai/fahadkamraan_sfk/mlops-transaction-classifier |
| Web Demo (Gradio) | https://huggingface.co/spaces/fahadkamraan/transaction-categorizer-demo |

---

## Contributing

All work follows a protected-branch, pull-request workflow:

```
main      ← protected · submission state · PR from develop only
  └── develop  ← protected · default branch · CI runs here
        └── feature/<description>  ← where all work happens
```

In short: branch off `develop`, commit using [Conventional Commits](https://www.conventionalcommits.org),
make sure `flake8 src/ --max-line-length=120` passes, then open a PR back into `develop`. Never push
directly to `main` or `develop`.

Full details — branch naming, commit format, the PR checklist, code style, and who owns which files —
are in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Team

| Name | Roll No. | GitHub |
|---|---|---|
| S Fahad Kamraan | G25AIT2091 | [@s-fk](https://github.com/s-fk) |
| Dhruvi Patel | G25AIT2030 | [@dhruvi9660](https://github.com/dhruvi9660) |
| Mahesh V | G25AIT2058 | [@maheshv2058-iitj](https://github.com/maheshv2058-iitj) |
| Himanshu Choubey | G25AIT2039 | [@g25ait2039-uid](https://github.com/g25ait2039-uid) |

---

## License

Released under the [MIT License](LICENSE).
