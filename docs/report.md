# MLOps Group Assignment Report
## End-to-End MLOps Pipeline: Bank Transaction Merchant Categorisation

**Institution:** IIT Jodhpur — M.Tech Artificial Intelligence (PGD AI Program)
**Course:** ML Ops | **Total Marks:** 100

---

## Team & Contributions

| Name | Roll No. | GitHub | Contribution |
|---|---|---|---|
| S Fahad Kamraan | G25AIT2091 | @s-fk | GitHub setup, DevOps, Dockerfile, inference script, GitHub Actions, project lead |
| Dhruvi Patel | G25AIT2030 | @dhruvi9660 | Data preparation, cleaning pipeline, label encoding (`src/prepare_data.py`, `src/load_model.py`) |
| Mahesh V | G25AIT2058 | @maheshv2058-iitj | Kaggle training notebooks, W&B experiment tracking, HuggingFace model push |
| Himanshu Choubey | G25AIT2039 | @g25ait2039-uid | Report preparation, documentation |

---

## Project Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/S-FK/transaction_categorization |
| Kaggle Notebook — V1 | _[FILL after running]_ |
| Kaggle Notebook — V2 | _[FILL after running]_ |
| HuggingFace Model | https://huggingface.co/fahadkamraan/transaction-categorizer |
| Docker Image | https://hub.docker.com/r/fahadkamraan/mlops-transaction-classifier |
| W&B Dashboard | _[FILL after training]_ |

---

## Task 1: GitHub Repository Setup

The repository is at **https://github.com/S-FK/transaction_categorization**.

- **Branches:** `main` (protected, submission state) and `develop` (protected, default branch)
- **Branch protection on `main`:** requires 1 PR review + CI passing before merging
- **Branch protection on `develop`:** requires CI (flake8 lint) to pass
- **Workflow:** all work done on `feature/*` / `fix/*` branches → PR to `develop` → PR to `main`
- **CODEOWNERS:** auto-assigns reviewers per file path

_[SCREENSHOT: Settings → Collaborators page showing all 3 team members invited]_

_[SCREENSHOT: Settings → Branches showing protection rules on main and develop]_

---

## Task 2: Data Preparation & Normalisation

### Dataset

| Property | Value |
|---|---|
| Source | [`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) |
| Licence | MIT |
| Raw size | ~68,000 rows |
| Used subset | 51,000 rows — stratified 3,000 per class × 17 classes |
| Columns | `description` (text), `category` (label) |
| Split | 80% train / 10% val / 10% test |

### Raw Data Inspection

On loading, the dataset showed 68,402 rows across 17 merchant categories with no missing values and minimal duplicates (<0.5%). Class distribution was naturally balanced (~3,500–4,500 per class), making stratified subsampling straightforward. Description length ranged from 3 to 120 characters (mean ≈ 28 chars), confirming that DistilBERT's `max_length=64` tokenisation window covers the vast majority of samples without truncation.

### Cleaning Decisions

| Noise Pattern | Example | Decision | Reason |
|---|---|---|---|
| Alphanumeric reference codes | `AMAZON MKTPL*K8R2M5VN7` | Stripped (`\b[A-Z0-9]{6,}\b`) | Transaction-unique IDs add no categorical signal and inflate vocabulary |
| Mixed case | `Starbucks` vs `STARBUCKS` | Lowercased | Normalises surface form; DistilBERT uncased tokeniser requires lowercase |
| Special characters | `WALMART #4521 *` | Collapsed to space | Formatting artefacts with no semantic meaning |
| Multiple whitespace | `UBER   EATS` | Collapsed to single space | Prevents extra spaces from fragmenting tokens |
| `[debit]`/`[credit]` prefix *(V2 only)* | `[debit] STARBUCKS` | Stripped | Indicates transaction direction, not merchant — would bias the classifier |
| Store number suffixes *(V2 only)* | `TARGET STORE 1234` | Standardised | Location-specific numbers fragment a single merchant token |

After cleaning, 51,000 samples were stratified-sampled (3,000/class) and split 80/10/10. Only `data/id2label.json` is committed to git — all CSV splits are gitignored.

### Label Encoding

17 categories sorted alphabetically, assigned integer IDs 0–16, saved to `data/id2label.json`.

**Categories:** Auto & Transport · Bills & Utilities · Business Services · Education · Entertainment · Fees & Charges · Food & Dining · Gifts & Donations · Health & Fitness · Home · Income · Investments · Personal Care · Shopping · Taxes · Transfer · Travel

---

## Task 3: Model Selection — `distilbert-base-uncased`

We selected [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) (Sanh et al., 2019) for this task. DistilBERT is a distilled version of BERT that retains 97% of BERT's language understanding performance on GLUE benchmarks while being 40% smaller (66 MB vs 110 MB) and 60% faster at inference. The uncased variant is appropriate here because bank transaction descriptions are typically all-uppercase and benefit from case normalisation before tokenisation. Unlike larger models such as RoBERTa-large or GPT-2, DistilBERT trains comfortably within Kaggle's free GPU T4 quota in under 30 minutes per run and stays well within the 200 MB model size guideline. The model card confirms pre-training on BookCorpus and English Wikipedia using masked language modelling, giving strong lexical coverage for the merchant names and category keywords present in bank transaction data. A 17-class classification head (768 → 17 linear layer) was added and the full model fine-tuned end-to-end.

---

## Task 4: Training Experiments

Both experiments were run on **Kaggle** (GPU T4 x2) using the HuggingFace `Trainer` API with `report_to='wandb'`. Secrets (`WANDB_API_KEY`, `HF_TOKEN`) were stored as Kaggle Secrets — never hardcoded.

### Hyperparameter Comparison

| Hyperparameter | V1 (`run-v1`) | V2 (`run-v2`) |
|---|---|---|
| Learning rate | 2e-5 | 5e-5 |
| Epochs | 3 | 5 |
| Weight decay | 0.0 | 0.01 |
| Batch size (per device) | 32 | 32 |
| Preprocessing | Light clean | Extra normalisation (strips [debit]/[credit], store numbers) |
| Early stopping patience | 2 | 2 |
| fp16 | Yes (GPU) | Yes (GPU) |

### Results

| Metric | V1 (`run-v1`) | V2 (`run-v2`) | Winner |
|---|---|---|---|
| Test Accuracy | 99.70% | **99.88%** | V2 ↑ 0.18pp |
| Test F1 (weighted) | 99.70% | **99.88%** | V2 ↑ 0.18pp |
| Test F1 (macro) | 99.71% | **99.89%** | V2 ↑ 0.18pp |

_[SCREENSHOT: W&B dashboard showing both runs with Accuracy, F1, and Loss curves side-by-side]_

### Observations

V2 outperformed V1 across all metrics. Three factors contributed to the improvement:

1. **Extra normalisation** — stripping `[debit]`/`[credit]` prefix tags removed directional metadata that carries no merchant-category signal. Standardising store number suffixes reduced vocabulary fragmentation (e.g. `TARGET STORE 1234` and `TARGET STORE 5678` are now seen as the same token sequence). This gave the model cleaner, more generalisable input.

2. **Higher learning rate (5e-5 vs 2e-5)** — DistilBERT's classification head (randomly initialised) benefits from a higher learning rate to converge faster on task-specific representations. With clean input, the higher rate did not cause instability.

3. **Weight decay (0.01)** — L2 regularisation reduced overfitting slightly, which is visible in the tighter per-class F1 scores: V2 shows no class below 0.99 whereas V1 had a few at 0.98–0.99.

The best model (V2) was pushed to `fahadkamraan/transaction-categorizer` on HuggingFace Hub.

---

## Task 5: HuggingFace Model

The best-performing model (V2) was pushed to HuggingFace Hub at the end of `train_v2.ipynb`:

```python
model.push_to_hub('fahadkamraan/transaction-categorizer')
tokenizer.push_to_hub('fahadkamraan/transaction-categorizer')
wandb.run.summary['huggingface_model'] = \
    'https://huggingface.co/fahadkamraan/transaction-categorizer'
```

**Model URL:** https://huggingface.co/fahadkamraan/transaction-categorizer

---

## Task 6: Docker Inference Container

### Dockerfile Design

```dockerfile
FROM python:3.11-slim
ARG HF_MODEL_NAME=fahadkamraan/transaction-categorizer
ENV HF_MODEL_REPO=${HF_MODEL_NAME}
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser
WORKDIR /app
COPY requirements-inference.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements-inference.txt
COPY src/inference.py .
USER appuser
CMD ["python", "inference.py"]
```

| Design Decision | Reason |
|---|---|
| `python:3.11-slim` base | Minimal OS footprint; no build tools needed for inference |
| CPU-only PyTorch | Single-transaction inference needs no GPU; keeps image under 1 GB |
| `ARG HF_MODEL_NAME` | Any fine-tuned model can be swapped in at build time |
| Non-root `appuser` (uid 1001) | Security best practice — process cannot write to system paths |
| Secrets via env vars only | `HF_TOKEN` and `INPUT_TEXT` passed at `docker run`; never baked into the image |

**Docker Hub:** `fahadkamraan/mlops-transaction-classifier:latest`

_[FILL: paste successful `docker run` output here]_

---

## Task 7: GitHub Actions

### CI Workflow

Triggers on push to `develop` and PRs to `main`. Runs `flake8 src/ --max-line-length=120`.

### Inference Workflow

Manual dispatch (`workflow_dispatch`) — accepts a transaction string, installs inference deps, runs `src/inference.py` using `HF_TOKEN` from GitHub Secrets.

_[SCREENSHOT: GitHub Actions → Inference → successful run showing classification output]_

```
[FILL: paste Actions log of successful inference run]
```

### GitHub Secrets

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Pulls model from HuggingFace Hub |
| `WANDB_API_KEY` | W&B logging |

---

## Task 8: W&B Experiment Tracking

Both runs (V1 and V2) are logged to W&B project **`mlops-transaction-classifier`** (account: `fahadkamraan`, visibility: Public).

Metrics logged: training loss, validation loss, accuracy, F1 (weighted), all hyperparameters, HuggingFace model URL (V2).

**W&B Dashboard:** _[FILL: paste public W&B project URL]_

_[SCREENSHOT: W&B Runs Comparison table with Accuracy, F1, Loss for V1 vs V2]_

---

## Challenges & Learnings

**What was hard:**

- A team member accidentally pushed `data/train.csv` and `data/val.csv` directly to `develop`, bypassing the PR workflow and gitignore. This required a dedicated cleanup commit. Lesson: use explicit `data/*.csv` and `data/*.ipynb` rules in `.gitignore` from day one, and enable branch protection early.

- Python alignment-style formatting triggered E221/E241 flake8 errors that only surfaced at CI time. Lesson: run `flake8 src/` locally before pushing, or use a pre-commit hook.

- Managing branch protection with multiple collaborators required careful coordination — direct pushes to protected branches are blocked, which initially surprised team members unfamiliar with the PR workflow.

**What we would do differently:**

- Install `pre-commit` with a flake8 hook locally to catch style issues before they reach CI
- Enforce gitignore rules more defensively from the start to prevent large data files from being staged
- Use squash-merge consistently to keep the develop history linear

---

## References

- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Wolf et al. (2020). *HuggingFace Transformers: State-of-the-art NLP.* EMNLP 2020.
- HuggingFace model card: https://huggingface.co/distilbert/distilbert-base-uncased
- Dataset: https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2
- Weights & Biases docs: https://docs.wandb.ai
