# Contributing Guide

Thank you for contributing to **transaction_categorization**.  
Please read this guide fully before opening a branch, commit, or pull request.

---

## Table of Contents

1. [Team & Ownership](#team--ownership)
2. [Branching Strategy](#branching-strategy)
3. [Commit Message Format](#commit-message-format)
4. [Pull Request Process](#pull-request-process)
5. [Code Style](#code-style)
6. [Local Setup](#local-setup)
7. [Who to Unblock](#who-to-unblock)

---

## Team & Ownership

| Name | GitHub | Owns |
|---|---|---|
| S Fahad Kamraan | @s-fk | `.github/`, `Dockerfile`, `src/inference.py`, `README.md` |
| Dhruvi Patel | @dhruvi9660 | `src/prepare_data.py`, `src/load_model.py`, `data/` |
| Mahesh V | @Mahesh2058 |  `notebooks/` |
| Himanshu Choubey | @(To Do) | `docs/` |


---

## Branching Strategy

```
main          ← production / submission state  (protected)
  └── develop ← integration / staging          (protected, default branch)
        ├── feature/<short-description>        ← new work
        ├── fix/<short-description>            ← bug fixes
        ├── docs/<short-description>           ← documentation only
        └── chore/<short-description>          ← config, deps, tooling
```

### Rules

| Rule | Detail |
|---|---|
| Never push directly to `main` or `develop` | Always go through a PR |
| Branch off `develop`, not `main` | `git checkout -b feature/xyz develop` |
| Keep branches short-lived | Merge within your task window; don't let branches drift |
| One logical change per branch | Don't bundle unrelated changes |
| Delete branch after merge | Keep the remote clean |

### Branch naming examples

```
feature/data-preparation
feature/kaggle-training-v1
feature/kaggle-training-v2
feature/dockerfile
feature/github-actions
fix/label-encoding-bug
docs/update-readme-links
chore/pin-dependency-versions
```

---

## Commit Message Format

This project follows **Conventional Commits** (<https://www.conventionalcommits.org>).

```
<type>(<scope>): <short summary>

[optional body — explain WHY, not WHAT]

[optional footer: issue refs, breaking changes]
```

### Types

| Type | When to use |
|---|---|
| `feat` | A new feature or script |
| `fix` | A bug fix |
| `data` | Data pipeline changes (prepare_data, id2label) |
| `model` | ML model or notebook changes |
| `ci` | Changes to GitHub Actions workflows |
| `docker` | Dockerfile or image changes |
| `docs` | README, CONTRIBUTING, report |
| `chore` | Dependency updates, config, tooling |
| `refactor` | Code restructure with no behaviour change |

### Scopes (optional but encouraged)

`data`, `model`, `inference`, `docker`, `ci`, `deps`, `readme`

### Examples

```
feat(data): add stratified sampling to prepare_data.py

data(model): load distilbert with correct num_labels from id2label.json

fix(inference): handle empty INPUT_TEXT with early exit

ci: add flake8 lint step to CI workflow

docker: use non-root appuser in Dockerfile

chore(deps): pin transformers to 4.41.2
```

### Rules

- Subject line: max **72 characters**, imperative mood ("add" not "added")
- No full stop at end of subject
- Body: explain **why** the change was needed, not what it does
- Reference issues/tasks in footer: `Refs: Task 2`

---

## Pull Request Process

### Before opening a PR

```bash
# 1. Make sure your branch is up to date with develop
git fetch origin
git rebase origin/develop

# 2. Run lint locally and fix all warnings
pip install flake8
flake8 src/ --max-line-length=120 --ignore=E501,W503

# 3. Verify your script runs without errors
python src/prepare_data.py   # (Dhruvi)
python src/load_model.py     # (Dhruvi)
python src/inference.py      # (Fahad)
```

### Opening the PR

1. Push your branch: `git push origin feature/<your-branch>`
2. Go to GitHub → **Pull requests → New pull request**
3. Base: `develop` ← Compare: `feature/<your-branch>`
4. Fill in the PR template completely — do not delete sections
5. Assign **S Fahad Kamraan (@s-fk)** as reviewer
6. Add the relevant label (`data`, `model`, `ci`, `docker`, `docs`)

### Merging rules

| Requirement | Detail |
|---|---|
| Approvals | Minimum **1** approval from a team member |
| CI | `Lint (flake8)` check must be green |
| Conflicts | Resolve all merge conflicts before requesting review |
| Merge style | **Squash and merge** — keeps the commit history clean |

### After merge

```bash
# Switch back to develop and pull the merged changes
git checkout develop
git pull origin develop

# Delete your local feature branch
git branch -d feature/<your-branch>
```

---

## Code Style

- **Formatter:** follow PEP 8; line length max 120
- **Linter:** `flake8` (runs automatically in CI on every push to `develop`)
- **Imports:** standard library → third-party → local; one blank line between groups
- **Type hints:** use them on all function signatures in `src/`
- **Comments:** only when the *why* is non-obvious; no block comments explaining what the code does
- **Secrets:** never hardcode tokens, API keys, or passwords — use environment variables only

---

## Local Setup

```bash
# Clone
git clone https://github.com/s-fk/transaction_categorization.git
cd transaction_categorization

# Always work from develop
git checkout develop

# Create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install lint tool
pip install flake8
```

---
