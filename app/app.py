#!/usr/bin/env python3
"""
Transaction Categoriser — Gradio Web App
Hosted on HuggingFace Spaces: fahadkamraan/transaction-categorizer-demo
"""

import os
import re
import gradio as gr
import pandas as pd
from transformers import pipeline

MODEL_REPO = os.environ.get("HF_MODEL_REPO", "fahadkamraan/transaction-categorizer")

_REF_CODE_RE = re.compile(r'\b[A-Z0-9]{6,}\b')
_SPACES_RE = re.compile(r'\s{2,}')
_SPECIAL_RE = re.compile(r'[*#@&]+')
_PREFIX_RE = re.compile(r'^\[(debit|credit)\]\s*', re.IGNORECASE)
_STORE_RE = re.compile(r'\b(store|branch|#)\s*\d+\b', re.IGNORECASE)


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = _PREFIX_RE.sub('', text)
    text = _STORE_RE.sub('', text)
    text = _REF_CODE_RE.sub('', text)
    text = _SPECIAL_RE.sub(' ', text)
    text = _SPACES_RE.sub(' ', text).strip()
    return text


def load_classifier():
    return pipeline(
        task="text-classification",
        model=MODEL_REPO,
        tokenizer=MODEL_REPO,
        device=-1,
        top_k=5,
    )


_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = load_classifier()
    return _classifier


def classify_single(text: str) -> dict:
    """Classify a single transaction description."""
    if not text or not text.strip():
        return {}, "Please enter a transaction description."

    text = text.strip()
    if len(text) > 512:
        return {}, "Input too long (max 512 characters)."

    clf = get_classifier()
    clean = preprocess(text)
    results = clf(clean)
    preds = results[0] if results and isinstance(results[0], list) else results

    scores = {p["label"]: round(p["score"], 4) for p in preds}
    top = preds[0]
    summary = f"**{top['label']}** ({top['score'] * 100:.1f}% confidence)"
    return scores, summary


def classify_csv(file) -> pd.DataFrame:
    """Classify all transactions in an uploaded CSV file."""
    if file is None:
        return pd.DataFrame({"error": ["No file uploaded."]})

    try:
        # Gradio 6.x passes filepath as str; Gradio 4.x passes object with .name
        filepath = file if isinstance(file, str) else file.name
        df = pd.read_csv(filepath)
    except Exception as e:
        return pd.DataFrame({"error": [f"Could not read CSV: {e}"]})

    desc_col = None
    for candidate in [
        "description", "Description", "DESCRIPTION",
        "transaction", "Transaction", "merchant", "Merchant",
        "details", "Details", "narration", "Narration",
    ]:
        if candidate in df.columns:
            desc_col = candidate
            break

    if desc_col is None:
        desc_col = df.columns[0]

    clf = get_classifier()
    categories, confidences = [], []

    for raw in df[desc_col].astype(str):
        clean = preprocess(raw)
        results = clf(clean)
        preds = results[0] if results and isinstance(results[0], list) else results
        categories.append(preds[0]["label"])
        confidences.append(f"{preds[0]['score'] * 100:.1f}%")

    out = df.copy()
    out["Predicted Category"] = categories
    out["Confidence"] = confidences
    return out


# ── UI ─────────────────────────────────────────────────────────────────────────

DESCRIPTION = """
## 🏦 Bank Transaction Categoriser

Fine-tuned **DistilBERT** model that classifies bank transaction descriptions
into **17 merchant spending categories**.

**Model:** [`fahadkamraan/transaction-categorizer`](https://huggingface.co/fahadkamraan/transaction-categorizer)
**Dataset:** `DoDataThings/us-bank-transaction-categories-v2` · 42,975 samples · 17 classes
**Accuracy:** 99.88% (test set)

**Categories:** Education · Entertainment · Fees · Groceries · Healthcare · Income · Insurance · Mortgage · Personal Care · Rent · Restaurants · Shopping · Subscription · Transfer · Transportation · Travel · Utilities
"""

EXAMPLES = [
    ["starbucks coffee purchase"],
    ["amazon prime subscription renewal"],
    ["shell gas station fill up"],
    ["dr smith medical clinic visit"],
    ["delta airlines flight booking"],
    ["state farm insurance premium"],
    ["apartment rent payment"],
    ["netflix monthly subscription"],
    ["uber eats food delivery"],
    ["paycheck direct deposit"],
]

with gr.Blocks(
    title="Transaction Categoriser",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="emerald"),
    css="""
    .gr-button-primary { background: #4f46e5 !important; }
    .label-pill { font-size: 0.85rem; font-weight: 600; }
    """
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        with gr.TabItem("📝 Single Transaction"):
            with gr.Row():
                with gr.Column(scale=3):
                    txt_input = gr.Textbox(
                        label="Transaction Description",
                        placeholder="e.g. STARBUCKS STORE 12345  or  [debit] AMAZON PRIME",
                        lines=2,
                    )
                    classify_btn = gr.Button("Classify", variant="primary")

                with gr.Column(scale=2):
                    result_label = gr.Markdown(label="Top Prediction")
                    result_scores = gr.Label(label="Top 5 Probabilities", num_top_classes=5)

            gr.Examples(examples=EXAMPLES, inputs=txt_input, label="Try an example")

            classify_btn.click(
                fn=classify_single,
                inputs=txt_input,
                outputs=[result_scores, result_label],
            )
            txt_input.submit(
                fn=classify_single,
                inputs=txt_input,
                outputs=[result_scores, result_label],
            )

        with gr.TabItem("📂 Upload Bank Statement (CSV)"):
            gr.Markdown("""
Upload a CSV file containing transaction descriptions.
The app will detect the description column automatically (or use the first column).

**Expected columns:** `description`, `transaction`, `merchant`, `details`, or `narration`
""")
            csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
            csv_btn = gr.Button("Categorise All Transactions", variant="primary")
            csv_output = gr.DataFrame(label="Results", wrap=True)

            csv_btn.click(fn=classify_csv, inputs=csv_input, outputs=csv_output)

    gr.Markdown("""
---
**IIT Jodhpur — ML Ops Group Assignment (CSL 7040)**
[GitHub](https://github.com/S-FK/transaction_categorization) · [HuggingFace Model](https://huggingface.co/fahadkamraan/transaction-categorizer) · [W&B Dashboard](https://wandb.ai/fahadkamraan_sfk/mlops-transaction-classifier)
""")

if __name__ == "__main__":
    demo.launch(share=False)
