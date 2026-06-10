# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim keeps the final image under ~200 MB before model download.
FROM python:3.11-slim

# ── Build argument ─────────────────────────────────────────────────────────────
# Override at build time:
#   docker build --build-arg HF_MODEL_NAME=fahadkamraan/transaction-categorizer .
ARG HF_MODEL_NAME=fahadkamraan/transaction-categorizer
ENV HF_MODEL_REPO=${HF_MODEL_NAME}

# ── Security: run as non-root user ────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# ── Install inference-only dependencies ───────────────────────────────────────
# CPU-only torch wheel keeps the layer small; no CUDA needed for inference.
COPY requirements-inference.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements-inference.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY src/inference.py .

# ── Drop to non-root ───────────────────────────────────────────────────────────
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
# Pass secrets and input via environment variables — never bake them into image.
# Example:
#   docker run --rm \
#     -e HF_TOKEN=<token> \
#     -e INPUT_TEXT="[debit] STARBUCKS STORE 12345" \
#     fahadkamraan/mlops-transaction-classifier:latest
CMD ["python", "inference.py"]
