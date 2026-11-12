FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    DATABASE_URL=sqlite:////app/db/textsense.db

WORKDIR /app

# torch en version CPU : image beaucoup plus légère qu'avec CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# Modèle fallback (rapide) + modèle principal téléchargé depuis le Hub (~270 Mo).
#   docker build --build-arg HF_MODEL_ID=AhamedT04/textsense-distilcamembert-sentiment-3c .   # modèle fine-tuné publié
#   docker build --build-arg DOWNLOAD_HF_MODEL=0 .               # fallback uniquement
ARG DOWNLOAD_HF_MODEL=1
ARG HF_MODEL_ID=cmarkea/distilcamembert-base-sentiment
ENV HF_MODEL_ID=${HF_MODEL_ID}
RUN python -m app.ml.train_model \
 && if [ "$DOWNLOAD_HF_MODEL" = "1" ]; then \
      python -c "import os; from transformers import pipeline; pipeline('text-classification', model=os.environ['HF_MODEL_ID'])"; \
    fi

RUN useradd --create-home appuser && mkdir -p /app/db && chown appuser /app/db
USER appuser

ENV HF_HUB_OFFLINE=1
VOLUME ["/app/db"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
