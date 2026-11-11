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

# Modèle fallback (rapide) + modèle Hugging Face (~250 Mo, désactivable :
# docker build --build-arg DOWNLOAD_HF_MODEL=0 .)
ARG DOWNLOAD_HF_MODEL=1
RUN python -m app.ml.train_model \
 && if [ "$DOWNLOAD_HF_MODEL" = "1" ]; then \
      python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cmarkea/distilcamembert-base-sentiment')"; \
    fi

RUN useradd --create-home appuser && mkdir -p /app/db && chown appuser /app/db
USER appuser

ENV HF_HUB_OFFLINE=1
VOLUME ["/app/db"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
