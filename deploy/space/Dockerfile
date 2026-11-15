# Image de la démo publique (Hugging Face Spaces, SDK Docker).
#
# Différences avec le Dockerfile principal :
# * modèle ONNX int8 (69 Mo, ~4 ms par texte) : ni torch ni CUDA dans l'image ;
# * mode démo activé (historique par visiteur, limite de débit, avertissement) ;
# * utilisateur d'UID 1000 et port 7860, comme l'attend Hugging Face Spaces.
#
# Ce fichier est copié à la racine du Space par deploy/deploy_space.py.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN useradd --create-home --uid 1000 user \
 && mkdir /home/user/app && chown user:user /home/user/app
WORKDIR /home/user/app

# Dépendances d'exécution sans torch (inutile avec ONNX Runtime).
COPY --chown=user requirements.txt .
RUN grep -v '^torch' requirements.txt > /tmp/requirements-space.txt \
 && pip install --no-cache-dir -r /tmp/requirements-space.txt

COPY --chown=user app ./app
COPY --chown=user data/sentiment_dataset.csv ./data/sentiment_dataset.csv
USER user

# Modèle fallback (TF-IDF) puis modèle principal ONNX téléchargé depuis le Hub,
# à une révision figée pour que l'image soit reproductible.
ARG MODEL_REPO=AhamedT04/textsense-distilcamembert-sentiment-3c
ARG MODEL_REVISION=main
ARG DOWNLOAD_MODEL=1
RUN python -m app.ml.train_model \
 && if [ "$DOWNLOAD_MODEL" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='models/textsense-3c', allow_patterns=['config.json', 'tokenizer*', 'special_tokens_map.json', 'sentencepiece.bpe.model', 'onnx/model.int8.onnx'])"; \
    fi

ENV MODEL_BACKEND=onnx \
    ONNX_MODEL_DIR=models/textsense-3c \
    ONNX_MODEL_FILE=onnx/model.int8.onnx \
    DEMO_MODE=true \
    RATE_LIMIT_PER_MINUTE=30

EXPOSE 7860
# --proxy-headers : l'adresse IP réelle du visiteur (limite de débit) est lue
# dans l'en-tête X-Forwarded-For posé par le proxy de Hugging Face.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
