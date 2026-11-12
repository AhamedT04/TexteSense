"""Classifieurs de sentiment avec une interface commune.

Chaque classifieur expose ``predict_proba(texts) -> np.ndarray`` de forme
``(n_textes, 3)`` dans l'ordre :data:`app.ml.labels.SENTIMENTS`.

Ce module est utilisé à la fois par l'API (``app.ml.predictor``) et par les
scripts d'évaluation (``textsense_ml``) : on évalue donc exactement le code
qui tourne en production.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from app.ml.labels import SENTIMENTS, build_mapping_matrix

logger = logging.getLogger(__name__)


# Attribut ajouté au config.json d'un modèle par `textsense_ml.calibrate` :
# le seuil de confiance calibré voyage avec le modèle.
THRESHOLD_CONFIG_KEY = "textsense_confidence_threshold"


class SentimentClassifier(Protocol):
    name: str
    recommended_threshold: float | None

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class HuggingFaceClassifier:
    """Modèle Transformers (identifiant du Hub ou dossier local)."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Nom court et sans chemin absolu : un dossier local est désigné par son nom.
        local_dir = Path(model_name_or_path)
        self.name = local_dir.name if local_dir.is_dir() else model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, revision=revision, local_files_only=local_files_only
        )
        self.model.eval()
        self.mapping = build_mapping_matrix(self.model.config.id2label)
        self.recommended_threshold = getattr(self.model.config, THRESHOLD_CONFIG_KEY, None)

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        """Logits bruts du modèle (avant softmax et regroupement)."""
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                outputs.append(self.model(**encoded).logits.float().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, len(self.mapping)))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return _softmax(self.predict_logits(texts)) @ self.mapping


class OnnxClassifier:
    """Modèle exporté au format ONNX, exécuté avec ONNX Runtime (CPU)."""

    def __init__(self, model_dir: str | Path, *, onnx_file: str = "model.onnx", max_length: int = 512) -> None:
        import json

        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        self.name = f"{model_dir.name}/{onnx_file}"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in config["id2label"].items()}
        self.mapping = build_mapping_matrix(id2label)
        self.recommended_threshold = config.get(THRESHOLD_CONFIG_KEY)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / onnx_file), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(SENTIMENTS)))
        encoded = self.tokenizer(
            list(texts), truncation=True, max_length=self.max_length, padding=True, return_tensors="np"
        )
        feed = {k: v.astype(np.int64) for k, v in encoded.items() if k in self._input_names}
        (logits,) = self.session.run(["logits"], feed)
        return _softmax(logits) @ self.mapping


class SklearnClassifier:
    """Modèle fallback TF-IDF + régression logistique (fichiers joblib)."""

    def __init__(self, model_dir: str | Path) -> None:
        import joblib

        model_dir = Path(model_dir)
        model_path = model_dir / "sentiment_model.pkl"
        vectorizer_path = model_dir / "vectorizer.pkl"
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                "Aucun modèle fallback dans "
                f"{model_dir} : exécutez `python -m app.ml.train_model`."
            )
        self.name = "tfidf-logistic-regression"
        self.recommended_threshold = None
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        # Réordonne les colonnes de predict_proba selon SENTIMENTS.
        self._order = [list(self.model.classes_).index(s) for s in SENTIMENTS]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(SENTIMENTS)))
        probabilities = self.model.predict_proba(self.vectorizer.transform(list(texts)))
        return probabilities[:, self._order]


def load_classifier(spec: str, **kwargs) -> SentimentClassifier:
    """Construit un classifieur à partir d'une spécification texte.

    * ``hf:<id-ou-dossier>[@révision]`` — modèle Transformers ;
    * ``onnx:<dossier>[/<fichier>.onnx]`` — modèle ONNX ;
    * ``sklearn:<dossier>`` — modèle fallback.
    """
    kind, _, target = spec.partition(":")
    if kind == "hf":
        name, _, revision = target.partition("@")
        return HuggingFaceClassifier(name, revision=revision or None, **kwargs)
    if kind == "onnx":
        path = Path(target)
        if path.suffix == ".onnx":
            return OnnxClassifier(path.parent, onnx_file=path.name, **kwargs)
        return OnnxClassifier(path, **kwargs)
    if kind == "sklearn":
        return SklearnClassifier(target)
    raise ValueError(f"Spécification de modèle invalide : {spec!r} (attendu hf:, onnx: ou sklearn:)")
