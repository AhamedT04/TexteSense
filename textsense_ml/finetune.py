"""Fine-tuning d'un modèle Transformers en classification 3 classes.

Usage (depuis la racine du projet) ::

    python -m textsense_ml.finetune --config configs/finetune_distilcamembert_3c.yaml
    python -m textsense_ml.finetune --config <...> --seed 13 --name <autre-nom>

Protocole :

* entraînement sur ``train_dataset`` ;
* choix de la meilleure époque sur ``validation_dataset`` (macro-F1) ;
* les jeux de test (``tweets_fr_test``, ``textsense_eval_v1``) ne sont
  **jamais** utilisés ici : ils sont évalués ensuite avec
  ``textsense_ml.evaluate``.

Sorties :

* ``models/<name>/`` — poids, tokenizer, ``training_info.json`` (non versionné) ;
* ``reports/training/<name>/`` — historique, courbes, configuration (versionné).
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from app.config import BASE_DIR
from app.ml.labels import SENTIMENT_INDEX, SENTIMENTS, label_to_sentiment
from textsense_ml.data import load_dataset
from textsense_ml.plots import plot_training_curves
from textsense_ml.tracking import run_metadata, write_json

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    name: str
    base_model: str
    base_model_revision: str | None = None
    head_init: str = "random"  # "random" ou "star_merge"
    train_dataset: str = "tweets_fr_train"
    validation_dataset: str = "tweets_fr_validation"
    max_length: int = 128
    learning_rate: float = 3e-5
    batch_size: int = 16
    epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    seed: int = 42
    models_dir: str = "models"
    reports_dir: str = "reports/training"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path, **overrides) -> FinetuneConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class EncodedDataset:
    """Jeu tokenisé à la volée une fois ; le padding est fait par lot (DataCollatorWithPadding)."""

    def __init__(self, tokenizer, texts: list[str], labels: list[str], max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [SENTIMENT_INDEX[label] for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _init_head_from_star_model(model, config: FinetuneConfig) -> None:
    """Initialise la couche de sortie 3 classes à partir de la tête « étoiles » d'origine.

    Pour chaque sentiment, on moyenne les lignes des étoiles correspondantes
    (négatif = moyenne(1★, 2★), neutre = 3★, positif = moyenne(4★, 5★)). Le
    modèle démarre ainsi avec la connaissance du modèle d'origine au lieu
    d'une tête aléatoire.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    source = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, revision=config.base_model_revision
    )
    src_layer = source.classifier.out_proj
    dst_layer = model.classifier.out_proj
    groups: dict[int, list[int]] = {i: [] for i in range(len(SENTIMENTS))}
    for index, label in source.config.id2label.items():
        groups[SENTIMENT_INDEX[label_to_sentiment(label)]].append(int(index))

    with torch.no_grad():
        for target, rows in groups.items():
            dst_layer.weight[target] = src_layer.weight[rows].mean(dim=0)
            dst_layer.bias[target] = src_layer.bias[rows].mean()
        # La couche dense intermédiaire a la même forme : elle est déjà chargée,
        # on vérifie simplement qu'elle est identique à la source.
        assert torch.equal(model.classifier.dense.weight, source.classifier.dense.weight)
    logger.info("Tête 3 classes initialisée depuis la tête étoiles : %s", groups)


def build_model(config: FinetuneConfig):
    from transformers import AutoModelForSequenceClassification

    id2label = dict(enumerate(SENTIMENTS))
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        num_labels=len(SENTIMENTS),
        id2label=id2label,
        label2id={label: i for i, label in id2label.items()},
        ignore_mismatched_sizes=True,  # l'ancienne tête 5 classes est remplacée
    )
    if config.head_init == "star_merge":
        _init_head_from_star_model(model, config)
    elif config.head_init != "random":
        raise ValueError(f"head_init inconnu : {config.head_init}")
    return model


def compute_metrics(eval_prediction) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }


def _relative(path: str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return Path(path).name


def train(config: FinetuneConfig) -> dict:
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    output_dir = BASE_DIR / config.models_dir / config.name
    report_dir = BASE_DIR / config.reports_dir / config.name
    checkpoints_dir = output_dir / "checkpoints"

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_model_revision)
    train_df = load_dataset(config.train_dataset)
    val_df = load_dataset(config.validation_dataset)
    train_ds = EncodedDataset(tokenizer, train_df["text"].tolist(), train_df["label"].tolist(), config.max_length)
    val_ds = EncodedDataset(tokenizer, val_df["text"].tolist(), val_df["label"].tolist(), config.max_length)
    logger.info("Entraînement : %d exemples — validation : %d", len(train_ds), len(val_ds))

    model = build_model(config)

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        use_cpu=True,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    start = time.perf_counter()
    trainer.train()
    duration = time.perf_counter() - start
    best_eval = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    info = {
        "config": asdict(config),
        # Chemin relatif : ce fichier est versionné, il ne doit pas contenir de chemin personnel.
        "best_checkpoint": _relative(trainer.state.best_model_checkpoint),
        "best_validation": {k: float(v) for k, v in best_eval.items() if isinstance(v, (int, float))},
        "train_examples": len(train_ds),
        "validation_examples": len(val_ds),
        "training_seconds": round(duration, 1),
        "metadata": run_metadata(),
    }
    write_json(output_dir / "training_info.json", info)
    write_json(report_dir / "training_info.json", info)
    write_json(report_dir / "log_history.json", {"log_history": trainer.state.log_history})
    (report_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True))
    plot_training_curves(trainer.state.log_history, report_dir / "training_curves.png")

    # Les checkpoints intermédiaires ne servent plus : le meilleur modèle est sauvegardé.
    import shutil

    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    logger.info("Modèle sauvegardé dans %s (%.0f s)", output_dir, duration)
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    config = FinetuneConfig.from_yaml(args.config, seed=args.seed, name=args.name)
    info = train(config)
    print(f"Meilleure validation : {info['best_validation']}")


if __name__ == "__main__":
    main()
