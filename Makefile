# Commandes du projet. `make help` liste les cibles.
# Toutes les commandes se lancent depuis la racine du projet, dans l'environnement virtuel.

PYTHON ?= python
BASE_MODEL := hf:cmarkea/distilcamembert-base-sentiment@5716940dead38a2685ae40c24a494e7a4e78b38b
MAIN_RUN := distilcamembert-3c-starinit-s42
EVAL_SETS := textsense_eval_v1 tweets_fr_validation tweets_fr_test

.PHONY: help install install-ml test run train-fallback eval-baselines finetune ablations seeds \
        eval-finetuned calibrate export-onnx eval-onnx report notebooks

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Dépendances de l'application + tests (torch CPU)
	$(PYTHON) -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements-dev.txt

install-ml: install ## + dépendances ML (fine-tuning, notebooks, ONNX)
	$(PYTHON) -m pip install -r requirements-ml.txt

test: ## Lance la suite de tests
	$(PYTHON) -m pytest

run: ## Lance l'API en local (http://127.0.0.1:8000)
	uvicorn app.main:app --reload

train-fallback: ## Entraîne le modèle fallback TF-IDF
	$(PYTHON) -m app.ml.train_model

eval-baselines: ## Évalue le modèle d'origine (étoiles) et le fallback
	$(PYTHON) -m textsense_ml.evaluate --name baseline-stars --model $(BASE_MODEL) --datasets $(EVAL_SETS)
	$(PYTHON) -m textsense_ml.evaluate --name tfidf-fallback --model sklearn:saved_models --datasets $(EVAL_SETS)

finetune: ## Fine-tuning principal (≈ 10 min sur CPU)
	$(PYTHON) -m textsense_ml.finetune --config configs/finetune_distilcamembert_3c.yaml

ablations: ## Ablations : tête aléatoire, modèle de départ générique
	$(PYTHON) -m textsense_ml.finetune --config configs/ablation_random_head.yaml
	$(PYTHON) -m textsense_ml.finetune --config configs/ablation_generic_base.yaml

seeds: ## Même configuration avec d'autres graines (variance entre entraînements)
	$(PYTHON) -m textsense_ml.finetune --config configs/finetune_distilcamembert_3c.yaml --seed 13 --name distilcamembert-3c-starinit-s13
	$(PYTHON) -m textsense_ml.finetune --config configs/finetune_distilcamembert_3c.yaml --seed 7 --name distilcamembert-3c-starinit-s7

eval-finetuned: ## Évalue tous les modèles présents dans models/
	@for dir in models/*/; do \
		name=$$(basename $$dir); \
		[ -f $$dir/config.json ] || continue; \
		$(PYTHON) -m textsense_ml.evaluate --name $$name --model hf:$$dir --datasets $(EVAL_SETS); \
	done

calibrate: ## Choisit le seuil de confiance sur la validation
	$(PYTHON) -m textsense_ml.calibrate --run $(MAIN_RUN) --compare baseline-stars --target-accuracy 0.85 --write-to-model models/$(MAIN_RUN)

export-onnx: ## Exporte le modèle principal en ONNX (+ version quantifiée int8)
	$(PYTHON) -m textsense_ml.export_onnx --model-dir models/$(MAIN_RUN)

eval-onnx: ## Évalue la version ONNX int8
	$(PYTHON) -m textsense_ml.evaluate --name $(MAIN_RUN)-onnx-int8 --model onnx:models/$(MAIN_RUN)-onnx/model.int8.onnx --datasets $(EVAL_SETS)

report: ## Régénère reports/RESULTS.md
	$(PYTHON) -m textsense_ml.report

notebooks: ## Ré-exécute les notebooks (sorties incluses)
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
