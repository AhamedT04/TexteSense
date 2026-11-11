# TextSense

[![Tests](https://github.com/AhamedT04/TexteSense/actions/workflows/tests.yml/badge.svg)](https://github.com/AhamedT04/TexteSense/actions/workflows/tests.yml)
[![Licence MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

API REST et application web d'**analyse de sentiment en français** (positif /
négatif / neutre), avec un modèle **DistilCamemBERT fine-tuné en 3 classes**,
une boucle de **retour utilisateur** pour collecter des données réelles, et un
pipeline ML reproductible (évaluation, fine-tuning, calibration, export ONNX).

## Résultats

Le modèle pré-entraîné utilisé à l'origine prédit une note de 1 à 5 étoiles,
convertie en 3 classes. L'évaluation a montré qu'il **ne reconnaît presque
jamais la classe neutre**. Le modèle fine-tuné directement en 3 classes corrige
ce défaut :

| Modèle | `textsense_eval_v1` (360 phrases) | `tweets_fr_test` (870 tweets) | F1 neutre (eval_v1) |
|---|---|---|---|
| Modèle d'origine (5 étoiles regroupées) | 0,510 | 0,363 | 0,031 |
| Fallback TF-IDF + régression logistique | 0,513 | 0,264 | 0,651 |
| **DistilCamemBERT fine-tuné 3 classes** | **0,885** | **0,723** | **0,882** |
| ↳ même modèle, ONNX int8 (69 Mo, ~4× plus rapide) | 0,882 | 0,705 | 0,870 |

Macro-F1. Résultat stable sur trois graines (0,883 ± 0,003 sur
`textsense_eval_v1`). Intervalles de confiance, ablations et
analyse d'erreurs : [`reports/RESULTS.md`](reports/RESULTS.md) ·
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) ·
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Fonctionnalités

- Sentiment `positive` / `negative` / `neutral`, score de confiance et drapeau
  `low_confidence` (seuil **calibré** sur la validation, stocké avec le modèle)
- Retour utilisateur : confirmer ou corriger chaque prédiction, export CSV
  des corrections pour constituer un jeu de données réel
- Traçabilité : chaque analyse enregistre le modèle qui l'a produite
- Deux moteurs d'inférence : PyTorch (Transformers) ou **ONNX Runtime**
  (quantifié int8)
- Modèle de secours TF-IDF si le modèle principal est indisponible
- Extraction de mots-clés, historique paginé, statistiques
- Documentation OpenAPI interactive sur `/docs`

## Démarrage rapide

Python **3.12** recommandé (les versions épinglées de torch, pandas et
scikit-learn n'ont pas de paquets pour Python 3.14).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install          # torch CPU + dépendances + outils de test
make train-fallback   # modèle de secours (quelques secondes)
make run              # http://127.0.0.1:8000
```

Sans `make` : voir les commandes équivalentes dans le [`Makefile`](Makefile).

### Choisir le modèle principal

Le modèle est désigné par `HF_MODEL_ID` (fichier `.env`, voir
[`.env.example`](.env.example)) :

```bash
# Modèle fine-tuné, après l'avoir entraîné localement (make finetune)
HF_MODEL_ID=models/distilcamembert-3c-starinit-s42

# Ou un modèle publié sur le Hub Hugging Face
HF_MODEL_ID=<compte>/textsense-distilcamembert-sentiment-3c

# Ou la version ONNX quantifiée (make export-onnx)
MODEL_BACKEND=onnx
ONNX_MODEL_DIR=models/distilcamembert-3c-starinit-s42-onnx
ONNX_MODEL_FILE=model.int8.onnx
```

Un identifiant du Hub doit être téléchargé une fois (l'application démarre
ensuite hors ligne) :

```bash
python -c "from transformers import pipeline; pipeline('text-classification', model='<HF_MODEL_ID>')"
```

Sans modèle principal, l'application bascule sur le modèle de secours.

## Pipeline ML

Toutes les étapes sont scriptées, versionnées et reproductibles (révisions de
modèles et de données figées, graines fixées, métadonnées d'exécution
enregistrées avec chaque résultat).

```bash
make install-ml       # accelerate, matplotlib, onnx, jupyter…
make eval-baselines   # 1. mesurer les modèles existants
make finetune         # 2. fine-tuning principal (~10 min sur CPU)
make ablations seeds  # 3. ablations + variance entre graines
make eval-finetuned   # 4. évaluer les modèles entraînés
make calibrate        # 5. choisir le seuil de confiance (validation)
make export-onnx      # 6. export ONNX + quantification int8 + benchmark
make report           # 7. régénérer reports/RESULTS.md
make notebooks        # 8. ré-exécuter les notebooks d'analyse
```

| Données | Rôle |
|---|---|
| `tweets_fr_train` (1 839) | entraînement |
| `tweets_fr_validation` (324) | choix de l'époque, calibration du seuil |
| `tweets_fr_test` (870) | test, domaine des tweets |
| [`textsense_eval_v1`](data/eval/DATASHEET.md) (360) | test, textes courants + cas difficiles annotés |

Les tweets proviennent de
[`cardiffnlp/tweet_sentiment_multilingual`](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)
(partie française), téléchargés à une révision figée et jamais redistribués
dans ce dépôt.

## Structure du projet

```text
TexteSense/
├── app/                        # Service en production (FastAPI)
│   ├── main.py                 # Application, logging, démarrage
│   ├── routes.py               # Endpoints
│   ├── schemas.py · models.py · database.py · config.py
│   ├── ml/
│   │   ├── predictor.py        # Choix du modèle, secours, seuil
│   │   ├── classifiers.py      # Transformers / ONNX / sklearn, interface commune
│   │   ├── labels.py           # Conversion des sorties de modèles en 3 sentiments
│   │   ├── keyword_extractor.py
│   │   └── train_model.py      # Modèle de secours TF-IDF
│   ├── static/ · templates/    # Interface web
├── textsense_ml/               # ML hors ligne (non utilisé par l'API)
│   ├── data.py                 # Jeux de données, révisions figées
│   ├── evaluate.py · metrics.py · report.py
│   ├── finetune.py · calibrate.py
│   ├── export_onnx.py · publish_model.py · export_feedback.py
├── configs/                    # Configurations d'entraînement (YAML)
├── data/eval/                  # Jeu de test + fiche + guide d'annotation
├── reports/                    # Résultats versionnés (JSON, PNG, RESULTS.md)
├── notebooks/                  # Analyses (générées par _build_notebooks.py)
├── docs/                       # Model card, protocole expérimental
├── tests/                      # Tests API, retours utilisateurs, ML
├── models/                     # Poids entraînés (non versionnés)
├── Dockerfile · Makefile · .github/workflows/tests.yml
└── requirements*.txt           # exécution · dev · ml
```

## API

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/analyze` | Analyser un texte |
| `GET` | `/history` | Historique paginé (`?skip=0&limit=50`) |
| `GET` | `/analysis/{id}` | Détail d'une analyse |
| `PUT` | `/analysis/{id}/feedback` | Confirmer ou corriger le sentiment (`{"label": "neutral"}`) |
| `DELETE` | `/analysis/{id}/feedback` | Retirer le retour utilisateur |
| `DELETE` | `/history/{id}` | Supprimer une analyse |
| `GET` | `/stats` | Statistiques (dont nombre de retours et de corrections) |
| `GET` | `/health` | État du service et du modèle chargé |

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "La réunion a lieu jeudi à 14 heures."}'
```

```json
{
  "id": 1,
  "text": "La réunion a lieu jeudi à 14 heures.",
  "sentiment": "neutral",
  "confidence": 0.9108,
  "low_confidence": false,
  "keywords": ["réunion", "lieu", "jeudi", "heures"],
  "created_at": "2026-11-11T20:29:40.223679Z",
  "model_name": "distilcamembert-3c-starinit-s42",
  "user_label": null,
  "user_label_at": null
}
```

`low_confidence` vaut `true` sous le seuil de confiance. Ordre de priorité du
seuil : variable `CONFIDENCE_THRESHOLD` si elle est définie, sinon le seuil
calibré enregistré dans le `config.json` du modèle, sinon 0,55.

Les retours utilisateurs s'exportent avec :

```bash
python -m textsense_ml.export_feedback --only-corrections
```

## Configuration

Voir [`.env.example`](.env.example). Principales variables :

| Variable | Défaut | Description |
|---|---|---|
| `DATABASE_URL` | `textsense.db` à la racine | SQLite ou PostgreSQL (`postgresql+psycopg://…`) |
| `MODEL_BACKEND` | `transformers` | `transformers` ou `onnx` |
| `HF_MODEL_ID` | `cmarkea/distilcamembert-base-sentiment` | Identifiant du Hub ou dossier local |
| `ONNX_MODEL_DIR` / `ONNX_MODEL_FILE` | – / `model.onnx` | Modèle ONNX (si `MODEL_BACKEND=onnx`) |
| `CONFIDENCE_THRESHOLD` | seuil du modèle, sinon 0,55 | Seuil `low_confidence` |
| `MODEL_DIR` | `saved_models` | Modèle de secours |
| `DEBUG` | `false` | Logs détaillés |

## Tests

```bash
make test
```

Les tests utilisent une base SQLite temporaire (votre base n'est jamais
modifiée). Ceux qui nécessitent un modèle sont ignorés si aucun n'est
disponible ; en intégration continue, le modèle de secours est entraîné puis
toute la suite est exécutée.

## Docker

```bash
docker build -t textsense .            # inclut le modèle HF (~250 Mo)
# docker build --build-arg DOWNLOAD_HF_MODEL=0 -t textsense .   # secours uniquement
docker run -p 8000:8000 -v textsense-db:/app/db textsense
```

## Limites connues

- **Ironie** : point faible principal, y compris après fine-tuning (voir
  l'analyse par phénomène dans [`reports/RESULTS.md`](reports/RESULTS.md)).
- **Domaine d'entraînement** : le fine-tuning utilise des tweets (environnement,
  politique) ; les performances sur un autre domaine (avis clients, support)
  sont à vérifier avec des données de ce domaine — d'où la collecte des
  retours utilisateurs.
- **Jeu d'évaluation** : 360 phrases rédigées pour le projet (avec
  assistance IA), distribution artificielle ; voir sa
  [fiche](data/eval/DATASHEET.md).
- **Calibration** : le seuil choisi sur la validation donne 85 % d'exactitude
  sur les prédictions gardées en validation, 80 % sur le test tweets (le modèle
  y est un peu trop confiant).
- **Pas d'authentification** : n'importe quel visiteur peut consulter,
  corriger et supprimer l'historique. À protéger avant une mise en ligne.
- **Données personnelles** : les textes saisis sont stockés en clair dans la
  base ; ne pas y saisir d'informations sensibles.

## Documentation

| Document | Contenu |
|---|---|
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Fiche du modèle : usage prévu, données, résultats, limites |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Protocole, ablations, décisions et leurs justifications |
| [`reports/RESULTS.md`](reports/RESULTS.md) | Tous les résultats (généré automatiquement) |
| [`data/eval/DATASHEET.md`](data/eval/DATASHEET.md) | Fiche du jeu d'évaluation |
| [`data/eval/ANNOTATION_GUIDELINES.md`](data/eval/ANNOTATION_GUIDELINES.md) | Règles d'annotation |
| [`notebooks/`](notebooks/) | Analyses des modèles de référence et du fine-tuning |

## Auteur

Ahamed Tidiane Tall

## Licence

Code distribué sous licence MIT — voir [LICENSE](LICENSE). Le modèle de base
`cmarkea/distilcamembert-base-sentiment` est sous licence MIT ; consulter la
fiche de `cardiffnlp/tweet_sentiment_multilingual` pour les conditions
d'utilisation des tweets.
