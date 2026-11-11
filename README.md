# TextSense

[![Tests](https://github.com/AhamedT04/TexteSense/actions/workflows/tests.yml/badge.svg)](https://github.com/AhamedT04/TexteSense/actions/workflows/tests.yml)
[![Licence MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

TextSense est une application web et une API REST d'analyse de sentiment en français, construite avec Python, FastAPI, SQLite et JavaScript. Elle permet de soumettre un texte, d'estimer son sentiment, d'extraire des mots-clés, d'enregistrer le résultat en base de données et de consulter l'historique depuis une interface web.

## Fonctionnalités

- Analyse de sentiment : `positive`, `negative`, `neutral`
- Score de confiance associé à la prédiction
- Flag `low_confidence` quand le modèle est incertain
- Extraction de mots-clés (stopwords français filtrés)
- Historique paginé des analyses en base SQLite
- Suppression d'une analyse enregistrée
- Statistiques globales sur les analyses effectuées
- Interface web connectée à l'API
- Logging structuré avec horodatage
- Documentation Swagger disponible via `/docs`

## Technologies

- Python 3.12
- FastAPI + Uvicorn
- SQLAlchemy + SQLite
- Pydantic Settings (configuration via `.env`)
- Hugging Face Transformers (`cmarkea/distilcamembert-base-sentiment`)
- scikit-learn (modèle fallback TF-IDF + Régression logistique)
- HTML / CSS / JavaScript (vanilla)

## Modèle de sentiment

Le projet utilise en priorité le modèle pré-entraîné `cmarkea/distilcamembert-base-sentiment` via Hugging Face Transformers. Ce modèle est spécialisé pour le français et offre une précision solide sur des textes courants.

Le modèle doit être téléchargé **une seule fois** avant le premier lancement (voir section [Premier lancement](#premier-lancement)). Les lancements suivants utilisent le cache local.

Si le modèle HF est indisponible, l'application bascule automatiquement sur un modèle de secours `TF-IDF + LogisticRegression` entraîné localement.

## Limites connues

- Certaines phrases très courtes (`je sors`, `je viens`) portent peu de contexte de sentiment.
- Le modèle pré-entraîné reste généraliste et n'est pas spécialisé sur un domaine métier.
- **Classe neutre fragile** : le modèle prédit une note de 1 à 5 étoiles, entraîné sur des *avis* (Amazon, Allociné). Le sentiment est obtenu en regroupant 1-2★ → négatif, 3★ → neutre, 4-5★ → positif. Une phrase factuelle (« La réunion a lieu à 14h ») n'est pas un avis, et sort souvent positive avec une confiance faible. Un fine-tuning directement sur 3 classes est la piste d'amélioration prévue.
- **Modèle fallback** : entraîné sur un petit jeu (100 phrases réelles + phrases synthétiques). Évalué uniquement sur des phrases réelles non vues, son score est modeste (macro-F1 ≈ 0,58 sur 20 phrases) : c'est un filet de sécurité, pas un modèle de production.
- L'extraction de mots-clés est basée sur des règles (fréquence + stopwords), sans analyse sémantique.
- **Pas d'authentification** : n'importe quel visiteur peut consulter et supprimer l'historique (`DELETE /history/{id}`). À protéger avant toute mise en ligne publique.

## Structure du projet

```text
TextSense/
├── app/
│   ├── main.py               # Initialisation FastAPI, logging, lifespan
│   ├── routes.py             # Endpoints API
│   ├── models.py             # Modèle SQLAlchemy
│   ├── database.py           # Configuration DB + migrations automatiques
│   ├── schemas.py            # Schémas Pydantic
│   ├── config.py             # Configuration centralisée (variables d'environnement)
│   ├── ml/
│   │   ├── predictor.py      # Prédiction HF + fallback sklearn
│   │   ├── keyword_extractor.py
│   │   └── train_model.py    # Entraînement du modèle fallback
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│   └── templates/
│       └── index.html
├── data/
│   └── sentiment_dataset.csv
├── saved_models/             # Modèle sklearn sérialisé
├── tests/
│   ├── conftest.py           # Base de test temporaire (la vraie base n'est jamais touchée)
│   └── test_api.py
├── .github/workflows/tests.yml  # Intégration continue (pytest)
├── .env.example              # Exemple de configuration
├── Dockerfile
├── requirements.txt          # Dépendances d'exécution
├── requirements-dev.txt      # + dépendances de test
├── pytest.ini
├── LICENSE
└── README.md
```

## Installation

### Prérequis (Ubuntu / WSL)

```bash
sudo apt install python3-full
```

### Créer l'environnement virtuel

```bash
# Créer le venv (dans le home Linux pour de meilleures performances sous WSL)
python3 -m venv ~/textsense_venv
source ~/textsense_venv/bin/activate
```

### Installer les dépendances

```bash
# torch en version CPU (~200 Mo au lieu de ~2 Go avec CUDA)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Exécution seule
pip install -r requirements.txt
# ou, pour développer et lancer les tests
pip install -r requirements-dev.txt
```

> Python 3.12 recommandé : les versions épinglées de torch, pandas et scikit-learn ne fournissent pas de paquets pour Python 3.14.

### Configuration (optionnel)

```bash
cp .env.example .env
# Éditez .env selon vos besoins
```

## Premier lancement

Le modèle Hugging Face doit être téléchargé **une seule fois** (~250 Mo) :

```bash
python -c "
from transformers import pipeline
pipeline('sentiment-analysis', model='cmarkea/distilcamembert-base-sentiment')
print('Modèle téléchargé avec succès.')
"
```

Si vous ne souhaitez pas télécharger le modèle HF, entraînez le modèle fallback sklearn :

```bash
python -m app.ml.train_model
```

Le script affiche le macro-F1 mesuré sur une partie des phrases **réelles** du CSV, jamais vues à l'entraînement. Les phrases synthétiques (générées par gabarits) servent uniquement à l'entraînement : les inclure dans le jeu de test gonflerait le score (≈ 95 % affiché contre ≈ 55 % réel).

## Lancer le projet

```bash
source ~/textsense_venv/bin/activate
cd TexteSense
uvicorn app.main:app --reload
```

### Avec Docker

```bash
docker build -t textsense .            # inclut le modèle HF (~250 Mo)
# docker build --build-arg DOWNLOAD_HF_MODEL=0 -t textsense .   # fallback sklearn uniquement
docker run -p 8000:8000 -v textsense-db:/app/db textsense
```

Puis ouvrez :

- Interface web : `http://127.0.0.1:8000/`
- Documentation API : `http://127.0.0.1:8000/docs`
- Health check : `http://127.0.0.1:8000/health`

## Endpoints principaux

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/analyze` | Analyser un texte |
| `GET` | `/history` | Historique paginé (`?skip=0&limit=50`) |
| `GET` | `/analysis/{id}` | Détail d'une analyse |
| `DELETE` | `/history/{id}` | Supprimer une analyse |
| `GET` | `/stats` | Statistiques globales |

### Exemple de requête

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Je suis très content de ce résultat, il est vraiment convaincant."}'
```

### Exemple de réponse

```json
{
  "id": 1,
  "text": "Je suis très content de ce résultat, il est vraiment convaincant.",
  "sentiment": "positive",
  "confidence": 0.9741,
  "low_confidence": false,
  "keywords": ["content", "résultat", "convaincant"],
  "created_at": "2026-04-17T01:05:22.000000+00:00"
}
```

Le champ `low_confidence` est `true` quand le score de confiance est inférieur au seuil configuré (défaut : 0.55). L'interface affiche alors un badge orange pour indiquer un résultat indicatif.

## Tests

```bash
pytest
```

Les tests utilisent une base SQLite temporaire : votre `textsense.db` n'est jamais modifiée. Les tests nécessitant un modèle sont automatiquement ignorés si aucun modèle (HF ou sklearn) n'est prêt. En intégration continue (GitHub Actions), le modèle fallback est entraîné puis toute la suite est exécutée.

```bash
pytest -v        # Détail des tests
pytest -q        # Mode silencieux
```

## Variables d'environnement

Voir [.env.example](.env.example) pour la liste complète. Principales variables :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DATABASE_URL` | `textsense.db` à la racine du projet | URL de la base de données (SQLite ou PostgreSQL via `postgresql+psycopg://…`) |
| `HF_MODEL_ID` | `cmarkea/distilcamembert-base-sentiment` | Modèle Hugging Face |
| `HF_CONFIDENCE_THRESHOLD` | `0.55` | Seuil de confiance minimum |
| `MODEL_DIR` | `saved_models` | Dossier du modèle fallback |
| `DEBUG` | `false` | Active les logs DEBUG |

## Auteur

Ahamed Tidiane Tall

## Licence

Distribué sous licence MIT — voir [LICENSE](LICENSE).
