# TextSense

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
- L'extraction de mots-clés est basée sur des règles (fréquence + stopwords), sans analyse sémantique.

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
│   ├── conftest.py
│   └── test_api.py
├── .env.example              # Exemple de configuration
├── requirements.txt
├── pytest.ini
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
pip install -r requirements.txt
```

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
python app/ml/train_model.py
```

## Lancer le projet

```bash
source ~/textsense_venv/bin/activate
cd /chemin/vers/Projet_IA
uvicorn app.main:app --reload
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

Les tests nécessitant un modèle disponible sont automatiquement ignorés si aucun modèle (HF ou sklearn) n'est prêt.

```bash
pytest -v        # Détail des tests
pytest -q        # Mode silencieux
```

## Variables d'environnement

Voir [.env.example](.env.example) pour la liste complète. Principales variables :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DATABASE_URL` | `sqlite:///./textsense.db` | URL de la base de données |
| `HF_MODEL_ID` | `cmarkea/distilcamembert-base-sentiment` | Modèle Hugging Face |
| `HF_CONFIDENCE_THRESHOLD` | `0.55` | Seuil de confiance minimum |
| `DEBUG` | `false` | Active les logs DEBUG |

## Auteur

Ahamed Tidiane Tall
