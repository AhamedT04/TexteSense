---
language: fr
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: cmarkea/distilcamembert-base-sentiment
datasets:
  - cardiffnlp/tweet_sentiment_multilingual
metrics:
  - f1
  - accuracy
tags:
  - sentiment-analysis
  - french
  - camembert
---

# TextSense — DistilCamemBERT sentiment 3 classes (`distilcamembert-3c-starinit-s42`)

Classifieur de sentiment pour le **français** en trois classes : `negative`,
`neutral`, `positive`. Modèle principal de l'application
[TextSense](https://github.com/AhamedT04/TexteSense).

## Détails du modèle

| | |
|---|---|
| Architecture | DistilCamemBERT (6 couches, ~68 M paramètres) + tête de classification 3 classes |
| Modèle de départ | [`cmarkea/distilcamembert-base-sentiment`](https://huggingface.co/cmarkea/distilcamembert-base-sentiment) (révision `5716940`), entraîné à prédire 1 à 5 étoiles sur des avis Amazon et Allociné |
| Modification | tête 5 étoiles remplacée par une tête 3 classes, **initialisée depuis la tête d'origine** (négatif = moyenne 1★/2★, neutre = 3★, positif = moyenne 4★/5★), puis fine-tuning complet |
| Étiquettes | `0: negative`, `1: neutral`, `2: positive` |
| Longueur max. | 128 tokens à l'entraînement, 512 à l'inférence |
| Seuil de confiance calibré | `textsense_confidence_threshold = 0.81` (dans `config.json`) |
| Auteur | Ahamed Tidiane Tall |
| Licence | MIT |

## Usage prévu

- **Prévu** : estimer l'attitude exprimée dans des textes courts à moyens en
  français (avis, messages, commentaires), avec une vraie classe « neutre »
  pour les textes factuels, questions et consignes.
- **Hors périmètre** : décisions individuelles à fort enjeu (recrutement,
  modération automatique sans contrôle humain, santé), détection d'émotions
  fines, autres langues, textes longs (> 512 tokens tronqués).
- **Usage recommandé** : traiter les prédictions sous le seuil calibré
  (`low_confidence`) comme indicatives et prévoir une relecture humaine.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="AhamedT04/textsense-distilcamembert-sentiment-3c")
classifier("La réunion a lieu jeudi à 14 heures.")
# [{'label': 'neutral', 'score': 0.886...}]
```

## Données d'entraînement

Partie française de
[`cardiffnlp/tweet_sentiment_multilingual`](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)
(révision `14b13ed`) : tweets annotés en 3 classes équilibrées.

| Split | Exemples | Usage |
|---|---|---|
| train | 1 839 | entraînement |
| validation | 324 | choix de l'époque, calibration du seuil |
| test | 870 | évaluation finale |

## Procédure d'entraînement

| Hyperparamètre | Valeur |
|---|---|
| Optimiseur | AdamW, learning rate 3e-5, weight decay 0,01 |
| Planning | linéaire, warmup 10 % |
| Batch | 16 |
| Époques | 4 max., meilleure époque selon le macro-F1 de validation (early stopping, patience 2) |
| Graine | 42 |
| Matériel | CPU Intel Core i7-1255U (12 threads), ~9 min |

Configuration complète : [`configs/finetune_distilcamembert_3c.yaml`](../configs/finetune_distilcamembert_3c.yaml).
Reproduction : `make finetune`.

La configuration (modèle de départ, initialisation de la tête, hyperparamètres)
a été fixée **avant** de regarder les jeux de test ; les variantes
(ablations, autres graines) servent à interpréter le résultat, pas à choisir
le meilleur score sur le test.

## Évaluation

Deux jeux de test, jamais utilisés pendant l'entraînement ni pour un réglage :

- `tweets_fr_test` — 870 tweets, même domaine que l'entraînement ;
- [`textsense_eval_v1`](../data/eval/DATASHEET.md) — 360 phrases courantes
  (avis, informations, questions, consignes) avec cas difficiles annotés,
  **autre domaine** que l'entraînement.

Macro-F1 avec intervalle de confiance à 95 % (bootstrap, 1 000 tirages) :

| Modèle | `textsense_eval_v1` | `tweets_fr_test` |
|---|---|---|
| Modèle d'origine (étoiles regroupées) | 0,510 [0,480–0,541] | 0,363 [0,335–0,390] |
| TF-IDF + régression logistique | 0,513 [0,462–0,564] | 0,264 [0,238–0,292] |
| **Ce modèle** | **0,885 [0,851–0,917]** | **0,723 [0,692–0,754]** |

Par classe (précision / rappel / F1) :

| Classe | `textsense_eval_v1` | `tweets_fr_test` |
|---|---|---|
| negative | 0,89 / 0,84 / 0,87 | 0,77 / 0,82 / 0,79 |
| neutral | 0,93 / 0,84 / 0,88 | 0,75 / 0,65 / 0,70 |
| positive | 0,85 / 0,97 / 0,91 | 0,66 / 0,70 / 0,68 |

Exactitude par phénomène sur `textsense_eval_v1` (n = nombre d'exemples ;
petits effectifs → tendances) :

| Phénomène | n | Modèle d'origine | Ce modèle |
|---|---|---|---|
| explicite | 80 | 0,99 | 1,00 |
| négation | 40 | 0,95 | 0,97 |
| contraste | 30 | 0,90 | 0,97 |
| implicite | 30 | 0,90 | 0,80 |
| **ironie** | 15 | 0,27 | **0,20** |
| factuel | 45 | 0,00 | 0,93 |
| question | 20 | 0,05 | 1,00 |
| consigne / annonce | 20 | 0,00 | 0,80 |
| lexique émotionnel sans jugement | 15 | 0,07 | 0,33 |

**Stabilité** : même configuration avec trois graines (42, 13, 7) → macro-F1
0,883 ± 0,003 sur `textsense_eval_v1` et 0,724 ± 0,004 sur `tweets_fr_test`.

**Version ONNX int8** (69 Mo, ~4× plus rapide sur CPU) : macro-F1 0,882 sur
`textsense_eval_v1` et 0,705 sur `tweets_fr_test`.

Ablations et analyse détaillée :
[`docs/EXPERIMENTS.md`](EXPERIMENTS.md) et
[`reports/RESULTS.md`](../reports/RESULTS.md).

### Calibration

Seuil `low_confidence` choisi sur la validation : le plus petit seuil tel que
les prédictions au-dessus atteignent 85 % d'exactitude → **0,81** (73,8 % des
prédictions de validation jugées fiables). Sur les jeux de test :

| Jeu | Prédictions au-dessus du seuil | Exactitude de ces prédictions | Exactitude des prédictions signalées |
|---|---|---|---|
| `textsense_eval_v1` | 73,6 % | 93,2 % | 75,8 % |
| `tweets_fr_test` | 73,1 % | 79,9 % | 52,1 % |

Sur les tweets de test, le modèle est un peu trop confiant (ECE 0,146) : le
seuil n'y atteint pas les 85 % visés.

## Limites et biais

- **Ironie** : très mal gérée (3 phrases ironiques sur 15 correctement
  classées) ; une phrase ironique aux mots positifs est en général prédite
  positive avec une confiance élevée.
- **Lexique émotionnel** : un texte qui *décrit* une émotion sans jugement
  (« un film sur un homme triste ») est souvent pris pour une opinion.
- **Domaine** : entraîné sur des tweets d'environnement et de politique
  (2010-2020 environ). Les performances sur d'autres domaines (support client,
  santé, finance) doivent être mesurées avant usage.
- **Jeu `textsense_eval_v1`** : rédigé pour le projet avec assistance IA,
  distribution artificielle (classes équilibrées, cas difficiles
  sur-représentés) ; il ne remplace pas une évaluation sur des données réelles.
- **Biais hérités** : le modèle de départ et les tweets peuvent véhiculer des
  biais (sujets, registres, groupes sociaux) qui n'ont pas été audités.

## Références

- Delestre, C. et Amar, A. (2022). *DistilCamemBERT : une distillation du
  modèle français CamemBERT*. CAp 2022.
- Barbieri, F., Espinosa Anke, L. et Camacho-Collados, J. (2022). *XLM-T:
  Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*.
  LREC 2022.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT 2019.
