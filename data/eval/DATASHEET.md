# Fiche du jeu de données — `textsense_eval_v1`

Format inspiré des *Datasheets for Datasets* (Gebru et al., 2021).

## En bref

| | |
|---|---|
| Fichier | [`textsense_eval_v1.csv`](textsense_eval_v1.csv) |
| Taille | 360 textes |
| Langue | français |
| Classes | `negative` 120 · `neutral` 120 · `positive` 120 |
| Usage | **évaluation finale uniquement** |
| Version | v1 (novembre 2026) |

## Pourquoi ce jeu existe

Les jeux publics de sentiment en français sont soit binaires (Allociné :
positif / négatif), soit d'un domaine très particulier (tweets politiques ou
environnementaux). Aucun ne mesure ce qui compte pour TextSense : des textes
courants (avis, messages, consignes) **avec une vraie classe neutre** et des
cas difficiles identifiés.

Ce jeu sert de **jeu de test** : il ne doit jamais être utilisé pour entraîner
un modèle, choisir une époque, régler un hyperparamètre ou un seuil. Sinon, le
score obtenu dessus ne mesurerait plus la capacité à généraliser.

## Composition

Colonnes : `id`, `text`, `label`, `phenomenon`, `domain`.

| Phénomène | Description | positive | negative | neutral |
|---|---|---|---|---|
| `explicit` | sentiment exprimé sans ambiguïté | 40 | 40 | – |
| `negation` | le sens dépend d'une négation (« pas mal du tout ») | 20 | 20 | – |
| `contrast` | « mais », « malgré »… : la conclusion l'emporte | 15 | 15 | – |
| `implicit` | sentiment sans mot évaluatif (« Plus jamais. ») | 15 | 15 | – |
| `irony` | sens inverse des mots employés | – | 15 | – |
| `comparative` | jugement par comparaison | 10 | – | – |
| `informal` | registre familier, argot, émojis | 10 | 10 | – |
| `long` | plusieurs phrases | 10 | 5 | 10 |
| `factual` | information factuelle | – | – | 45 |
| `question` | question sans jugement | – | – | 20 |
| `instruction` | consigne, annonce | – | – | 20 |
| `emotional_lexicon` | mots émotionnels sans jugement de l'auteur (« un homme triste ») | – | – | 15 |
| `short` | phrase courte et banale | – | – | 10 |

Domaines (15) : restauration, produit, tech, service client, voyage, travail,
éducation, cinéma, culture, santé, loisirs, sport, actualités, quotidien.

## Collecte et annotation

- **Origine** : phrases **rédigées pour ce projet** (pas de données
  collectées auprès de personnes réelles). Elles ont été écrites et
  pré-annotées avec l'aide d'un assistant IA, en suivant le
  [guide d'annotation](ANNOTATION_GUIDELINES.md).
- **Relecture** : chaque étiquette doit être validée par une relecture
  humaine. Les éventuelles corrections seront publiées dans une version v2,
  sans jamais modifier la v1 (pour garder les résultats comparables).
- **Contrôle de fuite** : aucune phrase n'apparaît dans les données
  d'entraînement (CSV du modèle fallback, tweets `train` / `validation` /
  `test`). Vérification par correspondance exacte après normalisation (casse,
  ponctuation).

## Limites connues

- **Taille** : 360 exemples, soit environ ±3 points d'incertitude sur le
  macro-F1 (voir les intervalles de confiance dans
  [`reports/RESULTS.md`](../../reports/RESULTS.md)). Les scores par phénomène
  portent sur 10 à 45 exemples : ce sont des tendances, pas des mesures
  précises.
- **Distribution artificielle** : classes équilibrées et cas difficiles
  sur-représentés par construction. Le score ne prédit pas directement la
  performance sur un trafic réel, où les proportions diffèrent.
- **Rédaction unique** : un seul style d'écriture, une seule source. Un jeu
  issu de vrais utilisateurs (retours collectés dans l'application) sera plus
  représentatif.
- **Ironie uniquement négative** : l'ironie positive est rare et ambiguë ;
  elle n'est pas couverte.

## Utilisation

```bash
python -m textsense_ml.evaluate --name <nom> --model <spec> --datasets textsense_eval_v1
```

Licence : identique à celle du projet (MIT).
