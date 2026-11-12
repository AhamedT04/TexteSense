# Résultats

> Fichier généré par `python -m textsense_ml.report` à partir de `reports/` — ne pas modifier à la main.
> Protocole et analyse : [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## Vue d'ensemble

Macro-F1 avec intervalle de confiance à 95 % (bootstrap, 1 000 rééchantillonnages).

### `textsense_eval_v1`

| Modèle | Macro-F1 [IC 95 %] | Exactitude | F1 négatif | F1 neutre | F1 positif | ECE |
|---|---|---|---|---|---|---|
| `baseline-stars` | **0.510** <sub>[0.480–0.541]</sub> | 0.617 | 0.805 | 0.031 | 0.694 | 0.154 |
| `distilcamembert-3c-randinit-s42` | **0.877** <sub>[0.844–0.911]</sub> | 0.878 | 0.860 | 0.874 | 0.897 | 0.026 |
| `distilcamembert-3c-starinit-s13` | **0.883** <sub>[0.848–0.916]</sub> | 0.883 | 0.872 | 0.876 | 0.900 | 0.042 |
| `distilcamembert-3c-starinit-s42` | **0.885** <sub>[0.851–0.917]</sub> | 0.886 | 0.867 | 0.882 | 0.907 | 0.049 |
| `distilcamembert-3c-starinit-s42-onnx-int8` | **0.882** <sub>[0.848–0.914]</sub> | 0.883 | 0.852 | 0.870 | 0.922 | 0.049 |
| `distilcamembert-3c-starinit-s7` | **0.880** <sub>[0.844–0.913]</sub> | 0.881 | 0.877 | 0.865 | 0.897 | 0.040 |
| `distilcamembert-generic-3c-s42` | **0.783** <sub>[0.738–0.826]</sub> | 0.786 | 0.731 | 0.807 | 0.812 | 0.094 |
| `tfidf-fallback` | **0.513** <sub>[0.462–0.564]</sub> | 0.542 | 0.481 | 0.651 | 0.407 | 0.033 |

### `tweets_fr_test`

| Modèle | Macro-F1 [IC 95 %] | Exactitude | F1 négatif | F1 neutre | F1 positif | ECE |
|---|---|---|---|---|---|---|
| `baseline-stars` | **0.363** <sub>[0.335–0.390]</sub> | 0.438 | 0.507 | 0.052 | 0.529 | 0.167 |
| `distilcamembert-3c-randinit-s42` | **0.723** <sub>[0.692–0.754]</sub> | 0.723 | 0.788 | 0.696 | 0.684 | 0.123 |
| `distilcamembert-3c-starinit-s13` | **0.728** <sub>[0.696–0.757]</sub> | 0.728 | 0.790 | 0.707 | 0.687 | 0.145 |
| `distilcamembert-3c-starinit-s42` | **0.723** <sub>[0.692–0.754]</sub> | 0.724 | 0.792 | 0.696 | 0.681 | 0.146 |
| `distilcamembert-3c-starinit-s42-onnx-int8` | **0.705** <sub>[0.672–0.737]</sub> | 0.705 | 0.774 | 0.696 | 0.645 | 0.148 |
| `distilcamembert-3c-starinit-s7` | **0.720** <sub>[0.688–0.749]</sub> | 0.720 | 0.783 | 0.699 | 0.677 | 0.153 |
| `distilcamembert-generic-3c-s42` | **0.743** <sub>[0.713–0.772]</sub> | 0.744 | 0.793 | 0.729 | 0.707 | 0.151 |
| `tfidf-fallback` | **0.264** <sub>[0.238–0.292]</sub> | 0.341 | 0.244 | 0.465 | 0.082 | 0.175 |

### `tweets_fr_validation`

| Modèle | Macro-F1 [IC 95 %] | Exactitude | F1 négatif | F1 neutre | F1 positif | ECE |
|---|---|---|---|---|---|---|
| `baseline-stars` | **0.373** <sub>[0.329–0.414]</sub> | 0.457 | 0.525 | 0.035 | 0.560 | 0.145 |
| `distilcamembert-3c-randinit-s42` | **0.776** <sub>[0.730–0.818]</sub> | 0.778 | 0.842 | 0.709 | 0.777 | 0.075 |
| `distilcamembert-3c-starinit-s13` | **0.760** <sub>[0.714–0.806]</sub> | 0.762 | 0.827 | 0.687 | 0.767 | 0.115 |
| `distilcamembert-3c-starinit-s42` | **0.757** <sub>[0.710–0.803]</sub> | 0.759 | 0.825 | 0.690 | 0.756 | 0.130 |
| `distilcamembert-3c-starinit-s42-onnx-int8` | **0.749** <sub>[0.702–0.797]</sub> | 0.750 | 0.827 | 0.691 | 0.730 | 0.130 |
| `distilcamembert-3c-starinit-s7` | **0.750** <sub>[0.704–0.793]</sub> | 0.753 | 0.820 | 0.673 | 0.758 | 0.124 |
| `distilcamembert-generic-3c-s42` | **0.765** <sub>[0.717–0.809]</sub> | 0.769 | 0.823 | 0.701 | 0.773 | 0.137 |
| `tfidf-fallback` | **0.274** <sub>[0.228–0.324]</sub> | 0.349 | 0.228 | 0.484 | 0.111 | 0.161 |

## Exactitude par phénomène linguistique (`textsense_eval_v1`)

| Phénomène | n | `baseline-stars` | `distilcamembert-3c-randinit-s42` | `distilcamembert-3c-starinit-s13` | `distilcamembert-3c-starinit-s42` | `distilcamembert-3c-starinit-s42-onnx-int8` | `distilcamembert-3c-starinit-s7` | `distilcamembert-generic-3c-s42` | `tfidf-fallback` |
|---|---|---|---|---|---|---|---|---|---|
| explicit | 80 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.51 |
| negation | 40 | 0.95 | 1.00 | 0.95 | 0.97 | 0.95 | 0.97 | 0.80 | 0.42 |
| contrast | 30 | 0.90 | 1.00 | 0.93 | 0.97 | 0.90 | 0.93 | 0.73 | 0.17 |
| implicit | 30 | 0.90 | 0.77 | 0.80 | 0.80 | 0.70 | 0.80 | 0.47 | 0.30 |
| irony | 15 | 0.27 | 0.07 | 0.13 | 0.20 | 0.27 | 0.20 | 0.07 | 0.07 |
| comparative | 10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.40 |
| informal | 20 | 1.00 | 0.80 | 0.95 | 0.90 | 0.95 | 0.95 | 0.80 | 0.40 |
| long | 25 | 0.60 | 0.88 | 0.92 | 0.92 | 0.92 | 0.88 | 0.88 | 0.48 |
| factual | 45 | 0.00 | 0.98 | 0.96 | 0.93 | 0.96 | 0.96 | 0.96 | 0.96 |
| question | 20 | 0.05 | 0.85 | 0.95 | 1.00 | 1.00 | 0.95 | 0.90 | 0.80 |
| instruction | 20 | 0.00 | 0.95 | 0.80 | 0.80 | 0.85 | 0.80 | 0.85 | 0.80 |
| emotional_lexicon | 15 | 0.07 | 0.27 | 0.40 | 0.33 | 0.40 | 0.27 | 0.07 | 1.00 |
| short | 10 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 |

## Entraînements

| Exécution | Modèle de départ | Tête | Graine | Macro-F1 validation | Durée (CPU) |
|---|---|---|---|---|---|
| `distilcamembert-3c-randinit-s42` | `cmarkea/distilcamembert-base-sentiment` | random | 42 | 0.776 | 11.0 min |
| `distilcamembert-3c-starinit-s13` | `cmarkea/distilcamembert-base-sentiment` | star_merge | 13 | 0.760 | 8.7 min |
| `distilcamembert-3c-starinit-s42` | `cmarkea/distilcamembert-base-sentiment` | star_merge | 42 | 0.757 | 8.7 min |
| `distilcamembert-3c-starinit-s7` | `cmarkea/distilcamembert-base-sentiment` | star_merge | 7 | 0.750 | 8.7 min |
| `distilcamembert-generic-3c-s42` | `cmarkea/distilcamembert-base` | random | 42 | 0.765 | 8.3 min |

## Calibration du seuil de confiance

- `distilcamembert-3c-starinit-s42` : seuil **0.81** — 73.8% des prédictions jugées fiables, exactitude 85.4% sur `tweets_fr_validation` (cible 85%).
