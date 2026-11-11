# Expériences — du modèle « étoiles » au modèle 3 classes

Ce document retrace le raisonnement : la question posée, le protocole, les
résultats et les décisions. Les chiffres détaillés sont dans
[`reports/RESULTS.md`](../reports/RESULTS.md) (généré automatiquement) et les
analyses dans [`notebooks/`](../notebooks/).

## 1. Point de départ et problème

L'API utilisait `cmarkea/distilcamembert-base-sentiment`, qui prédit une note
de 1 à 5 étoiles, convertie en sentiment : 1-2★ → négatif, 3★ → neutre,
4-5★ → positif (somme des probabilités).

**Hypothèse** : « 3 étoiles » (un avis mitigé) n'est pas la même chose que
« neutre » (pas d'opinion : information, question, consigne). Le modèle n'a
jamais vu de texte sans opinion pendant son entraînement ; il devrait donc
mal reconnaître la classe neutre.

Il n'existait aucune mesure pour le vérifier : le projet n'avait pas de jeu
d'évaluation, et la seule métrique affichée (95 % pour le modèle de secours)
était gonflée par une fuite de données (phrases synthétiques quasi
identiques entre entraînement et test).

## 2. Mesurer d'abord

### Jeux de test

| Jeu | Taille | Rôle |
|---|---|---|
| [`textsense_eval_v1`](../data/eval/DATASHEET.md) | 360 (120 / classe) | Textes courants, cas difficiles étiquetés par phénomène. Cible principale de l'application. |
| `tweets_fr_test` | 870 (290 / classe) | Jeu public, permet la comparaison avec d'autres travaux et vérifie le domaine d'entraînement. |

**Métrique principale : macro-F1.** L'exactitude seule masque le problème : le
modèle d'origine atteint 61,7 % d'exactitude sur `textsense_eval_v1` tout en
ratant 118 phrases neutres sur 120.

**Incertitude** : chaque score est accompagné d'un intervalle de confiance à
95 % par bootstrap (1 000 rééchantillonnages). Sur 360 exemples, il fait
environ ±3 points.

### Résultats des modèles de référence

| Modèle | macro-F1 eval_v1 | F1 neutre eval_v1 | macro-F1 tweets test |
|---|---|---|---|
| Étoiles regroupées | 0,510 | **0,031** | 0,363 |
| TF-IDF + régression logistique | 0,513 | 0,651 | 0,264 |

L'hypothèse est confirmée : sur 120 phrases neutres, le modèle d'origine en
classe 86 en **positif**, 32 en négatif et 2 seulement en neutre. Il est en
revanche excellent sur les négations (95 %) et les contrastes (90 %).

**Décision** : le défaut est structurel (définition de la classe), pas un
problème de seuil. Il faut entraîner un modèle qui prédit directement les
3 classes, sur des données contenant de vrais textes neutres.

## 3. Fine-tuning

### Choix des données

Les corpus français publics sont soit binaires (Allociné), soit sans classe
neutre. La partie française de `cardiffnlp/tweet_sentiment_multilingual` est
le seul jeu public trouvé avec **3 classes équilibrées** et des splits
train / validation / test officiels. Limite assumée : domaine différent
(tweets sur l'environnement et la politique) de l'usage visé — d'où
l'évaluation sur deux jeux.

Les données sont téléchargées à une **révision figée** du Hub
(`14b13ed`) et ne sont pas recopiées dans le dépôt.

### Protocole

```
tweets_fr_train ──► entraînement
tweets_fr_validation ──► meilleure époque (macro-F1) + seuil de confiance
tweets_fr_test, textsense_eval_v1 ──► évaluation finale, une seule fois
```

Aucune décision n'est prise en regardant les jeux de test. La configuration
principale (ci-dessous) a été fixée avant toute évaluation ; les variantes
servent à **expliquer** le résultat, pas à choisir le meilleur chiffre de test.

### Configuration principale

- Départ : `cmarkea/distilcamembert-base-sentiment` (il sait déjà distinguer
  positif et négatif en français).
- Tête 3 classes **initialisée depuis la tête étoiles** : les poids de sortie
  « négatif » sont la moyenne des lignes 1★ et 2★, « neutre » = 3★,
  « positif » = moyenne 4★ et 5★. Le modèle démarre avec un comportement
  proche de l'original au lieu d'une tête aléatoire.
- AdamW, lr 3e-5, warmup 10 %, weight decay 0,01, batch 16, 4 époques max,
  early stopping (patience 2), max 128 tokens, graine 42.
- Entraînement sur CPU (i7-1255U) : ~9 minutes.

### Courbes

| Époque | Perte validation | Macro-F1 validation |
|---|---|---|
| 1 | 0,728 | 0,681 |
| 2 | 0,615 | 0,732 |
| 3 | 0,620 | 0,741 |
| 4 | 0,653 | 0,757 |

La perte de validation remonte après l'époque 2 alors que le macro-F1
continue de progresser : le modèle devient **plus confiant** (y compris sur
ses erreurs) tout en classant mieux. C'est un début de sur-apprentissage qui
touche la calibration plus que le classement ; il explique en partie l'ECE
plus élevé sur les tweets. L'époque retenue est la 4 (critère : macro-F1).

### Résultats

| Modèle | macro-F1 eval_v1 | F1 neutre eval_v1 | macro-F1 tweets test |
|---|---|---|---|
| Étoiles regroupées | 0,510 [0,480–0,541] | 0,031 | 0,363 [0,335–0,390] |
| **Fine-tuné 3 classes** | **0,885 [0,851–0,917]** | **0,882** | **0,723 [0,692–0,754]** |

Les intervalles ne se chevauchent pas : l'amélioration est nette, bien
au-delà de l'incertitude due à la taille des jeux.

Par phénomène (`textsense_eval_v1`) : les classes neutres passent de 0 % à
80-100 % (factuel 93 %, questions 100 %, consignes 80 %), et les acquis sont
conservés en négation (97 %) et contraste (97 %).

**Ce qui ne s'améliore pas** :

- **ironie** : 20 % (3/15), contre 27 % avant — aucun progrès, l'écart est dans
  le bruit (15 exemples). Les tweets d'entraînement contiennent peu d'ironie
  étiquetée ;
- **lexique émotionnel sans jugement** (« un film sur un homme triste ») :
  33 % ;
- **implicite** : 80 % contre 90 % avant (3 phrases d'écart sur 30, dans la
  marge d'incertitude).

## 4. Variance et ablations

Toutes les variantes ont été évaluées sur les mêmes jeux, avec le même code.

### Variance entre graines

Même configuration, graines 42, 13 et 7 :

| Graine | macro-F1 eval_v1 | macro-F1 tweets test |
|---|---|---|
| 42 | 0,885 | 0,723 |
| 13 | 0,883 | 0,728 |
| 7 | 0,880 | 0,720 |
| **moyenne ± écart-type** | **0,883 ± 0,003** | **0,724 ± 0,004** |

Le résultat est **stable** : le hasard de l'entraînement fait varier le score
d'environ 0,3-0,4 point. Une différence de moins de ~1 point entre deux
configurations n'est donc pas interprétable avec une seule graine.

### Ablation 1 — initialisation de la tête

| Tête 3 classes | macro-F1 eval_v1 | macro-F1 tweets test | ECE eval_v1 / tweets | Meilleure époque |
|---|---|---|---|---|
| depuis la tête étoiles (principal) | 0,885 | 0,723 | 0,049 / 0,146 | 4 |
| aléatoire | 0,877 | 0,723 | **0,026 / 0,123** | 3 |

**L'hypothèse n'est pas confirmée** : initialiser la tête depuis les étoiles
n'améliore pas le classement (écart de 0,8 point sur eval_v1, du même ordre
que la variance entre graines ; identique sur les tweets). La tête aléatoire
s'arrête une époque plus tôt et est **mieux calibrée**. Interprétation : les
~68 M paramètres de l'encodeur sont fine-tunés ; la tête (2 300 paramètres
de sortie) se réapprend en quelques centaines de pas, son initialisation
compte peu.

**Décision** : on garde le modèle principal (choisi avant de voir le test ;
changer maintenant reviendrait à sélectionner sur le test). Pour un prochain
entraînement, la tête aléatoire est préférable : plus simple, aussi bonne et
mieux calibrée.

### Ablation 2 — modèle de départ

| Modèle de départ | macro-F1 eval_v1 | macro-F1 tweets test |
|---|---|---|
| `distilcamembert-base-sentiment` (déjà entraîné sur des avis) | **0,885** [0,851–0,917] | 0,723 [0,692–0,754] |
| `distilcamembert-base` (générique) | 0,783 [0,738–0,826] | **0,743** [0,713–0,772] |

C'est le résultat le plus instructif :

- **dans le domaine d'entraînement** (tweets), le modèle générique fait un peu
  mieux (+2 points, intervalles qui se chevauchent) ;
- **hors domaine** (eval_v1), il perd **10 points** (intervalles disjoints).
  Le détail par phénomène montre où : négation 0,80 contre 0,97, contraste
  0,73 contre 0,97, implicite 0,47 contre 0,80. Le pré-entraînement sur des
  avis apporte une compréhension de la négation et du contraste que 1 839
  tweets ne suffisent pas à apprendre.

**Leçon méthodologique** : avec le seul jeu de test public, on aurait choisi
le modèle générique — et dégradé l'application sur ses vrais textes. Évaluer
sur des données proches de l'usage réel est indispensable.

### Synthèse

| Question | Réponse |
|---|---|
| Le fine-tuning 3 classes corrige-t-il le neutre ? | Oui : F1 neutre 0,03 → 0,88 |
| Le résultat est-il stable ? | Oui : ± 0,3 point entre graines |
| L'initialisation de la tête depuis les étoiles aide-t-elle ? | Non (et la tête aléatoire est mieux calibrée) |
| Partir d'un modèle déjà spécialisé en sentiment aide-t-il ? | Oui, fortement hors domaine (+10 points) |

## 5. Calibration du seuil `low_confidence`

Le seuil 0,55 de départ n'avait pas de justification. Méthode (classification
sélective) : sur la **validation**, on cherche le plus petit seuil tel que les
prédictions au-dessus atteignent une exactitude cible.

| Cible | Seuil | Part jugée fiable (validation) |
|---|---|---|
| 80 % | 0,66 | 85,8 % |
| **85 %** | **0,81** | **73,8 %** |
| 90 % | 0,94 | 52,5 % |

**Décision : cible 85 %, seuil 0,81.** À 90 %, la moitié des prédictions
serait signalée, ce qui rend le drapeau inutile ; à 80 %, il signale trop peu
d'erreurs.

Vérification sur les jeux de test (jamais utilisés pour le choix) :

| Jeu | Part jugée fiable | Exactitude gardées | Exactitude signalées |
|---|---|---|---|
| `textsense_eval_v1` | 73,6 % | 93,2 % | 75,8 % |
| `tweets_fr_test` | 73,1 % | 79,9 % | 52,1 % |

Le seuil sépare bien les prédictions fiables des autres, mais sur les tweets
de test la cible de 85 % n'est pas tout à fait atteinte (79,9 %) : le seuil
estimé sur 324 exemples est optimiste. Piste : *temperature scaling* pour
recalibrer les probabilités.

Le seuil est enregistré **dans le `config.json` du modèle**
(`textsense_confidence_threshold`) : il voyage avec le modèle, et
l'application l'applique automatiquement (sauf si `CONFIDENCE_THRESHOLD` est
défini).

## 6. Export ONNX et quantification

**Objectif** : servir le modèle plus vite et plus léger sur CPU, sans
dégrader la qualité — et le vérifier.

Commande : `make export-onnx` (export opset 17, puis quantification dynamique
des poids en int8 avec ONNX Runtime). Latence mesurée en requêtes unitaires
(un texte à la fois, comme dans l'API) sur 100 textes × 3 répétitions, CPU
i7-1255U ; rapport : `reports/onnx/distilcamembert-3c-starinit-s42.json`.

| Variante | Taille | Latence p50 | Latence p95 | Accord avec PyTorch (eval_v1) | macro-F1 eval_v1 | macro-F1 tweets test |
|---|---|---|---|---|---|---|
| PyTorch float32 | 272 Mo | 17,2 ms | 32,7 ms | – | 0,885 | 0,723 |
| ONNX float32 | 272 Mo | 10,3 ms | 23,5 ms | 100 % | 0,885 | 0,723 |
| **ONNX int8 (per-channel)** | **69 Mo** | **4,4 ms** | **10,1 ms** | 95,8 % | 0,882 | 0,705 |

### Incident : quantification per-tensor

La première quantification (réglages par défaut : **une seule échelle par
matrice de poids**) donnait un modèle inutilisable : 31 % d'accord avec
PyTorch, 34 % d'exactitude, soit le hasard. Sans la mesure d'accord, ce modèle
aurait pu partir en production en ayant l'air de fonctionner (il renvoie des
prédictions plausibles).

Cause : quelques poids de valeur extrême imposent une échelle large à toute
la matrice, et les autres poids perdent leur précision. Variantes testées :

| Réglage | Accord | Exactitude eval_v1 |
|---|---|---|
| QInt8 per-tensor (défaut) | 30,8 % | 33,9 % |
| QInt8 per-tensor, `reduce_range` | 30,3 % | 33,3 % |
| QUInt8 per-tensor | 54,7 % | 57,2 % |
| **QInt8 per-channel** | **95,8 %** | **88,3 %** |
| QInt8 per-channel, MatMul seulement (144 Mo) | 97,2 % | 88,9 % |

**Décision** : per-channel sur tous les opérateurs (4× plus petit). Coût :
-0,3 point sur eval_v1, -1,8 point sur les tweets (au-delà de la variance entre
graines, intervalles qui se chevauchent).

**Recommandation** : ONNX float32 quand la qualité prime (résultats
identiques, 1,7× plus rapide) ; ONNX int8 quand la mémoire ou la latence
priment (4× plus léger et ~4× plus rapide). Le backend PyTorch reste le
défaut pour sa simplicité.

## 7. Pistes suivantes

1. **Données réelles** : exploiter les retours utilisateurs
   (`python -m textsense_ml.export_feedback`), les relire selon le
   [guide d'annotation](../data/eval/ANNOTATION_GUIDELINES.md), et fine-tuner
   sur ce domaine.
2. **Ironie** : ajouter des exemples ironiques annotés à l'entraînement
   (jamais ceux du jeu de test).
3. **Calibration** : *temperature scaling* sur la validation.
4. **Relecture humaine** de `textsense_eval_v1` → v2, et un second annotateur
   pour mesurer l'accord inter-annotateurs (kappa de Cohen).
