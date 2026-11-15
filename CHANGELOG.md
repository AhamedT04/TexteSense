# Journal des modifications

Toutes les évolutions notables de TextSense sont consignées ici.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le
projet respecte le [versionnage sémantique](https://semver.org/lang/fr/).

## [Non publié]

## [1.3.1] — 2026-11-15

### Corrigé

- **Démo : le modèle int8 prédisait « neutre » pour tous les textes** sur le
  serveur. Sur les processeurs x86 sans VNNI, ONNX Runtime sature avec des
  poids int8 sur 8 bits ; la quantification utilise maintenant
  `reduce_range=True` (macro-F1 0,887 sur `textsense_eval_v1`, 0,692 sur
  `tweets_fr_test`). Détails : `docs/EXPERIMENTS.md`, incident n° 2.

### Ajouté

- **Test de fumée** au chargement du modèle principal : un modèle qui prédit
  mal des phrases évidentes est rejeté (bascule sur le modèle de secours).
- `REQUIRE_PRIMARY_MODEL` : `/health` renvoie 503 si le modèle principal
  n'est pas validé (activé dans l'image de démo, pour que l'hébergeur ne
  mette pas en ligne une version cassée).

## [1.3.0] — 2026-11-14

### Ajouté

- **Démo publique** sur Render ([textesense.onrender.com](https://textesense.onrender.com)) :
  image dédiée sans PyTorch (ONNX int8, révision du modèle figée),
  `render.yaml` pour le déploiement continu depuis `main`, et script
  équivalent pour Hugging Face Spaces (`deploy/deploy_space.py`).
- **Mode démo** (`DEMO_MODE`) : chaque visiteur ne voit que ses analyses,
  limite de débit par adresse IP et globale (réponse 429 avec `Retry-After`),
  avertissement de confidentialité et liens vers le code et le modèle.
- **Sauvegarde des retours utilisateurs** vers un dataset privé du Hub
  (`FEEDBACK_REPO_ID`), sans identifiant de session ni adresse IP.
- Version ONNX int8 publiable dans le dossier `onnx/` du dépôt du modèle
  (`publish_model --onnx`).

### Modifié

- L'export CSV des retours et la sauvegarde de la démo partagent la même
  extraction (`app/feedback_backup.py`).

## [1.2.0] — 2026-11-12

### Ajouté

- **Modèle fine-tuné en 3 classes** (`distilcamembert-3c-starinit-s42`),
  publié sur le Hub :
  [`AhamedT04/textsense-distilcamembert-sentiment-3c`](https://huggingface.co/AhamedT04/textsense-distilcamembert-sentiment-3c).
  Macro-F1 0,885 sur `textsense_eval_v1` (contre 0,510) et 0,723 sur
  `tweets_fr_test` (contre 0,363).
- **Pipeline ML reproductible** (`textsense_ml/`, `configs/`, `Makefile`) :
  évaluation avec intervalles de confiance bootstrap, fine-tuning piloté par
  YAML, ablations, variance entre graines, calibration du seuil, export ONNX
  fp32/int8, publication sur le Hub.
- **Jeu d'évaluation** `data/eval/textsense_eval_v1.csv` (360 phrases annotées
  par phénomène), avec datasheet et guide d'annotation.
- **Retours utilisateurs** : confirmer ou corriger une prédiction depuis
  l'interface (`PUT`/`DELETE /analysis/{id}/feedback`), statistiques
  associées et export (`textsense_ml.export_feedback`).
- **Backend ONNX** (`MODEL_BACKEND=onnx`) et seuil de confiance calibré
  stocké avec le modèle.
- Fiche du modèle, protocole expérimental, rapport de résultats et notebooks
  d'analyse.
- CI GitHub Actions, Dockerfile, licence MIT, `requirements-dev.txt`.

### Modifié

- Interface commune pour les classifieurs Transformers, ONNX et sklearn.
- Chaque analyse enregistre le modèle qui l'a produite (`model_name`).
- `/health` indique le backend et le modèle chargé, sans chemin serveur.
- Variable `CONFIDENCE_THRESHOLD` (l'ancien nom `HF_CONFIDENCE_THRESHOLD`
  reste accepté).
- Le fallback TF-IDF est évalué sur des phrases non vues (≈ 55 % réels au lieu
  de ≈ 95 % gonflés).

### Corrigé

- Migration de la base portable (fonctionne aussi sur PostgreSQL).
- Dates renvoyées en UTC explicite (décalage horaire dans l'interface).
- Un échec de chargement du modèle n'est plus retenté à chaque requête.
- Les tests utilisent une base temporaire et ne modifient plus la vraie base.
- Mots-clés : stopwords complétés et tokenisation Unicode (œ, æ…).
- Chemins résolus depuis la racine du projet.

## [1.1.0] — 2026-04-17

### Ajouté

- Configuration centralisée par variables d'environnement (`.env`).
- Indicateur `low_confidence` et pagination de `/history`.
- Migration automatique de la base au démarrage.
- Suite de tests étendue (37 tests).

### Modifié

- Logging structuré et pré-chargement des modèles au démarrage.
- Liste de stopwords français complétée.

## [1.0.0] — 2026-03-27

### Ajouté

- Première version : API FastAPI, interface web, analyse de sentiment avec
  DistilCamemBERT et fallback TF-IDF, extraction de mots-clés, historique
  et statistiques.

[Non publié]: https://github.com/AhamedT04/TexteSense/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/AhamedT04/TexteSense/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/AhamedT04/TexteSense/compare/f033e3e...v1.3.0
[1.2.0]: https://github.com/AhamedT04/TexteSense/compare/7e63d72...f033e3e
[1.1.0]: https://github.com/AhamedT04/TexteSense/compare/abb7f22...7e63d72
[1.0.0]: https://github.com/AhamedT04/TexteSense/commit/abb7f22
