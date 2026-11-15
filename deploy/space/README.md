---
title: TextSense
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Analyse de sentiment en français, 3 classes
models:
  - AhamedT04/textsense-distilcamembert-sentiment-3c
---

# TextSense — démo

Analyse de sentiment en **français** (positif / neutre / négatif) avec un
DistilCamemBERT fine-tuné en 3 classes, exécuté en ONNX int8 sur CPU.

- Code source et documentation : [github.com/AhamedT04/TexteSense](https://github.com/AhamedT04/TexteSense)
- Fiche du modèle (résultats, limites) : [AhamedT04/textsense-distilcamembert-sentiment-3c](https://huggingface.co/AhamedT04/textsense-distilcamembert-sentiment-3c)

## Fonctionnement de la démo

- Vos analyses ne sont visibles que par vous et sont effacées au redémarrage
  du Space.
- Si vous confirmez ou corrigez un résultat, le texte et votre avis sont
  conservés de façon anonyme (sans identifiant ni adresse IP) dans un dataset
  privé, pour évaluer et améliorer le modèle.
- Les analyses sont limitées à 30 par minute et par adresse IP.

Limites connues du modèle : l'ironie est très mal reconnue, et il est un peu
trop confiant sur les tweets. Détails dans la fiche du modèle.
