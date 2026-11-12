# Guide d'annotation — sentiment 3 classes

Ce guide définit comment attribuer une étiquette à un texte. Il s'applique au
jeu d'évaluation `textsense_eval_v1` et à toute annotation future (retours
utilisateurs relus, nouveaux jeux). Une règle écrite rend l'annotation
**cohérente et vérifiable** : deux personnes qui suivent ce guide doivent
aboutir à la même étiquette.

## La question à se poser

> **Quelle attitude l'auteur exprime-t-il envers ce dont il parle ?**

On annote le **sentiment exprimé par l'auteur**, pas le sujet du texte ni les
mots qu'il contient.

| Étiquette | Définition | Exemple |
|---|---|---|
| `positive` | L'auteur exprime une évaluation favorable, une satisfaction, un plaisir. | « Je reviendrai sans hésiter. » |
| `negative` | L'auteur exprime une évaluation défavorable, un mécontentement, une frustration. | « Plus jamais. » |
| `neutral` | Aucune évaluation : information, question, consigne, description. | « Le musée est fermé le mardi. » |

## Règles

1. **Sentiment de l'auteur, pas du sujet.** « Le film raconte l'histoire d'un
   homme triste » → `neutral` : l'auteur décrit, il ne juge pas.
2. **Les mots chargés ne suffisent pas.** « Le sondage mesure la satisfaction
   des clients » → `neutral`. À l'inverse, un texte sans mot émotionnel peut
   être chargé : « On m'a fait payer deux fois la même prestation » → `negative`.
3. **Négation : on annote le sens final.** « Pas mal du tout » → `positive` ;
   « Je ne recommande pas » → `negative`.
4. **Ironie : on annote l'intention réelle.** « Génial, encore une panne » →
   `negative`.
5. **Contraste (« mais », « pourtant », « malgré ») : la conclusion l'emporte.**
   « L'attente était longue, mais le repas valait le coup » → `positive`.
   Si aucune conclusion ne domine, le texte est **exclu** du jeu plutôt que
   forcé dans une classe.
6. **Questions et consignes** → `neutral`, sauf si elles portent clairement un
   jugement (« Sérieux, 50 € pour ça ? » → `negative`).
7. **Faits objectivement bons ou mauvais.** Un fait rapporté sans prise de
   position est `neutral` (« La température atteindra 18 degrés »). Dès que
   l'auteur qualifie le fait (« Les chiffres du chômage sont alarmants »), on
   annote ce jugement.
8. **Doute persistant** → le texte n'entre pas dans le jeu d'évaluation. Un jeu
   d'évaluation doit contenir des étiquettes sûres.

## Métadonnées associées

Chaque exemple porte aussi :

- `phenomenon` — la difficulté linguistique principale (voir
  [`DATASHEET.md`](DATASHEET.md)) ; un seul phénomène par phrase, le plus
  marquant ;
- `domain` — le thème (restauration, tech, santé…), pour vérifier que le
  modèle ne réussit pas uniquement sur certains sujets.
