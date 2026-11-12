"""Génère les notebooks d'analyse à partir de ce script (source de vérité versionnable).

Les notebooks ne font que **lire** les rapports produits par ``textsense_ml``
(aucun entraînement) : ils sont rapides à ré-exécuter et leurs chiffres
correspondent toujours à ``reports/``.

Usage ::

    python notebooks/_build_notebooks.py
    make notebooks        # ré-exécute et enregistre les sorties
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """\
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
REPORTS = ROOT / "reports"
pd.set_option("display.precision", 3)


def load_metrics(run, dataset):
    path = REPORTS / "eval" / run / dataset / "metrics.json"
    return json.loads(path.read_text())["metrics"] if path.exists() else None


def overview(runs, datasets):
    rows = []
    for run in runs:
        for dataset in datasets:
            m = load_metrics(run, dataset)
            if m is None:
                continue
            low, high = m["ci95"]["macro_f1"]
            rows.append({
                "modèle": run, "jeu": dataset, "macro-F1": m["macro_f1"],
                "IC 95 %": f"[{low:.3f} – {high:.3f}]", "exactitude": m["accuracy"],
                "F1 nég.": m["per_class"]["negative"]["f1"],
                "F1 neutre": m["per_class"]["neutral"]["f1"],
                "F1 pos.": m["per_class"]["positive"]["f1"], "ECE": m["ece"],
            })
    return pd.DataFrame(rows).set_index(["jeu", "modèle"]).sort_index()


def by_phenomenon(runs):
    data = {}
    for run in runs:
        m = load_metrics(run, "textsense_eval_v1")
        if m:
            data[run] = {k: v["accuracy"] for k, v in m["by_group"]["phenomenon"].items()}
    df = pd.DataFrame(data)
    counts = load_metrics(runs[0], "textsense_eval_v1")["by_group"]["phenomenon"]
    df.insert(0, "n", [counts[p]["n"] for p in df.index])
    return df
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def notebook_baselines():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 01 — Évaluation des modèles de référence\n\n"
            "**Question** : que valent les modèles d'origine du projet, et où se trompent-ils ?\n\n"
            "**Modèles évalués**\n"
            "- `baseline-stars` : `cmarkea/distilcamembert-base-sentiment` (5 étoiles regroupées : "
            "1-2★ → négatif, 3★ → neutre, 4-5★ → positif) — le modèle utilisé par l'API à l'origine ;\n"
            "- `tfidf-fallback` : TF-IDF + régression logistique (modèle de secours).\n\n"
            "**Jeux de test**\n"
            "- `textsense_eval_v1` : 360 phrases du projet, 120 par classe, cas difficiles annotés "
            "([fiche](../data/eval/DATASHEET.md)) ;\n"
            "- `tweets_fr_test` : 870 tweets français annotés en 3 classes (jeu public, autre domaine).\n\n"
            "Les chiffres viennent de `reports/eval/` (produits par `make eval-baselines`)."
        ),
        code(SETUP),
        md("## Vue d'ensemble\n\nMétrique principale : **macro-F1** (chaque classe compte autant)."),
        code('overview(["baseline-stars", "tfidf-fallback"], ["textsense_eval_v1", "tweets_fr_test"])'),
        md(
            "## Matrices de confusion\n\n"
            "Lignes = vraie classe, colonnes = prédiction. Pourcentages par ligne (rappel)."
        ),
        code(
            'for run in ["baseline-stars", "tfidf-fallback"]:\n'
            '    display(Image(filename=str(REPORTS / "eval" / run / "textsense_eval_v1" / "confusion_matrix.png"), width=360))'
        ),
        md(
            "## Exactitude par phénomène linguistique\n\n"
            "Sur 10 à 45 exemples par ligne : ce sont des **tendances**, pas des mesures précises."
        ),
        code(
            'table = by_phenomenon(["baseline-stars", "tfidf-fallback"])\n'
            "ax = table.drop(columns='n').plot.barh(figsize=(7, 5))\n"
            'ax.set_xlabel("exactitude"); ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.3)\n'
            "plt.tight_layout(); plt.show()\n"
            "table"
        ),
        md("## Analyse d'erreurs : où va la classe neutre ?"),
        code(
            'pred = pd.read_csv(REPORTS / "eval" / "baseline-stars" / "textsense_eval_v1" / "predictions.csv")\n'
            'neutral = pred[pred.label == "neutral"]\n'
            'display(neutral.pred.value_counts().rename("prédictions pour les 120 phrases neutres").to_frame())\n'
            'neutral[neutral.pred != "neutral"].sort_values("confidence", ascending=False)'
            '[["text", "phenomenon", "pred", "confidence"]].head(10)'
        ),
        md(
            "## Conclusions\n\n"
            "- Le modèle « étoiles » est **très bon en positif / négatif** (négations, contrastes) mais "
            "**ne prédit presque jamais neutre** : une information factuelle ou une question n'est pas un "
            "avis, et la note « 3 étoiles » ne correspond pas à « pas d'opinion ». Les phrases neutres "
            "partent massivement en positif.\n"
            "- Le modèle TF-IDF fait l'erreur inverse : il range beaucoup de textes en neutre, et ne "
            "comprend ni la négation ni le contraste.\n"
            "- Le défaut est **structurel** (mauvaise définition de la classe neutre), pas un problème de "
            "seuil : il faut un modèle entraîné **directement en 3 classes** → notebook 02."
        ),
    ]
    return nb


def notebook_finetuning():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 02 — Fine-tuning en 3 classes : résultats et analyse\n\n"
            "**Question** : un DistilCamemBERT fine-tuné directement en 3 classes corrige-t-il la classe "
            "neutre sans perdre la qualité en positif / négatif ?\n\n"
            "**Protocole** (détails : [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md))\n"
            "- entraînement : `tweets_fr_train` (1 839 tweets) ;\n"
            "- choix de la meilleure époque : `tweets_fr_validation` (324 tweets) ;\n"
            "- test, une seule fois à la fin : `tweets_fr_test` (870) et `textsense_eval_v1` (360)."
        ),
        code(SETUP),
        code(
            'MAIN = "distilcamembert-3c-starinit-s42"\n'
            'SEEDS = ["distilcamembert-3c-starinit-s42", "distilcamembert-3c-starinit-s13", "distilcamembert-3c-starinit-s7"]\n'
            'ABLATIONS = ["distilcamembert-3c-randinit-s42", "distilcamembert-generic-3c-s42"]\n'
            'BASELINES = ["baseline-stars", "tfidf-fallback"]'
        ),
        md("## Courbes d'entraînement\n\nPerte d'entraînement et de validation, macro-F1 de validation par époque."),
        code('display(Image(filename=str(REPORTS / "training" / MAIN / "training_curves.png")))'),
        md("## Comparaison avec les modèles de référence"),
        code('overview(BASELINES + [MAIN], ["textsense_eval_v1", "tweets_fr_test"])'),
        code(
            'for run in ["baseline-stars", MAIN]:\n'
            '    display(Image(filename=str(REPORTS / "eval" / run / "textsense_eval_v1" / "confusion_matrix.png"), width=360))'
        ),
        md(
            "## Variance entre graines\n\n"
            "Même configuration, trois graines aléatoires. L'écart-type indique quelle différence entre deux "
            "modèles peut être due au hasard de l'entraînement."
        ),
        code(
            "rows = []\n"
            "for run in SEEDS:\n"
            '    for dataset in ["textsense_eval_v1", "tweets_fr_test"]:\n'
            "        m = load_metrics(run, dataset)\n"
            "        if m:\n"
            '            rows.append({"graine": run.rsplit("-s", 1)[1], "jeu": dataset, "macro-F1": m["macro_f1"],\n'
            '                         "F1 neutre": m["per_class"]["neutral"]["f1"]})\n'
            "seeds = pd.DataFrame(rows)\n"
            'display(seeds.pivot(index="graine", columns="jeu", values="macro-F1"))\n'
            'seeds.groupby("jeu")[["macro-F1", "F1 neutre"]].agg(["mean", "std"])'
        ),
        md(
            "## Ablations\n\n"
            "- `randinit` : même modèle de départ, tête 3 classes **aléatoire** (au lieu d'être initialisée "
            "depuis la tête « étoiles ») ;\n"
            "- `generic` : DistilCamemBERT **générique**, jamais entraîné au sentiment."
        ),
        code('overview([MAIN] + ABLATIONS, ["textsense_eval_v1", "tweets_fr_test"])'),
        md("## Par phénomène linguistique (`textsense_eval_v1`)"),
        code(
            'table = by_phenomenon(["baseline-stars", MAIN])\n'
            "ax = table.drop(columns='n').plot.barh(figsize=(7, 5))\n"
            'ax.set_xlabel("exactitude"); ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.3)\n'
            "plt.tight_layout(); plt.show()\n"
            "table"
        ),
        md("## Erreurs restantes du modèle fine-tuné"),
        code(
            'pred = pd.read_csv(REPORTS / "eval" / MAIN / "textsense_eval_v1" / "predictions.csv")\n'
            "errors = pred[pred.label != pred.pred]\n"
            'display(pd.crosstab(errors.label, errors.pred, rownames=["vérité"], colnames=["prédiction"]))\n'
            'errors.sort_values("confidence", ascending=False)[["text", "label", "pred", "confidence", "phenomenon"]].head(15)'
        ),
        md(
            "## Seuil de confiance\n\n"
            "Choisi sur la **validation** : le plus petit seuil tel que les prédictions au-dessus atteignent "
            "l'exactitude visée."
        ),
        code(
            'calibration = json.loads((REPORTS / "calibration" / f"{MAIN}.json").read_text())\n'
            'display(Markdown(f"Seuil retenu : **{calibration[\'chosen\'][\'threshold\']:.2f}** "\n'
            '                 f"(cible {calibration[\'target_accuracy\']:.0%} sur `{calibration[\'dataset\']}`)"))\n'
            'display(Image(filename=str(REPORTS / "calibration" / f"{MAIN}.png"), width=480))'
        ),
        md("## Export ONNX"),
        code(
            'onnx = json.loads((REPORTS / "onnx" / f"{MAIN}.json").read_text())["results"]\n'
            "pd.DataFrame({k: {\n"
            '    "taille (Mo)": v["size_mb"], "latence p50 (ms)": v["latency_ms_single_text"]["p50"],\n'
            '    "latence p95 (ms)": v["latency_ms_single_text"]["p95"],\n'
            '    "accord avec PyTorch": v["agreement_with_pytorch"],\n'
            '    "exactitude textsense_eval_v1": v["accuracy_textsense_eval_v1"],\n'
            "} for k, v in onnx.items()}).T"
        ),
        md(
            "## Conclusions\n\n"
            "- **Classe neutre corrigée** : F1 neutre 0,03 → 0,88 sur `textsense_eval_v1` ; macro-F1 "
            "0,51 → 0,885 (intervalles de confiance disjoints). Sur les tweets de test : 0,36 → 0,72.\n"
            "- **Acquis conservés** : négation et contraste restent à ~97 %.\n"
            "- **Résultat stable** : 0,883 ± 0,003 sur trois graines.\n"
            "- **Ablations** : initialiser la tête depuis les étoiles n'apporte rien (la tête aléatoire est "
            "même mieux calibrée) ; partir d'un modèle déjà spécialisé en sentiment est décisif hors "
            "domaine (+10 points sur `textsense_eval_v1`), alors que le modèle générique est un peu meilleur "
            "sur les tweets. Un seul jeu de test aurait conduit au mauvais choix.\n"
            "- **Limites** : ironie (~20 %), textes décrivant une émotion sans jugement (~33 %).\n"
            "- **ONNX int8 per-channel** : 4× plus léger et ~4× plus rapide, -0,3 point sur "
            "`textsense_eval_v1`, -1,8 point sur les tweets. La quantification per-tensor par défaut "
            "cassait le modèle (34 % d'exactitude) : toujours mesurer après quantification."
        ),
    ]
    return nb


def main() -> None:
    for name, builder in [
        ("01_baseline_evaluation.ipynb", notebook_baselines),
        ("02_finetuning_results.ipynb", notebook_finetuning),
    ]:
        nb = builder()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, HERE / name)
        print(f"Écrit : {HERE / name}")


if __name__ == "__main__":
    main()
