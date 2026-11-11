"""Agrège tous les rapports en un tableau de résultats : ``reports/RESULTS.md``.

Usage ::

    python -m textsense_ml.report
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import BASE_DIR

REPORTS_DIR = BASE_DIR / "reports"
DATASET_ORDER = ["textsense_eval_v1", "tweets_fr_test", "tweets_fr_validation"]
PHENOMENA_ORDER = [
    "explicit",
    "negation",
    "contrast",
    "implicit",
    "irony",
    "comparative",
    "informal",
    "long",
    "factual",
    "question",
    "instruction",
    "emotional_lexicon",
    "short",
]


def _load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(REPORTS_DIR.glob(pattern))]


def _fmt_ci(metrics: dict) -> str:
    low, high = metrics["ci95"]["macro_f1"]
    return f"**{metrics['macro_f1']:.3f}** <sub>[{low:.3f}–{high:.3f}]</sub>"


def build_markdown() -> str:
    evaluations = _load("eval/*/*/metrics.json")
    trainings = _load("training/*/training_info.json")
    runs = sorted({e["run"] for e in evaluations})
    by_key = {(e["run"], e["dataset"]): e for e in evaluations}
    datasets = [d for d in DATASET_ORDER if any(e["dataset"] == d for e in evaluations)]

    lines = [
        "# Résultats",
        "",
        "> Fichier généré par `python -m textsense_ml.report` à partir de `reports/` — ne pas modifier à la main.",
        "> Protocole et analyse : [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).",
        "",
        "## Vue d'ensemble",
        "",
        "Macro-F1 avec intervalle de confiance à 95 % (bootstrap, 1 000 rééchantillonnages).",
        "",
    ]
    for dataset in datasets:
        lines += [
            f"### `{dataset}`",
            "",
            "| Modèle | Macro-F1 [IC 95 %] | Exactitude | F1 négatif | F1 neutre | F1 positif | ECE |",
            "|---|---|---|---|---|---|---|",
        ]
        for run in runs:
            entry = by_key.get((run, dataset))
            if entry is None:
                continue
            m = entry["metrics"]
            per_class = m["per_class"]
            lines.append(
                f"| `{run}` | {_fmt_ci(m)} | {m['accuracy']:.3f} | {per_class['negative']['f1']:.3f} | "
                f"{per_class['neutral']['f1']:.3f} | {per_class['positive']['f1']:.3f} | {m['ece']:.3f} |"
            )
        lines.append("")

    eval_runs = [r for r in runs if (r, "textsense_eval_v1") in by_key]
    if eval_runs:
        lines += [
            "## Exactitude par phénomène linguistique (`textsense_eval_v1`)",
            "",
            "| Phénomène | n | " + " | ".join(f"`{r}`" for r in eval_runs) + " |",
            "|---|---|" + "---|" * len(eval_runs),
        ]
        groups = by_key[(eval_runs[0], "textsense_eval_v1")]["metrics"]["by_group"]["phenomenon"]
        for phenomenon in [p for p in PHENOMENA_ORDER if p in groups]:
            cells = [
                f"{by_key[(r, 'textsense_eval_v1')]['metrics']['by_group']['phenomenon'][phenomenon]['accuracy']:.2f}"
                for r in eval_runs
            ]
            lines.append(f"| {phenomenon} | {groups[phenomenon]['n']} | " + " | ".join(cells) + " |")
        lines.append("")

    if trainings:
        lines += [
            "## Entraînements",
            "",
            "| Exécution | Modèle de départ | Tête | Graine | Macro-F1 validation | Durée (CPU) |",
            "|---|---|---|---|---|---|",
        ]
        for info in sorted(trainings, key=lambda i: i["config"]["name"]):
            config = info["config"]
            lines.append(
                f"| `{config['name']}` | `{config['base_model']}` | {config['head_init']} | {config['seed']} | "
                f"{info['best_validation'].get('eval_macro_f1', float('nan')):.3f} | "
                f"{info['training_seconds'] / 60:.1f} min |"
            )
        lines.append("")

    calibrations = _load("calibration/*.json")
    if calibrations:
        lines += ["## Calibration du seuil de confiance", ""]
        for calibration in calibrations:
            chosen = calibration["chosen"]
            if chosen:
                lines.append(
                    f"- `{calibration['run']}` : seuil **{chosen['threshold']:.2f}** — "
                    f"{chosen['coverage']:.1%} des prédictions jugées fiables, exactitude "
                    f"{chosen['accuracy']:.1%} sur `{calibration['dataset']}` "
                    f"(cible {calibration['target_accuracy']:.0%})."
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    out = REPORTS_DIR / "RESULTS.md"
    out.write_text(build_markdown(), encoding="utf-8")
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
