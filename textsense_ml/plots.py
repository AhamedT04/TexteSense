"""Figures des rapports (matplotlib, rendu sans écran)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS_FR = {"negative": "négatif", "neutral": "neutre", "positive": "positif"}


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    counts = np.asarray(matrix)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    names = [LABELS_FR.get(label, label) for label in labels]
    ax.set_xticks(range(len(labels)), names)
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité")
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if normalized[i, j] > 0.6 else "black"
            ax.text(j, i, f"{counts[i, j]}\n{normalized[i, j]:.0%}", ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_selective_curve(curves: dict[str, list[dict]], chosen: float | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, rows in curves.items():
        ax.plot([r["coverage"] for r in rows], [r["accuracy"] for r in rows], marker=".", label=name)
    ax.set_xlabel("Couverture (part des prédictions jugées fiables)")
    ax.set_ylabel("Exactitude sur ces prédictions")
    ax.set_title("Compromis couverture / exactitude selon le seuil", fontsize=10)
    ax.grid(alpha=0.3)
    if chosen is not None:
        ax.set_title(f"Compromis couverture / exactitude (seuil retenu : {chosen:.2f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(log_history: list[dict], path: Path) -> None:
    train = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
    evals = [(e["epoch"], e["eval_loss"], e.get("eval_macro_f1")) for e in log_history if "eval_loss" in e]

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(9, 3.5))
    if train:
        ax_loss.plot(*zip(*train), label="entraînement", alpha=0.7)
    if evals:
        ax_loss.plot([e[0] for e in evals], [e[1] for e in evals], marker="o", label="validation")
        ax_f1.plot([e[0] for e in evals], [e[2] for e in evals], marker="o", color="tab:green")
    ax_loss.set_xlabel("Époque")
    ax_loss.set_ylabel("Perte (cross-entropy)")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    ax_f1.set_xlabel("Époque")
    ax_f1.set_ylabel("Macro-F1 (validation)")
    ax_f1.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
