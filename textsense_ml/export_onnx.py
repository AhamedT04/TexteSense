"""Exporte un modèle fine-tuné en ONNX (float32 + int8 quantifié) et mesure le gain.

Usage ::

    python -m textsense_ml.export_onnx --model-dir models/distilcamembert-3c-starinit-s42

Produit ``models/<nom>-onnx/`` :

* ``model.onnx``       — graphe ONNX float32 (mêmes poids que PyTorch) ;
* ``model.int8.onnx``  — quantification dynamique des poids en entiers 8 bits ;
* ``config.json`` + fichiers du tokenizer — nécessaires à l'inférence.

Et ``reports/onnx/<nom>.json`` : tailles, latences et accord des prédictions
entre PyTorch et ONNX sur ``textsense_eval_v1``.

Pourquoi : ONNX Runtime exécute le modèle sans PyTorch, plus vite sur CPU ;
la quantification int8 divise la taille par ~4 pour une perte de précision
généralement faible — à vérifier, d'où la mesure d'accord.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from app.config import BASE_DIR
from app.ml.classifiers import HuggingFaceClassifier, OnnxClassifier
from textsense_ml.data import load_dataset
from textsense_ml.tracking import run_metadata, write_json

OPSET = 17


def export(model_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.config.return_dict = False

    sample = tokenizer(["Exemple de phrase pour tracer le graphe."], return_tensors="pt")
    fp32_path = out_dir / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    int8_path = out_dir / "model.int8.onnx"
    # per_channel=True : une échelle de quantification par canal de sortie.
    # Avec une seule échelle par matrice (per-tensor, le défaut), quelques
    # poids extrêmes écrasent la précision de toute la matrice : mesuré sur
    # textsense_eval_v1, l'exactitude tombait à 34 % (hasard) contre 88 %
    # en per-channel. Voir docs/EXPERIMENTS.md, section ONNX.
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, per_channel=True)

    tokenizer.save_pretrained(out_dir)
    shutil.copy(model_dir / "config.json", out_dir / "config.json")
    return fp32_path, int8_path


def latency_ms(classifier, texts: list[str], repeats: int = 3) -> dict[str, float]:
    """Latence d'une requête unitaire (un texte à la fois, comme dans l'API)."""
    classifier.predict_proba(texts[:5])  # échauffement
    timings = []
    for _ in range(repeats):
        for text in texts:
            start = time.perf_counter()
            classifier.predict_proba([text])
            timings.append(1000 * (time.perf_counter() - start))
    timings.sort()
    return {
        "p50": round(statistics.median(timings), 2),
        "p95": round(timings[int(0.95 * (len(timings) - 1))], 2),
        "mean": round(statistics.fmean(timings), 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--benchmark-texts", type=int, default=100)
    args = parser.parse_args(argv)

    model_dir = args.model_dir if args.model_dir.is_absolute() else BASE_DIR / args.model_dir
    out_dir = args.out_dir or model_dir.with_name(f"{model_dir.name}-onnx")
    fp32_path, int8_path = export(model_dir, out_dir)

    eval_df = load_dataset("textsense_eval_v1")
    texts = eval_df["text"].tolist()
    bench_texts = texts[: args.benchmark_texts]

    candidates = {
        "pytorch_fp32": HuggingFaceClassifier(str(model_dir), max_length=512),
        "onnx_fp32": OnnxClassifier(out_dir, onnx_file="model.onnx"),
        "onnx_int8": OnnxClassifier(out_dir, onnx_file="model.int8.onnx"),
    }
    reference = candidates["pytorch_fp32"].predict_proba(texts)
    results = {}
    for name, classifier in candidates.items():
        probabilities = reference if name == "pytorch_fp32" else classifier.predict_proba(texts)
        results[name] = {
            "latency_ms_single_text": latency_ms(classifier, bench_texts),
            "agreement_with_pytorch": float((probabilities.argmax(1) == reference.argmax(1)).mean()),
            "max_abs_proba_diff": float(np.abs(probabilities - reference).max()),
            "accuracy_textsense_eval_v1": float(
                (np.array(["negative", "neutral", "positive"])[probabilities.argmax(1)] == eval_df["label"]).mean()
            ),
        }
    sizes = {
        "pytorch_fp32": (model_dir / "model.safetensors").stat().st_size,
        "onnx_fp32": fp32_path.stat().st_size,
        "onnx_int8": int8_path.stat().st_size,
    }
    for name, size in sizes.items():
        results[name]["size_mb"] = round(size / 1e6, 1)

    report = {
        "model_dir": str(model_dir.relative_to(BASE_DIR)),
        "onnx_dir": str(out_dir.relative_to(BASE_DIR)),
        "opset": OPSET,
        "benchmark": {
            "texts": len(bench_texts),
            "repeats": 3,
            "note": "requêtes unitaires sur CPU, threads par défaut ; ordre de grandeur, dépend de la machine",
        },
        "results": results,
        "metadata": run_metadata(),
    }
    write_json(BASE_DIR / "reports" / "onnx" / f"{model_dir.name}.json", report)

    print(f"{'variante':<14}{'taille':>10}{'p50 (ms)':>11}{'p95 (ms)':>11}{'accord':>9}{'exactitude':>12}")
    for name, r in results.items():
        print(
            f"{name:<14}{r['size_mb']:>8.1f}Mo{r['latency_ms_single_text']['p50']:>11.1f}"
            f"{r['latency_ms_single_text']['p95']:>11.1f}{r['agreement_with_pytorch']:>9.1%}"
            f"{r['accuracy_textsense_eval_v1']:>12.1%}"
        )


if __name__ == "__main__":
    main()
