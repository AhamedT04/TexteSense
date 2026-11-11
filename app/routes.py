import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.ml.keyword_extractor import extract_keywords
from app.ml.predictor import predict_sentiment
from app.models import Analysis
from app.schemas import AnalysisResponse, DeleteResponse, FeedbackInput, StatsResponse, TextInput

logger = logging.getLogger(__name__)
router = APIRouter()


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite ne conserve pas le fuseau : une date relue « naïve » est en UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _get_or_404(db: Session, analysis_id: int) -> Analysis:
    record = db.get(Analysis, analysis_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analyse introuvable.",
        )
    return record


def serialize_analysis(record: Analysis) -> AnalysisResponse:
    return AnalysisResponse(
        id=record.id,
        text=record.text,
        sentiment=record.sentiment,
        confidence=record.confidence,
        low_confidence=record.low_confidence,
        keywords=json.loads(record.keywords),
        created_at=_as_utc(record.created_at),
        model_name=record.model_name,
        user_label=record.user_label,
        user_label_at=_as_utc(record.user_label_at),
    )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyser un texte",
    description="Prédit le sentiment (positif / négatif / neutre) et extrait les mots-clés.",
)
def analyze_text(payload: TextInput, db: Session = Depends(get_db)) -> AnalysisResponse:
    try:
        prediction = predict_sentiment(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Erreur lors de la prédiction : %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'analyse. Réessayez.",
        ) from exc

    keywords = extract_keywords(payload.text)
    analysis = Analysis(
        text=payload.text,
        sentiment=prediction["sentiment"],
        confidence=prediction["confidence"],
        low_confidence=prediction["low_confidence"],
        keywords=json.dumps(keywords, ensure_ascii=False),
        model_name=prediction["model"],
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    logger.info(
        "Analyse id=%d : sentiment=%s confidence=%.4f low_confidence=%s model=%s",
        analysis.id,
        analysis.sentiment,
        analysis.confidence,
        analysis.low_confidence,
        analysis.model_name,
    )
    return serialize_analysis(analysis)


@router.get(
    "/history",
    response_model=list[AnalysisResponse],
    summary="Historique des analyses",
)
def get_history(
    skip: int = Query(default=0, ge=0, description="Nombre d'entrées à ignorer"),
    limit: int | None = Query(default=None, ge=1, description="Nombre maximum d'entrées à retourner"),
    db: Session = Depends(get_db),
) -> list[AnalysisResponse]:
    settings = get_settings()
    effective_limit = min(
        limit if limit is not None else settings.history_default_limit,
        settings.history_max_limit,
    )
    records = (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .offset(skip)
        .limit(effective_limit)
        .all()
    )
    return [serialize_analysis(record) for record in records]


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Détail d'une analyse",
)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisResponse:
    return serialize_analysis(_get_or_404(db, analysis_id))


@router.put(
    "/analysis/{analysis_id}/feedback",
    response_model=AnalysisResponse,
    summary="Donner le sentiment correct d'une analyse",
    description=(
        "Enregistre le sentiment jugé correct par l'utilisateur : identique à la "
        "prédiction pour la confirmer, différent pour la corriger. Ces retours "
        "servent à constituer un jeu de données réel (voir `textsense_ml.export_feedback`)."
    ),
)
def set_feedback(
    analysis_id: int, payload: FeedbackInput, db: Session = Depends(get_db)
) -> AnalysisResponse:
    record = _get_or_404(db, analysis_id)
    record.user_label = payload.label
    record.user_label_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    logger.info(
        "Retour utilisateur id=%d : prédit=%s, correct=%s", record.id, record.sentiment, record.user_label
    )
    return serialize_analysis(record)


@router.delete(
    "/analysis/{analysis_id}/feedback",
    response_model=AnalysisResponse,
    summary="Retirer le retour utilisateur d'une analyse",
)
def clear_feedback(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisResponse:
    record = _get_or_404(db, analysis_id)
    record.user_label = None
    record.user_label_at = None
    db.commit()
    db.refresh(record)
    return serialize_analysis(record)


@router.delete(
    "/history/{analysis_id}",
    response_model=DeleteResponse,
    summary="Supprimer une analyse",
)
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)) -> DeleteResponse:
    record = _get_or_404(db, analysis_id)
    db.delete(record)
    db.commit()
    logger.info("Analyse id=%d supprimée.", analysis_id)
    return DeleteResponse(message="Analyse supprimée.")


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Statistiques globales",
)
def get_stats(db: Session = Depends(get_db)) -> StatsResponse:
    sentiment_rows = (
        db.query(Analysis.sentiment, func.count(Analysis.id))
        .group_by(Analysis.sentiment)
        .all()
    )
    counts = {sentiment: total for sentiment, total in sentiment_rows}
    total = sum(counts.values())
    with_feedback = db.query(func.count(Analysis.id)).filter(Analysis.user_label.is_not(None)).scalar()
    corrected = (
        db.query(func.count(Analysis.id))
        .filter(Analysis.user_label.is_not(None), Analysis.user_label != Analysis.sentiment)
        .scalar()
    )
    return StatsResponse(
        total=total,
        positive=counts.get("positive", 0),
        negative=counts.get("negative", 0),
        neutral=counts.get("neutral", 0),
        with_feedback=with_feedback or 0,
        corrected=corrected or 0,
    )
