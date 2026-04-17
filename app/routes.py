import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.ml.keyword_extractor import extract_keywords
from app.ml.predictor import predict_sentiment
from app.models import Analysis
from app.schemas import AnalysisResponse, DeleteResponse, StatsResponse, TextInput

logger = logging.getLogger(__name__)
router = APIRouter()


def serialize_analysis(record: Analysis) -> AnalysisResponse:
    return AnalysisResponse(
        id=record.id,
        text=record.text,
        sentiment=record.sentiment,
        confidence=record.confidence,
        low_confidence=record.low_confidence,
        keywords=json.loads(record.keywords),
        created_at=record.created_at,
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
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    logger.info(
        "Analyse id=%d : sentiment=%s confidence=%.4f low_confidence=%s",
        analysis.id,
        analysis.sentiment,
        analysis.confidence,
        analysis.low_confidence,
    )
    return serialize_analysis(analysis)


@router.get(
    "/history",
    response_model=list[AnalysisResponse],
    summary="Historique des analyses",
)
def get_history(
    skip: int = Query(default=0, ge=0, description="Nombre d'entrées à ignorer"),
    limit: int = Query(default=None, description="Nombre maximum d'entrées à retourner"),
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
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analyse introuvable.",
        )
    return serialize_analysis(record)


@router.delete(
    "/history/{analysis_id}",
    response_model=DeleteResponse,
    summary="Supprimer une analyse",
)
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)) -> DeleteResponse:
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analyse introuvable.",
        )
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
    return StatsResponse(
        total=total,
        positive=counts.get("positive", 0),
        negative=counts.get("negative", 0),
        neutral=counts.get("neutral", 0),
    )
