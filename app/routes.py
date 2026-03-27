import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.keyword_extractor import extract_keywords
from app.ml.predictor import predict_sentiment
from app.models import Analysis
from app.schemas import AnalysisResponse, DeleteResponse, StatsResponse, TextInput

router = APIRouter()


def serialize_analysis(record: Analysis) -> AnalysisResponse:
    return AnalysisResponse(
        id=record.id,
        text=record.text,
        sentiment=record.sentiment,
        confidence=record.confidence,
        keywords=json.loads(record.keywords),
        created_at=record.created_at,
    )


@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze_text(payload: TextInput, db: Session = Depends(get_db)) -> AnalysisResponse:
    try:
        prediction = predict_sentiment(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    keywords = extract_keywords(payload.text)
    analysis = Analysis(
        text=payload.text,
        sentiment=prediction["sentiment"],
        confidence=prediction["confidence"],
        keywords=json.dumps(keywords, ensure_ascii=True),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return serialize_analysis(analysis)


@router.get("/history", response_model=list[AnalysisResponse])
def get_history(db: Session = Depends(get_db)) -> list[AnalysisResponse]:
    records = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    return [serialize_analysis(record) for record in records]


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisResponse:
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analyse introuvable.")
    return serialize_analysis(record)


@router.delete("/history/{analysis_id}", response_model=DeleteResponse)
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)) -> DeleteResponse:
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analyse introuvable.")
    db.delete(record)
    db.commit()
    return DeleteResponse(message="Analyse supprimee.")


@router.get("/stats", response_model=StatsResponse)
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
