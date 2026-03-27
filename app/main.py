from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.database import Base, engine
from app.ml.predictor import HF_MODEL_ID, is_model_ready
from app.routes import router

app = FastAPI(
    title="TextSense API",
    description="API d'analyse de sentiment et d'extraction de mots-cles.",
    version="1.0.0",
)

Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": "TextSense"},
    )


@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "ok",
        "model_ready": is_model_ready(),
        "primary_model": HF_MODEL_ID,
        "fallback_model": "tfidf-logistic-regression",
    }
