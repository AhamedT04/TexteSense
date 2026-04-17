import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.database import Base, engine, run_migrations
from app.ml.predictor import HF_MODEL_ID, is_model_ready
from app.routes import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "DEBUG" if get_settings().debug else "INFO",
        },
    }
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan (démarrage / arrêt)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialisation base de données
    Base.metadata.create_all(bind=engine)
    run_migrations()

    # Pré-chargement des modèles ML au démarrage (évite la latence à la 1ère requête)
    logger.info("Pré-chargement des modèles ML…")
    if is_model_ready():
        try:
            from app.ml.predictor import load_huggingface_pipeline
            load_huggingface_pipeline()
            logger.info("Modèle Hugging Face prêt.")
        except Exception as exc:
            logger.warning("Pré-chargement HF impossible : %s", exc)
    else:
        logger.warning(
            "Aucun modèle disponible au démarrage. "
            "Téléchargez le modèle HF ou entraînez le fallback sklearn."
        )

    logger.info("TextSense démarré — http://localhost:8000")
    yield
    logger.info("TextSense arrêté.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="TextSense API",
    description=(
        "API d'analyse de sentiment et d'extraction de mots-clés pour textes en français. "
        "Modèle principal : DistilCamemBERT (Hugging Face). "
        "Fallback : TF-IDF + Régression logistique."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": settings.app_name},
    )


@app.get("/health", include_in_schema=False)
def health():
    ready = is_model_ready()
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "primary_model": HF_MODEL_ID,
        "fallback_model": "tfidf-logistic-regression",
        "version": app.version,
    }
