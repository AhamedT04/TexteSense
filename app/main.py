import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR, get_settings
from app.database import init_db
from app.ml.predictor import is_model_ready, is_primary_model_ready
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
    init_db()

    # Pré-chargement des modèles ML au démarrage (évite la latence à la 1ère requête)
    logger.info("Pré-chargement des modèles ML…")
    if is_primary_model_ready():
        logger.info("Modèle principal prêt.")
    elif is_model_ready():
        logger.info("Modèle fallback scikit-learn prêt.")
    else:
        logger.warning(
            "Aucun modèle disponible au démarrage. "
            "Téléchargez le modèle principal ou entraînez le fallback sklearn."
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
        "Modèle principal : DistilCamemBERT (Transformers ou ONNX). "
        "Fallback : TF-IDF + Régression logistique."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": settings.app_name},
    )


def _public_model_name() -> str:
    """Nom du modèle principal sans chemin du serveur (inutile et indiscret côté client)."""
    if settings.model_backend == "onnx":
        return f"{settings.onnx_model_dir.name}/{settings.onnx_model_file}" if settings.onnx_model_dir else "onnx"
    local_dir = Path(settings.hf_model_id)
    return local_dir.name if local_dir.is_dir() else settings.hf_model_id


@app.get("/health", include_in_schema=False)
def health():
    ready = is_model_ready()
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "model_backend": settings.model_backend,
        "primary_model": _public_model_name(),
        "primary_model_loaded": is_primary_model_ready(),
        "fallback_model": "tfidf-logistic-regression",
        "version": app.version,
    }
