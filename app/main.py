import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from app.config import settings
from app.core import HybridAIEngine
from app.api.routes import tenants, schemas, data, models, rag, inference

logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HybridAIEngine.App")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Hybrid AI Engine state...")
    engine = HybridAIEngine()
    
    # Auto-seed baseline multi-tenant data & LightGBM model if running on a fresh container
    if not engine.ml_pipeline.has_model("fintech_corp"):
        try:
            logger.info("Fresh container startup detected. Auto-seeding multi-tenant datasets & LightGBM model...")
            from scripts.seed_demo_data import seed
            seed()
        except Exception as e:
            logger.error(f"Auto-seeding during container startup failed: {e}")

    app.state.engine = engine
    yield
    logger.info("Shutting down Hybrid AI Engine...")

app = FastAPI(
    title="Hybrid AI Engine API",
    description=(
        "Production-grade Multi-Tenant Enterprise Intelligence Platform featuring "
        "Dynamic Runtime Schema Validation, Automated Classical ML (LightGBM + Optuna), "
        "and Intelligent Cold-Start Fallback to Pinecone Serverless Vector RAG."
    ),
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers under /api/v1
api_v1_prefix = "/api/v1"
app.include_router(tenants.router, prefix=api_v1_prefix)
app.include_router(schemas.router, prefix=api_v1_prefix)
app.include_router(data.router, prefix=api_v1_prefix)
app.include_router(models.router, prefix=api_v1_prefix)
app.include_router(rag.router, prefix=api_v1_prefix)
app.include_router(inference.router, prefix=api_v1_prefix)

# Mount Static & UI files
UI_DIR = Path(__file__).parent / "ui"
STATIC_DIR = UI_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/health", tags=["Health"])
def health_check(request: Request):
    engine = getattr(request.app.state, "engine", None)
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "pinecone_connected": engine.pinecone.is_live if engine else False,
        "pinecone_index": settings.PINECONE_INDEX_NAME
    }

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard():
    index_file = UI_DIR / "templates" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Hybrid AI Engine</h1><p>UI is being initialized...</p>")
