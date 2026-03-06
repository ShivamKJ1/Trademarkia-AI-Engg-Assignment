"""FastAPI application entrypoint and startup/shutdown lifecycle wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from src.config import settings
from src.search_engine import SearchEngine
from src.utils import setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run expensive setup exactly once at process start.
    logger.info("Starting application and running startup pipeline")
    engine = SearchEngine(settings)
    engine.initialize()
    # Keep initialized engine in app state so all routes share one instance.
    app.state.search_engine = engine
    logger.info("Startup pipeline completed")
    yield
    logger.info("Application shutdown")


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)
