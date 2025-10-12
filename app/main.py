from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import geological_section, auth, user_sections, strato_ai
from app.core.database import engine
from app.models.user import Base
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== ЗАПУСК ПРИЛОЖЕНИЯ ===")
    logger.info("Геологический разрез API запускается...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    logger.info("=== ОСТАНОВКА ПРИЛОЖЕНИЯ ===")
    logger.info("Геологический разрез API останавливается...")
    await engine.dispose()

app = FastAPI(
    title="Геологический разрез API",
    description="API для построения геологического разреза по карте и легенде",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(geological_section.router, prefix="/api/v1")
app.include_router(strato_ai.router, prefix="/api/v1")
app.include_router(auth.router)
app.include_router(user_sections.router, prefix="/api/v1")