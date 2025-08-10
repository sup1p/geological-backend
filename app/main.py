from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import geological_section
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Геологический разрез API",
    description="API для построения геологического разреза по карте и легенде",
    version="1.0.0",
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


@app.get("/")
async def root():
    logger.info("Запрос к корневому эндпоинту")
    return {"message": "Геологический разрез API работает"}


@app.on_event("startup")
async def startup_event():
    logger.info("=== ЗАПУСК ПРИЛОЖЕНИЯ ===")
    logger.info("Геологический разрез API запускается...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=== ОСТАНОВКА ПРИЛОЖЕНИЯ ===")
    logger.info("Геологический разрез API останавливается...")
