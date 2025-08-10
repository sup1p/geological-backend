from pydantic_settings import BaseSettings
import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
    ],
)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_name: str = "Геологический разрез API"
    debug: bool = False
    upload_dir: str = "uploads"
    output_dir: str = "outputs"

    # Настройки для обработки изображений
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: list = ["png", "jpg", "jpeg"]

    # Настройки для кластеризации цветов
    color_tolerance: float = 30.0  # Допуск для сопоставления цветов в LAB
    min_cluster_size: int = 5  # Минимальный размер кластера цветов

    class Config:
        env_file = ".env"


settings = Settings()

# Создание директорий при запуске
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

logger.info(f"Приложение {settings.app_name} инициализировано")
logger.info(
    f"Директории созданы: uploads={settings.upload_dir}, outputs={settings.output_dir}"
)
