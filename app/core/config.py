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

    # GEMINI
    google_api_key: str

    # Настройки базы данных
    postgres_user: str
    postgres_password: str
    postgres_db: str
    db_host: str
    db_port: int
    database_url: str
    
    # Настройки JWT
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    
    # gcp
    gcs_bucket_name: str
    gcs_bucket_image_path: str

    class Config:
        env_file = ".env"


settings = Settings()

# Создание директорий при запуске
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

logger.info(f"Приложение {settings.app_name} инициализировано")
logger.info(
    f"Директории созданы: uploads={settings.upload_dir}, output={settings.output_dir}"
)
