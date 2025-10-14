"""
Тестирование API для работы с секциями пользователей
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os

from app.main import app
from app.core.database import get_db
from app.models.user import Base

# Тестовая база данных
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)


async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
async def setup_database():
    """Создание и очистка тестовой базы данных"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def test_get_sections_unauthorized(client):
    """Тест получения секций без авторизации"""
    response = client.get("/api/v1/sections/")
    assert response.status_code == 401


def test_health_check(client):
    """Тест health check"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# TODO: Добавить тесты с авторизацией после настройки JWT