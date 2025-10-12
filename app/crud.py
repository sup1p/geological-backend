from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from google.cloud import storage
from typing import Optional
import math

from app.core.config import settings
from app.models.user import User, UserSections


# GCS функции
async def upload_image_to_gcs(file_bytes: bytes, content_type: str, user_id: str, filename: str) -> str:
    print("Uploading image to GCS...")
    client = storage.Client()
    bucket = client.bucket(settings.gcs_bucket_name)
    blob = bucket.blob(f"{settings.gcs_bucket_image_path}/{user_id}/{filename}")
    blob.upload_from_string(file_bytes, content_type=content_type)
    # Можно сделать публичным:
    print(f"Файл загружен в GCS: {blob.public_url}")
    return blob.public_url

# UserSections CRUD
async def create_user_section(db: AsyncSession, user_id: int, section_url: str) -> UserSections:
    """Создать новую секцию пользователя"""
    db_section = UserSections(
        user_id=user_id,
        section_url=section_url
    )
    db.add(db_section)
    await db.commit()
    await db.refresh(db_section)
    return db_section


async def get_user_sections_paginated(
    db: AsyncSession, 
    user_id: int, 
    page: int = 1, 
    page_size: int = 10
) -> dict:
    """
    Получить секции пользователя с пагинацией
    
    Args:
        db: Асинхронная сессия базы данных
        user_id: ID пользователя
        page: Номер страницы (начиная с 1)
        page_size: Количество элементов на странице
        
    Returns:
        dict: Словарь с данными пагинации и секциями
    """
    # Подсчет общего количества секций
    count_stmt = select(func.count()).select_from(UserSections).where(UserSections.user_id == user_id)
    count_result = await db.execute(count_stmt)
    total_items = count_result.scalar()
    
    # Вычисление смещения
    offset = (page - 1) * page_size
    
    # Получение секций с пагинацией
    stmt = (
        select(UserSections)
        .where(UserSections.user_id == user_id)
        .order_by(UserSections.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    
    result = await db.execute(stmt)
    sections = result.scalars().all()
    
    # Вычисление общего количества страниц
    total_pages = math.ceil(total_items / page_size) if total_items > 0 else 0
    
    return {
        "items": sections,
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": page,
        "page_size": page_size,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }


async def get_user_section_by_id(db: AsyncSession, section_id: int, user_id: int) -> Optional[UserSections]:
    """Получить конкретную секцию пользователя по ID"""
    stmt = select(UserSections).where(
        UserSections.id == section_id,
        UserSections.user_id == user_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def delete_user_section(db: AsyncSession, section_id: int, user_id: int) -> bool:
    """Удалить секцию пользователя"""
    section = await get_user_section_by_id(db, section_id, user_id)
    if section:
        await db.delete(section)
        await db.commit()
        return True
    return False

