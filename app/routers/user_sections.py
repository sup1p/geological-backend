from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.schemas import User, PaginatedSectionsResponse, UserSectionResponse, UserSectionCreate
from app import crud

router = APIRouter(prefix="/sections", tags=["User Sections"])

@router.get("/", response_model=PaginatedSectionsResponse)
async def get_user_sections(
    page: int = Query(1, ge=1, description="Номер страницы (начиная с 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Количество элементов на странице"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Получить все секции текущего пользователя с пагинацией
    
    Args:
        page: Номер страницы (начиная с 1)
        page_size: Количество элементов на странице (1-100)
        current_user: Текущий аутентифицированный пользователь
        db: Сессия базы данных
        
    Returns:
        PaginatedSectionsResponse: Пагинированный список секций пользователя
    """
    result = await crud.get_user_sections_paginated(
        db=db, 
        user_id=current_user.id, 
        page=page, 
        page_size=page_size
    )
    
    return PaginatedSectionsResponse(**result)


@router.get("/{section_id}", response_model=UserSectionResponse)
async def get_user_section(
    section_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Получить конкретную секцию пользователя по ID
    
    Args:
        section_id: ID секции
        current_user: Текущий аутентифицированный пользователь
        db: Сессия базы данных
        
    Returns:
        UserSectionResponse: Данные секции
    """
    section = await crud.get_user_section_by_id(db, section_id, current_user.id)
    
    if not section:
        raise HTTPException(
            status_code=404,
            detail="Секция не найдена или не принадлежит пользователю"
        )
    
    return section


@router.post("/", response_model=UserSectionResponse)
async def create_user_section(
    section_data: UserSectionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Создать новую секцию для пользователя
    
    Args:
        section_data: Данные новой секции
        current_user: Текущий аутентифицированный пользователь
        db: Сессия базы данных
        
    Returns:
        UserSectionResponse: Созданная секция
    """
    section = await crud.create_user_section(
        db=db,
        user_id=current_user.id,
        section_url=section_data.section_url
    )
    
    return section


@router.delete("/{section_id}")
async def delete_user_section(
    section_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Удалить секцию пользователя
    
    Args:
        section_id: ID секции для удаления
        current_user: Текущий аутентифицированный пользователь
        db: Сессия базы данных
        
    Returns:
        dict: Сообщение об успешном удалении
    """
    deleted = await crud.delete_user_section(db, section_id, current_user.id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Секция не найдена или не принадлежит пользователю"
        )
    
    return {"message": "Секция успешно удалена"}