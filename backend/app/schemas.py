from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class Point(BaseModel):
    x: int
    y: int


class SectionRequest(BaseModel):
    start_point: Point
    end_point: Point


class GeologicalLayer(BaseModel):
    color: List[int]  # RGB цвет
    name: str = ""
    order: int  # Порядок в легенде (сверху вниз)
    text: str = ""  # Описание слоя
    length: Optional[int] = None  # Длина слоя в пикселях
    original_color: Optional[List[int]] = None  # Оригинальный цвет с карты


class LegendEntry(BaseModel):
    color: List[int]  # RGB цвет
    symbol: str  # Геологический символ
    text: str  # Описание
    y_position: int  # Позиция в легенде


class LegendTestResponse(BaseModel):
    success: bool
    legend_data: List[Dict]
    image_url: str
    message: str
    uploaded_file: str


class SectionResponse(BaseModel):
    layers: List[GeologicalLayer]
    image_url: str
    map_with_line_url: str
    message: str = "Разрез успешно построен"
    uploaded_files: dict = {}  # Информация о сохраненных файлах
    legend_data: Optional[List[Dict]] = None  # Данные легенды
    line_pixels_count: Optional[int] = None  # Количество пикселей вдоль линии


class StratoAnswer(BaseModel):
    answer: str = Field(..., description="Short and polite response to the user in language of the question. Information based on the provided context.")


# Auth schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=4, max_length=72)


class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=72)


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class User(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class UserSectionCreate(BaseModel):
    section_url: str


class UserSectionResponse(BaseModel):
    id: int
    user_id: int
    section_url: str
    created_at: datetime

    class Config:
        from_attributes = True


class PaginatedSectionsResponse(BaseModel):
    items: List[UserSectionResponse]
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    has_next: bool
    has_prev: bool