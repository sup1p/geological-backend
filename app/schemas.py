from pydantic import BaseModel
from typing import List, Tuple, Dict, Optional


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
    message: str = "Разрез успешно построен"
    uploaded_files: dict = {}  # Информация о сохраненных файлах
    legend_data: Optional[List[Dict]] = None  # Данные легенды
