from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
from typing import List
import os
import logging
from app.schemas import (
    SectionRequest,
    SectionResponse,
    GeologicalLayer,
    Point,
    LegendTestResponse,
)
from app.core.geological_processor import GeologicalProcessor
from app.core.config import settings
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geological-section", tags=["Геологический разрез"])


@router.post("/create-section", response_model=SectionResponse)
async def create_geological_section(
    map_image: UploadFile = File(..., description="Изображение геологической карты"),
    legend_image: UploadFile = File(..., description="Изображение легенды"),
    start_x: int = Form(..., description="X координата начальной точки"),
    start_y: int = Form(..., description="Y координата начальной точки"),
    end_x: int = Form(..., description="X координата конечной точки"),
    end_y: int = Form(..., description="Y координата конечной точки"),
):
    """
    Создает геологический разрез по карте и легенде
    """
    logger.info("=== НАЧАЛО ЗАПРОСА СОЗДАНИЯ РАЗРЕЗА ===")
    logger.info(f"Файлы: карта={map_image.filename}, легенда={legend_image.filename}")
    logger.info(f"Координаты: ({start_x}, {start_y}) -> ({end_x}, {end_y})")

    try:
        # Проверяем формат файлов
        supported_formats = [".png", ".jpg", ".jpeg"]
        map_ext = os.path.splitext(map_image.filename.lower())[1]
        legend_ext = os.path.splitext(legend_image.filename.lower())[1]

        logger.info(f"Проверяю форматы: карта={map_ext}, легенда={legend_ext}")

        if map_ext not in supported_formats:
            logger.error(f"Неподдерживаемый формат карты: {map_ext}")
            raise HTTPException(
                status_code=400, detail="Карта должна быть в формате PNG, JPG или JPEG"
            )
        if legend_ext not in supported_formats:
            logger.error(f"Неподдерживаемый формат легенды: {legend_ext}")
            raise HTTPException(
                status_code=400,
                detail="Легенда должна быть в формате PNG, JPG или JPEG",
            )

        # Читаем изображения
        logger.info("Читаю файлы изображений...")
        map_content = await map_image.read()
        legend_content = await legend_image.read()

        logger.info(
            f"Размеры файлов: карта={len(map_content)} байт, легенда={len(legend_content)} байт"
        )

        # Сохраняем файлы в папку uploads
        logger.info("Сохраняю файлы в папку uploads...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Генерируем уникальные имена файлов
        map_filename = f"map_{timestamp}_{map_image.filename}"
        legend_filename = f"legend_{timestamp}_{legend_image.filename}"

        map_filepath = os.path.join(settings.upload_dir, map_filename)
        legend_filepath = os.path.join(settings.upload_dir, legend_filename)

        # Сохраняем файлы
        with open(map_filepath, "wb") as f:
            f.write(map_content)
        with open(legend_filepath, "wb") as f:
            f.write(legend_content)

        logger.info(f"Файлы сохранены: карта={map_filepath}, легенда={legend_filepath}")

        # Конвертируем в numpy массивы
        map_np = np.frombuffer(map_content, np.uint8)
        legend_np = np.frombuffer(legend_content, np.uint8)

        map_img = cv2.imdecode(map_np, cv2.IMREAD_COLOR)
        legend_img = cv2.imdecode(legend_np, cv2.IMREAD_COLOR)

        if map_img is None or legend_img is None:
            logger.error("Не удалось декодировать изображения")
            raise HTTPException(
                status_code=400, detail="Не удалось прочитать изображения"
            )

        # Проверяем размеры изображений
        logger.info(
            f"Размеры изображений: карта={map_img.shape}, легенда={legend_img.shape}"
        )

        if map_img.shape[0] == 0 or map_img.shape[1] == 0:
            logger.error("Карта имеет недопустимые размеры")
            raise HTTPException(
                status_code=400, detail="Карта имеет недопустимые размеры"
            )
        if legend_img.shape[0] == 0 or legend_img.shape[1] == 0:
            logger.error("Легенда имеет недопустимые размеры")
            raise HTTPException(
                status_code=400, detail="Легенда имеет недопустимые размеры"
            )

        # Проверяем координаты точек
        logger.info(f"Проверяю координаты: карта {map_img.shape[1]}x{map_img.shape[0]}")

        if (
            start_x < 0
            or start_y < 0
            or end_x < 0
            or end_y < 0
            or start_x >= map_img.shape[1]
            or start_y >= map_img.shape[0]
            or end_x >= map_img.shape[1]
            or end_y >= map_img.shape[0]
        ):
            logger.error(
                f"Координаты выходят за границы: ({start_x}, {start_y}), ({end_x}, {end_y})"
            )
            raise HTTPException(
                status_code=400,
                detail="Координаты точек выходят за границы изображения",
            )

        # Обрабатываем геологический разрез
        logger.info("Запускаю обработку геологического разреза...")
        processor = GeologicalProcessor()
        result = processor.process_geological_section(
            map_img, legend_img, (start_x, start_y), (end_x, end_y)
        )

        if not result["success"]:
            logger.error(f"Ошибка обработки: {result['error']}")
            raise HTTPException(
                status_code=500, detail=f"Ошибка обработки: {result['error']}"
            )

        # Формируем ответ
        logger.info("Формирую ответ...")
        layers = []
        for layer in result["layers"]:
            layers.append(
                GeologicalLayer(
                    color=list(layer["color"]),
                    order=layer["order"],
                    text=layer.get("text", ""),
                    length=layer.get("length"),
                )
            )

        response = SectionResponse(
            layers=layers,
            image_url=f"/api/v1/geological-section/download/{os.path.basename(result['output_path'])}",
            message=f"Разрез построен успешно. Найдено {len(layers)} слоев.",
            uploaded_files={
                "map_file": map_filename,
                "legend_file": legend_filename,
                "map_path": map_filepath,
                "legend_path": legend_filepath,
            },
            legend_data=result.get("legend_data", []),
        )

        logger.info("=== УСПЕШНОЕ ЗАВЕРШЕНИЕ ЗАПРОСА ===")
        logger.info(f"Результат: {len(layers)} слоев, файл: {result['output_path']}")

        return response

    except HTTPException:
        logger.error("HTTPException перехвачена")
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        logger.exception("Детали ошибки:")
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.post("/test-legend", response_model=LegendTestResponse)
async def test_legend_extraction(
    legend_image: UploadFile = File(
        ..., description="Изображение легенды для тестирования"
    ),
):
    """
    Тестирует извлечение данных из легенды и создает изображение со всеми слоями
    """
    logger.info("=== НАЧАЛО ТЕСТИРОВАНИЯ ЛЕГЕНДЫ ===")
    logger.info(f"Файл легенды: {legend_image.filename}")

    try:
        # Проверяем формат файла
        supported_formats = [".png", ".jpg", ".jpeg"]
        legend_ext = os.path.splitext(legend_image.filename.lower())[1]

        if legend_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail="Легенда должна быть в формате PNG, JPG или JPEG",
            )

        # Читаем изображение
        legend_content = await legend_image.read()
        legend_np = np.frombuffer(legend_content, np.uint8)
        legend_img = cv2.imdecode(legend_np, cv2.IMREAD_COLOR)

        if legend_img is None:
            raise HTTPException(
                status_code=400, detail="Не удалось прочитать изображение легенды"
            )

        # Сохраняем файл легенды
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        legend_filename = f"test_legend_{timestamp}_{legend_image.filename}"
        legend_filepath = os.path.join(settings.upload_dir, legend_filename)

        with open(legend_filepath, "wb") as f:
            f.write(legend_content)

        logger.info(f"Легенда сохранена: {legend_filepath}")

        # Обрабатываем легенду
        processor = GeologicalProcessor()
        legend_data = processor.extract_legend_data(legend_img)

        if not legend_data:
            logger.warning("Не удалось извлечь данные из легенды")
            raise HTTPException(
                status_code=400, detail="Не удалось извлечь данные из легенды"
            )

        # Генерируем названия на основе символов
        legend_data = processor._generate_geological_names_from_symbols(legend_data)

        # Создаем тестовое изображение
        output_path = processor.create_legend_test_visualization(
            legend_data, settings.output_dir
        )

        # Формируем ответ
        response = {
            "success": True,
            "legend_data": legend_data,
            "image_url": f"/api/v1/geological-section/download/{os.path.basename(output_path)}",
            "message": f"Извлечено {len(legend_data)} блоков легенды",
            "uploaded_file": legend_filename,
        }

        logger.info("=== УСПЕШНОЕ ЗАВЕРШЕНИЕ ТЕСТИРОВАНИЯ ЛЕГЕНДЫ ===")
        logger.info(f"Результат: {len(legend_data)} блоков, файл: {output_path}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {str(e)}")
        logger.exception("Детали ошибки:")
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_section_image(filename: str):
    """
    Скачивает изображение геологического разреза
    """
    filepath = os.path.join(settings.output_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(filepath, media_type="image/png", filename=filename)


@router.get("/download-upload/{filename}")
async def download_uploaded_file(filename: str):
    """
    Скачивает загруженный файл из папки uploads
    """
    filepath = os.path.join(settings.upload_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")

    # Определяем MIME тип на основе расширения файла
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = mime_types.get(ext, "application/octet-stream")

    return FileResponse(filepath, media_type=media_type, filename=filename)


@router.get("/list-uploads")
async def list_uploaded_files():
    """
    Возвращает список загруженных файлов
    """
    try:
        files = []
        if os.path.exists(settings.upload_dir):
            for filename in os.listdir(settings.upload_dir):
                filepath = os.path.join(settings.upload_dir, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    files.append(
                        {
                            "filename": filename,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "download_url": f"/api/v1/geological-section/download-upload/{filename}",
                        }
                    )

        return {"upload_dir": settings.upload_dir, "files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Ошибка при получении списка файлов: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка при получении списка файлов: {str(e)}"
        )


@router.delete("/delete-upload/{filename}")
async def delete_uploaded_file(filename: str):
    """
    Удаляет загруженный файл из папки uploads
    """
    filepath = os.path.join(settings.upload_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")

    try:
        os.remove(filepath)
        logger.info(f"Файл удален: {filepath}")
        return {"message": f"Файл {filename} успешно удален"}
    except Exception as e:
        logger.error(f"Ошибка при удалении файла {filename}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка при удалении файла: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Проверка состояния API
    """
    logger.info("Health check запрос")
    return {"status": "healthy", "message": "Геологический разрез API работает"}
