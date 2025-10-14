from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.geological_processor import GeologicalProcessor
from app.models.user import User
from app.core.dependencies import get_current_user
from app.core.database import get_db
from app import crud

import cv2
import numpy as np
import os
import logging
import traceback
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geological-section", tags=["Геологический разрез"])


@router.post("/create-enhanced-section")
async def create_enhanced_geological_section(
    map_image: UploadFile = File(..., description="Изображение геологической карты"),
    legend_image: UploadFile = File(..., description="Изображение легенды"),
    column_image: UploadFile = File(..., description="Изображение стратиграфической колонки"),
    start_x: int = Form(..., description="X координата начальной точки разреза"),
    start_y: int = Form(..., description="Y координата начальной точки разреза"),
    end_x: int = Form(..., description="X координата конечной точки разреза"),
    end_y: int = Form(..., description="Y координата конечной точки разреза"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Создает улучшенный геологический разрез с использованием трех изображений:
    карты, легенды и стратиграфической колонки.
    
    Требует аутентификации. Возвращает PNG изображение разреза через StreamingResponse
    с дополнительными заголовками, включая user_id, количество слоев и координаты разреза.
    Все файлы обрабатываются в памяти и автоматически удаляются.
    """
    logger.info("=== НАЧАЛО СОЗДАНИЯ УЛУЧШЕННОГО РАЗРЕЗА ===")
    logger.info(f"Файлы: карта={map_image.filename}, легенда={legend_image.filename}, колонка={column_image.filename}")
    logger.info(f"Линия разреза: ({start_x}, {start_y}) -> ({end_x}, {end_y})")

    temp_column_path = None
    temp_section_path = None

    try:
        # Проверяем форматы файлов
        supported_formats = [".png", ".jpg", ".jpeg"]
        
        for file, name in [(map_image, "карта"), (legend_image, "легенда"), (column_image, "колонка")]:
            ext = os.path.splitext(file.filename.lower())[1]
            if ext not in supported_formats:
                logger.error(f"Неподдерживаемый формат {name}: {ext}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"{name.capitalize()} должна быть в формате PNG, JPG или JPEG"
                )

        # Читаем изображения в память
        logger.info("Читаю файлы изображений в память...")
        map_content = await map_image.read()
        legend_content = await legend_image.read()
        column_content = await column_image.read()

        logger.info(f"Размеры файлов: карта={len(map_content)}, легенда={len(legend_content)}, колонка={len(column_content)} байт")

        # Конвертируем в numpy массивы (работаем в памяти)
        map_np = np.frombuffer(map_content, np.uint8)
        legend_np = np.frombuffer(legend_content, np.uint8)
        column_np = np.frombuffer(column_content, np.uint8)

        map_img = cv2.imdecode(map_np, cv2.IMREAD_COLOR)
        legend_img = cv2.imdecode(legend_np, cv2.IMREAD_COLOR)
        column_img = cv2.imdecode(column_np, cv2.IMREAD_COLOR)

        # Проверяем что изображения декодировались
        if map_img is None or legend_img is None or column_img is None:
            logger.error("Не удалось декодировать одно или несколько изображений")
            raise HTTPException(
                status_code=400, detail="Не удалось прочитать одно или несколько изображений"
            )

        # Проверяем размеры изображений
        logger.info(f"Размеры: карта={map_img.shape}, легенда={legend_img.shape}, колонка={column_img.shape}")

        for img, name in [(map_img, "карта"), (legend_img, "легенда"), (column_img, "колонка")]:
            if img.shape[0] == 0 or img.shape[1] == 0:
                logger.error(f"{name.capitalize()} имеет недопустимые размеры")
                raise HTTPException(
                    status_code=400, detail=f"{name.capitalize()} имеет недопустимые размеры"
                )

        # Проверяем координаты разреза
        height, width = map_img.shape[:2]
        logger.info(f"Проверяю координаты разреза для карты {width}x{height}")

        if start_x < 0 or start_x >= width or start_y < 0 or start_y >= height:
            raise HTTPException(
                status_code=400,
                detail=f"Начальная точка ({start_x}, {start_y}) выходит за границы карты {width}x{height}"
            )
        
        if end_x < 0 or end_x >= width or end_y < 0 or end_y >= height:
            raise HTTPException(
                status_code=400,
                detail=f"Конечная точка ({end_x}, {end_y}) выходит за границы карты {width}x{height}"
            )

        # Создаем ТОЛЬКО временный файл для колонки (т.к. StrategraphicColumnProcessor требует путь к файлу)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_column:
            temp_column_path = temp_column.name
            cv2.imwrite(temp_column_path, column_img)

        logger.info("Временный файл колонки создан, начинаю обработку...")

        # Создаем GeologicalProcessor и обрабатываем
        geo_processor = GeologicalProcessor()
        
        # Используем новый метод который обрабатывает все три изображения
        result = geo_processor.process_enhanced_geological_section(
            map_img, legend_img, temp_column_path, (start_x, start_y), (end_x, end_y)
        )

        # Проверяем что результат получен
        if not result or 'section_path' not in result:
            logger.error("Не удалось создать геологический разрез")
            raise HTTPException(
                status_code=500, detail="Не удалось создать геологический разрез"
            )

        temp_section_path = result['section_path']
        layers_count = len(result.get('layers', []))

        logger.info("=== УСПЕШНО СОЗДАН РАЗРЕЗ ===")
        logger.info(f"Временный файл: {temp_section_path}")
        logger.info(f"Слоев: {layers_count}")

        # Проверяем что файл разреза создан
        if not os.path.exists(temp_section_path):
            logger.error(f"Файл разреза не найден: {temp_section_path}")
            raise HTTPException(
                status_code=500, detail="Файл разреза не найден"
            )

        # Читаем файл в память
        with open(temp_section_path, 'rb') as f:
            section_data = f.read()

        logger.info(f"Размер файла разреза: {len(section_data)} байт")

        # Сохраняем информацию о секции в базу данных
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        section_filename = f"geological_section_{timestamp}.png"
        
        try:
            section_url = await crud.upload_image_to_gcs(file_bytes=section_data, content_type="image/png", user_id=current_user.id, filename=section_filename)
        except Exception as e:
            logger.warning(f"Не удалось загрузить изображение в GCS: {e}")
            raise HTTPException(status_code=500, detail="Не удалось загрузить изображение в облачное хранилище")

        try:
            await crud.create_user_section(
                db=db,
                user_id=current_user.id,
                section_url=section_url
            )
            logger.info(f"Секция сохранена в базу данных для пользователя {current_user.id}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить секцию в базу данных: {e}")
            raise HTTPException(status_code=500, detail="Не удалось сохранить секцию в базу данных")

        # Создаем генератор для streaming response с автоудалением файлов
        def generate_response():
            try:
                # Отдаем данные
                yield section_data
            finally:
                # Удаляем временные файлы после отдачи
                for temp_file in [temp_column_path, temp_section_path]:
                    try:
                        if temp_file and os.path.exists(temp_file):
                            os.remove(temp_file)
                            logger.info(f"Удален временный файл: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить временный файл {temp_file}: {e}")

        # Возвращаем streaming response
        return StreamingResponse(
            generate_response(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={section_filename}",
                "X-Layers-Count": str(layers_count),
                "X-Section-Coordinates": f"{start_x},{start_y},{end_x},{end_y}",
                "X-User-ID": str(current_user.id)
            }
        )

    except HTTPException:
        # Убираем временные файлы в случае ошибки
        for temp_file in [temp_column_path, temp_section_path]:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        raise
    except Exception as e:
        # Убираем временные файлы в случае ошибки
        for temp_file in [temp_column_path, temp_section_path]:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        
        logger.error(f"Непредвиденная ошибка: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Проверка состояния API
    """
    logger.info("Health check запрос")
    return {"status": "healthy", "message": "Геологический разрез API работает"}
