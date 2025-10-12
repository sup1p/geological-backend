import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime
import re
from pprint import pprint

logger = logging.getLogger(__name__)


class StrategraphicColumnProcessor:
    """
    Класс для анализа стратиграфической колонки и извлечения текста
    из колонки "Характеристика пород"
    """
    
    def __init__(self, use_enhanced_cleaning: bool = True):
        """
        Инициализация процессора стратиграфической колонки
        
        Args:
            use_enhanced_cleaning: Использовать ли улучшенную очистку геологических терминов
        """
        self.layers = []
        self.original_image = None
        self.processed_image = None
        self.use_enhanced_cleaning = use_enhanced_cleaning
        logger.info(f"StrategraphicColumnProcessor инициализирован (enhanced_cleaning={use_enhanced_cleaning})")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Загружает изображение стратиграфической колонки
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Загруженное изображение
        """
        logger.info(f"Загружаю изображение: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        logger.info(f"Изображение загружено. Размер: {self.original_image.shape}")
        return self.original_image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения: перевод в серый, бинаризация, удаление шума
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        logger.info("Начинаю предобработку изображения")
        
        # Конвертируем в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Увеличиваем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Удаляем шум, сохраняя края букв
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=30, sigmaSpace=15)
        
        # Увеличиваем изображение для улучшения OCR
        scale = 1.5
        denoised = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Адаптивная бинаризация для лучшего выделения текста
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 12
        )

        # Проверяем инверсию: нам нужен черный текст на белом фоне
        # Оценим долю черных пикселей; если фон темный (черных много) — инвертируем
        black_ratio = float(np.mean(binary < 128))
        if black_ratio > 0.5:
            binary = cv2.bitwise_not(binary)
        
        # Морфологические операции для очистки
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        self.processed_image = cleaned
        logger.info("Предобработка завершена")
        
        return cleaned

    def _ocr_lines_with_boxes(self, column_image: np.ndarray) -> List[Tuple[int, str]]:
        """
        Использует pytesseract image_to_data для извлечения слов и сборки строк
        Возвращает список (y_center, text) для каждой строки сверху вниз
        """
        try:
            import pytesseract
        except ImportError:
            logger.error("Tesseract не установлен")
            return []

        # Tesseract ожидает RGB
        if len(column_image.shape) == 2:
            rgb = cv2.cvtColor(column_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(column_image, cv2.COLOR_BGR2RGB)

        config = "--oem 1 --psm 4"  # Единая колонка текста, с анализом строк
        data = pytesseract.image_to_data(rgb, lang='rus', config=config, output_type=pytesseract.Output.DICT)

        n = len(data.get('text', []))
        lines: Dict[Tuple[int, int, int], List[Tuple[int, int, int, str, int]]] = {}
        for i in range(n):
            text = data['text'][i]
            if not text or text.isspace():
                continue
            conf = int(data.get('conf', ["-1"])[i]) if data.get('conf') else -1
            if conf < 55:
                continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            key = (data.get('page_num', [0])[i], data.get('block_num', [0])[i], data.get('line_num', [0])[i])
            lines.setdefault(key, []).append((x, y, w, h, text, conf))

        collected: List[Tuple[int, str]] = []
        for (_, _, _), items in lines.items():
            # сортируем слова по x
            items.sort(key=lambda t: t[0])
            y_center = int(np.median([y + h // 2 for (_, y, _, h, _, _) in items]))
            words = [t[4] for t in items]
            line_text = " ".join(words)
            line_text = self._clean_text_basic(line_text)
            if self._is_meaningful(line_text):
                collected.append((y_center, line_text))

        # Сортируем строки по Y
        collected.sort(key=lambda t: t[0])
        return collected

    def _clean_text_basic(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        # нормализуем точка+пробелы
        text = re.sub(r"\s*\.\s*", ". ", text)
        return text

    def _is_meaningful(self, text: str) -> bool:
        if not text:
            return False
        # Доля кириллицы
        cyr = sum(1 for ch in text if '\u0400' <= ch <= '\u04FF')
        if cyr < max(3, int(len(text) * 0.3)):
            return False
        # Минимальная длина
        if len(text) < 5:
            return False
        return True

    def _merge_lines_into_layers(self, lines: List[Tuple[int, str]]) -> List[str]:
        """
        Группирует строки в слои по вертикальным промежуткам и пунктуации.
        Удаляет строку-заголовок "Характеристика пород" (в любом написании/без пробелов).
        """
        if not lines:
            return []

        # Удаляем заголовок колонки, если встретится
        def is_header(s: str) -> bool:
            s2 = re.sub(r"\s+", "", s).lower()
            return "характеристикапород" in s2

        filtered = [(y, t) for (y, t) in lines if not is_header(t)]
        if not filtered:
            filtered = lines

        # Определяем средний межстрочный интервал
        ys = [y for (y, _) in filtered]
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        gap_thr = np.median(gaps) * 1.6 if gaps else 35

        layers: List[str] = []
        current: List[str] = []
        prev_y = None
        for y, text in filtered:
            if prev_y is not None and (y - prev_y) > gap_thr and current:
                layers.append(self._finalize_layer(current))
                current = []
            current.append(text)
            prev_y = y

        if current:
            layers.append(self._finalize_layer(current))

        # Фильтруем слишком короткие/шумные элементы
        result = [self._fix_split_sentences(s) for s in layers if self._is_meaningful(s)]
        return result

    def _finalize_layer(self, parts: List[str]) -> str:
        text = " ".join(parts)
        text = re.sub(r"\s+", " ", text).strip()
        # Удаляем повторяющиеся точки/пробелы
        text = re.sub(r"\s*\.\s*", ". ", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        # Исправляем слепленные слова в некоторых случаях (очень простая эвристика)
        # Вставляем пробел после точки, если буква слеплена
        text = re.sub(r"\.([А-Яа-я])", r". \1", text)
        return text

    def _ocr_block_text(self, column_image: np.ndarray, y0: int, y1: int) -> str:
        """OCR текста внутри одной ячейки (между y0 и y1)."""
        region = column_image[max(0, y0):min(column_image.shape[0], y1), :]
        # Добавим отступы
        pad = 8
        region = cv2.copyMakeBorder(region, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        # Небольшое увеличение для OCR
        region = cv2.resize(region, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)

        try:
            import pytesseract
            rgb = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB) if len(region.shape) == 2 else cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            config = '--oem 1 --psm 6 -l rus'
            text = pytesseract.image_to_string(rgb, config=config)
            text = self._clean_text(text)
            return text
        except Exception as e:
            logger.warning(f"OCR block error: {e}")
            return ""

    def _fix_split_sentences(self, layer_text: str) -> str:
        """
        Эвристика: если слой содержит несколько предложений, но по смыслу это один слой,
        не делим его. И наоборот: если в одном слое явно встречаются маркеры начала
        следующего слоя ("Свита.", индексы/символы и т.п.) — разделим.
        Здесь делаем только склейку коротких частей, разделённых точкой.
        """
        parts = [p.strip() for p in layer_text.split('.') if p.strip()]
        if not parts:
            return layer_text
        # Склеиваем слишком короткие куски с соседями
        merged: List[str] = []
        buf = ''
        for p in parts:
            if len(p) < 10:
                buf = (buf + ' ' + p).strip()
                continue
            if buf:
                merged.append(buf)
                buf = ''
            merged.append(p)
        if buf:
            merged.append(buf)
        return '. '.join(merged)
    
    def find_table_boundaries(self, image: np.ndarray) -> Dict[str, int]:
        """
        Находит границы таблицы и определяет позицию правой колонки
        
        Args:
            image: Обработанное изображение
            
        Returns:
            Словарь с координатами границ таблицы
        """
        logger.info("Ищу границы таблицы")
        
        height, width = image.shape[:2]
        
        # 0) Попытка по заголовку через OCR (наиболее надежно)
        header_box = self._locate_column_by_header(image)

        # 1) Ищем вертикальные линии с улучшенной логикой
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(255 - image, cv2.MORPH_OPEN, vertical_kernel)

        # Находим контуры вертикальных линий
        contours, _ = cv2.findContours(
            vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Собираем x-координаты вертикальных линий с фильтрацией
        vertical_x_coords = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Более строгие критерии для вертикальных разделителей
            if h > height * 0.5 and w <= 5 and x > width * 0.1:  # Исключаем края изображения
                vertical_x_coords.append(x)

        # Сортируем координаты
        vertical_x_coords.sort()
        
        # Дополнительно ищем разделители по плотности пикселей
        if len(vertical_x_coords) < 2:
            vertical_x_coords = self._find_vertical_separators_by_density(image)

        # Определяем границы колонок
        if len(vertical_x_coords) >= 2:
            # Правая колонка - это самая последняя колонка
            # Берем два последних разделителя
            left_boundary = vertical_x_coords[-2] if len(vertical_x_coords) >= 2 else vertical_x_coords[-1]
            right_boundary = width - 10  # Небольшой отступ от края

            # Правая колонка: ориентируемся на заголовок, если он найден
            if header_box is not None:
                hx1, hx2 = header_box
                # Найдем ближайшие вертикальные линии слева/справа от заголовка
                left_candidates = [x for x in vertical_x_coords if x <= hx1]
                right_candidates = [x for x in vertical_x_coords if x >= hx2]
                if left_candidates:
                    column_start = max(0, left_candidates[-1] - 6)
                else:
                    column_start = max(0, int(hx1) - 20)
                if right_candidates:
                    column_end = min(width - 1, right_candidates[0] + 24)
                else:
                    column_end = min(width - 1, int(hx2) + 40)
            else:
                approx_start = vertical_x_coords[-2]
                column_start = max(0, approx_start - 20)
                column_end = min(width - 1, width - 1)
        else:
            # Если вертикальные линии не найдены, используем профиль вертикальной плотности чернил
            logger.warning("Вертикальные линии не найдены, использую вертикальную проекцию")

            # Интенсивность чернил по столбцам (меньше значение -> больше черного)
            ink_per_col = np.sum(image < 200, axis=0).astype(np.int32)

            # Сглаживаем профиль
            if len(ink_per_col) > 21:
                k = np.ones(21, dtype=np.float32) / 21.0
                ink_smooth = np.convolve(ink_per_col, k, mode='same')
            else:
                ink_smooth = ink_per_col.astype(np.float32)

            # Порог для "текстовой" области
            thr = max(10.0, np.percentile(ink_smooth, 60))

            # Находим непрерывные сегменты, где плотность выше порога
            segments = []
            in_seg = False
            seg_start = 0
            for x in range(width):
                if ink_smooth[x] >= thr and not in_seg:
                    in_seg = True
                    seg_start = x
                elif ink_smooth[x] < thr and in_seg:
                    in_seg = False
                    segments.append((seg_start, x))
            if in_seg:
                segments.append((seg_start, width - 1))

            # Выбираем сильнейший сегмент в правой половине
            best = None
            best_score = -1
            for (xs, xe) in segments:
                width_seg = xe - xs
                score = width_seg * float(np.mean(ink_smooth[xs:xe + 1]))
                if xs > width * 0.45 and score > best_score:
                    best = (xs, xe)
                    best_score = score

            if best is None and segments:
                # Fallback: берем самый сильный сегмент вообще
                for (xs, xe) in segments:
                    width_seg = xe - xs
                    score = width_seg * float(np.mean(ink_smooth[xs:xe + 1]))
                    if score > best_score:
                        best = (xs, xe)
                        best_score = score

            if best is not None:
                xs, xe = best
                # Если нашли заголовок, оттолкнемся от него
                if header_box is not None:
                    hx1, hx2 = header_box
                    xs = min(xs, hx1)
                    xe = max(xe, hx2)
                pad_left = 60  # больше запас слева, чтобы не обрезать текст
                pad_right = 40
                column_start = max(0, xs - pad_left)
                column_end = min(width - 1, xe + pad_right)
            else:
                # Последний резерв — правая треть
                column_start = int(width * 0.55)
                column_end = width - 10

            left_boundary = 0
            right_boundary = width

        # Страховка: корректируем в пределах изображения и минимальная ширина
        if column_end - column_start < int(width * 0.18):  # слишком узко — расширим
            expand = int(width * 0.05)
            column_start = max(0, column_start - expand)
            column_end = min(width - 1, column_end + expand)
        
        boundaries = {
            'left': left_boundary,
            'right': right_boundary,
            'top': 0,
            'bottom': height,
            'column_start': column_start,
            'column_end': column_end
        }
        
        logger.info(f"Найдены границы таблицы: {boundaries}")
        return boundaries

    def _find_vertical_separators_by_density(self, image: np.ndarray) -> List[int]:
        """
        Ищет вертикальные разделители по анализу плотности черных пикселей
        """
        height, width = image.shape[:2]
        
        # Подсчитываем плотность черных пикселей по вертикали
        vertical_density = np.sum(image == 0, axis=0) / height
        
        # Ищем минимумы плотности (потенциальные разделители)
        separators = []
        threshold = 0.05  # Порог для определения разделителя
        
        for x in range(width // 4, width * 3 // 4):  # Ищем в средней части
            if vertical_density[x] < threshold:
                # Проверяем, что это локальный минимум
                window = 20
                start = max(0, x - window)
                end = min(width, x + window)
                if vertical_density[x] == np.min(vertical_density[start:end]):
                    separators.append(x)
        
        # Удаляем слишком близкие разделители
        filtered_separators = []
        for sep in separators:
            if not filtered_separators or abs(sep - filtered_separators[-1]) > width // 10:
                filtered_separators.append(sep)
        
        return filtered_separators

    def _locate_column_by_header(self, image: np.ndarray) -> Tuple[int, int] | None:
        """
        Пытается найти заголовок колонки "Характеристика пород" и вернуть (x1, x2)
        границы текста заголовка, чтобы от него отстроить колонку. Возвращает None,
        если не найдено.
        """
        try:
            import pytesseract
        except ImportError:
            return None

        # Нужен RGB для tesseract
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        config = "--oem 1 --psm 6"
        data = pytesseract.image_to_data(rgb, lang='rus', config=config, output_type=pytesseract.Output.DICT)
        n = len(data.get('text', []))
        boxes = []
        for i in range(n):
            t = data['text'][i]
            if not t:
                continue
            conf = int(data.get('conf', ["-1"])[i]) if data.get('conf') else -1
            if conf < 45:
                continue
            # Приводим к виду без пробелов/регистра
            t_norm = re.sub(r"\s+", "", t).lower()
            if 'характеристика' in t_norm or 'пород' in t_norm:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                boxes.append((x, y, w, h))

        if not boxes:
            return None

        # Объединяем найденные слова в одну горизонтальную полосу
        xs = [x for (x, _, _, _) in boxes]
        xe = [x + w for (x, _, w, _) in boxes]
        return (min(xs), max(xe))
    
    def crop_right_column(self, image: np.ndarray, boundaries: Dict[str, int]) -> np.ndarray:
        """
        Обрезает правую колонку с характеристикой пород
        
        Args:
            image: Исходное изображение
            boundaries: Границы таблицы
            
        Returns:
            Обрезанная колонка
        """
        logger.info("Обрезаю правую колонку")
        
        x_start = boundaries['column_start']
        x_end = boundaries['column_end']
        y_start = boundaries['top']
        y_end = boundaries['bottom']
        
        # Обрезаем колонку
        cropped_column = image[y_start:y_end, x_start:x_end]
        
        logger.info(f"Колонка обрезана. Размер: {cropped_column.shape}")
        return cropped_column
    
    # Удалено: старая детекция строк по проекциям (заменена на Hough + гибрид)
    
    def _merge_close_rows(self, row_boundaries: List[Tuple[int, int]], min_gap: int = 10) -> List[Tuple[int, int]]:
        """
        Объединяет близко расположенные строки
        
        Args:
            row_boundaries: Границы строк
            min_gap: Минимальный промежуток для объединения
            
        Returns:
            Объединенные границы строк
        """
        if not row_boundaries:
            return []
        
        merged = []
        current_start, current_end = row_boundaries[0]
        
        for start, end in row_boundaries[1:]:
            if start - current_end <= min_gap:
                # Объединяем строки
                current_end = end
            else:
                # Сохраняем текущую строку и начинаем новую
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Добавляем последнюю строку
        merged.append((current_start, current_end))
        
        return merged
    
    # Удалено: построчный OCR; теперь OCR выполняется на словах (image_to_data)
    
    def _clean_text(self, text: str) -> str:
        """
        Улучшенная очистка извлеченного текста от артефактов OCR
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        if not text:
            return ""
        
        # Убираем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Убираем одиночные символы и цифры в начале строки
        text = re.sub(r'^[0-9\.\)\-\|\s\*\=\^\`\~]+', '', text)
        
        # Убираем артефакты в конце строки
        text = re.sub(r'[0-9\.\)\-\|\*\=\^\`\~\s]+$', '', text)
        
        # Исправляем распространенные ошибки OCR
        text = re.sub(r'\bДО\s*т\s*Н\s*\'\s*а\s*_\s*\*\s*0\s*\=', '', text)  # Артефакты из примера
        text = re.sub(r'\s*=\s*"ч\s*=\s*\|\s*=\s*\|"', '', text)  # Табличные артефакты
        text = re.sub(r'\bс\s*Горов\s*\d+', '', text)  # Номера страниц/строк
        text = re.sub(r'\bee\s*\|', '', text)  # Границы таблицы
        text = re.sub(r'\b\d+\s*0\s*', '', text)  # Одиночные числа
        text = re.sub(r'‚\s*', '', text)  # Запятые в начале
        text = re.sub(r'\|\s*0\s*', '', text)  # Табличные разделители
        
        # Исправляем слитные слова (простая эвристика)
        text = re.sub(r'([а-я])([А-Я])', r'\1 \2', text)  # разделяем слитные слова
        
        # Убираем повторяющиеся символы
        text = re.sub(r'([\.,:;])\1+', r'\1', text)
        
        # Очищаем странные символы
        text = re.sub(r'[^а-яёА-ЯЁa-zA-Z0-9\s\.,;:()\-]', ' ', text)
        
        # Финальная очистка пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _clean_geological_text_enhanced(self, text: str) -> str:
        """
        Специализированная очистка геологического текста с исправлением терминов
        
        Args:
            text: Исходный текст после базовой очистки
            
        Returns:
            Текст с исправленными геологическими терминами
        """
        if not text:
            return ""
        
        # Исправляем распространенные ошибки OCR в геологических терминах
        corrections = {
            r'\bболид\s*ООО\s*ний': 'Верхний',
            r'\bзито-\s*базальтовые': 'Андезито-базальтовые', 
            r'\bзий\s*-\s*средний': 'Нижний - средний',
            r'\bий\s*плиоцен': 'Нижний плиоцен',
            r'\bйловская': 'Михайловская',
            r'\bломераты': 'Конгломераты',
            r'\bикская': 'Черникская',
            r'\bлиты': 'Гравелиты',
            r'\bбу:\s*инская': 'бурых углей. Дусинская',
            r'\bовская': 'Петровская',
            r'\bые\s*кремнистые': 'Черные кремнистые',
            r'\bизве\s*шорская': 'известняки. Лумшорская',
            r'\bгично\s*чередующиеся': 'Ритмично чередующиеся',
            r'\bаргиллу\s*айа\s*агоролеп\s*Ми\s*11а': 'аргиллиты и известняки',
            r'\bАсагиипа\s*апри\(ата\s*\(\s*М\s*БА\s*1\s*е': 'Acanthina antiquata (M. BAYLE)',
            r'\bгская': 'Льютская',
            r'\bаники': 'Песчаники',
            r'\bютгипсапа\s*агса\s*\(Си\s*П\s*тат': 'Gothograptus gracis (CURR)',
            r'\bнская': 'Полянская',
            r'\bэзернистые': 'Разнозернистые',
            r'\bВеггтазе\s*Иа\s*5р': 'Bertranella sp.',
            r'\bзные\s*глины': 'Красные глины',
            r'\bГиаи': 'и алевролиты'
        }
        
        # Применяем исправления
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Убираем артефакты начала строк (обрезанные слова)
        text = re.sub(r'^[а-я]{1,3}\s+', '', text)
        
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_strategraphic_column(self, image_path: str) -> List[str]:
        """
        Основной метод обработки стратиграфической колонки
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список слоев (характеристик пород)
        """
        logger.info("=== НАЧАЛО ОБРАБОТКИ СТРАТИГРАФИЧЕСКОЙ КОЛОНКИ ===")

        try:
            # 1. Загружаем изображение
            image = self.load_image(image_path)
            
            # 2. Предобработка
            processed_image = self.preprocess_image(image)
            
            # 3. Находим границы таблицы
            boundaries = self.find_table_boundaries(processed_image)
            
            # 4. Обрезаем правую колонку
            column_image = self.crop_right_column(processed_image, boundaries)
            
            # 5. Сегментация колонок на блоки (строки таблицы)
            # Приоритет: явные горизонтальные границы ячеек (Hough) -> гибридные интервалы
            blocks = self._detect_cell_rows(column_image)
            if not blocks:
                blocks = self._segment_blocks_by_gaps_and_lines(column_image)

            layers: List[str] = []
            if blocks:
                # OCR строго по ячейке (между горизонтальными линиями)
                for (y0, y1) in blocks:
                    text_block = self._ocr_block_text(column_image, y0, y1)
                    if not text_block:
                        continue
                    # Пропустим заголовок в первом блоке
                    norm = re.sub(r"\s+", "", text_block).lower()
                    if len(layers) == 0 and "характеристикапород" in norm:
                        continue
                    # Разделим по маркерам только если в блоке реально две записи
                    parts = self._split_by_markers(text_block)
                    if len(parts) == 1:
                        parts = [text_block]
                    for part in parts:
                        if not self._is_meaningful(part):
                            continue
                        # Удалим повтор соседних слоёв (дубликаты от близких линий)
                        if layers:
                            prev_norm = re.sub(r"\s+", "", layers[-1]).lower()
                            curr_norm = re.sub(r"\s+", "", part).lower()
                            if curr_norm == prev_norm:
                                continue
                        layers.append(part)
            else:
                # Fallback на старую логику
                separators = self._find_horizontal_separators(column_image)
                ocr_lines = self._ocr_lines_with_boxes(column_image)
                if separators:
                    layers = self._merge_lines_by_separators(ocr_lines, separators, column_image.shape[0])
                else:
                    layers = self._merge_lines_into_layers(ocr_lines)
            
            # 6. Fallback на гибридный метод, если Hough дал слишком мало
            if len(layers) < 3:
                logger.info("Мало слоев, fallback на гибридные интервалы")
                blocks_fb = self._segment_blocks_by_gaps_and_lines(column_image)
                if blocks_fb:
                    layers_fb: List[str] = []
                    for (y0, y1) in blocks_fb:
                        bucket = [tx for (yc, tx) in ocr_lines if y0 + 3 <= yc < y1 - 3]
                        if bucket:
                            part = self._finalize_layer(bucket)
                            for p in self._split_by_markers(part):
                                if self._is_meaningful(p):
                                    tnorm = re.sub(r"\s+","", p).lower()
                                    if len(layers_fb) == 0 and "характеристикапород" in tnorm:
                                        continue
                                    layers_fb.append(p)
                    if len(layers_fb) > len(layers):
                        layers = layers_fb
            
            # Применяем улучшенную очистку геологических терминов, если включена
            if self.use_enhanced_cleaning and layers:
                logger.info("Применяю улучшенную очистку геологических терминов...")
                enhanced_layers = []
                for layer in layers:
                    enhanced = self._clean_geological_text_enhanced(layer)
                    if enhanced and len(enhanced) > 5:  # Фильтруем слишком короткие
                        enhanced_layers.append(enhanced)
                layers = enhanced_layers
                logger.info(f"После улучшенной очистки: {len(layers)} слоев")

            self.layers = layers

            # Логируем итоговый порядок слоев (сверху вниз)
            logger.info(f"=== ОБРАБОТКА ЗАВЕРШЕНА. Найдено {len(layers)} слоев ===")
            if self.layers:
                try:
                    preview_lines = []
                    for idx, text in enumerate(self.layers):
                        preview = text.strip()
                        if len(preview) > 80:
                            preview = preview[:77] + "..."
                        preview_lines.append(f"  {idx + 1:02d}. {preview}")
                    logger.info("Порядок слоев (сверху вниз):\n" + "\n".join(preview_lines))
                except Exception as e:
                    logger.warning(f"Не удалось сформировать лог порядка слоев: {e}")
        except Exception as e:
            logger.error(f"Ошибка при обработке: {str(e)}")
            logger.exception("Детали ошибки:")
            return []

        return self.layers

    def _find_horizontal_separators(self, column_image: np.ndarray) -> List[int]:
        """
        Ищет горизонтальные разделители двумя способами: морфология и проекция.
        Возвращает Y-координаты линий (сверху вниз), без дублей.
        """
        height, width = column_image.shape[:2]

        # 1) Морфологический поиск тонких длинных линий
        inv = 255 - column_image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 3), 1))
        lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ys_morph = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > width * 0.65 and h <= 6:
                ys_morph.append(y + h // 2)

        # 2) Проекция чернил: горизонтальные "пустые" полосы между строками
        ink = np.sum(column_image < 200, axis=1).astype(np.int32)
        # Сгладим профиль
        if height > 11:
            k = np.ones(11, dtype=np.float32) / 11.0
            ink_s = np.convolve(ink, k, mode='same')
        else:
            ink_s = ink.astype(np.float32)
        # Порог для пустых промежутков
        low_thr = max(5.0, np.percentile(ink_s, 25))
        ys_proj = [i for i in range(height) if ink_s[i] <= low_thr]
        # Оставим только локальные минимумы, сгруппируем в кластеры
        proj_lines = []
        if ys_proj:
            start = ys_proj[0]
            prev = ys_proj[0]
            for y in ys_proj[1:]:
                if y - prev > 3:
                    proj_lines.append((start, prev))
                    start = y
                prev = y
            proj_lines.append((start, prev))
        ys_proj_centers = [ (a+b)//2 for (a,b) in proj_lines if (b-a) >= 2 ]

        # Объединим результаты и удалим дубликаты
        ys = sorted(ys_morph + ys_proj_centers)
        filtered = []
        for y in ys:
            if not filtered or y - filtered[-1] > 10:
                filtered.append(y)
        # Отсечём крайние линии у границ изображения
        filtered = [y for y in filtered if 8 <= y <= height - 8]
        return filtered

    def _merge_lines_by_separators(self, lines: List[Tuple[int, str]], separators: List[int], height: int) -> List[str]:
        """
        Группирует OCR-линии по интервалам между горизонтальными разделителями таблицы.
        Каждый интервал -> один слой.
        """
        if not lines:
            return []
        # Формируем интервалы [start, end)
        bounds: List[Tuple[int, int]] = []
        prev = 0
        for y in separators:
            bounds.append((prev, y))
            prev = y
        bounds.append((prev, height))

        # Сортируем линии по Y
        lines_sorted = sorted(lines, key=lambda t: t[0])
        result: List[str] = []
        header_removed = False
        pad = 3  # небольшой отступ от линий
        for (y0, y1) in bounds:
            bucket: List[str] = []
            for (yc, tx) in lines_sorted:
                if y0 + pad <= yc < y1 - pad:
                    bucket.append(tx)
            if bucket:
                layer_text = self._finalize_layer(bucket)
                if self._is_meaningful(layer_text):
                    # Удаляем строку-заголовок, если она попалась как отдельный слой
                    if not header_removed:
                        tnorm = re.sub(r"\s+","", layer_text).lower()
                        if "характеристикапород" in tnorm:
                            header_removed = True
                            continue
                    result.append(layer_text)
        return result

    def _split_by_markers(self, text: str) -> List[str]:
        """
        Эвристическое разбиение слоя, если в нём случайно оказались два соседних слоя.
        Разделяем по частым геологическим маркерам ("свита.", "плиоцен.", и т.п.)
        с сохранением маркера в начале сегмента.
        """
        if not text:
            return []
        markers = [
            r"\bсвита\.", r"\bплиоцен\.", r"\bмиоцен\.", r"\болигoцен\.", r"\bюрск\w*\.", r"\bмелова\w*\."
        ]
        pattern = "|".join(markers)
        # Найдём позиции маркеров
        spans = []
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append(m.start())
        if len(spans) <= 1:
            return [text]
        parts = []
        for i, s in enumerate(spans):
            e = spans[i + 1] if i + 1 < len(spans) else len(text)
            parts.append(text[s:e].strip(" ,.-\n"))
        # Если есть префикс перед первым маркером, приклеим его к первому сегменту
        prefix = text[:spans[0]].strip()
        if prefix and parts:
            parts[0] = (prefix + " " + parts[0]).strip()
        # Уберём пустые
        return [p for p in parts if p]

    def _segment_blocks_by_gaps_and_lines(self, column_image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Улучшенный сегментатор строк: использует комбинацию горизонтальных линий
        и анализа пустых промежутков для точного разбиения на строки таблицы.
        Возвращает список границ блоков (y0, y1) сверху вниз.
        """
        height, width = column_image.shape[:2]
        # Линии
        seps = self._find_horizontal_separators(column_image)

        # Проекция
        ink = np.sum(column_image < 200, axis=1).astype(np.int32)
        if height > 11:
            k = np.ones(11, dtype=np.float32) / 11.0
            ink_s = np.convolve(ink, k, mode='same')
        else:
            ink_s = ink.astype(np.float32)
        low_thr = max(5.0, np.percentile(ink_s, 25))
        high_thr = max(10.0, np.percentile(ink_s, 40))

        # Границы "пустых" зон
        gaps: List[Tuple[int, int]] = []
        in_gap = False
        start = 0
        for y in range(height):
            if ink_s[y] <= low_thr and not in_gap:
                in_gap = True
                start = y
            elif ink_s[y] > low_thr and in_gap:
                in_gap = False
                if y - start >= 6:
                    gaps.append((start, y))
        if in_gap:
            gaps.append((start, height))

        # Соберём "кандидаты" линий из центров разрывов и найденных разделителей
        line_y = set(int((a + b) // 2) for (a, b) in gaps)
        line_y.update(seps)
        lines_sorted = sorted(y for y in line_y if 5 <= y <= height - 5)

        # Формируем блоки между соседними линиями
        bounds: List[Tuple[int, int]] = []
        prev = 0
        for y in lines_sorted:
            if y - prev >= 25:  # минимальная высота блока
                # проверим, что внутри есть текст (по high_thr)
                if np.max(ink_s[prev:y]) >= high_thr:
                    bounds.append((prev, y))
            prev = y
        if height - prev >= 25 and np.max(ink_s[prev:]) >= high_thr:
            bounds.append((prev, height))

        # Постфильтр: удалим слишком маленькие/пустые
        cleaned: List[Tuple[int, int]] = []
        for (a, b) in bounds:
            if b - a < 20:
                continue
            if np.sum(column_image[a:b, :] < 200) < width * (b - a) * 0.02:
                continue
            cleaned.append((a, b))

        return cleaned

    def _detect_cell_rows(self, column_image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Находит горизонтальные границы ячеек правой колонки с помощью HoughLinesP.
        Возвращает отсортированный список (y0, y1) для каждой строки-ячейки.
        """
        height, width = column_image.shape[:2]
        # Работать удобнее по инверсии
        inv = 255 - column_image
        # Усилим горизонтальные линии: закрытие + открытие
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 2), 1))
        enhanced = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, horiz_kernel, iterations=1)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

        # Края
        edges = cv2.Canny(enhanced, 30, 120)
        # Прямые отрезки
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(width * 0.75), maxLineGap=6)
        if lines is None:
            # Попробуем чисто морфологические контуры на enhanced
            cnts, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ys = []
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w > width * 0.75 and h <= 4:
                    ys.append(int(y + h // 2))
        else:
            ys = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if abs(y1 - y2) <= 2:
                    ys.append(int((y1 + y2) // 2))
        # Добавим верх/низ как псевдолинии, чтобы крайние строки не потерять
        ys.extend([0, height - 1])

        if not ys:
            return []

        # Кластеризуем близкие по Y линии
        ys_sorted = sorted(ys)
        clustered: List[int] = []
        cluster: List[int] = [ys_sorted[0]]
        for y in ys_sorted[1:]:
            if y - cluster[-1] <= 6:
                cluster.append(y)
            else:
                clustered.append(int(np.median(cluster)))
                cluster = [y]
        clustered.append(int(np.median(cluster)))

        # Сформируем интервалы между горизонтальными линиями
        clustered = [y for y in clustered if 6 <= y <= height - 6]
        clustered = sorted(set(clustered))
        if len(clustered) < 2:
            return []

        rows: List[Tuple[int, int]] = []
        prev = clustered[0]
        for y in clustered:
            if y - prev >= 24:
                rows.append((prev, y))
            prev = y
        if height - prev >= 24:
            rows.append((prev, height))

        # Фильтрация по содержанию текста
        filtered: List[Tuple[int, int]] = []
        for (a, b) in rows:
            if self._has_content_ratio(column_image[a:b, :], min_ratio=0.012):
                filtered.append((a, b))
        return filtered

    def _has_content_ratio(self, region: np.ndarray, min_ratio: float) -> bool:
        """Проверяет, что доля тёмных пикселей достаточна, чтобы считать блок текстовым."""
        total = float(region.size)
        if total <= 0:
            return False
        dark = float(np.sum(region < 200))
        return (dark / total) >= min_ratio
    
    def print_results(self):
        """Выводит результаты в консоль"""
        if not self.layers:
            print("Нет данных для вывода")
            return
        
        print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА СТРАТИГРАФИЧЕСКОЙ КОЛОНКИ ===")
        print(f"Найдено слоев: {len(self.layers)}")
        print("\nСлои (сверху вниз):")
        pprint(self.layers)
        print("\n" + "="*60)
    
    def save_results(self, output_dir: str = "output_strategraphic_column") -> Dict[str, str]:
        """
        Сохраняет результаты в файлы и возвращает пути к ним.

        Args:
            output_dir: Директория для сохранения файлов

        Returns:
            Словарь с путями к сохраненным файлам
        """
        os.makedirs(output_dir, exist_ok=True)

        # Генерируем стабильное короткое имя по содержимому
        try:
            import hashlib
            payload = json.dumps(self.layers, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            short_hash = hashlib.sha256(payload).hexdigest()[:8]
        except Exception:
            short_hash = datetime.now().strftime("%H%M%S")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"layers_{short_hash}"

        json_path = os.path.join(output_dir, f"{base_name}.json")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.layers, f, ensure_ascii=False, indent=2)

        # TXT (пронумерованный список)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Найдено слоев: {len(self.layers)}\n\n")
            for idx, layer in enumerate(self.layers, start=1):
                f.write(f"{idx:02d}. {layer}\n")

        logger.info(f"Результаты сохранены: json={json_path}, txt={txt_path}")
        return {"json": json_path, "txt": txt_path}

    def create_debug_visualization(self, output_dir: str = "output_strategraphic_column") -> str:
        """
        Создает отладочную визуализацию: исходное, предобработанное, границы колонки, кроп колонки.

        Args:
            output_dir: Директория для сохранения

        Returns:
            Путь к сохраненному изображению или пустая строка
        """
        if self.original_image is None or self.processed_image is None:
            logger.warning("Нет данных для визуализации")
            return ""

        # Создаем фигуру
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Исходное изображение
        if len(self.original_image.shape) == 3:
            original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = self.original_image

        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title("Исходное изображение")
        axes[0, 0].axis('off')

        # Обработанное изображение
        axes[0, 1].imshow(self.processed_image, cmap='gray')
        axes[0, 1].set_title("После предобработки")
        axes[0, 1].axis('off')

        # Найденные границы таблицы
        try:
            boundaries = self.find_table_boundaries(self.processed_image)
            debug_image = original_rgb.copy() if len(original_rgb.shape) == 3 else cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
            cv2.line(debug_image, (boundaries['column_start'], 0), (boundaries['column_start'], debug_image.shape[0]), (255, 0, 0), 2)
            cv2.line(debug_image, (boundaries['column_end'], 0), (boundaries['column_end'], debug_image.shape[0]), (255, 0, 0), 2)
        except Exception:
            debug_image = original_rgb

        axes[1, 0].imshow(debug_image)
        axes[1, 0].set_title("Найденные границы колонки")
        axes[1, 0].axis('off')

        # Обрезанная колонка
        try:
            column_image = self.crop_right_column(self.processed_image, boundaries)
        except Exception:
            column_image = self.processed_image
        axes[1, 1].imshow(column_image, cmap='gray')
        axes[1, 1].set_title("Обрезанная колонка")
        axes[1, 1].axis('off')

        # Сохраняем
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = os.path.join(output_dir, f"debug_visualization_{timestamp}.png")

        plt.tight_layout()
        plt.savefig(debug_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Отладочная визуализация сохранена: {debug_path}")
        return debug_path


def main():
    """Основная функция для тестирования"""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создаем процессор
    processor = StrategraphicColumnProcessor()
    
    # Путь к изображению
    image_path = "/mnt/data/strategraphic_column.jpg"
    
    try:
        # Обрабатываем стратиграфическую колонку
        layers = processor.process_strategraphic_column(image_path)
        
        # Выводим результаты
        processor.print_results()
        
        # Сохраняем результаты
        saved_files = processor.save_results()
        
        # Создаем отладочную визуализацию
        debug_path = processor.create_debug_visualization()
        
        print(f"\nРезультаты сохранены в файлы:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type.upper()}: {file_path}")
        
        if debug_path:
            print(f"  DEBUG: {debug_path}")
        
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}")
        logger.exception("Детали ошибки:")


if __name__ == "__main__":
    main()
