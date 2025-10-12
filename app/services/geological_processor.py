import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
import random
from app.core.config import settings

from collections import Counter
from app.services.strategraphic_column import StrategraphicColumnProcessor



try:
    # Опционально: используем rapidfuzz при наличии
    from rapidfuzz import fuzz
    def _similarity(a: str, b: str) -> float:
        return float(fuzz.token_set_ratio(a, b))
except Exception:
    # Fallback на difflib, если rapidfuzz недоступен
    from difflib import SequenceMatcher
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0

def _normalize_text(text: str) -> str:
    """Нормализует текст: нижний регистр, удаление лишних пробелов/переводов/декора."""
    if not text:
        return ""
    import re as _re
    t = str(text).lower().replace('\n', ' ')
    t = _re.sub(r"\s+", " ", t).strip()
    t = _re.sub(r"[‘’`´“”\[\]{}|\\]+", "", t)
    return t

logger = logging.getLogger(__name__)


class GeologicalProcessor:
    def __init__(self):
        self.legend_colors = []
        self.legend_order = []
        logger.info("GeologicalProcessor инициализирован")

    def rgb_to_lab(self, color: Tuple[int, int, int]) -> np.ndarray:
        """Конвертирует RGB цвет в LAB цветовое пространство"""
        # Нормализуем RGB в диапазон [0, 1]
        rgb_normalized = np.array(color) / 255.0
        # Конвертируем в LAB
        lab = cv2.cvtColor(
            np.array([[rgb_normalized]], dtype=np.float32), cv2.COLOR_RGB2LAB
        )
        return lab[0, 0]

    def lab_distance(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]
    ) -> float:
        """Вычисляет расстояние между цветами в LAB пространстве"""
        lab1 = self.rgb_to_lab(color1)
        lab2 = self.rgb_to_lab(color2)
        return np.linalg.norm(lab1 - lab2)

    def cluster_colors(
        self, colors: List[Tuple[int, int, int]], n_clusters: int = None
    ) -> List[Tuple[int, int, int]]:
        """Кластеризует цвета для уменьшения шума"""
        if len(colors) < 3:
            return colors

        if n_clusters is None:
            n_clusters = min(len(colors) // 10 + 1, len(colors))

        # Конвертируем цвета в LAB
        lab_colors = np.array([self.rgb_to_lab(color) for color in colors])

        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lab_colors)

        # Находим центры кластеров
        clustered_colors = []
        for i in range(n_clusters):
            cluster_colors = [colors[j] for j in range(len(colors)) if labels[j] == i]
            if cluster_colors:
                # Берем наиболее частый цвет в кластере
                color_counts = Counter(cluster_colors)
                most_common_color = color_counts.most_common(1)[0][0]
                clustered_colors.append(most_common_color)

        logger.info(f"Кластеризация: {len(colors)} -> {len(clustered_colors)} цветов")
        return clustered_colors

    def adaptive_color_tolerance(self, color: Tuple[int, int, int]) -> float:
        """Адаптивный допуск на основе яркости цвета - более мягкий подход для лучшего обнаружения"""
        # Увеличены допуски для лучшего сопоставления цветов с карты
        brightness = (color[0] + color[1] + color[2]) / 3.0
        if brightness < 100:
            return 100  # Темные цвета - умеренный допуск (было 40)
        elif brightness < 200:
            return 120  # Средние цвета - умеренный допуск (было 60)
        else:
            return 150  # Светлые цвета - мягкий допуск (было 80)

    def extract_legend_colors(
        self, legend_image: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Извлекает цвета из легенды сверху вниз"""
        logger.info(
            f"Начинаю извлечение цветов из легенды. Размер: {legend_image.shape}"
        )

        # Конвертируем в RGB если нужно
        if len(legend_image.shape) == 3 and legend_image.shape[2] == 3:
            legend_rgb = cv2.cvtColor(legend_image, cv2.COLOR_BGR2RGB)
            logger.debug("Конвертировал BGR в RGB")
        else:
            legend_rgb = legend_image
            logger.debug("Изображение уже в RGB формате")

        height, width = legend_rgb.shape[:2]
        colors = []

        # Улучшенный алгоритм извлечения цветов
        # Ищем цветные области, пропуская белый фон
        for y in range(0, height, 30):  # Увеличиваем шаг для лучшего покрытия
            row_colors = legend_rgb[y, :]
            # Находим доминирующий цвет в строке
            unique_colors, counts = np.unique(
                row_colors.reshape(-1, 3), axis=0, return_counts=True
            )

            if len(unique_colors) > 0:
                # Фильтруем белые/светлые цвета
                for color in unique_colors:
                    r, g, b = int(color[0]), int(color[1]), int(color[2])
                    # Пропускаем очень светлые цвета (белый фон)
                    if r < 250 and g < 250 and b < 250:  # Более мягкая фильтрация
                        # Проверяем, что это не серый цвет
                        if (
                            abs(r - g) > 15 or abs(g - b) > 15 or abs(r - b) > 15
                        ):  # Более мягкая проверка
                            color_tuple = (r, g, b)
                            if (
                                color_tuple not in colors and len(colors) < 20
                            ):  # Увеличиваем лимит
                                colors.append(color_tuple)
                                logger.debug(
                                    f"Добавлен цвет: {color_tuple} на позиции y={y}"
                                )
                                break  # Берем только первый не-белый цвет из строки

        # Если не нашли достаточно цветов, пробуем другой подход
        if len(colors) < 3:
            logger.warning(
                f"Найдено мало цветов ({len(colors)}), пробую альтернативный метод"
            )
            colors = self._extract_colors_alternative(legend_rgb)

        # Ограничиваем количество цветов до разумного значения
        if len(colors) > 15:
            logger.info(f"Слишком много цветов ({len(colors)}), ограничиваю до 15")
            colors = colors[:15]

        self.legend_colors = colors
        self.legend_order = list(range(len(colors)))

        logger.info(f"Извлечено {len(colors)} уникальных цветов из легенды")
        logger.debug(f"Цвета легенды: {colors}")

        return colors

    def extract_legend_names_ocr(self, legend_image: np.ndarray) -> List[str]:
        """Извлекает названия слоев из легенды с помощью OCR"""
        logger.info("Пытаюсь извлечь названия слоев с помощью Tesseract")

        try:
            import pytesseract

            logger.info("Tesseract импортирован успешно")

            # Конвертируем в RGB если нужно
            if len(legend_image.shape) == 3 and legend_image.shape[2] == 3:
                legend_rgb = cv2.cvtColor(legend_image, cv2.COLOR_BGR2RGB)
                logger.debug("Конвертировал изображение из BGR в RGB")
            else:
                legend_rgb = legend_image
                logger.debug("Изображение уже в RGB формате")

            logger.info(f"Размер изображения легенды: {legend_rgb.shape}")

            # Увеличиваем контраст для лучшего OCR
            legend_gray = cv2.cvtColor(legend_rgb, cv2.COLOR_RGB2GRAY)
            logger.debug("Конвертировал изображение в оттенки серого")

            _, legend_binary = cv2.threshold(
                legend_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            logger.debug("Применил бинаризацию с методом OTSU")

            # Извлекаем текст с помощью Tesseract
            logger.info("Запускаю Tesseract OCR для извлечения текста")
            text = pytesseract.image_to_string(legend_binary, lang="rus")
            logger.info(f"Tesseract вернул сырой текст длиной {len(text)} символов")

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            logger.info(f"Обработал текст в {len(lines)} строк")

            if lines:
                logger.info(f"Tesseract извлек текст: {lines}")
                return lines
            else:
                logger.warning("Tesseract не извлек ни одной строки текста")

        except ImportError:
            logger.error("Tesseract не установлен - импорт pytesseract не удался")
        except Exception as e:
            logger.error(f"Ошибка Tesseract OCR: {str(e)}")
            logger.exception("Детали ошибки:")

        # Если OCR не сработал, возвращаем стандартные названия
        logger.info("OCR не сработал, использую стандартные названия")
        return self.extract_legend_names()

    def extract_legend_names(self, legend_image: np.ndarray = None) -> List[str]:
        """Извлекает названия слоев из легенды"""
        logger.info("Извлекаю названия слоев из легенды")

        # Генерируем названия на основе геологических формаций
        geological_names = [
            "Кварциты",
            "Песчаники",
            "Глины",
            "Известняки",
            "Мергели",
            "Алевролиты",
            "Аргиллиты",
            "Конгломераты",
            "Брекчии",
            "Туфы",
            "Базальты",
            "Граниты",
            "Гнейсы",
            "Сланцы",
            "Мрамор",
        ]

        # Используем названия в порядке извлечения цветов
        names = []
        for i in range(len(self.legend_colors)):
            if i < len(geological_names):
                names.append(geological_names[i])
            else:
                names.append(f"Формация {i + 1}")

        logger.info(f"Сгенерировано {len(names)} названий слоев")
        logger.debug(f"Названия слоев: {names}")

        return names

    def _extract_colors_alternative(
        self, legend_rgb: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Альтернативный метод извлечения цветов"""
        height, width = legend_rgb.shape[:2]
        colors = []

        # Ищем цветные пиксели по всей легенде
        for y in range(0, height, 25):
            for x in range(0, width, 25):
                color = legend_rgb[y, x]
                r, g, b = int(color[0]), int(color[1]), int(color[2])

                # Фильтруем белые/светлые цвета
                if r < 250 and g < 250 and b < 250:
                    # Проверяем, что это не серый цвет
                    if abs(r - g) > 10 or abs(g - b) > 10 or abs(r - b) > 10:
                        color_tuple = (r, g, b)
                        if (
                            color_tuple not in colors and len(colors) < 20
                        ):  # Увеличиваем лимит
                            colors.append(color_tuple)
                            logger.debug(
                                f"Альтернативный метод: добавлен цвет {color_tuple}"
                            )

        return colors

    def get_line_pixels(
        self, start_point: Tuple[int, int], end_point: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Получает пиксели вдоль линии между двумя точками"""
        x1, y1 = start_point
        x2, y2 = end_point

        logger.info(f"Строю линию от ({x1}, {y1}) до ({x2}, {y2})")

        # Используем алгоритм Брезенхэма для построения линии
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > dy:
            steps = dx
        else:
            steps = dy

        x_increment = float(dx) / float(steps)
        y_increment = float(dy) / float(steps)

        x = float(x1)
        y = float(y1)

        for i in range(int(steps + 1)):
            points.append((int(round(x)), int(round(y))))
            x = x + x_increment
            y = y + y_increment

        logger.info(f"Построена линия с {len(points)} точками")
        logger.debug(f"Первые 5 точек: {points[:5]}")

        return points

    def find_closest_color(
        self, color: Tuple[int, int, int], legend_colors: List[Tuple[int, int, int]]
    ) -> Tuple[int, int]:
        """Находит ближайший цвет в легенде с использованием LAB пространства"""
        if not legend_colors:
            logger.warning("Список цветов легенды пуст")
            return -1, float("inf")

        # Сначала ищем точное совпадение
        for i, legend_color in enumerate(legend_colors):
            if color == legend_color:
                logger.debug(f"Найдено точное совпадение: {color} с индексом {i}")
                return i, 0.0

        # Если точного совпадения нет, ищем близкий цвет в LAB пространстве
        min_distance = float("inf")
        closest_index = -1
        adaptive_tolerance = self.adaptive_color_tolerance(color)

        for i, legend_color in enumerate(legend_colors):
            # Вычисляем расстояние в LAB пространстве
            distance = self.lab_distance(color, legend_color)

            if distance < min_distance and distance < adaptive_tolerance:
                min_distance = distance
                closest_index = i

        # Если не нашли в LAB, пробуем RGB с более мягким допуском
        if closest_index == -1:
            for i, legend_color in enumerate(legend_colors):
                # RGB расстояние
                rgb_distance = (
                    sum((a - b) ** 2 for a, b in zip(color, legend_color)) ** 0.5
                )

                if (
                    rgb_distance < min_distance and rgb_distance < 100
                ):  # Более мягкий допуск для RGB
                    min_distance = rgb_distance
                    closest_index = i

        if closest_index >= 0:
            logger.debug(
                f"Найдено близкое совпадение: {color} с {legend_colors[closest_index]}, "
                f"расстояние: {min_distance:.2f}, допуск: {adaptive_tolerance:.2f}"
            )
            return closest_index, min_distance
        else:
            logger.debug(f"Близкое совпадение не найдено для цвета: {color}")
            return -1, float("inf")

    def analyze_section_line(
        self,
        map_image: np.ndarray,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> List[Tuple[int, int, int]]:
        """Анализирует цвета вдоль линии разреза с кластеризацией"""
        logger.info(
            f"Анализирую цвета вдоль линии разреза. Размер карты: {map_image.shape}"
        )

        # Используем оригинальное изображение в BGR формате для соответствия с легендой
        if len(map_image.shape) == 3 and map_image.shape[2] == 3:
            map_bgr = map_image  # Оставляем в BGR формате
            logger.debug("Использую карту в BGR формате для соответствия с легендой")
        else:
            map_bgr = map_image
            logger.debug("Карта уже в нужном формате")

        line_pixels = self.get_line_pixels(start_point, end_point)
        colors_along_line = []
        unique_colors = set()

        for x, y in line_pixels:
            if 0 <= y < map_bgr.shape[0] and 0 <= x < map_bgr.shape[1]:
                color = tuple(map_bgr[y, x])  # BGR цвет
                colors_along_line.append(color)
                unique_colors.add(color)

        logger.info(f"Проанализировано {len(colors_along_line)} пикселей вдоль линии")
        logger.debug(f"Уникальных цветов вдоль линии: {len(unique_colors)}")

        # Кластеризуем цвета для уменьшения шума
        if len(unique_colors) > 5:
            clustered_colors = self.cluster_colors(list(unique_colors))
            logger.info(f"После кластеризации: {len(clustered_colors)} цветов")

            # Заменяем цвета в линии на ближайшие кластеризованные
            processed_colors = []
            for color in colors_along_line:
                closest_cluster = min(
                    clustered_colors, key=lambda c: self.lab_distance(color, c)
                )
                processed_colors.append(closest_cluster)

            colors_along_line = processed_colors

        logger.debug(f"Первые 10 цветов (BGR): {list(set(colors_along_line))[:10]}")

        return colors_along_line

    def build_geological_layers(
        self,
        colors_along_line: List[Tuple[int, int, int]],
        legend_colors: List[Tuple[int, int, int]],
    ) -> List[Dict]:
        """Строит геологические слои с улучшенным алгоритмом"""
        logger.info(f"Строю геологические слои из {len(colors_along_line)} цветов")
        logger.info(f"Цвета легенды для сопоставления: {legend_colors}")

        # Анализируем последовательность цветов вдоль линии
        layer_sequence = []
        current_color = None
        current_length = 0

        for i, color in enumerate(colors_along_line):
            closest_index, distance = self.find_closest_color(color, legend_colors)

            if closest_index >= 0:
                if current_color is None or current_color != closest_index:
                    # Начинаем новый слой
                    if current_color is not None or current_length > 0:
                        layer_sequence.append(
                            {
                                "index": current_color,
                                "color": legend_colors[current_color],
                                "order": current_color,
                                "length": current_length,
                                "start_pos": i - current_length,
                            }
                        )

                    current_color = closest_index
                    current_length = 1
                else:
                    # Продолжаем текущий слой
                    current_length += 1
            else:
                # Пропускаем цвета, не найденные в легенде
                if current_color is not None and current_length > 0:
                    layer_sequence.append(
                        {
                            "index": current_color,
                            "color": legend_colors[current_color],
                            "order": current_color,
                            "length": current_length,
                            "start_pos": i - current_length,
                        }
                    )
                    current_color = None
                    current_length = 0

        # Добавляем последний слой
        if current_color is not None and current_length > 0:
            layer_sequence.append(
                {
                    "index": current_color,
                    "color": legend_colors[current_color],
                    "order": current_color,
                    "length": current_length,
                    "start_pos": len(colors_along_line) - current_length,
                }
            )

        # Если не нашли слои, пробуем более простой подход
        if not layer_sequence:
            logger.warning("Не найдено слоев, пробую альтернативный подход")
            unique_colors = list(set(colors_along_line))
            for color in unique_colors:
                closest_index, distance = self.find_closest_color(color, legend_colors)
                if closest_index >= 0:
                    layer_sequence.append(
                        {
                            "index": closest_index,
                            "color": legend_colors[closest_index],
                            "order": closest_index,
                            "length": colors_along_line.count(color),
                            "start_pos": 0,
                        }
                    )

        # Фильтруем слишком короткие слои (шум)
        min_layer_length = max(
            1, len(colors_along_line) // 100
        )  # Уменьшаем минимальную длину
        filtered_layers = [
            layer for layer in layer_sequence if layer["length"] >= min_layer_length
        ]

        # Удаляем дубликаты, оставляя только первое вхождение каждого слоя
        unique_layers = []
        seen_indices = set()

        for layer in filtered_layers:
            if layer["index"] not in seen_indices:
                unique_layers.append(layer)
                seen_indices.add(layer["index"])

        # Сортируем слои по порядку в легенде
        unique_layers.sort(key=lambda x: x["order"], reverse=False)

        logger.info(f"Построено {len(unique_layers)} геологических слоев")
        logger.info(
            f"Слои: {[(layer['order'], layer['length']) for layer in unique_layers]}"
        )

        return unique_layers

    def create_section_visualization(self, layers: List[Dict], output_path: str) -> str:
        """Создает реалистичную визуализацию геологического разреза с неровными границами"""
        logger.info(f"Создаю реалистичную визуализацию для {len(layers)} слоев")

        if not layers:
            logger.warning("Нет слоев для визуализации")
            # Создаем пустую визуализацию с сообщением
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.text(
                0.5,
                0.5,
                "Не найдено слоев для отображения",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Геологический разрез", fontsize=16, fontweight="bold")

            # Сохраняем изображение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"section_{timestamp}.png"
            filepath = os.path.join(settings.output_dir, filename)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Пустая визуализация сохранена: {filepath}")
            return filepath

        # Создаем фигуру для реалистичного разреза
        fig, ax = plt.subplots(figsize=(20, 12))

        # Параметры разреза
        section_width = 16.0
        section_height = 10.0
        base_y = 0.5  # Отступ снизу для подписей

        # Высокое разрешение для плавности
        x_points = np.linspace(0, section_width, 200)
        
        # Сортируем слои по порядку появления
        sorted_layers = sorted(layers, key=lambda x: x["order"])
        
        # Создаем реалистичные границы слоев
        layer_boundaries = []
        current_depth = section_height + base_y
        
        logger.info("Отображаемые слои в визуализации (сверху вниз):")
        
        # Генерируем границы всех слоев сразу для непрерывности
        for i, layer in enumerate(sorted_layers):
            # Вычисляем толщину слоя
            layer_length = layer.get("length", 100)
            total_pixels = sum(layer_item.get("length", 100) for layer_item in sorted_layers)
            normalized_length = layer_length / total_pixels if total_pixels > 0 else 1.0
            
            # Более реалистичное распределение толщин
            min_thickness = 0.8
            max_thickness = 2.5
            layer_thickness = min_thickness + (max_thickness - min_thickness) * normalized_length
            
            # Создаем естественную неровную границу
            np.random.seed(42 + i)
            
            # Комбинируем несколько волн разной частоты для реалистичности
            main_wave = np.sin(x_points * 0.4 + i * 0.3) * 0.12
            secondary_wave = np.sin(x_points * 1.2 + i * 0.7) * 0.05
            fine_details = np.sin(x_points * 2.8 + i * 1.1) * 0.02
            
            # Добавляем случайные геологические нарушения
            disturbances = np.random.normal(0, 0.04, len(x_points))
            
            # Сглаживаем для плавности
            try:
                from scipy.ndimage import gaussian_filter1d
                disturbances = gaussian_filter1d(disturbances, sigma=3)
            except ImportError:
                # Простое сглаживание без scipy
                kernel_size = 5
                disturbances = np.convolve(disturbances, np.ones(kernel_size)/kernel_size, mode='same')
            
            # Комбинируем все компоненты
            boundary_variation = main_wave + secondary_wave + fine_details + disturbances
            
            # Вычисляем нижнюю границу слоя
            next_depth = current_depth - layer_thickness + boundary_variation
            
            # Обеспечиваем непрерывность - нижняя граница не должна пересекать верхнюю
            next_depth = np.minimum(next_depth, current_depth - 0.3)
            
            layer_boundaries.append({
                'top': current_depth if i == 0 else layer_boundaries[i-1]['bottom'],
                'bottom': next_depth,
                'layer': layer,
                'thickness': layer_thickness
            })
            
            current_depth = np.mean(next_depth)
        
        # Рисуем слои
        for i, boundary in enumerate(layer_boundaries):
            layer = boundary['layer']
            
            # Нормализация цвета
            r, g, b = layer["color"]
            color = (r / 255.0, g / 255.0, b / 255.0)
            
            # Создаем полигон слоя
            top_boundary = boundary['top'] if isinstance(boundary['top'], np.ndarray) else np.full(len(x_points), boundary['top'])
            bottom_boundary = boundary['bottom']
            
            # Создаем замкнутый полигон
            polygon_x = np.concatenate([x_points, x_points[::-1]])
            polygon_y = np.concatenate([top_boundary, bottom_boundary[::-1]])
            
            # Основной слой
            layer_polygon = plt.Polygon(
                list(zip(polygon_x, polygon_y)),
                facecolor=color,
                edgecolor='none',  # Убираем контур для более естественного вида
                alpha=0.95
            )
            ax.add_patch(layer_polygon)
            
            # Добавляем тонкие границы только там, где нужно подчеркнуть контакт
            if i > 0:  # Не рисуем верхнюю границу первого слоя
                ax.plot(x_points, top_boundary, color='black', linewidth=0.5, alpha=0.7)
            
            # Добавляем текстуры для разных типов пород
            self._add_layer_texture(ax, x_points, top_boundary, bottom_boundary, layer, color)
            
            # Подпись слоя (очищаем от переносов строк)
            raw_text = layer.get("text", f"Слой {layer['order'] + 1}")
            clean_text = self._clean_legend_text(raw_text)
            layer_text = clean_text
            if "length" in layer:
                layer_text += f" ({layer['length']}px)"
            
            # Размещаем текст в центре слоя
            center_x = section_width / 2
            center_y = np.mean(top_boundary) - (np.mean(top_boundary) - np.mean(bottom_boundary)) / 2
            
            # Текст с полупрозрачным фоном
            ax.text(
                center_x, center_y, layer_text,
                ha="center", va="center", 
                fontsize=9, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none")
            )
            
            logger.info(f"  - {layer_text}: цвет_легенды_BGR {layer['color']}, толщина={boundary['thickness']:.2f}")
        
        # Настраиваем оси
        ax.set_xlim(-0.5, section_width + 8.0)  # Расширяем справа для легенды
        ax.set_ylim(base_y - 0.5, section_height + base_y + 0.5)
        ax.set_aspect("equal")
        ax.set_title("Геологический разрез", fontsize=16, fontweight="bold")
        
        # Убираем оси для более чистого вида
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Убираем рамку вокруг графика
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        # Добавляем сетку для масштаба
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        
        # Добавляем легенду справа от разреза
        legend_x_start = section_width + 1.0
        legend_y_start = section_height + base_y - 1.0
        legend_item_height = 0.6
        legend_item_width = 0.8
        legend_text_x_offset = 1.2
        
        # Сортируем слои по порядку (сверху вниз)
        for i, layer in enumerate(sorted_layers):
            legend_y = legend_y_start - i * legend_item_height
            
            # Цветной квадратик
            b, g, r = layer["color"]  # BGR формат
            color_normalized = (r / 255.0, g / 255.0, b / 255.0)
            
            legend_rect = plt.Rectangle(
                (legend_x_start, legend_y),
                legend_item_width,
                legend_item_height * 0.8,
                facecolor=color_normalized,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(legend_rect)
            
            # Текст слоя (очищаем от переносов строк)
            raw_text = layer.get("text", f"Слой {layer['order'] + 1}")
            clean_text = self._clean_legend_text(raw_text)
            layer_text = clean_text
            if "length" in layer:
                layer_text += f" ({layer['length']}px)"
                
            ax.text(
                legend_x_start + legend_text_x_offset,
                legend_y + legend_item_height * 0.4,
                layer_text,
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
        
        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"section_{timestamp}.png"
        filepath = os.path.join(settings.output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Реалистичная визуализация сохранена: {filepath}")
        return filepath
    
    def _add_layer_texture(self, ax, x_points, top_boundary, bottom_boundary, layer, base_color):
        """Добавляет текстуру слоя в зависимости от типа породы"""
        layer_text = layer.get("text", "").lower()
        
        # Определяем тип породы по тексту
        if any(word in layer_text for word in ['песчан', 'песок', 'алевро']):
            # Песчаные породы - точечная текстура
            self._add_sandstone_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
        elif any(word in layer_text for word in ['глин', 'аргил', 'мерг']):
            # Глинистые породы - горизонтальные линии
            self._add_shale_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
        elif any(word in layer_text for word in ['известняк', 'карбон']):
            # Известняки - блочная текстура
            self._add_limestone_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
        elif any(word in layer_text for word in ['лав', 'базальт', 'андези']):
            # Вулканические породы - неправильная текстура
            self._add_volcanic_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
    
    def _add_sandstone_texture(self, ax, x_points, top_boundary, bottom_boundary, base_color):
        """Песчаная текстура - мелкие точки"""
        np.random.seed(42)
        n_points = int(len(x_points) * 0.3)
        for _ in range(n_points):
            x = np.random.choice(x_points)
            x_idx = np.argmin(np.abs(x_points - x))
            y_range = top_boundary[x_idx] - bottom_boundary[x_idx]
            if y_range > 0:
                y = bottom_boundary[x_idx] + np.random.random() * y_range
                ax.plot(x, y, 'o', color='black', markersize=0.5, alpha=0.3)
    
    def _add_shale_texture(self, ax, x_points, top_boundary, bottom_boundary, base_color):
        """Глинистая текстура - тонкие горизонтальные линии"""
        n_lines = 8
        for i in range(n_lines):
            progress = (i + 1) / (n_lines + 1)
            y_line = []
            for j, x in enumerate(x_points[::5]):  # Каждая 5-я точка
                y_range = top_boundary[j*5] - bottom_boundary[j*5]
                if y_range > 0:
                    y = bottom_boundary[j*5] + progress * y_range
                    y_line.append(y)
                else:
                    y_line.append(bottom_boundary[j*5])
            
            if len(y_line) > 1:
                ax.plot(x_points[::5], y_line, '-', color='black', linewidth=0.3, alpha=0.4)
    
    def _add_limestone_texture(self, ax, x_points, top_boundary, bottom_boundary, base_color):
        """Известняковая текстура - прямоугольные блоки"""
        block_width = 0.4
        n_blocks = int(len(x_points) * 0.1)
        np.random.seed(42)
        
        for _ in range(n_blocks):
            x = np.random.choice(x_points[:-10])
            x_idx = np.argmin(np.abs(x_points - x))
            y_range = top_boundary[x_idx] - bottom_boundary[x_idx]
            if y_range > 0.2:
                y = bottom_boundary[x_idx] + np.random.random() * (y_range - 0.2)
                rect = plt.Rectangle((x, y), block_width, 0.15, 
                                   facecolor='none', edgecolor='black', 
                                   linewidth=0.4, alpha=0.5)
                ax.add_patch(rect)
    
    def _add_volcanic_texture(self, ax, x_points, top_boundary, bottom_boundary, base_color):
        """Вулканическая текстура - неправильные формы"""
        np.random.seed(42)
        n_shapes = int(len(x_points) * 0.05)
        
        for _ in range(n_shapes):
            x = np.random.choice(x_points[:-20])
            x_idx = np.argmin(np.abs(x_points - x))
            y_range = top_boundary[x_idx] - bottom_boundary[x_idx]
            if y_range > 0.3:
                y = bottom_boundary[x_idx] + np.random.random() * (y_range - 0.3)
                # Неправильная форма
                angles = np.linspace(0, 2*np.pi, 8)
                radii = np.random.uniform(0.1, 0.2, 8)
                shape_x = x + radii * np.cos(angles)
                shape_y = y + radii * np.sin(angles)
                shape = plt.Polygon(list(zip(shape_x, shape_y)), 
                                  facecolor='black', alpha=0.2, edgecolor='none')
                ax.add_patch(shape)

    def create_section_visualization_with_names(
        self, layers: List[Dict], legend_data: List[Dict], output_path: str
    ) -> str:
        """Создает реалистичную визуализацию геологического разреза с естественными геологическими структурами"""
        logger.info(
            f"Создаю улучшенную реалистичную визуализацию для {len(layers)} слоев с геологическими структурами"
        )

        if not layers:
            logger.warning("Нет слоев для визуализации")
            # Создаем пустую визуализацию с сообщением
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.text(
                0.5,
                0.5,
                "Не найдено слоев для отображения",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Геологический разрез", fontsize=16, fontweight="bold")

            # Сохраняем изображение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"section_{timestamp}.png"
            filepath = os.path.join(settings.output_dir, filename)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Пустая визуализация сохранена: {filepath}")
            return filepath

        # Создаем фигуру для реалистичного разреза
        fig, ax = plt.subplots(figsize=(24, 14))

        # Параметры разреза
        section_width = 20.0
        section_height = 12.0
        base_y = 1.0

        # Высокое разрешение для плавных границ
        x_points = np.linspace(0, section_width, 300)
        
        # Сортируем слои по порядку (сверху вниз)
        sorted_layers = sorted(layers, key=lambda x: x.get("order", 0))
        
        logger.info("Отображаемые слои в визуализации (сверху вниз):")

        # Генерируем реалистичные геологические структуры
        layer_boundaries = self._generate_geological_boundaries(sorted_layers, x_points, section_height, base_y)
        
        # Рисуем слои с улучшенным реализмом
        for i, boundary_data in enumerate(layer_boundaries):
            layer = boundary_data['layer']
            top_boundary = boundary_data['top']
            bottom_boundary = boundary_data['bottom']
            
            # Получаем цвет из легенды
            legend_color = layer["color"]
            b, g, r = legend_color
            base_color = np.array([r / 255.0, g / 255.0, b / 255.0])
            
            # Создаем градиентный эффект для объемности
            colors_array = self._create_layer_gradient(base_color, len(x_points))
            
            # Рисуем основной полигон слоя
            
            # Основной слой с градиентом
            for j in range(len(x_points) - 1):
                # Создаем мини-полигон для каждого сегмента
                segment_x = [x_points[j], x_points[j+1], x_points[j+1], x_points[j]]
                segment_y = [top_boundary[j], top_boundary[j+1], bottom_boundary[j+1], bottom_boundary[j]]
                
                # Применяем градиент цвета
                segment_color = colors_array[j]
                
                segment_polygon = plt.Polygon(
                    list(zip(segment_x, segment_y)),
                    facecolor=segment_color,
                    edgecolor='none',
                    alpha=0.9
                )
                ax.add_patch(segment_polygon)
            
            # Добавляем контурные линии для подчеркивания структуры
            if i > 0:  # Не рисуем верхнюю границу первого слоя
                ax.plot(x_points, top_boundary, color='black', linewidth=1.2, alpha=0.6)
            
            # Добавляем геологические текстуры
            self._add_enhanced_geological_texture(ax, x_points, top_boundary, bottom_boundary, layer, base_color)
            
            # Подпись слоя
            raw_text = layer.get("text", f"Слой {i + 1}")
            clean_text = self._clean_legend_text(raw_text)
            
            # Вычисляем толщину слоя для логирования
            avg_top = np.mean(top_boundary)
            avg_bottom = np.mean(bottom_boundary)
            layer_thickness = avg_top - avg_bottom
            logger.info(f"  - {clean_text[:50]}... : цвет_легенды_BGR {layer['color']}, толщина={layer_thickness:.2f}")
        
        # Добавляем геологические особенности
        self._add_geological_features(ax, x_points, section_height, base_y)
        
        # Настраиваем оси и внешний вид
        ax.set_xlim(-1.0, section_width + 10.0)
        ax.set_ylim(base_y - 1.0, section_height + base_y + 1.0)
        ax.set_aspect("equal")
        
        # Убираем стандартные оси
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        
        # Добавляем масштабную сетку
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.8, color='gray')
        
        # Добавляем заголовок в верхний центр
        ax.text(section_width / 2, section_height + base_y + 0.5, 
               "Геологический разрез", ha="center", va="bottom",
               fontsize=16, fontweight="bold", color="black")
        
        # Добавляем боковую легенду
        self._add_enhanced_legend(ax, sorted_layers, section_width, section_height, base_y)
        
        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"section_{timestamp}.png"
        filepath = os.path.join(settings.output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        
        logger.info(f"Улучшенная реалистичная визуализация сохранена: {filepath}")
        return filepath

    def _generate_geological_boundaries(self, layers: List[Dict], x_points: np.ndarray, section_height: float, base_y: float) -> List[Dict]:
        """Генерирует реалистичные геологические границы с естественными структурами"""
        boundaries = []
        current_top = section_height + base_y
        
        # Глобальные геологические параметры (все случайные)
        regional_tilt = random.uniform(0.05, 0.2)  # Региональный наклон слоев (случайный)
        folding_amplitude = random.uniform(0.1, 0.4)  # Амплитуда складчатости (случайная)
        folding_wavelength = random.uniform(6.0, 12.0)  # Длина волны складок (случайная)
        
        for i, layer in enumerate(layers):
            # Определяем толщину слоя
            layer_length = layer.get("length", 100)
            total_pixels = sum(layer_item.get("length", 100) for layer_item in layers)
            normalized_length = layer_length / total_pixels if total_pixels > 0 else 1.0
            
            # Адаптивная толщина на основе геологического типа
            base_thickness = 0.6 + 2.0 * normalized_length
            
            # Добавляем геологические структуры
            np.random.seed(42 + i)
            
            # Региональный наклон
            regional_slope = regional_tilt * (x_points / max(x_points))
            
            # Складчатость (антиклинали и синклинали)
            folding = folding_amplitude * np.sin(2 * np.pi * x_points / folding_wavelength + i * 0.3)
            
            # Локальные нарушения и неровности
            local_variations = []
            for j in range(len(x_points)):
                # Случайные геологические нарушения
                fault_influence = 0.0
                if np.random.random() < 0.05:  # 5% вероятность разлома
                    fault_influence = np.random.uniform(-0.2, 0.2)
                
                # Эрозионные процессы
                erosion = np.random.normal(0, 0.03)
                
                local_variations.append(fault_influence + erosion)
            
            local_variations = np.array(local_variations)
            
            # Сглаживание для плавности
            try:
                from scipy.ndimage import gaussian_filter1d
                local_variations = gaussian_filter1d(local_variations, sigma=2)
            except ImportError:
                pass
            
            # Комбинируем все эффекты
            bottom_variation = regional_slope + folding + local_variations
            
            # Создаем верхнюю границу (нижняя граница предыдущего слоя)
            if i == 0:
                top_boundary = np.full(len(x_points), current_top)
            else:
                top_boundary = boundaries[i-1]['bottom'].copy()
            
            # Создаем нижнюю границу
            bottom_boundary = top_boundary - base_thickness + bottom_variation
            
            boundaries.append({
                'layer': layer,
                'top': top_boundary,
                'bottom': bottom_boundary,
                'thickness': base_thickness
            })
            
            current_top = np.mean(bottom_boundary)
        
        return boundaries

    def _create_layer_gradient(self, base_color: np.ndarray, num_points: int) -> np.ndarray:
        """Создает градиент цвета для придания объемности слою"""
        colors = np.zeros((num_points, 3))
        
        for i in range(num_points):
            # Создаем вариации яркости для имитации освещения
            brightness_factor = 0.9 + 0.2 * np.sin(np.pi * i / num_points)
            
            # Добавляем небольшие случайные вариации для естественности
            noise = np.random.normal(0, 0.02, 3)
            
            # Применяем эффекты
            color_variant = base_color * brightness_factor + noise
            color_variant = np.clip(color_variant, 0, 1)
            
            colors[i] = color_variant
        
        return colors

    def _add_enhanced_geological_texture(self, ax, x_points: np.ndarray, top_boundary: np.ndarray, 
                                       bottom_boundary: np.ndarray, layer: Dict, base_color: np.ndarray):
        """Добавляет детализированные геологические текстуры"""
        layer_text = layer.get("text", "").lower()
        
        # Определяем тип породы и добавляем соответствующую текстуру
        if any(word in layer_text for word in ['песчан', 'песок', 'алевро']):
            self._add_sandstone_enhanced_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
        elif any(word in layer_text for word in ['глин', 'аргил']):
            self._add_shale_enhanced_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
        elif any(word in layer_text for word in ['известняк', 'мергел']):
            self._add_limestone_enhanced_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
        elif any(word in layer_text for word in ['лав', 'базальт', 'андези', 'вулкан']):
            self._add_volcanic_enhanced_texture(ax, x_points, top_boundary, bottom_boundary, base_color)
        else:
            self._add_generic_enhanced_texture(ax, x_points, top_boundary, bottom_boundary, base_color)

    def _add_sandstone_enhanced_texture(self, ax, x_points: np.ndarray, top_boundary: np.ndarray, 
                                      bottom_boundary: np.ndarray, base_color: np.ndarray):
        """Улучшенная текстура песчаника"""
        np.random.seed(42)
        
        # Горизонтальная слоистость
        n_layers = max(3, int((np.mean(top_boundary) - np.mean(bottom_boundary)) * 8))
        for i in range(n_layers):
            y_pos = np.mean(bottom_boundary) + (np.mean(top_boundary) - np.mean(bottom_boundary)) * i / n_layers
            ax.axhline(y=y_pos, xmin=0, xmax=max(x_points)/24, color='black', alpha=0.2, linewidth=0.3)
        
        # Зернистая текстура
        n_grains = int(len(x_points) * 0.4)
        for _ in range(n_grains):
            x = np.random.choice(x_points)
            x_idx = np.argmin(np.abs(x_points - x))
            if x_idx < len(top_boundary):
                y_range = top_boundary[x_idx] - bottom_boundary[x_idx]
                if y_range > 0:
                    y = bottom_boundary[x_idx] + np.random.random() * y_range
                    ax.plot(x, y, 'o', color='black', markersize=0.3, alpha=0.4)

    def _add_shale_enhanced_texture(self, ax, x_points: np.ndarray, top_boundary: np.ndarray, 
                                  bottom_boundary: np.ndarray, base_color: np.ndarray):
        """Улучшенная текстура глинистых пород"""
        # Тонкая слоистость
        layer_thickness = np.mean(top_boundary) - np.mean(bottom_boundary)
        n_laminae = max(5, int(layer_thickness * 15))
        
        for i in range(n_laminae):
            progress = i / n_laminae
            y_line = []
            x_subset = x_points[::3]  # Каждая третья точка для производительности
            
            for j, x in enumerate(x_subset):
                if j * 3 < len(top_boundary):
                    y_range = top_boundary[j * 3] - bottom_boundary[j * 3]
                    if y_range > 0:
                        y = bottom_boundary[j * 3] + progress * y_range
                        # Добавляем небольшие волнистости
                        y += 0.01 * np.sin(x * 2 + i * 0.5)
                        y_line.append(y)
            
            if len(y_line) > 1:
                ax.plot(x_subset[:len(y_line)], y_line, color='black', alpha=0.3, linewidth=0.2)

    def _add_limestone_enhanced_texture(self, ax, x_points: np.ndarray, top_boundary: np.ndarray, 
                                      bottom_boundary: np.ndarray, base_color: np.ndarray):
        """Улучшенная текстура известняков"""
        # Блочная структура
        np.random.seed(42)
        
        layer_thickness = np.mean(top_boundary) - np.mean(bottom_boundary)
        n_joints = max(2, int(layer_thickness * 6))
        
        # Вертикальные трещины
        for _ in range(n_joints):
            x_joint = np.random.uniform(min(x_points), max(x_points))
            x_idx = np.argmin(np.abs(x_points - x_joint))
            if x_idx < len(top_boundary):
                y_top = top_boundary[x_idx]
                y_bottom = bottom_boundary[x_idx]
                ax.plot([x_joint, x_joint], [y_bottom, y_top], color='black', alpha=0.4, linewidth=0.3)

    def _add_volcanic_enhanced_texture(self, ax, x_points: np.ndarray, top_boundary: np.ndarray, 
                                     bottom_boundary: np.ndarray, base_color: np.ndarray):
        """Улучшенная текстура вулканических пород"""
        np.random.seed(42)
        
        # Хаотичная структура вулканических пород
        n_fragments = int(len(x_points) * 0.2)
        for _ in range(n_fragments):
            x = np.random.choice(x_points)
            x_idx = np.argmin(np.abs(x_points - x))
            if x_idx < len(top_boundary):
                y_range = top_boundary[x_idx] - bottom_boundary[x_idx]
                if y_range > 0:
                    y = bottom_boundary[x_idx] + np.random.random() * y_range
                    size = np.random.uniform(0.5, 1.5)
                    ax.plot(x, y, 's', color='darkred', markersize=size, alpha=0.5)

    def _add_generic_enhanced_texture(self, ax, x_points: np.ndarray, top_boundary: np.ndarray, 
                                    bottom_boundary: np.ndarray, base_color: np.ndarray):
        """Общая текстура для неопределенных пород"""
        # Простая точечная текстура
        np.random.seed(42)
        n_points = int(len(x_points) * 0.15)
        
        for _ in range(n_points):
            x = np.random.choice(x_points)
            x_idx = np.argmin(np.abs(x_points - x))
            if x_idx < len(top_boundary):
                y_range = top_boundary[x_idx] - bottom_boundary[x_idx]
                if y_range > 0:
                    y = bottom_boundary[x_idx] + np.random.random() * y_range
                    ax.plot(x, y, '.', color='black', markersize=0.5, alpha=0.3)

    def _add_geological_features(self, ax, x_points: np.ndarray, section_height: float, base_y: float):
        """Добавляет дополнительные геологические особенности"""
        # Линии разломов
        np.random.seed(123)
        if np.random.random() < 0.3:  # 30% вероятность разлома
            fault_x = np.random.uniform(min(x_points) * 0.3, max(x_points) * 0.7)
            fault_angle = np.random.uniform(-0.3, 0.3)
            
            y_top = section_height + base_y
            y_bottom = base_y
            
            fault_x_top = fault_x
            fault_x_bottom = fault_x + fault_angle * (y_top - y_bottom)
            
            ax.plot([fault_x_top, fault_x_bottom], [y_top, y_bottom], 
                   color='red', linewidth=2, alpha=0.7, linestyle='--', label='Geological fault')

    def _clean_legend_text(self, text: str) -> str:
        """Очищает текст легенды от артефактов и специальных символов"""
        import re
        
        # Убираем специальные символы и артефакты
        text = re.sub(r'[=\[\]@&{}|<>~`^\\]', '', text)
        
        # Убираем множественные пробелы и переносы строк
        text = ' '.join(text.replace('\n', ' ').replace('\r', ' ').split())
        
        # Убираем ведущие и trailing пробелы
        text = text.strip()
        
        return text

    def _add_enhanced_legend(self, ax, layers: List[Dict], section_width: float, section_height: float, base_y: float):
        """Добавляет улучшенную легенду с детальной информацией"""
        legend_x_start = section_width + 1.5
        legend_y_start = section_height + base_y - 0.5
        legend_item_height = 0.8
        legend_item_width = 1.0
        
        # Заголовок легенды
        ax.text(legend_x_start + legend_item_width/2, legend_y_start + 0.3, 
               "ЛЕГЕНДА", ha="center", va="bottom",
               fontsize=12, fontweight="bold", color="black")
        
        for i, layer in enumerate(layers):
            legend_y = legend_y_start - i * legend_item_height - 0.8
            
            # Цветной прямоугольник
            b, g, r = layer["color"]
            color_normalized = (r / 255.0, g / 255.0, b / 255.0)
            
            legend_rect = plt.Rectangle(
                (legend_x_start, legend_y),
                legend_item_width, legend_item_height * 0.7,
                facecolor=color_normalized, edgecolor="black", linewidth=1.5
            )
            ax.add_patch(legend_rect)
            
            # Текст описания с полной очисткой от артефактов
            raw_text = layer.get("text", f"Слой {i + 1}")
            clean_text = self._clean_legend_text(raw_text)
            
            # Ограничиваем длину текста
            if len(clean_text) > 40:
                clean_text = clean_text[:37] + "..."
            
            ax.text(legend_x_start + legend_item_width + 0.3, legend_y + legend_item_height * 0.35,
                   clean_text, ha="left", va="center", fontsize=9, fontweight="bold")

    def create_debug_legend_visualization(
        self, legend_data: List[Dict], output_path: str = "outputs"
    ) -> str:
        """
        Создает отладочную визуализацию легенды с извлеченными блоками, цветами и текстами
        
        Args:
            legend_data: Данные легенды с блоками
            output_path: Путь для сохранения
            
        Returns:
            Путь к созданному изображению
        """
        logger.info(f"Создаю отладочную визуализацию легенды для {len(legend_data)} блоков")
        
        # Создаем директорию если не существует
        os.makedirs(output_path, exist_ok=True)
        
        # Генерируем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_legend_{timestamp}.png"
        filepath = os.path.join(output_path, filename)
        
        # Определяем размеры изображения
        block_height = 60  # Высота каждого блока
        legend_width = 800  # Ширина легенды
        color_width = 60   # Ширина цветного квадрата
        # text_width = legend_width - color_width - 20  # Ширина для текста (не используется)
        
        total_height = len(legend_data) * block_height + 40  # +40 для заголовка
        
        # Создаем фигуру
        fig, ax = plt.subplots(1, 1, figsize=(legend_width/80, total_height/80))
        ax.set_xlim(0, legend_width)
        ax.set_ylim(0, total_height)
        ax.set_aspect('equal')
        
        # Убираем оси
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Добавляем заголовок
        ax.text(legend_width/2, total_height - 20, 
                f"Отладочная легенда ({len(legend_data)} блоков)", 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Отображаем каждый блок легенды
        for i, block in enumerate(legend_data):
            y_pos = total_height - 40 - (i + 1) * block_height
            
            # Извлекаем данные блока
            color = block.get('color', [128, 128, 128])  # BGR по умолчанию
            text = block.get('text', 'Нет текста')
            column_order = block.get('column_order', -1)
            match_score = block.get('match_score', 0.0)
            
            # Конвертируем цвет из BGR в RGB для matplotlib
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                rgb_color = (color[2]/255, color[1]/255, color[0]/255)  # BGR -> RGB
            else:
                rgb_color = (0.5, 0.5, 0.5)  # Серый по умолчанию
            
            # Рисуем цветной квадрат
            color_rect = plt.Rectangle((10, y_pos + 10), color_width - 20, block_height - 20,
                                     facecolor=rgb_color, edgecolor='black', linewidth=1)
            ax.add_patch(color_rect)
            
            # Добавляем текст (удаляем переносы строк и лишние пробелы)
            clean_text = ' '.join(text.replace('\n', ' ').split())
            text_preview = clean_text[:80] + "..." if len(clean_text) > 80 else clean_text
            
            # Основной текст
            ax.text(color_width + 10, y_pos + block_height - 15, 
                    f"{i+1}. {text_preview}", 
                    ha='left', va='top', fontsize=8, fontweight='bold')
            
            # Информация о сопоставлении
            if column_order >= 0:
                info_text = f"Порядок: {column_order}, Совпадение: {match_score:.1f}%"
                color_info = 'green' if match_score >= 70 else 'orange' if match_score >= 40 else 'red'
            else:
                info_text = "Не сопоставлено"
                color_info = 'gray'
                
            ax.text(color_width + 10, y_pos + 25, 
                    info_text, 
                    ha='left', va='center', fontsize=7, color=color_info)
            
            # Цвет в формате BGR
            ax.text(color_width + 10, y_pos + 10, 
                    f"BGR: {color}", 
                    ha='left', va='bottom', fontsize=7, color='gray')
        
        # Добавляем статистику внизу
        matched_count = sum(1 for block in legend_data if block.get('column_order', -1) >= 0)
        stats_text = f"Сопоставлено: {matched_count}/{len(legend_data)} блоков"
        ax.text(legend_width/2, 10, stats_text, 
                ha='center', va='bottom', fontsize=10, style='italic')
        
        # Сохраняем изображение
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close()
        
        logger.info(f"Отладочная легенда сохранена: {filepath}")
        return filepath

    def process_geological_section(
        self,
        map_image: np.ndarray,
        legend_image: np.ndarray,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> Dict:
        """Основной метод обработки геологического разреза"""
        logger.info("=== НАЧАЛО ОБРАБОТКИ ГЕОЛОГИЧЕСКОГО РАЗРЕЗА ===")
        logger.debug(f"Размер карты: {map_image.shape}, легенды: {legend_image.shape}, точки: {start_point}->{end_point}")

        try:
            # 1. Извлекаем данные из легенды
            logger.debug("Шаг 1: Извлечение данных из легенды")
            legend_data = self.extract_legend_data(legend_image)

            if not legend_data:
                logger.warning(
                    "Не удалось извлечь данные из легенды, использую старый метод"
                )
                legend_colors = self.extract_legend_colors(legend_image)
                legend_data = [
                    {"color": color, "text": f"Формация {i}", "symbol": f"F{i}"}
                    for i, color in enumerate(legend_colors)
                ]
            else:
                # Генерируем названия на основе символов
                legend_data = self._generate_geological_names_from_symbols(legend_data)

            # 1.1 Извлекаем последовательность текстов из стратиграфической колонки
            logger.debug("Шаг 1.1: Извлечение последовательности слоев из стратиграфической колонки")
            column_layers: List[str] = []
            try:
                # Используем улучшенную очистку геологических терминов
                sc_processor = StrategraphicColumnProcessor(use_enhanced_cleaning=True)
                column_layers = sc_processor.process_strategraphic_column(
                    "uploads/strategraphic_column.jpg"
                )
                logger.info(f"Извлечено {len(column_layers)} слоев из колонки с улучшенной очисткой")
                
                # Логируем извлеченные слои для отладки
                if column_layers:
                    logger.debug("Последовательность слоев из стратиграфической колонки:")
                    for i, layer in enumerate(column_layers):
                        preview = layer[:60] + "..." if len(layer) > 60 else layer
                        logger.debug(f"  {i+1:2d}. {preview}")
                        
            except Exception as e:
                logger.warning(f"Не удалось извлечь последовательность из колонки: {e}")

            # 1.2 Привязываем порядок к легенде по нечеткому совпадению с колонкой
            if column_layers:
                logger.debug("Сопоставляю тексты легенды с колонкой по fuzzy >= 70%")
                matched_count = 0
                normalized_column_layers = [_normalize_text(t) for t in column_layers]
                for i, entry in enumerate(legend_data):
                    raw_text = (entry.get("text") or "").strip()
                    text = _normalize_text(raw_text)
                    if not text:
                        # Помещаем неназванные в конец
                        entry["column_order"] = len(column_layers) + i
                        logger.info(
                            f"[LEGEND {i+1:02d}] пустой текст → NO MATCH | order={entry['column_order']}"
                        )
                        continue
                    best_idx = None
                    best_score = -1.0
                    for idx, lay in enumerate(normalized_column_layers):
                        score = _similarity(text, lay)
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    if best_idx is not None and best_score >= 40.0:
                        entry["column_order"] = int(best_idx)
                        entry["matched_column_text"] = column_layers[best_idx]
                        entry["match_score"] = float(best_score)
                        matched_count += 1
                        col_preview = (column_layers[best_idx] or "").strip()
                        if len(col_preview) > 70:
                            col_preview = col_preview[:67] + "..."
                        legend_preview = raw_text
                        if len(legend_preview) > 70:
                            legend_preview = legend_preview[:67] + "..."
                        logger.info(
                            f"[LEGEND {i+1:02d}] MATCH | legend='{legend_preview}' → column[{best_idx}]='{col_preview}' "
                            f"score={best_score:.1f} | order={entry['column_order']}"
                        )
                    else:
                        # Не нашли уверенного соответствия — отправляем в конец
                        entry["column_order"] = len(column_layers) + i
                        entry["match_score"] = float(best_score)
                        legend_preview = raw_text
                        if len(legend_preview) > 70:
                            legend_preview = legend_preview[:67] + "..."
                        logger.info(
                            f"[LEGEND {i+1:02d}] NO MATCH | legend='{legend_preview}' "
                            f"best_score={best_score:.1f} | order={entry['column_order']}"
                        )
                logger.info(f"Порядок из колонки: использован для {matched_count}/{len(legend_data)}")
            else:
                logger.info("Колонка не получена — используем порядок легенды")

            # Извлекаем цвета для обратной совместимости
            legend_colors = [entry["color"] for entry in legend_data]
            self.legend_colors = legend_colors

            logger.info(f"Извлечено блоков легенды: {len(legend_data)}")

            # 2. Анализируем линию разреза
            logger.debug("Шаг 2: Анализ линии разреза")
            colors_along_line = self.analyze_section_line(
                map_image, start_point, end_point
            )

            # 3. Строим геологические слои, используя порядок из стратиграфической колонки (если удалось сопоставить)
            logger.debug("Шаг 3: Построение слоев")
            layers = self.build_geological_layers_with_legend_data(
                colors_along_line, legend_data
            )

            # 4. Создаем визуализацию с названиями из легенды
            logger.debug("Шаг 4: Визуализация")
            output_path = self.create_section_visualization_with_names(
                layers, legend_data, settings.output_dir
            )

            # 5. Создаем изображение карты с линией разреза
            logger.debug("Шаг 5: Карта с линией")
            map_with_line_path = self.create_map_with_section_line(
                map_image, start_point, end_point, settings.output_dir
            )

            # 6. Создаем отладочную легенду
            logger.debug("Шаг 6: Отладочная легенда")
            debug_legend_path = self.create_debug_legend_visualization(
                legend_data, settings.output_dir
            )

            result = {
                "success": True,
                "layers": layers,
                "output_path": output_path,
                "map_with_line_path": map_with_line_path,
                "debug_legend_path": debug_legend_path,
                "legend_data": legend_data,
                "line_pixels_count": len(colors_along_line),
                "column_layers_count": len(column_layers),
            }

            logger.info("=== УСПЕШНОЕ ЗАВЕРШЕНИЕ ОБРАБОТКИ ===")
            logger.info(f"Результат: слоев={len(layers)}, пикселей={len(colors_along_line)}")

            return result

        except Exception as e:
            logger.error(f"ОШИБКА ПРИ ОБРАБОТКЕ: {str(e)}")
            logger.exception("Детали ошибки:")
            return {"success": False, "error": str(e)}

    def extract_legend_data(self, legend_image: np.ndarray) -> List[Dict]:
        """Извлекает данные из легенды: названия, символы и цвета"""
        logger.info("Извлекаю данные из легенды (названия, символы, цвета)")

        # Используем оригинальное изображение без изменений
        height, width = legend_image.shape[:2]
        logger.info(f"Размер легенды: {width}x{height}")

        # Находим контуры цветных блоков
        legend_data = self._find_color_blocks(legend_image)

        logger.info(f"Извлечено {len(legend_data)} блоков легенды")

        # Выводим краткую статистику всех блоков
        logger.info("=== СТАТИСТИКА ИЗВЛЕЧЕННЫХ БЛОКОВ ===")
        logger.info(f"Всего блоков: {len(legend_data)}")

        # Подсчитываем блоки с текстом
        blocks_with_text = sum(
            1 for entry in legend_data if entry.get("text", "").strip()
        )
        logger.info(f"Блоков с текстом: {blocks_with_text}")
        logger.info(f"Блоков без текста: {len(legend_data) - blocks_with_text}")

        # Показываем диапазон координат
        if legend_data:
            y_coords = [entry["y"] for entry in legend_data]
            logger.info(f"Диапазон Y-координат: {min(y_coords)} - {max(y_coords)}")

        return legend_data

    def _find_color_blocks(self, legend_image: np.ndarray) -> List[Dict]:
        """Находит контуры цветных блоков в легенде"""
        height, width = legend_image.shape[:2]

        # Конвертируем в градации серого для поиска контуров
        gray = cv2.cvtColor(legend_image, cv2.COLOR_BGR2GRAY)

        # Бинаризация с инверсией для выделения не-белых областей
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Находим контуры
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        logger.info(f"Найдено {len(contours)} контуров")

        blocks = []
        filtered_count = 0

        for i, contour in enumerate(contours):
            # Получаем прямоугольную область контура
            x, y, w, h = cv2.boundingRect(contour)

            # Ищем только основные цветные блоки легенды
            # Основные блоки обычно имеют размеры ~190x100 пикселей
            # и находятся в левой части изображения

            # Фильтруем слишком маленькие области
            if w < 50 or h < 50:
                filtered_count += 1
                continue

            # Фильтруем слишком большие области
            if w > width * 0.5 or h > height * 0.2:
                filtered_count += 1
                continue

            # Основные блоки должны быть в левой части (первые 30% ширины)
            if x > width * 0.3:
                filtered_count += 1
                continue

            # Блоки должны иметь примерно прямоугольную форму
            aspect_ratio = w / h
            if aspect_ratio < 1.0 or aspect_ratio > 3.0:
                filtered_count += 1
                continue

            # Извлекаем цвет из блока
            block_color = self._extract_color_from_block(legend_image, x, y, w, h)

            if block_color:
                # Извлекаем текст для этого блока
                block_text = self._extract_text_for_block(
                    legend_image, x, y, w, h, width
                )

                block_entry = {
                    "color": block_color,
                    "text": block_text,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "y_position": y,
                }

                blocks.append(block_entry)

        logger.info(
            f"Отфильтровано контуров: {filtered_count}, найдено основных блоков: {len(blocks)}"
        )

        # Сортируем блоки по Y-координате (сверху вниз)
        blocks.sort(key=lambda x: x["y"])

        return blocks

    def _extract_color_from_block(
        self, legend_image: np.ndarray, x: int, y: int, w: int, h: int
    ) -> Tuple[int, int, int]:
        """Извлекает цвет из блока по одному пикселю"""
        # Выбираем точку в левом краю блока (избегаем центр со символами)
        pixel_x = x + w // 4  # Четверть от ширины от левого края
        pixel_y = y + h // 2  # Центр по высоте

        # Проверяем, что точка внутри изображения
        if pixel_x >= legend_image.shape[1] or pixel_y >= legend_image.shape[0]:
            pixel_x = x + 10  # Отступ от левого края
            pixel_y = y + h // 2

        # Получаем цвет напрямую из изображения без изменений
        pixel_color = legend_image[pixel_y, pixel_x]
        b, g, r = int(pixel_color[0]), int(pixel_color[1]), int(pixel_color[2])

        # Возвращаем как есть (BGR формат)
        color = (b, g, r)

        # Отладочная информация
        logger.debug(
            f"Извлечен цвет из левого края блока ({pixel_x}, {pixel_y}): BGR({b}, {g}, {r})"
        )

        return color

    def _extract_text_for_block(
        self, legend_image: np.ndarray, x: int, y: int, w: int, h: int, total_width: int
    ) -> str:
        """Извлекает текст для блока"""
        # Вырезаем область справа от блока
        text_start_x = x + w + 10  # 10 пикселей отступ от блока
        text_end_x = total_width

        # Проверяем границы
        if text_start_x >= total_width:
            return ""

        # Вырезаем текстовую область
        text_region = legend_image[y : y + h, text_start_x:text_end_x]

        # Конвертируем в оттенки серого
        text_gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

        # Применяем бинаризацию OTSU
        _, text_binary = cv2.threshold(
            text_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        try:
            import pytesseract

            # OCR с Tesseract
            text = pytesseract.image_to_string(
                text_binary, lang="rus", config="--psm 6"
            )
            text = text.strip()

            if text and len(text) > 2:
                return text

        except ImportError:
            logger.error("Tesseract не установлен")
        except Exception as e:
            logger.error(f"Tesseract ошибка: {str(e)}")

        return ""

    def _extract_block_symbol(
        self, legend_rgb: np.ndarray, y_start: int, height: int
    ) -> str:
        """Извлекает символ блока легенды (геологический символ) - УСТАРЕЛО"""
        return ""

    def build_geological_layers_with_legend_data(
        self,
        colors_along_line: List[Tuple[int, int, int]],
        legend_data: List[Dict],
    ) -> List[Dict]:
        """Строит геологические слои с использованием данных легенды по тексту"""
        logger.info(f"Строю геологические слои из {len(colors_along_line)} цветов")
        logger.info(f"Данные легенды: {len(legend_data)} блоков")

        # Создаем словарь легенды: текст -> {color, index, order}
        legend_text_dict = {}
        for i, entry in enumerate(legend_data):
            raw_text = entry.get("text", "").strip()
            # Очищаем текст от переносов строк для ключей словаря
            clean_text = ' '.join(raw_text.replace('\n', ' ').split())
            if clean_text:  # Только записи с текстом
                legend_text_dict[clean_text] = {
                    "color": entry["color"],  # BGR формат
                    "index": i,
                    # Если был рассчитан порядок по колонке, используем его
                    "order": entry.get("column_order", i),
                    "text": clean_text,  # Используем очищенный текст
                }

        logger.info(
            f"Создан словарь легенды по тексту с {len(legend_text_dict)} записями"
        )
        logger.info(f"Тексты легенды: {list(legend_text_dict.keys())}")

        # Анализируем уникальные цвета вдоль линии
        unique_line_colors = list(set(colors_along_line))
        logger.info(f"Уникальных цветов вдоль линии: {len(unique_line_colors)}")

        # Показываем статистику цветов
        color_counts = {}
        for color in colors_along_line:
            color_counts[color] = color_counts.get(color, 0) + 1

        # Сортируем по частоте
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info("Топ-10 самых частых цветов на линии:")
        for i, (color, count) in enumerate(sorted_colors[:10]):
            logger.info(f"  {i + 1}. BGR{color}: {count} раз")

        # Анализируем последовательность цветов вдоль линии
        layer_sequence = []
        current_layer_text = None
        current_length = 0
        matched_colors = 0
        unmatched_colors = 0

        for i, color in enumerate(colors_along_line):
            # Ищем ближайший цвет в легенде для определения текста слоя
            closest_entry = self._find_closest_legend_entry(color, legend_data)

            if closest_entry:
                matched_colors += 1
                raw_layer_text = closest_entry.get("text", "").strip()
                layer_text = ' '.join(raw_layer_text.replace('\n', ' ').split())  # Очищаем от переносов строк

                # Логируем сопоставление для отладки
                if i % 100 == 0:  # Логируем каждый 100-й пиксель
                    logger.debug(
                        f"Пиксель {i}: цвет_карты_BGR{color} -> текст: {layer_text[:30]}..."
                    )

                if current_layer_text is None or current_layer_text != layer_text:
                    # Начинаем новый слой
                    if current_layer_text is not None and current_length > 0:
                        # Берем данные слоя из легенды по тексту
                        legend_info = legend_text_dict.get(current_layer_text)
                        if legend_info:
                            layer_sequence.append(
                                {
                                    "index": legend_info["index"],
                                    "color": legend_info[
                                        "color"
                                    ],  # Правильный цвет из легенды
                                    "original_color": colors_along_line[
                                        i - current_length
                                    ],  # Оригинальный цвет с карты (BGR)
                                    "order": legend_info["order"],
                                    "length": current_length,
                                    "start_pos": i - current_length,
                                    "text": current_layer_text,  # Текст из легенды
                                }
                            )

                    current_layer_text = layer_text
                    current_length = 1
                else:
                    # Продолжаем текущий слой
                    current_length += 1
            else:
                unmatched_colors += 1
                # Пропускаем цвета, не найденные в легенде
                if current_layer_text is not None and current_length > 0:
                    # Берем данные слоя из легенды по тексту
                    legend_info = legend_text_dict.get(current_layer_text)
                    if legend_info:
                        layer_sequence.append(
                            {
                                "index": legend_info["index"],
                                "color": legend_info[
                                    "color"
                                ],  # Правильный цвет из легенды
                                "original_color": colors_along_line[
                                    i - current_length
                                ],  # Оригинальный цвет с карты (BGR)
                                "order": legend_info["order"],
                                "length": current_length,
                                "start_pos": i - current_length,
                                "text": current_layer_text,  # Текст из легенды
                            }
                        )
                    current_layer_text = None
                    current_length = 0

        # Добавляем последний слой
        if current_layer_text is not None and current_length > 0:
            legend_info = legend_text_dict.get(current_layer_text)
            if legend_info:
                layer_sequence.append(
                    {
                        "index": legend_info["index"],
                        "color": legend_info["color"],  # Правильный цвет из легенды
                        "original_color": colors_along_line[
                            len(colors_along_line) - current_length
                        ],  # Оригинальный цвет с карты (BGR)
                        "order": legend_info["order"],
                        "length": current_length,
                        "start_pos": len(colors_along_line) - current_length,
                        "text": current_layer_text,  # Текст из легенды
                    }
                )

        # Логируем статистику сопоставления
        logger.info(
            f"Статистика сопоставления: {matched_colors} сопоставлено, {unmatched_colors} не найдено"
        )
        logger.info(f"Найдено {len(layer_sequence)} слоев до фильтрации")

        # Если не нашли слои, пробуем более простой подход
        if not layer_sequence:
            logger.warning("Не найдено слоев, пробую альтернативный подход")
            unique_colors = list(set(colors_along_line))
            for color in unique_colors:
                closest_entry = self._find_closest_legend_entry(color, legend_data)
                if closest_entry:
                    layer_text = closest_entry.get("text", "").strip()
                    legend_info = legend_text_dict.get(layer_text)
                    if legend_info:
                        layer_sequence.append(
                            {
                                "index": legend_info["index"],
                                "color": legend_info[
                                    "color"
                                ],  # Правильный цвет из легенды
                                "original_color": color,  # Оригинальный цвет с карты (BGR)
                                "order": legend_info["order"],
                                "length": colors_along_line.count(color),
                                "start_pos": 0,
                                "text": layer_text,  # Текст из легенды
                            }
                        )

        # Фильтруем слишком короткие слои (уменьшаем минимальную длину)
        min_layer_length = max(1, len(colors_along_line) // 350)  # Было 100, стало 200
        filtered_layers = [
            layer for layer in layer_sequence if layer["length"] >= min_layer_length
        ]

        logger.info(
            f"После фильтрации по длине: {len(filtered_layers)} слоев (мин. длина: {min_layer_length})"
        )

        # Удаляем дубликаты по тексту, но оставляем больше слоев
        unique_layers = []
        seen_texts = set()

        for layer in filtered_layers:
            layer_text = layer.get("text", "")
            if layer_text and layer_text not in seen_texts:
                # Проверяем, что текст слоя действительно есть в легенде
                if layer_text in legend_text_dict:
                    unique_layers.append(layer)
                    seen_texts.add(layer_text)
                else:
                    logger.warning(f"Обнаружен некорректный текст слоя: {layer_text}")

        # НОВАЯ ЛОГИКА: Разделяем слои на найденные в колонке и не найденные
        column_layers = []  # Слои из стратиграфической колонки (order < 20)
        legend_only_layers = []  # Слои только из легенды, но присутствующие на карте (order >= 20)
        
        for layer in unique_layers:
            if layer["order"] < 20:  # Найден в колонке
                column_layers.append(layer)
            else:  # Не найден в колонке, но присутствует на карте
                legend_only_layers.append(layer)
        
        # Сортируем слои из колонки по порядку (сверху вниз)
        column_layers.sort(key=lambda x: x["order"], reverse=False)
        
        # Сортируем слои только из легенды по индексу в легенде
        legend_only_layers.sort(key=lambda x: x["index"], reverse=False)
        
        # Объединяем: сначала слои из колонки, потом слои только из легенды (в самый низ)
        unique_layers = column_layers + legend_only_layers
        
        logger.info(f"Разделение слоев: колонка={len(column_layers)}, только_легенда={len(legend_only_layers)}")
        if column_layers:
            logger.info("Слои из стратиграфической колонки (порядок сверху-вниз):")
            for layer in column_layers:
                logger.info(f"  - {layer['order']:2d}: {layer['text'][:60]}...")
        if legend_only_layers:
            logger.info("Слои только из легенды (добавлены в низ разреза):")
            for layer in legend_only_layers:
                logger.info(f"  - {layer['index']:2d}: {layer['text'][:60]}...")

        logger.info(f"Построено {len(unique_layers)} геологических слоев:")
        for layer in unique_layers:
            text = layer.get("text", "")
            layer_index = layer["index"]
            layer_color = layer["color"]
            original_color = layer.get("original_color", layer_color)

            # Проверяем соответствие цвета и текста
            if text in legend_text_dict:
                legend_info = legend_text_dict[text]
                expected_color = legend_info["color"]
                expected_text = legend_info["text"]
                color_match = layer_color == expected_color
                text_match = text == expected_text

                logger.info(
                    f"  - Слой {layer['order']} (индекс {layer_index}): "
                    f"цвет_легенды={layer_color}, цвет_карты={original_color}, "
                    f"текст='{text[:50]}...', "
                    f"длина={layer['length']}, "
                    f"цвет_совпадает={color_match}, текст_совпадает={text_match}"
                )

                if not color_match:
                    logger.warning(
                        f"    НЕСООТВЕТСТВИЕ ЦВЕТА: ожидался {expected_color}, получен {layer_color}"
                    )
                if not text_match:
                    logger.warning(
                        f"    НЕСООТВЕТСТВИЕ ТЕКСТА: ожидался '{expected_text[:50]}...', получен '{text[:50]}...'"
                    )
            else:
                logger.error(
                    f"  - Слой {layer['order']}: текст {text} не найден в словаре легенды!"
                )

        return unique_layers

    def _find_closest_legend_entry(
        self, color: Tuple[int, int, int], legend_data: List[Dict]
    ) -> Dict:
        """Находит ближайшую запись в легенде для данного цвета с более строгим сопоставлением"""
        if not legend_data:
            return None

        min_distance = float("inf")
        closest_entry = None
        best_distance_type = "none"

        # Сначала ищем точное совпадение
        for entry in legend_data:
            legend_color = entry["color"]
            if color == legend_color:
                logger.debug(f"Найдено точное совпадение: {color} с {legend_color}")
                return entry

        for entry in legend_data:
            legend_color = entry["color"]

            # Сначала пробуем LAB расстояние (более точное для восприятия)
            lab_distance = self.lab_distance(color, legend_color)
            adaptive_tolerance = self.adaptive_color_tolerance(color)

            # Увеличиваем допуск для лучшего сопоставления
            strict_tolerance = adaptive_tolerance * 1.2  # Было 0.7

            if lab_distance < min_distance and lab_distance < strict_tolerance:
                min_distance = lab_distance
                closest_entry = entry
                best_distance_type = "LAB"

        # Если не нашли в LAB, пробуем RGB с более строгим допуском
        if closest_entry is None:
            for entry in legend_data:
                legend_color = entry["color"]
                # Используем явное вычисление для избежания переполнения
                rgb_distance = (
                    (color[0] - legend_color[0]) ** 2
                    + (color[1] - legend_color[1]) ** 2
                    + (color[2] - legend_color[2]) ** 2
                ) ** 0.5

                # Умеренный допуск для RGB (было 100, увеличено до 180)
                if rgb_distance < min_distance and rgb_distance < 180:
                    min_distance = rgb_distance
                    closest_entry = entry
                    best_distance_type = "RGB"

        # Если все еще не нашли, пробуем очень строгий подход
        if closest_entry is None:
            for entry in legend_data:
                legend_color = entry["color"]
                # Простое манхэттенское расстояние
                manhattan_distance = (
                    abs(color[0] - legend_color[0])
                    + abs(color[1] - legend_color[1])
                    + abs(color[2] - legend_color[2])
                )

                # Умеренный допуск для манхэттенского расстояния (было 150, увеличено до 250)
                if manhattan_distance < min_distance and manhattan_distance < 250:
                    min_distance = manhattan_distance
                    closest_entry = entry
                    best_distance_type = "Manhattan"

        # Логируем информацию о сопоставлении для отладки
        if closest_entry:
            logger.debug(
                f"Сопоставлен цвет {color} с {closest_entry['color']} "
                f"(расстояние: {min_distance:.2f}, тип: {best_distance_type})"
            )
        else:
            logger.debug(
                f"Не найден подходящий цвет в легенде для {color} "
                f"(минимальное расстояние: {min_distance:.2f})"
            )

        # Добавляем дополнительное логирование для отладки
        if closest_entry:
            closest_index = legend_data.index(closest_entry)
            logger.debug(
                f"  -> Индекс в легенде: {closest_index}, "
                f"текст: '{closest_entry.get('text', '')[:30]}...'"
            )

        return closest_entry

    def _generate_geological_names_from_symbols(
        self, legend_data: List[Dict]
    ) -> List[Dict]:
        """Обрабатывает названия из извлеченного текста"""
        logger.info("Обрабатываю названия из извлеченного текста")

        # Обновляем данные легенды, используя только извлеченный текст
        for i, entry in enumerate(legend_data):
            text = entry.get("text", "").strip()
            if text:
                # Используем извлеченный текст как название
                entry["text"] = text
                logger.info(f"Блок {i + 1}: '{text[:50]}...'")
            else:
                # Если текст не извлечен, оставляем пустую строку
                entry["text"] = ""
                logger.info(f"Блок {i + 1}: текст не извлечен")

        return legend_data

    def create_legend_test_visualization(
        self, legend_data: List[Dict], output_path: str
    ) -> str:
        """Создает тестовое изображение легенды со всеми слоями"""
        logger.info(
            f"Создаю тестовое изображение легенды для {len(legend_data)} блоков"
        )

        # Выводим все слои легенды в консоль
        print("=== СЛОИ ЛЕГЕНДЫ ===")
        for i, entry in enumerate(legend_data):
            text = entry.get("text", "")
            color = entry.get("color", [])
            print(f"Блок {i + 1}:")
            # print(f"  - Цвет: BGR{color}")
            print(f"  - Текст: '{text[:50]}...'")
            # print(f"  - Координаты: ({entry.get('x', 'N/A')}, {entry.get('y', 'N/A')})")

        if not legend_data:
            logger.warning("Нет данных легенды для визуализации")
            # Создаем пустое изображение
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.text(
                0.5,
                0.5,
                "Не найдено блоков легенды для отображения",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Тест легенды", fontsize=16, fontweight="bold")

            # Сохраняем изображение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"legend_test_{timestamp}.png"
            filepath = os.path.join(settings.output_dir, filename)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Пустое тестовое изображение сохранено: {filepath}")
            return filepath

        # Создаем фигуру для отображения легенды
        fig, ax = plt.subplots(figsize=(20, max(8, len(legend_data) * 0.8)))

        # Рисуем блоки легенды
        layer_height = 1.0
        layer_width = 18.0
        current_y = len(legend_data)  # Начинаем сверху

        logger.info("Отображаемые блоки легенды (сверху вниз):")

        # Собираем информацию о всех блоках в один список
        blocks_info = []
        for i, entry in enumerate(legend_data):
            text = entry.get("text", "")
            # Обрезаем текст до 10 символов
            short_text = text[:10] + "..." if len(text) > 10 else text
            color = entry["color"]
            y_pos = current_y - layer_height
            coords = (entry.get("x", "N/A"), entry.get("y", "N/A"))

            blocks_info.append(
                f"{i + 1}. '{short_text}': BGR{color}, y={y_pos:.1f}, coords{coords}"
            )

            # Отладочная информация о цвете
            b, g, r = color
            logger.debug(f"Блок {i + 1}: BGR({b}, {g}, {r}) -> RGB({r}, {g}, {b})")

            # Рисуем блок (конвертируем BGR в RGB для matplotlib)
            color_normalized = (r / 255.0, g / 255.0, b / 255.0)

            rect = plt.Rectangle(
                (0, current_y - layer_height),
                layer_width,
                layer_height,
                facecolor=color_normalized,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Добавляем подпись блока
            # Формируем текст для отображения
            if text:
                layer_text = text
            else:
                layer_text = f"Блок {i + 1} (без названия)"

            # Обрезаем длинный текст
            if len(layer_text) > 80:
                layer_text = layer_text[:77] + "..."

            # Добавляем информацию о цвете и координатах
            color_info = f"BGR: {entry['color']}"
            coord_info = f"Коорд: ({entry.get('x', 'N/A')}, {entry.get('y', 'N/A')})"

            # Размещаем текст в три строки
            ax.text(
                layer_width / 2,
                current_y - layer_height / 2 + 0.4,
                layer_text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

            ax.text(
                layer_width / 2,
                current_y - layer_height / 2,
                color_info,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                style="italic",
            )

            ax.text(
                layer_width / 2,
                current_y - layer_height / 2 - 0.4,
                coord_info,
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                style="italic",
            )

            current_y -= layer_height

        # Выводим один общий лог со всеми блоками
        logger.info("Блоки: " + " | ".join(blocks_info))

        ax.set_xlim(0, layer_width)
        ax.set_ylim(0, len(legend_data))
        ax.set_aspect("auto")
        ax.set_title(
            "Тест новой логики извлечения данных из легенды (контуры блоков)",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("Порядок блоков (сверху вниз)")

        # Добавляем пояснение
        ax.text(
            layer_width / 2,
            len(legend_data) + 0.5,
            f"Извлечено {len(legend_data)} блоков легенды",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
        )

        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"legend_test_{timestamp}.png"
        filepath = os.path.join(settings.output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Тестовое изображение легенды сохранено: {filepath}")
        return filepath

    def create_map_with_section_line(
        self,
        map_image: np.ndarray,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        output_path: str,
    ) -> str:
        """Создает изображение карты с отмеченными точками и линией разреза"""
        logger.info("Создаю изображение карты с линией разреза")

        # Конвертируем в RGB для matplotlib
        if len(map_image.shape) == 3 and map_image.shape[2] == 3:
            map_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        else:
            map_rgb = map_image

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(20, 16))

        # Отображаем карту
        ax.imshow(map_rgb)
        ax.set_title(
            "Геологическая карта с линией разреза", fontsize=16, fontweight="bold"
        )

        # Рисуем линию разреза
        x1, y1 = start_point
        x2, y2 = end_point

        # Рисуем линию красным цветом с толщиной
        ax.plot([x1, x2], [y1, y2], "r-", linewidth=3, alpha=0.8, label="Линия разреза")

        # Рисуем точки начала и конца
        ax.plot(
            x1,
            y1,
            "go",
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"Начало ({x1}, {y1})",
        )
        ax.plot(
            x2,
            y2,
            "ro",
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"Конец ({x2}, {y2})",
        )

        # Добавляем легенду
        ax.legend(loc="upper right", fontsize=12)

        # Убираем оси
        ax.set_xticks([])
        ax.set_yticks([])

        # Добавляем информацию о размере карты
        height, width = map_rgb.shape[:2]
        ax.text(
            0.02,
            0.98,
            f"Размер карты: {width} × {height}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"map_with_section_line_{timestamp}.png"
        filepath = os.path.join(settings.output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Изображение карты с линией разреза сохранено: {filepath}")
        return filepath

    def process_enhanced_geological_section(
        self,
        map_image: np.ndarray,
        legend_image: np.ndarray,
        column_image_path: str,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> Dict:
        """
        Улучшенный метод обработки геологического разреза с использованием трех изображений.
        Возвращает только путь к разрезу без создания отладочных файлов.
        """
        logger.info("=== НАЧАЛО УЛУЧШЕННОЙ ОБРАБОТКИ ГЕОЛОГИЧЕСКОГО РАЗРЕЗА ===")
        logger.info("Извлекаю данные из легенды (названия, символы, цвета)")

        try:
            # 1. Извлекаем данные из легенды
            legend_data = self.extract_legend_data(legend_image)
            
            if not legend_data:
                logger.warning("Не удалось извлечь данные из легенды")
                raise Exception("Не удалось обработать легенду")

            logger.info(f"Извлечено {len(legend_data)} блоков легенды")

            # 2. Извлекаем данные из стратиграфической колонки с улучшенной очисткой
            logger.info("Обрабатываю стратиграфическую колонку с улучшенной очисткой")
            sc_processor = StrategraphicColumnProcessor(use_enhanced_cleaning=True)
            column_layers = sc_processor.process_strategraphic_column(column_image_path)
            logger.info(f"Извлечено {len(column_layers)} слоев из колонки с улучшенной очисткой")

            # 3. Сопоставляем легенду с колонкой
            logger.info("Сопоставляю данные легенды с колонкой")
            for i, entry in enumerate(legend_data):
                legend_text = _normalize_text(entry.get("text", ""))
                
                best_score = 0
                best_order = 999  # По умолчанию в конец
                
                for j, column_text in enumerate(column_layers):
                    norm_column = _normalize_text(column_text)
                    score = _similarity(legend_text, norm_column)
                    
                    if score >= 70 and score > best_score:  # Порог для считания совпадением
                        best_score = score
                        best_order = j
                
                if best_score >= 70:
                    logger.info(f"[LEGEND {i+1:02d}] MATCH | legend='{legend_text[:60]}...' → column[{best_order}]='{column_layers[best_order][:60]}...' score={best_score:.1f} | order={best_order}")
                    entry["order"] = best_order
                else:
                    logger.info(f"[LEGEND {i+1:02d}] NO MATCH | legend='{legend_text[:60]}...' best_score={best_score:.1f} | order={i+20}")
                    entry["order"] = i + 20  # Неопределенные в конец
            
            logger.info(f"Порядок из колонки: использован для {sum(1 for e in legend_data if e.get('order', 999) < 20)}/{len(legend_data)}")

            # 4. Анализируем цвета вдоль линии разреза  
            logger.info(f"Анализирую цвета вдоль линии разреза. Размер карты: {map_image.shape}")
            
            colors_along_line = self.analyze_section_line(
                map_image, start_point, end_point
            )
            logger.info(f"Проанализировано {len(colors_along_line)} пикселей вдоль линии")

            # 5. Кластеризуем цвета для упрощения
            unique_colors = list(set(colors_along_line))
            logger.info(f"Уникальных цветов: {len(unique_colors)}")
            
            if len(unique_colors) > 50:
                # Кластеризуем только если много уникальных цветов
                clustered_colors = self.cluster_colors(unique_colors, n_clusters=min(50, len(unique_colors)))
                logger.info(f"Кластеризация: {len(unique_colors)} -> {len(clustered_colors)} цветов")
                
                # Заменяем цвета на кластерные
                color_mapping = {}
                for original_color in unique_colors:
                    closest_cluster = min(
                        clustered_colors,
                        key=lambda c: sum((a - b) ** 2 for a, b in zip(original_color, c))
                    )
                    color_mapping[original_color] = closest_cluster
                
                colors_along_line = [color_mapping[color] for color in colors_along_line]
                unique_colors = clustered_colors
            
            logger.info(f"После кластеризации: {len(unique_colors)} цветов")

            # 6. Строим геологические слои
            logger.info(f"Строю геологические слои из {len(colors_along_line)} цветов")
            layers = self.build_geological_layers_with_legend_data(
                colors_along_line, legend_data
            )
            
            if not layers:
                logger.warning("Не удалось построить геологические слои")
                raise Exception("Не удалось найти геологические слои")

            logger.info(f"Построено {len(layers)} геологических слоев")

            # 7. Создаем ТОЛЬКО визуализацию разреза (без отладочных файлов)
            logger.info("Создаю улучшенную реалистичную визуализацию геологического разреза")
            section_path = self.create_section_visualization_with_names(
                layers, legend_data, settings.output_dir
            )

            logger.info("=== УСПЕШНОЕ ЗАВЕРШЕНИЕ ОБРАБОТКИ ===")
            logger.info(f"Результат: слоев={len(layers)}, пикселей={len(colors_along_line)}")

            return {
                "section_path": section_path,
                "layers": layers,
                "legend_data": legend_data,
                "colors_count": len(colors_along_line),
                "unique_colors": len(unique_colors)
            }

        except Exception as e:
            logger.error(f"Ошибка в процессе обработки: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
