import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from app.core.config import settings
from sklearn.metrics import pairwise_distances
from collections import Counter

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
        """Адаптивный допуск на основе яркости цвета - более строгий подход"""
        # Более строгие допуски для лучшего сопоставления
        brightness = (color[0] + color[1] + color[2]) / 3.0
        if brightness < 100:
            return 40  # Темные цвета - очень строгий допуск (было 80)
        elif brightness < 200:
            return 60  # Средние цвета - строгий допуск (было 120)
        else:
            return 80  # Светлые цвета - умеренный допуск (было 180)

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
        logger.info(f"Уникальных цветов вдоль линии: {len(unique_colors)}")

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

        # Генерируем неровные границы для каждого слоя
        x_points = np.linspace(
            0, section_width, 100
        )  # Увеличил с 50 до 100 точек для плавности

        # Начинаем с верхней границы
        current_top = section_height + base_y

        # Получаем названия слоев
        legend_names = self.extract_legend_names()

        logger.info("Отображаемые слои в визуализации (сверху вниз):")
        for i, layer in enumerate(layers):
            # Правильная нормализация цветов для matplotlib (RGB в диапазоне [0,1])
            r, g, b = layer["color"]
            color = (r / 255.0, g / 255.0, b / 255.0)

            # Вычисляем толщину слоя на основе его длины
            layer_length = layer.get("length", 100)
            max_thickness = 2.0  # Максимальная толщина слоя
            min_thickness = 0.5  # Минимальная толщина слоя

            # Нормализуем толщину на основе длины слоя
            total_pixels = sum(l.get("length", 100) for l in layers)
            normalized_length = layer_length / total_pixels if total_pixels > 0 else 1.0
            layer_thickness = (
                min_thickness + (max_thickness - min_thickness) * normalized_length
            )

            # Генерируем неровную нижнюю границу слоя
            # Используем синусоиду с случайными вариациями для реалистичности
            np.random.seed(42 + i)  # Фиксированный seed для воспроизводимости

            # Базовая синусоида с более плавными волнами
            base_wave = (
                np.sin(x_points * 0.3 + i * 0.2) * 0.15
            )  # Уменьшил частоту и увеличил амплитуду

            # Добавляем случайные вариации с более плавным сглаживанием
            random_variations = np.random.normal(
                0, 0.08, len(x_points)
            )  # Увеличил стандартное отклонение

            # Сглаживаем вариации более интенсивно для плавности
            try:
                from scipy.ndimage import gaussian_filter1d

                smoothed_variations = gaussian_filter1d(
                    random_variations, sigma=4
                )  # Увеличил sigma для более плавного сглаживания
            except ImportError:
                # Если scipy нет, используем простое сглаживание
                smoothed_variations = random_variations

            # Добавляем дополнительные плавные волны для более естественного вида
            additional_waves = (
                np.sin(x_points * 0.1 + i * 0.5) * 0.05  # Длинные волны
                + np.sin(x_points * 0.7 + i * 0.1) * 0.03  # Короткие волны
            )

            # Комбинируем все вариации для создания плавной границы
            boundary_variations = base_wave + smoothed_variations + additional_waves

            # Вычисляем нижнюю границу слоя
            layer_bottom = current_top - layer_thickness + boundary_variations

            # Убеждаемся, что все массивы имеют одинаковую длину
            assert len(x_points) == len(layer_bottom), (
                f"Разные длины: x_points={len(x_points)}, layer_bottom={len(layer_bottom)}"
            )

            # Рисуем слой как полигон с неровными границами
            # Создаем точки для полигона: верхняя граница + нижняя граница в обратном порядке
            polygon_x = list(x_points) + list(x_points[::-1])
            polygon_y = [current_top] * len(x_points) + list(layer_bottom[::-1])

            # Создаем полигон слоя с более мягкими краями
            layer_polygon = plt.Polygon(
                list(zip(polygon_x, polygon_y)),
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,  # Уменьшил толщину линии для более мягкого вида
                alpha=0.95,  # Увеличил прозрачность для более естественного вида
            )
            ax.add_patch(layer_polygon)

            # Добавляем легкую тень для объема
            shadow_offset = 0.02
            shadow_polygon = plt.Polygon(
                list(
                    zip(
                        [x + shadow_offset for x in polygon_x],
                        [y + shadow_offset for y in polygon_y],
                    )
                ),
                facecolor="black",
                alpha=0.1,
                linewidth=0,
            )
            ax.add_patch(shadow_polygon)

            # Добавляем текстуру слоя (точки для имитации породы)
            # Временно отключено из-за проблем с размерами массивов
            pass

            # Добавляем подпись слоя с названием из легенды
            layer_index = layer["order"]
            layer_text = layer.get("text", "")

            # Если нет текста, не показываем подпись
            if not layer_text:
                layer_text = f"Слой {layer_index + 1} (без названия)"

            # Добавляем информацию о длине слоя
            if "length" in layer:
                layer_text += f" ({layer['length']}px)"

            # Размещаем текст в центре слоя
            center_x = section_width / 2
            center_y = current_top - layer_thickness / 2  # Это скалярное значение

            # Добавляем текст с фоном для лучшей читаемости
            ax.text(
                center_x,
                center_y,
                layer_text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
            )

            logger.info(
                f"  - {layer_text}: цвет {layer['color']}, толщина={layer_thickness:.2f}, позиция y={center_y:.2f}"
            )

            # Переходим к следующему слою
            current_top = float(
                np.mean(layer_bottom)
            )  # Используем среднее значение для плавности

        # Настраиваем оси
        ax.set_xlim(-0.5, section_width + 8.0)  # Расширяем справа для легенды
        ax.set_ylim(
            base_y - 0.5, section_height + base_y + 0.5
        )  # Возвращаем нормальную высоту
        ax.set_aspect("equal")
        ax.set_title("Геологический разрез", fontsize=16, fontweight="bold")
        # Убираем подписи осей
        # ax.set_xlabel("Расстояние (м)", fontsize=12)
        # ax.set_ylabel("Глубина (м)", fontsize=12)

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
        legend_x_start = section_width + 1.0  # Позиция начала легенды справа
        legend_y_start = section_height + base_y - 1.0  # Начинаем сверху
        legend_item_height = 0.6
        legend_item_width = 0.8
        legend_text_x_offset = 1.2

        # Сортируем слои по порядку (сверху вниз)
        sorted_layers = sorted(layers, key=lambda x: x["order"])

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

            # Текст слоя
            layer_text = layer.get("text", f"Слой {layer['order'] + 1}")
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

    def create_section_visualization_with_names(
        self, layers: List[Dict], legend_data: List[Dict], output_path: str
    ) -> str:
        """Создает реалистичную визуализацию геологического разреза с неровными границами"""
        logger.info(
            f"Создаю реалистичную визуализацию для {len(layers)} слоев с названиями из легенды"
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
        fig, ax = plt.subplots(figsize=(20, 12))

        # Параметры разреза
        section_width = 16.0
        section_height = 10.0
        base_y = 0.5  # Отступ снизу для подписей

        # Генерируем неровные границы для каждого слоя
        x_points = np.linspace(
            0, section_width, 100
        )  # Увеличил с 50 до 100 точек для плавности

        logger.info("Отображаемые слои в визуализации (сверху вниз):")

        # Начинаем с верхней границы
        current_top = section_height + base_y

        for i, layer in enumerate(layers):
            # ИСПОЛЬЗУЕМ ТОЛЬКО ЦВЕТ ИЗ ЛЕГЕНДЫ для визуализации (BGR -> RGB для matplotlib)
            legend_color = layer["color"]  # Цвет из легенды
            b, g, r = legend_color  # BGR формат
            color = (
                r / 255.0,
                g / 255.0,
                b / 255.0,
            )  # Конвертируем в RGB для matplotlib

            # Вычисляем толщину слоя на основе его длины
            layer_length = layer.get("length", 100)
            max_thickness = 2.0  # Максимальная толщина слоя
            min_thickness = 0.5  # Минимальная толщина слоя

            # Нормализуем толщину на основе длины слоя
            total_pixels = sum(l.get("length", 100) for l in layers)
            normalized_length = layer_length / total_pixels if total_pixels > 0 else 1.0
            layer_thickness = (
                min_thickness + (max_thickness - min_thickness) * normalized_length
            )

            # Генерируем неровную нижнюю границу слоя
            # Используем синусоиду с случайными вариациями для реалистичности
            np.random.seed(42 + i)  # Фиксированный seed для воспроизводимости

            # Базовая синусоида с более плавными волнами
            base_wave = (
                np.sin(x_points * 0.3 + i * 0.2) * 0.15
            )  # Уменьшил частоту и увеличил амплитуду

            # Добавляем случайные вариации с более плавным сглаживанием
            random_variations = np.random.normal(
                0, 0.08, len(x_points)
            )  # Увеличил стандартное отклонение

            # Сглаживаем вариации более интенсивно для плавности
            try:
                from scipy.ndimage import gaussian_filter1d

                smoothed_variations = gaussian_filter1d(
                    random_variations, sigma=4
                )  # Увеличил sigma для более плавного сглаживания
            except ImportError:
                # Если scipy нет, используем простое сглаживание
                smoothed_variations = random_variations

            # Добавляем дополнительные плавные волны для более естественного вида
            additional_waves = (
                np.sin(x_points * 0.1 + i * 0.5) * 0.05  # Длинные волны
                + np.sin(x_points * 0.7 + i * 0.1) * 0.03  # Короткие волны
            )

            # Комбинируем все вариации для создания плавной границы
            boundary_variations = base_wave + smoothed_variations + additional_waves

            # Вычисляем нижнюю границу слоя
            layer_bottom = current_top - layer_thickness + boundary_variations

            # Убеждаемся, что все массивы имеют одинаковую длину
            assert len(x_points) == len(layer_bottom), (
                f"Разные длины: x_points={len(x_points)}, layer_bottom={len(layer_bottom)}"
            )

            # Рисуем слой как полигон с неровными границами
            # Создаем точки для полигона: верхняя граница + нижняя граница в обратном порядке
            polygon_x = list(x_points) + list(x_points[::-1])
            polygon_y = [current_top] * len(x_points) + list(layer_bottom[::-1])

            # Создаем полигон слоя с более мягкими краями
            layer_polygon = plt.Polygon(
                list(zip(polygon_x, polygon_y)),
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,  # Уменьшил толщину линии для более мягкого вида
                alpha=0.95,  # Увеличил прозрачность для более естественного вида
            )
            ax.add_patch(layer_polygon)

            # Добавляем легкую тень для объема
            shadow_offset = 0.02
            shadow_polygon = plt.Polygon(
                list(
                    zip(
                        [x + shadow_offset for x in polygon_x],
                        [y + shadow_offset for y in polygon_y],
                    )
                ),
                facecolor="black",
                alpha=0.1,
                linewidth=0,
            )
            ax.add_patch(shadow_polygon)

            # Добавляем текстуру слоя (точки для имитации породы)
            # Временно отключено из-за проблем с размерами массивов
            pass

            # Добавляем подпись слоя
            layer_index = layer["order"]
            layer_text = layer.get("text", f"Формация {layer_index + 1}")

            # Добавляем информацию о длине слоя
            if "length" in layer:
                layer_text += f" ({layer['length']}px)"

            # Убираем текст из слоя - он будет в легенде снизу
            # Размещаем текст в центре слоя
            # center_x = section_width / 2
            # center_y = current_top - layer_thickness / 2  # Это скалярное значение

            # Добавляем текст с фоном для лучшей читаемости
            # ax.text(
            #     center_x,
            #     center_y,
            #     layer_text,
            #     ha="center",
            #     va="center",
            #     fontsize=10,
            #     fontweight="bold",
            #     color="black",
            #     bbox=dict(
            #         boxstyle="round,pad=0.3",
            #         facecolor="white",
            #         alpha=0.8,
            #         edgecolor="gray",
            #     ),
            # )

            logger.info(
                f"  - {layer_text}: цвет_легенды_BGR {legend_color}, "
                f"толщина={float(layer_thickness):.2f}"
            )

            # Переходим к следующему слою
            current_top = float(
                np.mean(layer_bottom)
            )  # Используем среднее значение для плавности

        # Настраиваем оси
        ax.set_xlim(-0.5, section_width + 8.0)  # Расширяем справа для легенды
        ax.set_ylim(
            base_y - 0.5, section_height + base_y + 0.5
        )  # Возвращаем нормальную высоту
        ax.set_aspect("equal")
        ax.set_title("Геологический разрез", fontsize=16, fontweight="bold")
        # Убираем подписи осей
        # ax.set_xlabel("Расстояние (м)", fontsize=12)
        # ax.set_ylabel("Глубина (м)", fontsize=12)

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
        legend_x_start = section_width + 1.0  # Позиция начала легенды справа
        legend_y_start = section_height + base_y - 1.0  # Начинаем сверху
        legend_item_height = 0.6
        legend_item_width = 0.8
        legend_text_x_offset = 1.2

        # Сортируем слои по порядку (сверху вниз)
        sorted_layers = sorted(layers, key=lambda x: x["order"])

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

            # Текст слоя
            layer_text = layer.get("text", f"Слой {layer['order'] + 1}")
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

    def process_geological_section(
        self,
        map_image: np.ndarray,
        legend_image: np.ndarray,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
    ) -> Dict:
        """Основной метод обработки геологического разреза"""
        logger.info("=== НАЧАЛО ОБРАБОТКИ ГЕОЛОГИЧЕСКОГО РАЗРЕЗА ===")
        logger.info(f"Размер карты: {map_image.shape}")
        logger.info(f"Размер легенды: {legend_image.shape}")
        logger.info(f"Точки: ({start_point}) -> ({end_point})")

        try:
            # 1. Извлекаем данные из легенды
            logger.info("Шаг 1: Извлечение данных из легенды")
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

            # Извлекаем цвета для обратной совместимости
            legend_colors = [entry["color"] for entry in legend_data]
            self.legend_colors = legend_colors

            logger.info(f"Извлечено {len(legend_data)} блоков легенды:")
            for i, entry in enumerate(legend_data):
                text = entry.get("text", "")
                if text:
                    logger.info(
                        f"  {i + 1}. Цвет: {entry['color']}, Текст: {text[:100]}..."
                    )
                else:
                    logger.info(
                        f"  {i + 1}. Цвет: {entry['color']}, Текст: не извлечен"
                    )

            # 2. Анализируем линию разреза
            logger.info("Шаг 2: Анализ линии разреза")
            colors_along_line = self.analyze_section_line(
                map_image, start_point, end_point
            )

            # 3. Строим геологические слои с использованием данных легенды
            logger.info("Шаг 3: Построение геологических слоев с данными легенды")
            layers = self.build_geological_layers_with_legend_data(
                colors_along_line, legend_data
            )

            # 4. Создаем визуализацию с названиями из легенды
            logger.info("Шаг 4: Создание визуализации с названиями из легенды")
            output_path = self.create_section_visualization_with_names(
                layers, legend_data, settings.output_dir
            )

            # 5. Создаем изображение карты с линией разреза
            logger.info("Шаг 5: Создание изображения карты с линией разреза")
            map_with_line_path = self.create_map_with_section_line(
                map_image, start_point, end_point, settings.output_dir
            )

            result = {
                "success": True,
                "layers": layers,
                "output_path": output_path,
                "map_with_line_path": map_with_line_path,
                "legend_data": legend_data,
                "line_pixels_count": len(colors_along_line),
            }

            logger.info("=== УСПЕШНОЕ ЗАВЕРШЕНИЕ ОБРАБОТКИ ===")
            logger.info(
                f"Результат: {len(layers)} слоев, {len(colors_along_line)} пикселей"
            )

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
            text = entry.get("text", "").strip()
            if text:  # Только записи с текстом
                legend_text_dict[text] = {
                    "color": entry["color"],  # BGR формат
                    "index": i,
                    "order": i,
                    "text": text,
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
        logger.info(f"Топ-10 самых частых цветов на линии:")
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
                layer_text = closest_entry.get("text", "").strip()

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
        min_layer_length = max(1, len(colors_along_line) // 200)  # Было 100, стало 200
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

        # Сортируем слои по порядку в легенде
        unique_layers.sort(key=lambda x: x["order"], reverse=False)

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

            # Уменьшаем допуск для более строгого сопоставления
            strict_tolerance = adaptive_tolerance * 0.7

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

                # Более строгий допуск для RGB (было 200, стало 100)
                if rgb_distance < min_distance and rgb_distance < 100:
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

                # Очень строгий допуск для манхэттенского расстояния (было 300, стало 150)
                if manhattan_distance < min_distance and manhattan_distance < 150:
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
