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
        """Адаптивный допуск на основе яркости цвета"""
        # Более темные цвета требуют меньшего допуска
        brightness = (color[0] + color[1] + color[2]) / 3.0  # Используем явное сложение
        if brightness < 100:
            return 30  # Темные цвета - строгий допуск
        elif brightness < 200:
            return 50  # Средние цвета - средний допуск
        else:
            return 70  # Светлые цвета - мягкий допуск

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

        # Конвертируем в RGB если нужно
        if len(map_image.shape) == 3 and map_image.shape[2] == 3:
            map_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
            logger.debug("Конвертировал карту из BGR в RGB")
        else:
            map_rgb = map_image
            logger.debug("Карта уже в RGB формате")

        line_pixels = self.get_line_pixels(start_point, end_point)
        colors_along_line = []
        unique_colors = set()

        for x, y in line_pixels:
            if 0 <= y < map_rgb.shape[0] and 0 <= x < map_rgb.shape[1]:
                color = tuple(map_rgb[y, x])
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

        logger.debug(f"Первые 10 цветов: {list(set(colors_along_line))[:10]}")

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
        """Создает визуализацию геологического разреза"""
        logger.info(f"Создаю визуализацию для {len(layers)} слоев")

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

        # Создаем фигуру для длинного прямоугольного разреза
        fig, ax = plt.subplots(
            figsize=(20, 8)
        )  # Увеличиваем размер для более широких слоев

        # Рисуем слои горизонтально (сверху вниз согласно легенде)
        layer_height = 1.0
        layer_width = 15.0  # Увеличиваем ширину слоя
        current_y = len(layers)  # Начинаем сверху

        # Получаем названия слоев
        legend_names = self.extract_legend_names()  # Убираем None параметр

        logger.info("Отображаемые слои в визуализации (сверху вниз):")
        for i, layer in enumerate(layers):
            color = np.array(layer["color"]) / 255.0  # Нормализуем цвета
            rect = plt.Rectangle(
                (0, current_y - layer_height),
                layer_width,
                layer_height,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Добавляем подпись слоя с названием из легенды
            layer_index = layer["order"]
            layer_text = layer.get("text", "")

            # Если нет текста, не показываем подпись
            if not layer_text:
                layer_text = f"Слой {layer_index + 1} (без названия)"

            # Добавляем информацию о длине слоя
            if "length" in layer:
                layer_text += f" ({layer['length']}px)"

            ax.text(
                layer_width / 2,
                current_y - layer_height / 2,
                layer_text,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )

            logger.info(
                f"  - {layer_text}: цвет {layer['color']}, позиция y={current_y - layer_height}"
            )

            current_y -= layer_height

        ax.set_xlim(0, layer_width)
        ax.set_ylim(0, len(layers))
        ax.set_aspect("auto")  # Убираем фиксированные пропорции
        ax.set_title("Геологический разрез", fontsize=16, fontweight="bold")
        ax.set_xlabel("Расстояние")
        ax.set_ylabel("Глубина")

        # Добавляем пояснение
        ax.text(
            layer_width / 2,
            len(layers) + 0.5,
            "Порядок слоев: сверху (выше) → вниз (глубже)",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
        )

        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"section_{timestamp}.png"
        filepath = os.path.join(settings.output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Визуализация сохранена: {filepath}")
        return filepath

    def create_section_visualization_with_names(
        self, layers: List[Dict], legend_data: List[Dict], output_path: str
    ) -> str:
        """Создает визуализацию геологического разреза с названиями из легенды"""
        logger.info(
            f"Создаю визуализацию для {len(layers)} слоев с названиями из легенды"
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

        # Создаем фигуру для длинного прямоугольного разреза
        fig, ax = plt.subplots(figsize=(24, 10))  # Увеличиваем размер для названий

        # Рисуем слои горизонтально (сверху вниз согласно легенде)
        layer_height = 1.0
        layer_width = 18.0  # Увеличиваем ширину для названий
        current_y = len(layers)  # Начинаем сверху

        logger.info("Отображаемые слои в визуализации (сверху вниз):")
        for i, layer in enumerate(layers):
            color = np.array(layer["color"]) / 255.0  # Нормализуем цвета
            rect = plt.Rectangle(
                (0, current_y - layer_height),
                layer_width,
                layer_height,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Добавляем подпись слоя с названием из легенды
            layer_index = layer["order"]
            layer_text = layer.get("text", f"Формация {layer_index + 1}")

            # Добавляем информацию о длине слоя
            if "length" in layer:
                layer_text += f" ({layer['length']}px)"

            ax.text(
                layer_width / 2,
                current_y - layer_height / 2,
                layer_text,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )

            logger.info(
                f"  - {layer_text}: цвет {layer['color']}, позиция y={current_y - layer_height}"
            )

            current_y -= layer_height

        ax.set_xlim(0, layer_width)
        ax.set_ylim(0, len(layers))
        ax.set_aspect("auto")  # Убираем фиксированные пропорции
        ax.set_title("Геологический разрез", fontsize=16, fontweight="bold")
        ax.set_xlabel("Расстояние")
        ax.set_ylabel("Глубина")

        # Добавляем пояснение
        ax.text(
            layer_width / 2,
            len(layers) + 0.5,
            "Порядок слоев: сверху (выше) → вниз (глубже)",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
        )

        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"section_{timestamp}.png"
        filepath = os.path.join(settings.output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Визуализация сохранена: {filepath}")
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

            # 3. Строим геологические слои
            logger.info("Шаг 3: Построение геологических слоев")
            layers = self.build_geological_layers(colors_along_line, legend_colors)

            # 4. Создаем визуализацию
            logger.info("Шаг 4: Создание визуализации")
            output_path = self.create_section_visualization(layers, settings.output_dir)

            result = {
                "success": True,
                "layers": layers,
                "output_path": output_path,
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

        # Конвертируем в RGB если нужно
        if len(legend_image.shape) == 3 and legend_image.shape[2] == 3:
            legend_rgb = cv2.cvtColor(legend_image, cv2.COLOR_BGR2RGB)
            logger.debug("Конвертировал изображение из BGR в RGB")
        else:
            legend_rgb = legend_image
            logger.debug("Изображение уже в RGB формате")

        height, width = legend_rgb.shape[:2]
        logger.info(f"Размер легенды: {width}x{height}")

        # Простой подход: делим легенду на равные блоки по вертикали
        # и извлекаем цвет из левой части, текст из правой
        legend_data = []

        # Определяем количество блоков на основе высоты
        # Примерно 22 блока в легенде
        block_height = height // 22

        logger.info(f"Высота блока: {block_height} пикселей")

        for i in range(22):
            start_y = i * block_height
            end_y = min((i + 1) * block_height, height)

            if end_y - start_y < 20:  # Пропускаем слишком маленькие блоки
                continue

            # Извлекаем цвет из левой части блока
            color_region = legend_rgb[
                start_y:end_y, : width // 6
            ]  # Увеличиваем область для цветов
            block_color = self._extract_dominant_color(color_region)

            if block_color:
                # Извлекаем текст из правой части блока
                text_region = legend_rgb[
                    start_y:end_y, width // 6 :
                ]  # Соответственно уменьшаем область для текста
                block_text = self._extract_text_simple(text_region)

                legend_entry = {
                    "color": block_color,
                    "text": block_text,
                    "y_position": start_y,
                    "height": end_y - start_y,
                }

                legend_data.append(legend_entry)
                logger.info(
                    f"Блок {len(legend_data)}: цвет={block_color}, текст='{block_text[:50]}...'"
                )

        logger.info(f"Извлечено {len(legend_data)} блоков легенды")
        return legend_data

    def _extract_dominant_color(self, color_region: np.ndarray) -> Tuple[int, int, int]:
        """Извлекает доминирующий цвет из области - простая логика как в тексте"""
        # Берем пиксель не в центре, а чуть левее от центра, чтобы избежать символов
        height, width = color_region.shape[:2]
        center_y = height // 2
        # Смещаемся на 1/4 от центра влево
        center_x = width // 4

        # Берем цвет пикселя
        pixel_color = color_region[center_y, center_x]
        r, g, b = int(pixel_color[0]), int(pixel_color[1]), int(pixel_color[2])

        return (r, g, b)

    def _extract_text_simple(self, text_region: np.ndarray) -> str:
        """Простое извлечение текста"""
        # Конвертируем в оттенки серого
        text_gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)

        # Применяем бинаризацию
        _, text_binary = cv2.threshold(
            text_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        try:
            import pytesseract

            # Пробуем извлечь текст
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
        """Строит геологические слои с использованием данных легенды"""
        logger.info(f"Строю геологические слои из {len(colors_along_line)} цветов")
        logger.info(f"Данные легенды: {len(legend_data)} блоков")

        # Анализируем последовательность цветов вдоль линии
        layer_sequence = []
        current_color = None
        current_length = 0

        for i, color in enumerate(colors_along_line):
            # Ищем ближайший цвет в легенде
            closest_entry = self._find_closest_legend_entry(color, legend_data)

            if closest_entry:
                closest_index = legend_data.index(closest_entry)
                if current_color is None or current_color != closest_index:
                    # Начинаем новый слой
                    if current_color is not None and current_length > 0:
                        layer_sequence.append(
                            {
                                "index": current_color,
                                "color": legend_data[current_color]["color"],
                                "order": current_color,
                                "length": current_length,
                                "start_pos": i - current_length,
                                "text": legend_data[current_color]["text"],
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
                            "color": legend_data[current_color]["color"],
                            "order": current_color,
                            "length": current_length,
                            "start_pos": i - current_length,
                            "text": legend_data[current_color]["text"],
                        }
                    )
                    current_color = None
                    current_length = 0

        # Добавляем последний слой
        if current_color is not None and current_length > 0:
            layer_sequence.append(
                {
                    "index": current_color,
                    "color": legend_data[current_color]["color"],
                    "order": current_color,
                    "length": current_length,
                    "start_pos": len(colors_along_line) - current_length,
                    "text": legend_data[current_color]["text"],
                }
            )

        # Если не нашли слои, пробуем более простой подход
        if not layer_sequence:
            logger.warning("Не найдено слоев, пробую альтернативный подход")
            unique_colors = list(set(colors_along_line))
            for color in unique_colors:
                closest_entry = self._find_closest_legend_entry(color, legend_data)
                if closest_entry:
                    closest_index = legend_data.index(closest_entry)
                    layer_sequence.append(
                        {
                            "index": closest_index,
                            "color": closest_entry["color"],
                            "order": closest_index,
                            "length": colors_along_line.count(color),
                            "start_pos": 0,
                            "text": closest_entry["text"],
                        }
                    )

        # Фильтруем слишком короткие слои (шум)
        min_layer_length = max(1, len(colors_along_line) // 100)
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

        logger.info(f"Построено {len(unique_layers)} геологических слоев:")
        for layer in unique_layers:
            text = layer.get("text", "")
            if text:
                logger.info(
                    f"  - Слой {layer['order']}: '{text[:50]}...', длина={layer['length']}"
                )
            else:
                logger.info(
                    f"  - Слой {layer['order']}: без названия, длина={layer['length']}"
                )

        return unique_layers

    def _find_closest_legend_entry(
        self, color: Tuple[int, int, int], legend_data: List[Dict]
    ) -> Dict:
        """Находит ближайшую запись в легенде для данного цвета"""
        if not legend_data:
            return None

        min_distance = float("inf")
        closest_entry = None

        for entry in legend_data:
            legend_color = entry["color"]

            # Сначала пробуем LAB расстояние
            lab_distance = self.lab_distance(color, legend_color)
            adaptive_tolerance = self.adaptive_color_tolerance(color)

            if lab_distance < min_distance and lab_distance < adaptive_tolerance:
                min_distance = lab_distance
                closest_entry = entry

        # Если не нашли в LAB, пробуем RGB
        if closest_entry is None:
            for entry in legend_data:
                legend_color = entry["color"]
                # Используем явное вычисление для избежания переполнения
                rgb_distance = (
                    (color[0] - legend_color[0]) ** 2
                    + (color[1] - legend_color[1]) ** 2
                    + (color[2] - legend_color[2]) ** 2
                ) ** 0.5

                if rgb_distance < min_distance and rgb_distance < 100:
                    min_distance = rgb_distance
                    closest_entry = entry

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
        for i, entry in enumerate(legend_data):
            color = np.array(entry["color"]) / 255.0  # Нормализуем цвета
            rect = plt.Rectangle(
                (0, current_y - layer_height),
                layer_width,
                layer_height,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Добавляем подпись блока
            text = entry.get("text", "")

            # Формируем текст для отображения
            if text:
                layer_text = text
            else:
                layer_text = f"Блок {i + 1} (без названия)"

            # Обрезаем длинный текст
            if len(layer_text) > 80:
                layer_text = layer_text[:77] + "..."

            # Добавляем информацию о цвете
            color_info = f"RGB: {entry['color']}"

            # Размещаем текст в две строки
            ax.text(
                layer_width / 2,
                current_y - layer_height / 2 + 0.3,
                layer_text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

            ax.text(
                layer_width / 2,
                current_y - layer_height / 2 - 0.3,
                color_info,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                style="italic",
            )

            logger.info(
                f"  - {layer_text}: цвет {entry['color']}, позиция y={current_y - layer_height}"
            )

            current_y -= layer_height

        ax.set_xlim(0, layer_width)
        ax.set_ylim(0, len(legend_data))
        ax.set_aspect("auto")
        ax.set_title(
            "Тест извлечения данных из легенды", fontsize=16, fontweight="bold"
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
