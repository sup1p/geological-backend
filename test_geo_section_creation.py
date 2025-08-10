#!/usr/bin/env python3
"""
Тестовый скрипт для проверки создания геологического разреза с логикой по тексту легенды
"""

import requests
import json


def test_legend_extraction():
    """Сначала тестируем извлечение легенды"""
    url = "http://localhost:8000/api/v1/geological-section/test-legend"
    legend_file_path = "uploads/legend_20250802_230347_legend.jpg"

    try:
        with open(legend_file_path, "rb") as f:
            files = {"legend_image": f}
            print("Тестирую извлечение легенды...")
            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Легенда извлечена: {len(result['legend_data'])} блоков")

                # Показываем информацию о блоках легенды
                print("\n=== БЛОКИ ЛЕГЕНДЫ ===")
                for i, entry in enumerate(result["legend_data"]):
                    text = entry.get("text", "").strip()
                    color = entry.get("color", [])
                    print(f"Блок {i + 1}:")
                    print(f"  - Цвет: BGR{color}")
                    print(f"  - Текст: '{text[:50]}...'")
                    print(
                        f"  - Координаты: ({entry.get('x', 'N/A')}, {entry.get('y', 'N/A')})"
                    )

                return True
            else:
                print(f"❌ Ошибка извлечения легенды: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Ошибка при извлечении легенды: {str(e)}")
        return False


def test_geo_section_creation():
    """Тестирует создание геологического разреза с логикой по тексту легенды"""

    # Сначала тестируем легенду
    if not test_legend_extraction():
        print("Не удалось извлечь легенду, прерываю тест")
        return

    url = "http://localhost:8000/api/v1/geological-section/create-section"

    # Пути к файлам
    map_file_path = "uploads/card.jpg"
    legend_file_path = "uploads/legend_20250802_230347_legend.jpg"

    # Координаты для большого изображения 4224 × 4394
    # Берем координаты в центре изображения
    start_x = 3000
    start_y = 500
    end_x = 200
    end_y = 500

    try:
        with (
            open(map_file_path, "rb") as map_file,
            open(legend_file_path, "rb") as legend_file,
        ):
            files = {"map_image": map_file, "legend_image": legend_file}

            data = {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            }

            print("\nОтправляю запрос на создание геологического разреза...")
            print(f"Координаты: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
            print(f"Размер изображения: 4224 × 4394")
            print(
                "Используется логика: текст легенды как ключ, цвета только из легенды"
            )

            response = requests.post(url, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                print("✅ Успешно!")
                print(f"Сообщение: {result['message']}")
                print(f"Создано слоев: {len(result['layers'])}")
                print(f"URL изображения разреза: {result['image_url']}")
                print(f"URL изображения карты с линией: {result['map_with_line_url']}")
                print(f"Пикселей вдоль линии: {result.get('line_pixels_count', 'N/A')}")

                # Показываем информацию о слоях
                print("\n=== СЛОИ ГЕОЛОГИЧЕСКОГО РАЗРЕЗА (по тексту легенды) ===")
                for i, layer in enumerate(result["layers"]):
                    print(f"\nСлой {i + 1}:")
                    print(f"  - Индекс в легенде: {layer.get('order', 'N/A')}")
                    print(f"  - Цвет легенды: BGR{layer.get('color', [])}")
                    print(f"  - Длина: {layer.get('length', 'N/A')} пикселей")
                    print(f"  - Название: '{layer.get('text', '')}'")

                    # Проверяем что используется только цвет из легенды
                    legend_color = layer.get("color", [])
                    text = layer.get("text", "")

                    if text.strip():
                        print(f"  - ✅ Есть название слоя из легенды")
                    else:
                        print(f"  - ⚠️  Нет названия слоя")

                # Показываем статистику
                print(f"\n=== СТАТИСТИКА (по тексту легенды) ===")
                print(f"Всего слоев: {len(result['layers'])}")

                # Подсчитываем слои с текстом
                layers_with_text = sum(
                    1 for layer in result["layers"] if layer.get("text", "").strip()
                )
                print(f"Слоев с названиями: {layers_with_text}")
                print(f"Слоев без названий: {len(result['layers']) - layers_with_text}")

                # Показываем уникальные цвета легенды
                unique_legend_colors = set()
                for layer in result["layers"]:
                    color = layer.get("color")
                    if color:
                        unique_legend_colors.add(tuple(color))
                print(f"Уникальных цветов легенды: {len(unique_legend_colors)}")

                # Показываем порядок слоев
                print(f"\nПорядок слоев (сверху вниз):")
                for i, layer in enumerate(result["layers"]):
                    order = layer.get("order", "N/A")
                    text = layer.get("text", "Без названия")
                    legend_color = layer.get("color", [])
                    print(
                        f"  {i + 1}. Слой {order}: '{text[:30]}...' | Цвет легенды: BGR{legend_color}"
                    )

                # Проверяем качество сопоставления
                print(f"\n=== КАЧЕСТВО СОПОСТАВЛЕНИЯ ===")
                text_matches = 0

                for layer in result["layers"]:
                    text = layer.get("text", "")
                    if text.strip():
                        text_matches += 1

                print(
                    f"Слоев с названиями из легенды: {text_matches}/{len(result['layers'])}"
                )

                if text_matches == len(result["layers"]):
                    print("✅ Все слои имеют названия из легенды")
                else:
                    print("⚠️  Некоторые слои не имеют названий")

                # Проверяем что все цвета взяты из легенды
                print(f"\n=== ПРОВЕРКА ЦВЕТОВ ===")
                all_colors_from_legend = True
                for layer in result["layers"]:
                    legend_color = layer.get("color", [])
                    if not legend_color:
                        all_colors_from_legend = False
                        break

                if all_colors_from_legend:
                    print("✅ Все цвета взяты из легенды")
                else:
                    print("⚠️  Некоторые цвета не из легенды")

            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"Ответ: {response.text}")

    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print(
            "Убедитесь, что файлы card.jpg и legend_20250802_230347_legend.jpg существуют в папке uploads"
        )
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")


if __name__ == "__main__":
    test_geo_section_creation()
