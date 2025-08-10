#!/usr/bin/env python3
"""
Тестовый скрипт для проверки новой логики извлечения данных из легенды
"""

import requests
import json


def test_legend_endpoint():
    """Тестирует новый эндпоинт /test-legend с новой логикой"""

    url = "http://localhost:8000/api/v1/geological-section/test-legend"

    # Путь к файлу легенды (замените на реальный путь)
    legend_file_path = "uploads/legend_20250802_230347_legend.jpg"

    try:
        with open(legend_file_path, "rb") as f:
            files = {"legend_image": f}

            print("Отправляю запрос на тестирование новой логики легенды...")
            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                print("✅ Успешно!")
                print(f"Сообщение: {result['message']}")
                print(f"Извлечено блоков: {len(result['legend_data'])}")
                print(f"URL изображения: {result['image_url']}")
                print(f"Сохраненный файл: {result['uploaded_file']}")

                # Показываем полную структуру блоков
                print("\n=== ПОЛНАЯ СТРУКТУРА ИЗВЛЕЧЕННЫХ БЛОКОВ ===")
                for i, block in enumerate(result["legend_data"]):
                    print(f"\nБлок {i + 1}:")
                    print(
                        f"  - Координаты: x={block.get('x', 'N/A')}, y={block.get('y', 'N/A')}, w={block.get('width', 'N/A')}, h={block.get('height', 'N/A')}"
                    )
                    print(f"  - Цвет: BGR{block.get('color', [])}")
                    print(f"  - Текст: '{block.get('text', 'не извлечен')}'")
                    print(f"  - Позиция Y: {block.get('y_position', 'N/A')}")

                # Показываем статистику
                print(f"\n=== СТАТИСТИКА ===")
                print(f"Всего блоков: {len(result['legend_data'])}")

                # Подсчитываем блоки с текстом
                blocks_with_text = sum(
                    1
                    for block in result["legend_data"]
                    if block.get("text", "").strip()
                )
                print(f"Блоков с текстом: {blocks_with_text}")
                print(
                    f"Блоков без текста: {len(result['legend_data']) - blocks_with_text}"
                )

                # Показываем уникальные цвета
                unique_colors = set()
                for block in result["legend_data"]:
                    color = block.get("color")
                    if color:
                        unique_colors.add(tuple(color))
                print(f"Уникальных цветов: {len(unique_colors)}")

                # Показываем диапазон размеров блоков
                if result["legend_data"]:
                    widths = [block.get("width", 0) for block in result["legend_data"]]
                    heights = [
                        block.get("height", 0) for block in result["legend_data"]
                    ]
                    print(
                        f"Размеры блоков: ширина {min(widths)}-{max(widths)}px, высота {min(heights)}-{max(heights)}px"
                    )

            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"Ответ: {response.text}")

    except FileNotFoundError:
        print(f"❌ Файл не найден: {legend_file_path}")
        print("Убедитесь, что файл легенды существует в папке uploads")
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")


if __name__ == "__main__":
    test_legend_endpoint()
