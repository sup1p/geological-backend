#!/usr/bin/env python3
"""
Тестовый скрипт для проверки эндпоинта тестирования легенды
"""

import requests
import json


def test_legend_endpoint():
    """Тестирует новый эндпоинт /test-legend"""

    url = "http://localhost:8000/api/v1/geological-section/test-legend"

    # Путь к файлу легенды (замените на реальный путь)
    legend_file_path = "uploads/legend_20250802_230347_legend.jpg"

    try:
        with open(legend_file_path, "rb") as f:
            files = {"legend_image": f}

            print("Отправляю запрос на тестирование легенды...")
            response = requests.post(url, files=files)

            if response.status_code == 200:
                result = response.json()
                print("✅ Успешно!")
                print(f"Сообщение: {result['message']}")
                print(f"Извлечено блоков: {len(result['legend_data'])}")
                print(f"URL изображения: {result['image_url']}")
                print(f"Сохраненный файл: {result['uploaded_file']}")

                # Показываем первые несколько блоков
                print("\nПервые 5 блоков легенды:")
                for i, block in enumerate(result["legend_data"][:5]):
                    text = block.get("text", "")
                    if text:
                        print(f"  {i + 1}. Цвет: {block.get('color', [])}")
                        print(f"     Текст: {text[:50]}...")
                    else:
                        print(f"  {i + 1}. Цвет: {block.get('color', [])}")
                        print(f"     Текст: не извлечен")

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
