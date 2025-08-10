#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы API геологического разреза
"""

import requests
import numpy as np
from PIL import Image, ImageDraw
import os


def create_test_images():
    """Создает тестовые изображения карты и легенды"""

    # Создаем тестовую карту
    map_img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(map_img)

    # Рисуем несколько цветных областей
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, color in enumerate(colors):
        y_start = i * 60
        y_end = (i + 1) * 60
        draw.rectangle([0, y_start, 400, y_end], fill=color)

    # Создаем тестовую легенду
    legend_img = Image.new("RGB", (200, 300), color="white")
    draw_legend = ImageDraw.Draw(legend_img)

    for i, color in enumerate(colors):
        y_start = i * 60
        y_end = (i + 1) * 60
        draw_legend.rectangle([0, y_start, 200, y_end], fill=color)
        draw_legend.text((10, y_start + 20), f"Слой {i}", fill="black")

    # Сохраняем изображения
    os.makedirs("test_images", exist_ok=True)
    map_img.save("test_images/test_map.png")
    legend_img.save("test_images/test_legend.png")

    return "test_images/test_map.png", "test_images/test_legend.png"


def test_api():
    """Тестирует API геологического разреза"""

    # Создаем тестовые изображения
    map_path, legend_path = create_test_images()

    # URL API
    base_url = "http://localhost:8000"

    # Проверяем доступность API
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ API доступен: {response.json()}")
    except requests.exceptions.ConnectionError:
        print(
            "✗ API недоступен. Убедитесь, что сервер запущен на http://localhost:8000"
        )
        return

    # Проверяем health endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/geological-section/health")
        print(f"✓ Health check: {response.json()}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")

    # Тестируем создание разреза
    try:
        with open(map_path, "rb") as map_file, open(legend_path, "rb") as legend_file:
            files = {
                "map_image": ("test_map.png", map_file, "image/png"),
                "legend_image": ("test_legend.png", legend_file, "image/png"),
            }

            data = {"start_x": 50, "start_y": 50, "end_x": 350, "end_y": 250}

            response = requests.post(
                f"{base_url}/api/v1/geological-section/create-section",
                files=files,
                data=data,
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Разрез создан успешно!")
                print(f"  Найдено слоев: {len(result['layers'])}")
                print(f"  Изображение: {result['image_url']}")
                print(f"  Сообщение: {result['message']}")

                # Пытаемся скачать изображение
                image_url = f"{base_url}{result['image_url']}"
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    with open("test_images/generated_section.png", "wb") as f:
                        f.write(img_response.content)
                    print(f"✓ Изображение сохранено: test_images/generated_section.png")
                else:
                    print(
                        f"✗ Не удалось скачать изображение: {img_response.status_code}"
                    )

            else:
                print(f"✗ Ошибка создания разреза: {response.status_code}")
                print(f"  Ответ: {response.text}")

    except Exception as e:
        print(f"✗ Ошибка тестирования: {e}")


if __name__ == "__main__":
    print("Тестирование API геологического разреза...")
    test_api()
    print("\nТестирование завершено!")
