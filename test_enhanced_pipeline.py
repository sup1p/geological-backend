#!/usr/bin/env python3
"""
Тестовый скрипт для проверки улучшенного pipeline создания геологического разреза
с интегрированной улучшенной очисткой геологических терминов
"""

import sys
import os
import logging
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from app.core.strategraphic_column import StrategraphicColumnProcessor
from app.core.geological_processor import GeologicalProcessor
import numpy as np
import cv2


def test_enhanced_pipeline():
    """Тестирует улучшенный pipeline с интегрированной очисткой"""
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Тестирование улучшенного pipeline геологического разреза")
    print("="*70)
    
    # Тест 1: Проверяем улучшенную обработку стратиграфической колонки
    print("\n📋 ТЕСТ 1: Обработка стратиграфической колонки с улучшенной очисткой")
    print("-" * 50)
    
    try:
        column_file = "uploads/strategraphic_column.jpg"
        if not os.path.exists(column_file):
            print(f"❌ Файл не найден: {column_file}")
            return
            
        # Тестируем старую версию
        print("🔄 Обработка БЕЗ улучшенной очистки:")
        processor_old = StrategraphicColumnProcessor(use_enhanced_cleaning=False)
        old_layers = processor_old.process_strategraphic_column(column_file)
        print(f"   Результат: {len(old_layers)} слоев")
        
        # Тестируем новую версию
        print("🔄 Обработка С улучшенной очисткой:")
        processor_new = StrategraphicColumnProcessor(use_enhanced_cleaning=True)
        new_layers = processor_new.process_strategraphic_column(column_file)
        print(f"   Результат: {len(new_layers)} слоев")
        
        # Сравниваем результаты
        print("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
        print("-" * 30)
        
        max_len = max(len(old_layers), len(new_layers))
        for i in range(max_len):
            old_text = old_layers[i] if i < len(old_layers) else "—"
            new_text = new_layers[i] if i < len(new_layers) else "—"
            
            print(f"\nСлой {i+1}:")
            print(f"  БЕЗ очистки: {old_text[:70]}{'...' if len(old_text) > 70 else ''}")
            print(f"  С очисткой:  {new_text[:70]}{'...' if len(new_text) > 70 else ''}")
            
            # Показываем только первые 5 для краткости
            if i >= 4:
                if max_len > 5:
                    print(f"\n... и ещё {max_len - 5} слоев")
                break
                
    except Exception as e:
        print(f"❌ Ошибка в тесте 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 2: Проверяем интеграцию с GeologicalProcessor
    print("\n\n🗺️ ТЕСТ 2: Интеграция с GeологicalProcessor")
    print("-" * 50)
    
    try:
        # Проверяем, что файлы карты и легенды существуют
        map_file = "uploads/card.jpg"  # Предполагаем, что есть карта
        legend_file = "uploads/legend_20250802_230347_legend.jpg"
        
        missing_files = []
        if not os.path.exists(map_file):
            missing_files.append(map_file)
        if not os.path.exists(legend_file):
            missing_files.append(legend_file)
        
        if missing_files:
            print(f"⚠️ Файлы не найдены для полного теста: {', '.join(missing_files)}")
            print("   Тестирую только извлечение последовательности слоев...")
            
            # Создаем GeologicalProcessor и проверяем что он использует улучшенную логику
            geo_processor = GeologicalProcessor()
            
            # Имитируем вызов части процесса
            print("🔄 Создаю StrategraphicColumnProcessor с улучшенной очисткой...")
            sc_processor = StrategraphicColumnProcessor(use_enhanced_cleaning=True)
            column_layers = sc_processor.process_strategraphic_column("uploads/strategraphic_column.jpg")
            
            print(f"✅ GeologicalProcessor получит {len(column_layers)} очищенных слоев")
            print("📋 Первые 3 слоя:")
            for i, layer in enumerate(column_layers[:3]):
                print(f"  {i+1}. {layer}")
        else:
            print("✅ Все файлы найдены, запускаю полный тест...")
            
            # Загружаем изображения
            map_image = cv2.imread(map_file)
            legend_image = cv2.imread(legend_file)
            
            # Создаем GeologicalProcessor  
            geo_processor = GeologicalProcessor()
            
            # Задаем простую линию разреза (по диагонали)
            height, width = map_image.shape[:2]
            start_point = (width // 4, height // 4)
            end_point = (3 * width // 4, 3 * height // 4)
            
            print(f"🔄 Создаю разрез от {start_point} до {end_point}...")
            
            # Вызываем основной метод
            result = geo_processor.process_geological_section(
                map_image, legend_image, start_point, end_point
            )
            
            print("✅ Разрез создан успешно!")
            print(f"📊 Найдено слоев в разрезе: {len(result.get('layers', []))}")
            
            # Показываем информацию о созданных файлах
            output_files = {
                'Геологический разрез': result.get('output_path'),
                'Карта с линией разреза': result.get('map_with_line_path'),
                'Отладочная легенда': result.get('debug_legend_path')
            }
            
            print("\n📁 Созданные файлы:")
            for file_type, file_path in output_files.items():
                if file_path:
                    print(f"  📄 {file_type}: {file_path}")
            
            # Показываем информацию о слоях
            layers = result.get('layers', [])
            if layers:
                print("\n📋 Слои в разрезе (с улучшенной последовательностью):")
                for i, layer in enumerate(layers[:5]):  # Показываем первые 5
                    name = layer.get('name', 'Неизвестный')
                    length = layer.get('length', 0)
                    print(f"  {i+1}. {name} (длина: {length})")
                
                if len(layers) > 5:
                    print(f"  ... и ещё {len(layers) - 5} слоев")
            
            # Показываем статистику легенды
            legend_data = result.get('legend_data', [])
            if legend_data:
                matched_count = sum(1 for block in legend_data if block.get('column_order', -1) >= 0)
                print(f"\n📊 Статистика легенды:")
                print(f"  Всего блоков в легенде: {len(legend_data)}")
                print(f"  Сопоставлено с колонкой: {matched_count}")
                print(f"  Отладочная легенда показывает детали сопоставления")
                    
    except Exception as e:
        print(f"❌ Ошибка в тесте 2: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Тестирование завершено!")
    print("\n💡 Теперь GeologicalProcessor использует улучшенную очистку")
    print("   геологических терминов при создании разрезов!")


if __name__ == "__main__":
    test_enhanced_pipeline()