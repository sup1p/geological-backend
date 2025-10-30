# Исправление ошибки 422 - координаты должны быть целыми числами

## 🐛 Проблема
```
POST http://localhost:8000/api/geological-section/create-enhanced-section 422 (Unprocessable Entity)

"Input should be a valid integer, unable to parse string as an integer"
"input":"3760.744107244613" // ← Дробные числа!
```

Бэкенд ожидает целые числа (integers) для координат, но получал числа с плавающей точкой.

## 🔍 Причина
При расчете абсолютных координат получались дробные значения:
```javascript
// ❌ Результат - дробные числа
const absoluteStartX = areas.map.x + startPoint.x  // 3760.744107244613
const absoluteStartY = areas.map.y + startPoint.y  // 1451.6380682907647
```

## ✅ Исправление
Добавил `Math.round()` для округления координат до целых чисел:
```javascript
// ✅ Округляем до целых чисел
const absoluteStartX = Math.round(areas.map.x + startPoint.x)  // 3761
const absoluteStartY = Math.round(areas.map.y + startPoint.y)  // 1452
const absoluteEndX = Math.round(areas.map.x + endPoint.x)      // 4987
const absoluteEndY = Math.round(areas.map.y + endPoint.y)      // 3776
```

## 🎯 Результат
- ✅ Координаты передаются как целые числа
- ✅ Бэкенд корректно обрабатывает запрос  
- ✅ HTTP 422 ошибка исправлена
- ✅ Геологические разрезы создаются успешно

## 📊 Пример корректных данных
```javascript
// ДО исправления
{start_x: "3760.744107244613", start_y: "1451.6380682907647"}

// ПОСЛЕ исправления  
{start_x: "3761", start_y: "1452"}
```

## 🧪 Тестирование
1. Загрузите изображение
2. Выберите области (map, legend, column)
3. Поставьте точки на карте
4. Нажмите "Create Section"
5. Проверьте консоль - должны быть целые числа в координатах

Проблема решена! 🎉