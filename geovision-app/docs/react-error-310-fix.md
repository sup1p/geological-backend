# Исправление React Error #310

## 🐛 Проблема
```
Uncaught Error: Minified React error #310
```
Ошибка возникала при нажатии на "Upload Map" из-за неправильного использования хуков React.

## 🔍 Причина
**Неправильное использование useCallback внутри JSX:**
```javascript
// ❌ НЕПРАВИЛЬНО - useCallback внутри JSX
<MapPointSelector 
  onPointsChange={useCallback((start, end) => {
    setStartPoint(start)
    setEndPoint(end)
  }, [])}
/>
```

React хуки (включая `useCallback`) должны вызываться только на верхнем уровне компонента, а не внутри условий, циклов или JSX.

## ✅ Исправление

### 1. Вынес useCallback на верхний уровень компонента
```javascript
// ✅ ПРАВИЛЬНО - на верхнем уровне
const handlePointsChange = useCallback((start: Point | null, end: Point | null) => {
  console.log('[UploadModal] Points changed:', { start, end })
  setStartPoint(start)
  setEndPoint(end)
}, [])

const handleAreasChange = useCallback((selectedAreas: { map: Area | null; legend: Area | null; column: Area | null }) => {
  setAreas(selectedAreas)
}, [])
```

### 2. Использую стабильные функции в JSX
```javascript
// ✅ ПРАВИЛЬНО - используем предварительно созданные callback'и
<MapPointSelector 
  sourceImage={sourceImage}
  mapArea={areas.map}
  onPointsChange={handlePointsChange}
/>

<ImageAreaSelector 
  imageFile={sourceImage}
  onAreasChange={handleAreasChange}
/>
```

## 🎯 Результат
- ✅ React Error #310 исправлен
- ✅ Компонент корректно загружается
- ✅ Upload Map работает без ошибок
- ✅ Хуки используются согласно Rules of Hooks

## 📋 Rules of Hooks (напоминание)
1. **Только на верхнем уровне** - не внутри циклов, условий или вложенных функций
2. **Только в React функциях** - в компонентах или кастомных хуках
3. **Одинаковый порядок** - хуки должны вызываться в одном порядке при каждом рендере

Проблема полностью решена! 🎉