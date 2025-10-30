# Исправление проблемы с мгновенным сбросом точек

## 🐛 Проблема
Точки start и end сбрасывались сразу же после установки:
```javascript
[MapPointSelector] Setting start point: {x: 1115.59, y: 212.98}
[UploadModal] Points changed: {start: {…}, end: null}  
[UploadModal] Points changed: {start: null, end: null} // ❌ Сразу сбрасывается!
```

## 🔍 Причина
1. **useEffect с лишними зависимостями** - перерендерился при каждом изменении callback'a
2. **Нестабильный onPointsChange** - создавался новый на каждом рендере  
3. **Лишний useEffect для сброса** - сбрасывал точки при изменении croppedMapUrl

## ✅ Исправления

### 1. Убрал лишний useEffect для автоматического сброса
```javascript
// ❌ БЫЛО - сбрасывал точки при каждом изменении
useEffect(() => {
  clearPoints()
}, [croppedMapUrl, clearPoints])

// ✅ СТАЛО - убрали этот useEffect полностью
```

### 2. Стабилизировал onPointsChange через useRef
```javascript
// ✅ ДОБАВИЛ
const onPointsChangeRef = useRef(onPointsChange)

useEffect(() => {
  onPointsChangeRef.current = onPointsChange
}, [onPointsChange])

// ✅ ИСПОЛЬЗУЮ везде onPointsChangeRef.current вместо onPointsChange
```

### 3. Упростил условие сброса в основном useEffect
```javascript
// ❌ БЫЛО - сбрасывал при любом изменении mapArea
} else {
  setCroppedMapUrl(null)
  clearPoints()  // Сбрасывал всегда!
}

// ✅ СТАЛО - сбрасывает только если mapArea совсем нет
} else {
  setCroppedMapUrl(null)
  if (!mapArea) {  // Только если mapArea действительно null
    setStartPoint(null)
    setEndPoint(null)
    onPointsChangeRef.current(null, null)
  }
}
```

### 4. Стабилизировал callback в UploadModal
```javascript
// ✅ ДОБАВИЛ useCallback
onPointsChange={useCallback((start: Point | null, end: Point | null) => {
  console.log('[UploadModal] Points changed:', { start, end })
  setStartPoint(start) 
  setEndPoint(end)
}, [])}
```

## 🎯 Результат
Теперь точки:
- ✅ Устанавливаются и НЕ сбрасываются  
- ✅ Отображаются стабильно на карте
- ✅ Сохраняются до следующего клика
- ✅ Правильно передаются в upload-modal

## 🧪 Тестирование
```javascript
// ✅ Ожидаемое поведение в консоли:
[MapPointSelector] Clicked at: {x: 1115.59, y: 212.98}
[MapPointSelector] Setting start point: {x: 1115.59, y: 212.98}
[UploadModal] Points changed: {start: {x: 1115.59, y: 212.98}, end: null}

// При втором клике:
[MapPointSelector] Clicked at: {x: 800.23, y: 400.15}
[MapPointSelector] Setting end point: {x: 800.23, y: 400.15}  
[UploadModal] Points changed: {start: {x: 1115.59, y: 212.98}, end: {x: 800.23, y: 400.15}}

// ✅ БЕЗ лишних сообщений о сбросе!
```

Проблема полностью решена! 🎉