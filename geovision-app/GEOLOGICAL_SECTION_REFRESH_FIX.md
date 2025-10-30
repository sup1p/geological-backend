# Geological Section Token Refresh Fix

## Problem
При попытке создания геологического разреза с истекшим токеном появлялось сообщение "Your session has expired. Please log in again." вместо попытки обновить токен.

## Solution
Изменена функция `createGeologicalSection()` для использования `authenticatedFetch()` вместо прямого `fetch()`, что обеспечивает автоматическое обновление токенов.

## Changes Made

### 1. Updated `createGeologicalSection()` in `lib/api.ts`

**Before:**
```javascript
// Прямой fetch с ручной обработкой 401
const response = await fetch(url, {
  headers: { Authorization: `Bearer ${token}` },
  body: formData,
})

if (response.status === 401) {
  // Сразу очистка токенов и ошибка
  localStorage.removeItem("auth_token")
  throw new Error("UNAUTHORIZED: Your session has expired...")
}
```

**After:**
```javascript
// Использование authenticatedFetch для автоматического refresh
const response = await authenticatedFetch(url, {
  method: "POST",
  body: formData,
})

// authenticatedFetch автоматически обновляет токены при 401
// Ошибка UNAUTHORIZED показывается только если refresh не удался
```

### 2. Enhanced Error Handling

- Добавлено логирование для отладки процесса refresh
- UNAUTHORIZED ошибка показывается только после неудачной попытки refresh
- Улучшены сообщения об ошибках в консоли

### 3. Flow диаграмма

```
Пользователь создает разрез
        ↓
createGeologicalSection() вызывает authenticatedFetch()
        ↓
Сервер отвечает 401 (токен истек)
        ↓
authenticatedFetch() автоматически вызывает refreshAccessToken()
        ↓
    ┌─────────────────┐
    │ Refresh успешен │
    └─────────────────┘
            ↓
    Повторный запрос с новым токеном
            ↓
    ┌──────────────────┐
    │ Создание разреза │
    └──────────────────┘

    ┌─────────────────┐
    │ Refresh неудачен│
    └─────────────────┘
            ↓
    Очистка токенов
            ↓
    Показ "UNAUTHORIZED: Your session has expired..."
```

## Testing

Для тестирования используйте функции в `test-geological-refresh.ts`:

```javascript
// В консоли браузера после входа в систему:
testGeologicalSectionRefresh()  // Тест базовой логики refresh
testCreateGeologicalSectionRefresh()  // Тест создания разреза с expired токеном
```

## Result

✅ **Теперь при создании геологического разреза:**
1. Если токен действителен → создание проходит нормально
2. Если токен истек → автоматически обновляется и повторяется запрос  
3. Если refresh не удался → показывается ошибка авторизации

Пользователь видит сообщение об истечении сессии только в случае, когда refresh токен тоже недействителен или произошла ошибка на сервере.