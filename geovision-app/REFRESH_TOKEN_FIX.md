# Refresh Token Logic Fix

## Changes Made

### 1. Enhanced WebSocket Authentication Handling (`lib/api.ts`)

- Updated `createWebSocketConnection()` to handle JSON error messages
- Added automatic token refresh when receiving `{"error": "Authentication failed"}`
- Added callbacks for successful token refresh and authentication failures
- WebSocket connection properly closes and recreates with new token after refresh

### 2. Improved Chatbot Modal (`components/chatbot-modal.tsx`)

- Added `tokenRefreshTrigger` state to trigger reconnection after token refresh
- Updated useEffect dependencies to include refresh trigger
- Added proper error message filtering to prevent auth errors from showing as chat messages
- Implemented automatic reconnection with new token after successful refresh

### 3. Fixed API Functions (`lib/api.ts`)

- Updated `getUserSections()`, `getUserSection()`, `deleteUserSection()` to use `authenticatedFetch()`
- These functions now automatically refresh tokens on 401 errors
- Consistent error handling across all authenticated endpoints

### 4. Token Refresh Flow

The refresh logic now works as follows:

1. **WebSocket receives auth error**: `{"error": "Authentication failed"}`
2. **Automatic refresh attempt**: Call `/api/auth/refresh` with current refresh token
3. **On success**: 
   - Store new access_token and refresh_token
   - Close current WebSocket connection
   - Trigger reconnection with new token
4. **On failure**:
   - Clear all tokens from localStorage
   - Show unauthorized error to user
   - Redirect to login

## API Endpoint Used

```
POST /api/auth/refresh
Content-Type: application/json

Body:
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

## Testing

Use the test functions in `lib/test-refresh.ts`:

```javascript
// In browser console after logging in:
testRefreshToken()  // Test refresh API endpoint
```

## Key Improvements

1. ✅ **WebSocket auto-refresh**: No more infinite unauthorized messages
2. ✅ **Seamless reconnection**: Chat continues after token refresh
3. ✅ **All API calls protected**: Consistent refresh logic across all endpoints
4. ✅ **Proper error handling**: Clear distinction between auth errors and other errors
5. ✅ **Token storage**: Both access and refresh tokens properly updated

The refresh token logic now works correctly for both WebSocket connections and HTTP API calls!