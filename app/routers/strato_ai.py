from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.agent import ask_with_context
from app.core.database import get_db
from app.services.auth import verify_token, get_user_by_username

import asyncio



router = APIRouter(tags=["Strato AI"])

async def authenticate_websocket_user(token: str, db: AsyncSession):
    """Authenticate user for WebSocket connection using JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = verify_token(token, credentials_exception)
        user = await get_user_by_username(db, username=token_data.username)
        if user is None:
            raise credentials_exception
        return user
    except Exception:
        return None


@router.websocket("/ws/strato")
async def websocket_endpoint(websocket: WebSocket):
    # Extract token from query parameters
    token = websocket.query_params.get("token")
    
    if not token:
        await websocket.close(code=4001, reason="Authentication token required")
        return
    
    # Get database session
    db_gen = get_db()
    db = await db_gen.__anext__()
    
    try:
        # Authenticate user
        user = await authenticate_websocket_user(token, db)
        if not user:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return
        
        await websocket.accept()
        
        while True:
            # Receive user message
            data = await websocket.receive_text()

            # Use authenticated user ID instead of websocket client ID
            response = await ask_with_context(str(user.id), data)

            # Send response back to client
            await websocket.send_text(response)

            # Optional delay for streaming simulation
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print(f"User {user.id if 'user' in locals() else 'Unknown'} disconnected")
    finally:
        # Close database session
        try:
            await db.close()
        except Exception:
            pass