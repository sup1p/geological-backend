from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.agent import ask_with_context
from app.core.database import get_db
from app.services.auth import verify_token, get_user_by_email

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
        # token_data.username actually contains email now
        user = await get_user_by_email(db, email=token_data.username)
        if user is None:
            raise credentials_exception
        return user
    except Exception:
        return None


@router.websocket("/ws/strato")
async def websocket_endpoint(websocket: WebSocket):
    user = None
    db = None
    
    try:
        # Extract token from query parameters
        token = websocket.query_params.get("token")
        
        if not token:
            print("WebSocket connection attempt without token")
            await websocket.close(code=4001, reason="Authentication token required")
            return
        
        # Get database session
        db_gen = get_db()
        db = await db_gen.__anext__()
        
        # First accept the connection, then authenticate
        await websocket.accept()
        
        # Authenticate user
        user = await authenticate_websocket_user(token, db)
        if not user:
            print(f"WebSocket authentication failed for token: {token[:10]}...")
            await websocket.send_text('{"error": "Authentication failed"}')
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        print(f"User {user.id} connected via WebSocket")
        
        while True:
            # Receive user message
            data = await websocket.receive_text()
            print(f"User {user.id} sent message: {data[:50]}...")

            # Use authenticated user ID instead of websocket client ID
            response = await ask_with_context(str(user.id), data)

            # Send response back to client
            await websocket.send_text(response)

            # Optional delay for streaming simulation
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print(f"User {user.id if user else 'Unknown'} disconnected")
    except Exception as e:
        print(f"WebSocket error for user {user.id if user else 'Unknown'}: {str(e)}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass
    finally:
        # Close database session
        if db:
            try:
                await db.close()
            except Exception:
                pass