from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[str]] = {}

    def get_history(self, session_id: str) -> List[str]:
        return self.sessions.setdefault(session_id, [])

    def add_message(self, session_id: str, message: str):
        self.sessions.setdefault(session_id, []).append(message)

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)
        
session_manager = SessionManager()


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    from app.services.auth import verify_token, get_user_by_email
    token_data = verify_token(token, credentials_exception)
    # token_data.username actually contains email now
    user = await get_user_by_email(db, email=token_data.username)
    if user is None:
        raise credentials_exception
    return user