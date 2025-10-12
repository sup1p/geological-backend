from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.config import settings
from app.models.user import User
from app.schemas import UserCreate, TokenData


def _truncate_password(password: str) -> str:
    """Безопасно обрезает пароль до 72 байт для bcrypt"""
    password_bytes = password.encode('utf-8')
    if len(password_bytes) <= 72:
        return password
    
    # Обрезаем по байтам, затем декодируем обратно в строку
    truncated_bytes = password_bytes[:72]
    # Удаляем неполные UTF-8 символы в конце
    while truncated_bytes:
        try:
            return truncated_bytes.decode('utf-8')
        except UnicodeDecodeError:
            truncated_bytes = truncated_bytes[:-1]
    # Fallback - если что-то пошло не так, возвращаем первые 72 символа
    return password[:72]


def verify_password(plain_password: str, hashed_password: str) -> bool:
    truncated_password = _truncate_password(plain_password)
    return bcrypt.checkpw(truncated_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    truncated_password = _truncate_password(password)
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(truncated_password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


async def authenticate_user(db: AsyncSession, username: str, password: str):
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalars().first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


async def get_user_by_username(db: AsyncSession, username: str):
    result = await db.execute(select(User).where(User.username == username))
    return result.scalars().first()


async def create_user(db: AsyncSession, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data