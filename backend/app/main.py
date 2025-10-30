from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import geological_section, auth, user_sections, strato_ai
from app.core.database import engine
from app.models.user import Base
import logging
from contextlib import asynccontextmanager


from datetime import timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.auth import authenticate_user, create_access_token, create_refresh_token
from app.schemas import Token
from app.core.config import settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== ЗАПУСК ПРИЛОЖЕНИЯ ===")
    logger.info("Геологический разрез API запускается...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    logger.info("=== ОСТАНОВКА ПРИЛОЖЕНИЯ ===")
    logger.info("Геологический разрез API останавливается...")
    await engine.dispose()

app = FastAPI(
    title="Геологический разрез API",
    description="API для построения геологического разреза по карте и легенде",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/auth/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token using email as subject (since we authenticate by email)
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token_expires = timedelta(days=settings.refresh_token_expire_days)
    refresh_token = create_refresh_token(
        data={"sub": user.email}, expires_delta=refresh_token_expires
    )
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
    
@app.get("/")
async def health_check():
    return {"status": "ok"}

# Подключение роутеров
app.include_router(geological_section.router, prefix="/api")
app.include_router(strato_ai.router, prefix="/api")
app.include_router(auth.router)
app.include_router(user_sections.router, prefix="/api")