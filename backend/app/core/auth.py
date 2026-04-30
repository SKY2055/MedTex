"""
Authentication and Authorization Module
Implements JWT-based authentication for clinical NER system
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

class User(BaseModel):
    """User model for authentication"""
    username: str
    email: Optional[str] = None
    role: str = "clinician"  # clinician, admin, reviewer
    disabled: bool = False

class UserInDB(User):
    """User model with hashed password"""
    hashed_password: str

class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User

class TokenData(BaseModel):
    """Token data for validation"""
    username: Optional[str] = None
    role: Optional[str] = None

# Mock user database (replace with real database in production)
# For demo purposes, using simple password comparison
# In production, use proper bcrypt hashing with database storage
fake_users_db: Dict[str, UserInDB] = {
    "admin": {
        "username": "admin",
        "email": "admin@medtex.com",
        "role": "admin",
        "hashed_password": "admin123",  # Plain text for demo - CHANGE IN PRODUCTION
        "disabled": False
    },
    "clinician": {
        "username": "clinician",
        "email": "clinician@medtex.com",
        "role": "clinician",
        "hashed_password": "clinician123",  # Plain text for demo - CHANGE IN PRODUCTION
        "disabled": False
    },
    "reviewer": {
        "username": "reviewer",
        "email": "reviewer@medtex.com",
        "role": "reviewer",
        "hashed_password": "reviewer123",  # Plain text for demo - CHANGE IN PRODUCTION
        "disabled": False
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    # For demo: simple string comparison (INSECURE - CHANGE IN PRODUCTION)
    # In production, use: return pwd_context.verify(plain_password, hashed_password)
    return plain_password == hashed_password

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user credentials"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None:
            raise credentials_exception
            
        token_data = TokenData(username=username, role=role)
        
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (not disabled)"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def require_role(required_role: str, current_user: User = Depends(get_current_active_user)) -> User:
    """Require specific user role"""
    if current_user.role != required_role and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{required_role}' required"
        )
    return current_user

def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Require admin role"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return current_user
