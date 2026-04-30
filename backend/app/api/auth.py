"""
Authentication API Endpoints
Handles user login, token generation, and user management
"""

import logging
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from backend.app.core.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_current_user,
    User,
    Token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

logger = logging.getLogger(__name__)

router = APIRouter()

class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str

class UserCreate(BaseModel):
    """User creation model"""
    username: str
    email: str
    password: str
    role: str = "clinician"

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return access token.
    
    OAuth2 compatible endpoint for token generation.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username} (role: {user.role})")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=User(username=user.username, email=user.email, role=user.role)
    )

@router.post("/login/json", response_model=Token)
async def login_json(login_data: LoginRequest):
    """
    Authenticate user and return access token (JSON endpoint).
    
    Alternative to OAuth2 form-based login for easier client integration.
    """
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {login_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {user.username} (role: {user.role})")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=User(username=user.username, email=user.email, role=user.role)
    )

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user information.
    
    Requires valid JWT token in Authorization header.
    """
    return current_user

@router.post("/users", response_model=User)
async def create_user(user_data: UserCreate, current_user: User = Depends(get_current_active_user)):
    """
    Create a new user (admin only).
    
    Requires admin role for security.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to create users"
        )
    
    # In production, this would add user to database
    # For now, just log the request
    logger.info(f"User creation request: {user_data.username} (role: {user_data.role})")
    
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User creation not yet implemented - use database integration"
    )

@router.get("/verify")
async def verify_token(current_user: User = Depends(get_current_active_user)):
    """
    Verify if a token is valid.
    
    Returns user information if token is valid.
    """
    return {"valid": True, "user": current_user}
