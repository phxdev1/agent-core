#!/usr/bin/env python3
"""
Authentication and authorization for Agent Core API
"""

import os
import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

# Configuration
SECRET_KEY = os.getenv("API_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# In-memory user store (replace with database in production)
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("changeme"),
        "api_key": "sk-admin-" + hashlib.sha256(b"admin-key").hexdigest()[:32],
        "role": "admin",
        "active": True
    },
    "user": {
        "username": "user",
        "hashed_password": pwd_context.hash("password"),
        "api_key": "sk-user-" + hashlib.sha256(b"user-key").hexdigest()[:32],
        "role": "user",
        "active": True
    }
}

# API key store (for simpler auth)
api_keys_db = {
    users_db["admin"]["api_key"]: "admin",
    users_db["user"]["api_key"]: "user"
}


class AuthHandler:
    """Handle authentication and authorization"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode and verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password"""
        user = users_db.get(username)
        if not user:
            return None
        if not AuthHandler.verify_password(password, user["hashed_password"]):
            return None
        return user
    
    @staticmethod
    def verify_api_key(api_key: str) -> Optional[str]:
        """Verify API key and return username"""
        return api_keys_db.get(api_key)


# Dependency functions for FastAPI

async def get_current_user_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = AuthHandler.decode_token(token)
    
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = users_db.get(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.get("active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_user_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from API key"""
    api_key = credentials.credentials
    
    # Check if it's an API key (starts with 'sk-')
    if not api_key.startswith("sk-"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = AuthHandler.verify_api_key(api_key)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = users_db.get(username)
    if user is None or not user.get("active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled or not found"
        )
    
    return user


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from either JWT or API key"""
    token = credentials.credentials
    
    # Try API key first (faster)
    if token.startswith("sk-"):
        return await get_current_user_api_key(credentials)
    else:
        return await get_current_user_jwt(credentials)


async def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# Rate limiting (simple in-memory implementation)
class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if rate limit is exceeded"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=60)


async def check_rate_limit(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Check rate limit for current user"""
    username = current_user.get("username", "unknown")
    
    if not rate_limiter.check_rate_limit(username):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    return current_user