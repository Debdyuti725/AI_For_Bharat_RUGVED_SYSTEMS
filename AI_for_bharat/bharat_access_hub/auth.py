"""
Auth Module — Sign up / Sign in with password hashing and JWT tokens.

Usage:
  - signup(email, password, name) → user dict
  - login(email, password) → JWT token
  - get_current_user(token) → user dict (for protected routes)
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt

from .database import create_user, get_user_by_email, get_user_by_id

# ─── Config ───────────────────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "bharat-access-hub-hackathon-secret-key-2025")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24


# ─── Password helpers (using hashlib — no bcrypt dependency issues) ───────────

def hash_password(password: str) -> str:
    """Hash a password with a random salt using SHA-256."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hashed}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its salted hash."""
    try:
        salt, stored_hash = hashed_password.split("$", 1)
        check_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
        return check_hash == stored_hash
    except (ValueError, AttributeError):
        return False


# ─── JWT helpers ──────────────────────────────────────────────────────────────

def create_access_token(user_id: int, email: str) -> str:
    """Create a JWT access token."""
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ─── Auth functions ───────────────────────────────────────────────────────────

def signup(email: str, password: str, name: str = "") -> Dict[str, Any]:
    """
    Register a new user.

    Returns: {"user_id": int, "email": str, "token": str}
    Raises: ValueError if email already exists or password too short.
    """
    email = email.lower().strip()

    if not email or "@" not in email:
        raise ValueError("Invalid email address.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    existing = get_user_by_email(email)
    if existing:
        raise ValueError("An account with this email already exists.")

    hashed = hash_password(password)
    user_id = create_user(email, hashed, name)
    token = create_access_token(user_id, email)

    return {
        "user_id": user_id,
        "email": email,
        "name": name,
        "token": token,
        "message": "Account created successfully!",
    }


def login(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user.

    Returns: {"user_id": int, "email": str, "token": str}
    Raises: ValueError if credentials are invalid.
    """
    email = email.lower().strip()
    user = get_user_by_email(email)

    if not user:
        raise ValueError("No account found with this email.")

    if not verify_password(password, user["hashed_password"]):
        raise ValueError("Incorrect password.")

    token = create_access_token(user["id"], user["email"])

    return {
        "user_id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "token": token,
        "message": "Login successful!",
    }


def get_current_user(token: str) -> Dict[str, Any]:
    """
    Get the current user from a JWT token.

    Returns: user dict from DB
    Raises: ValueError if token is invalid or user not found.
    """
    payload = decode_token(token)
    if not payload:
        raise ValueError("Invalid or expired token. Please log in again.")

    user_id = int(payload["sub"])
    user = get_user_by_id(user_id)

    if not user:
        raise ValueError("User not found.")

    return user
