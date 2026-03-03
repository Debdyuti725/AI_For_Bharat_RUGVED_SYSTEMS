"""
SQLite Database — Stores users, profiles, and chat history.

Tables:
  - users: id, email, hashed_password, name, created_at
  - profiles: user_id (FK), profile_json, tier1_complete, tier2_categories, updated_at
  - chat_history: id, user_id (FK), role, message, timestamp
"""

import sqlite3
import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bharat_access_hub.db")


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row_factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            profile_json TEXT NOT NULL DEFAULT '{}',
            tier1_complete INTEGER NOT NULL DEFAULT 0,
            tier2_categories TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id);
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")


# ─── User CRUD ────────────────────────────────────────────────────────────────

def create_user(email: str, hashed_password: str, name: str = "") -> int:
    """Create a new user. Returns user ID."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, hashed_password, name) VALUES (?, ?, ?)",
            (email.lower().strip(), hashed_password, name)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ─── Profile CRUD ─────────────────────────────────────────────────────────────

def save_profile(user_id: int, profile_dict: Dict[str, Any], tier1_complete: bool = False, tier2_categories: List[str] = None):
    """Save or update a user's profile."""
    conn = get_connection()
    tier2_json = json.dumps(tier2_categories or [])
    profile_json = json.dumps(profile_dict, ensure_ascii=False)
    try:
        conn.execute("""
            INSERT INTO profiles (user_id, profile_json, tier1_complete, tier2_categories, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                tier1_complete = excluded.tier1_complete,
                tier2_categories = excluded.tier2_categories,
                updated_at = datetime('now')
        """, (user_id, profile_json, int(tier1_complete), tier2_json))
        conn.commit()
    finally:
        conn.close()


def get_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Get a user's profile."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["profile_json"] = json.loads(result["profile_json"])
        result["tier2_categories"] = json.loads(result["tier2_categories"])
        return result
    finally:
        conn.close()


# ─── Chat History CRUD ────────────────────────────────────────────────────────

def save_chat_message(user_id: int, role: str, message: str):
    """Save a single chat message."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)",
            (user_id, role, message)
        )
        conn.commit()
    finally:
        conn.close()


def get_chat_history(user_id: int, limit: int = 20) -> List[Dict[str, str]]:
    """Get recent chat history for a user."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT role, message, timestamp FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def clear_chat_history(user_id: int):
    """Clear all chat history for a user."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


# Initialize on import
init_db()
