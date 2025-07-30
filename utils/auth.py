import os
import json
import hashlib
import re
from pathlib import Path

# USERS_PATH constant (points to project_root/data/users.json)
USERS_PATH = Path(__file__).resolve().parent.parent / "data" / "users.json"

def load_users():
    """Load users from USERS_PATH. Returns dict {username: password_hash}."""
    if not USERS_PATH.exists():
        return {}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users_dict):
    """Save users to USERS_PATH. Creates parent dir if needed."""
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, indent=2)

def hash_password(password: str) -> str:
    """Hash password with SHA-256. Returns hex digest."""
    h = hashlib.sha256()
    h.update(password.encode("utf-8"))
    return h.hexdigest()

def password_valid(password: str) -> bool:
    """
    Password must be at least 8 chars, have at least one lowercase letter,
    one uppercase letter, and one special (non-alphanumeric) character.
    """
    if len(password) < 8:
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[^a-zA-Z0-9]', password):
        return False
    return True

def verify_credentials(username, password) -> bool:
    """Check username exists and password matches hash."""
    users = load_users()
    hash_pw = hash_password(password)
    return username in users and users[username] == hash_pw

def register_user(username, password):
    """
    Register new user. Returns (success: bool, message: str).
    Rejects existing usernames, invalid password, empty fields.
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if not password_valid(password):
        return False, "Password must be at least 8 characters, include lowercase, uppercase, and a special character."
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": hash_password(password),
        "progress": {}
    }
    save_users(users)
    return True, "Registration successful."

def get_user_progress(username) -> dict:
    """
    Returns a copy of the user's progress dict, or {} if missing.
    """
    users = load_users()
    user = users.get(username)
    if not user or "progress" not in user:
        return {}
    # Return a copy to avoid accidental mutation
    return dict(user["progress"])

def save_user_progress(username, progress: dict):
    """
    Updates the user's progress dict and saves to users.json.
    """
    users = load_users()
    if username not in users:
        # Optionally scaffold new user if missing (shouldn't happen)
        users[username] = {"password": "", "progress": {}}
    users[username]["progress"] = progress
    save_users(users)

def ensure_user_exists(username):
    """
    Ensures that a user exists in users.json with an empty progress dict if not present.
    """
    users = load_users()
    if username not in users:
        users[username] = {"password": "", "progress": {}}
        save_users(users)