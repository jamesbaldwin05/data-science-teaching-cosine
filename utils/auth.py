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
    """
    Authenticate a user against stored credentials, supporting both legacy and new user record formats.

    - Loads users from disk.
    - Retrieves the record for the given username.
    - Ensures the user record is in dict form (migrates legacy string records if needed).
    - If migration occurred, persists the updated users to disk.
    - Returns True if the supplied password matches the stored hash; otherwise, False.
    - Returns False if the user does not exist.

    This function transparently migrates legacy user records (where the value is a string hash)
    to the newer dict format. After migration, updated records are saved back to disk.
    """
    users = load_users()
    rec = _ensure_record_dict(users, username)
    if rec is None:
        # User does not exist
        return False

    # If migration happened, save back to disk.
    if isinstance(users.get(username), dict) and "password" in users[username] and isinstance(users[username]["password"], str):
        # Only save if we just migrated from legacy string to dict (i.e., rec was previously a str)
        # This is safe to call every time after _ensure_record_dict; it will only update if needed.
        save_users(users)

    # Compare supplied password (after hashing) to stored hash.
    hash_pw = hash_password(password)
    return rec.get("password") == hash_pw

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


def _ensure_record_dict(users, username):
    """
    If user record is just a string (legacy hash), migrate it to dict form.
    Returns the user record (dict) or None if not present.
    """
    rec = users.get(username)
    if rec is None:
        return None
    if isinstance(rec, str):
        users[username] = {"password": rec, "progress": {}}
        return users[username]
    return rec


def get_user_progress(username) -> dict:
    """
    Returns a copy of the user's progress dict, or {} if missing.
    Transparently migrates legacy hash-only records.
    """
    users = load_users()
    rec = _ensure_record_dict(users, username)
    if rec is None:
        return {}
    if isinstance(rec, str):
        save_users(users)  # Shouldn't happen, but just in case
        return {}
    # Save after migration if needed
    if username in users and isinstance(users[username], dict) and "progress" not in users[username]:
        users[username]["progress"] = {}
        save_users(users)
    return dict(rec.get('progress', {}))


def save_user_progress(username, progress: dict):
    """
    Updates the user's progress dict and saves to users.json.
    Transparently migrates legacy hash-only records.
    """
    users = load_users()
    rec = _ensure_record_dict(users, username)
    if rec is None:
        # create blank password
        users[username] = {"password": "", "progress": progress}
    else:
        rec['progress'] = progress
    save_users(users)


def ensure_user_exists(username):
    """
    Ensures that a user exists in users.json with an empty progress dict if not present.
    Migrates legacy hash-only records.
    """
    users = load_users()
    _ensure_record_dict(users, username)
    if username not in users:
        users[username] = {"password": "", "progress": {}}
    save_users(users)