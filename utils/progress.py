import json
from pathlib import Path

def load_progress(progress_path: Path):
    if progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_progress(progress_path: Path, progress: dict):
    try:
        # Ensure parent directory exists
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"[save_progress] Warning: Failed to save progress to {progress_path}: {e}")