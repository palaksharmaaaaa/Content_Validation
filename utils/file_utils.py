from pathlib import Path
import hashlib


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def get_file_size_mb(file_path: str) -> float:
    size_bytes = Path(file_path).stat().st_size
    return size_bytes / (1024 * 1024)


def calculate_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()