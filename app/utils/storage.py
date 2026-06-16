import os
import shutil
from typing import Any, Union
from PIL import Image
from app.utils.logger import logger

class StorageManager:
    """
    Handles all file system operations within a dedicated storage directory.
    Designed for flexibility to support various pipelines and data types.
    """
    
    def __init__(self, base_dir: str = "storage"):
        self.base_dir = base_dir
        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """Creates the base storage directory if it doesn't exist."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            logger.debug(f"Created base storage directory: {self.base_dir}")

    def get_full_path(self, relative_path: str) -> str:
        """Returns the full absolute path for a given relative path within storage."""
        return os.path.join(self.base_dir, relative_path)

    def ensure_dir(self, relative_path: str) -> str:
        """Ensures a directory exists within the storage folder."""
        full_path = self.get_full_path(relative_path)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def clear_dir(self, relative_path: str):
        """Deletes and recreates a directory within storage."""
        full_path = self.get_full_path(relative_path)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
        os.makedirs(full_path)
        logger.info(f"Cleared directory: {relative_path}")

    def save_text(self, relative_path: str, content: str, encoding: str = "utf-8"):
        """Saves text content to a file."""
        full_path = self.get_full_path(relative_path)
        self.ensure_dir(os.path.dirname(relative_path))
        
        with open(full_path, "w", encoding=encoding) as f:
            f.write(content)
        logger.debug(f"Saved text file: {relative_path}")
        return full_path

    def save_image(self, relative_path: str, image: Image.Image, format: str = "PNG"):
        """Saves a PIL image to a file."""
        full_path = self.get_full_path(relative_path)
        self.ensure_dir(os.path.dirname(relative_path))
        
        image.save(full_path, format=format)
        logger.debug(f"Saved image: {relative_path}")
        return full_path

    def save_file(self, relative_path: str, content: bytes):
        """Saves binary content to a file."""
        full_path = self.get_full_path(relative_path)
        self.ensure_dir(os.path.dirname(relative_path))
        
        with open(full_path, "wb") as f:
            f.write(content)
        logger.debug(f"Saved binary file: {relative_path}")
        return full_path

    def exists(self, relative_path: str) -> bool:
        """Checks if a path exists within storage."""
        return os.path.exists(self.get_full_path(relative_path))
