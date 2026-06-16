import yaml
import os
from typing import Dict, Any

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its content as a dictionary.
    """
    if not os.path.exists(file_path):
        return {}
        
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError:
            return {}
