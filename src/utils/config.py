# src/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert file paths to Path objects
    for category in ['raw', 'processed', 'embeddings']:
        for key, value in config['paths'][category].items():
            config['paths'][category][key] = Path(value)
    
    return config

# Usage example:
if __name__ == "__main__":
    config = load_config()
    print(f"Model name: {config['model']['name']}")
    print(f"Training data path: {config['paths']['raw']['train']}")