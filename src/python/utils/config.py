"""
Configuration management for training and inference.

Loads and validates YAML configuration files for models and datasets.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path (str): Path to YAML config file
        
    Returns:
        Dict containing configuration
        
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")
    
    if config is None:
        config = {}
    
    return config


def get_config(section: str, key: str, config: Dict[str, Any], default: Any = None) -> Any:
    """
    Get a config value with optional default.
    
    Args:
        section (str): Top-level config section (e.g., 'training', 'data')
        key (str): Key within section (e.g., 'batch_size')
        config (Dict): Configuration dictionary
        default (Any): Default value if key not found
        
    Returns:
        Config value or default
    """
    if section not in config:
        return default
    
    section_config = config[section]
    if isinstance(section_config, dict):
        return section_config.get(key, default)
    
    return default


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override_config into base_config.
    
    Args:
        base_config (Dict): Base configuration
        override_config (Dict): Configuration to merge in (takes precedence)
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
