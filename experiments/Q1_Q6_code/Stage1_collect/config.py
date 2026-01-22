import os
import json
from typing import Any, Dict, Optional

# Default configuration if file is missing
DEFAULT_CONFIG = {
    "model_id": "Qwen/Qwen1.5-MoE-A2.7B",
    "dataset_config": {
        "source": "huggingface",
        "name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text"
    },
    "max_samples": -1,
    "batch_size": 4,
    "max_length": 2048
}

CONFIG_FILE = "config/qwen_config.json"


def _load_config(config_file):
    """Load a JSON config file and validate required keys."""
    if not os.path.exists(config_file):
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        required_keys = [
            "model_id",
            "dataset_config",
            "max_samples",
            "batch_size",
            "max_length",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None


def load_exp_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """加载项目配置。优先使用传入路径，其次尝试默认路径，最后使用内置默认值。"""
    
    config = None
    
    # 1. 尝试传入的路径
    if config_path and os.path.exists(config_path):
        config = _load_config(config_path)
    
    # 2. 尝试默认文件路径
    if config is None:
        default_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
        if os.path.exists(default_path):
            config = _load_config(default_path)
            
    # 3. 使用内置默认值
    if config is None:
        print("Warning: Config file not found. Using default configuration.")
        config = DEFAULT_CONFIG.copy()

    # 验证并设置 MAX_SAMPLES 默认值
    max_samples = config.get("max_samples")
    if not isinstance(max_samples, int) or max_samples <= 0:
        config["max_samples"] = None  # 不限制

    return config
