import yaml
import os

def load_config(config_path="config/default.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config:
    def __init__(self, config_dict):
        self._config = config_dict

    def __getattr__(self, name):
        if name in self._config:
            val = self._config[name]
            if isinstance(val, dict):
                return Config(val)
            return val
        raise AttributeError(f"Config object has no attribute {name}")

    def get(self, name, default=None):
        return self._config.get(name, default)
