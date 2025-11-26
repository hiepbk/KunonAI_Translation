"""
Config class similar to mmcv Config that supports dot notation access.
Can load from dict, yaml, json, or python file.
"""
from pathlib import Path
from typing import Any, Dict, Union
import yaml
import json


class Config:
    """
    Config class similar to mmcv Config that supports dot notation access.
    Can load from dict, yaml, json, or python file.
    """
    
    def __init__(self, cfg_dict: Dict[str, Any] = None, filename: str = None):
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = self._convert_to_dict(cfg_dict)
        self.filename = filename
        
    def _convert_to_dict(self, cfg: Any) -> Dict[str, Any]:
        """Convert input to dict recursively."""
        if isinstance(cfg, dict):
            return {k: self._convert_to_dict(v) for k, v in cfg.items()}
        elif isinstance(cfg, (list, tuple)):
            return [self._convert_to_dict(item) for item in cfg]
        elif isinstance(cfg, Config):
            return self._convert_to_dict(cfg._cfg_dict)
        else:
            return cfg
    
    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access like config.model_path."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        if name in self._cfg_dict:
            value = self._cfg_dict[name]
            # Wrap dict values in Config for nested access
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access like config['model_path']."""
        return self._cfg_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment."""
        self._cfg_dict[key] = value
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute assignment."""
        if name.startswith('_') or name in ['filename']:
            super().__setattr__(name, value)
        else:
            self._cfg_dict[name] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cfg_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self._cfg_dict.get(key, default)
    
    def keys(self):
        """Return keys."""
        return self._cfg_dict.keys()
    
    def values(self):
        """Return values."""
        return self._cfg_dict.values()
    
    def items(self):
        """Return items."""
        return self._cfg_dict.items()
    
    def update(self, other: Union[Dict, 'Config']) -> None:
        """Update config with another dict or Config."""
        if isinstance(other, Config):
            other = other._cfg_dict
        self._cfg_dict.update(other)
    
    def merge_from_dict(self, options: Dict[str, Any]) -> None:
        """Merge config from a dict, supporting nested keys with dot notation.
        
        Example:
            cfg.merge_from_dict({'model.path': './new/path', 'image.image_size': 1024})
        """
        for key, value in options.items():
            if '.' in key:
                # Handle nested keys like 'model.path'
                keys = key.split('.')
                d = self._cfg_dict
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = value
            else:
                self._cfg_dict[key] = value
    
    def merge_from_list(self, options: list) -> None:
        """Merge config from a list of key-value pairs (like argparse).
        
        Example:
            cfg.merge_from_list(['model.path', './new/path', 'image.image_size', '1024'])
        """
        assert len(options) % 2 == 0, f"Options list must have even number of elements, got {len(options)}"
        options_dict = {}
        for i in range(0, len(options), 2):
            key = options[i]
            value = options[i + 1]
            # Try to convert value to appropriate type
            try:
                # Try int
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    value = int(value)
                # Try float
                elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
                # Try bool
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
            except:
                pass
            options_dict[key] = value
        self.merge_from_dict(options_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict."""
        return self._convert_to_dict(self._cfg_dict)
    
    def dump(self, filepath: str = None, format: str = 'yaml') -> str:
        """Dump config to file or return as string."""
        if format.lower() == 'yaml':
            content = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        elif format.lower() == 'json':
            content = json.dumps(self.to_dict(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(content)
        return content
    
    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dict."""
        return cls(cfg_dict)
    
    @classmethod
    def from_file(cls, filename: str) -> 'Config':
        """Load Config from yaml, json, or python file."""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filename}")
        
        suffix = filepath.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                cfg_dict = yaml.safe_load(f)
        elif suffix == '.json':
            with open(filepath, 'r') as f:
                cfg_dict = json.load(f)
        elif suffix == '.py':
            # Execute python file and get config dict
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Get all non-private, non-callable attributes
            cfg_dict = {}
            for k, v in module.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg_dict[k] = v
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
        
        return cls(cfg_dict, filename=str(filepath))
    
    @staticmethod
    def parse_cfg_options(cfg_options: list) -> Dict[str, Any]:
        """Parse cfg_options list into dict (like mmdet3d).
        
        Args:
            cfg_options: List of key=value strings, e.g., ['model.path=./new/path', 'image.image_size=1024']
            
        Returns:
            Dict with parsed options
        """
        options = {}
        for opt in cfg_options:
            if '=' not in opt:
                raise ValueError(f"Invalid cfg-option format: {opt}. Expected 'key=value'")
            key, value = opt.split('=', 1)
            # Try to convert value to appropriate type
            try:
                # Try int
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    value = int(value)
                # Try float
                elif '.' in value and value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                    value = float(value)
                # Try bool
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # Try list (comma-separated)
                elif ',' in value:
                    value = [v.strip() for v in value.split(',')]
            except:
                pass
            options[key] = value
        return options
    
    def merge_from_dict(self, options: Dict[str, Any]) -> None:
        """Merge config from a dict, supporting nested keys with dot notation.
        
        Example:
            cfg.merge_from_dict({'model.path': './new/path', 'image.image_size': 1024})
        """
        for key, value in options.items():
            if '.' in key:
                # Handle nested keys like 'model.path'
                keys = key.split('.')
                d = self._cfg_dict
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = value
            else:
                self._cfg_dict[key] = value
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config(filename={self.filename}, keys={list(self.keys())})"

