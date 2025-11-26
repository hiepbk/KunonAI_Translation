"""
Config module - provides Config class and loads default config.
"""
from deepseekocr_net.utils.config import Config
from . import config

# Load default config from config.py
cfg = Config.from_file(__file__.replace('__init__.py', 'config.py'))
