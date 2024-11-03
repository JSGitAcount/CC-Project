import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
"""A path variable to get access to other directories coming with the package. For example,
```python 
from moonlander_environment import ROOT_DIR

standard_environment_config_path = ROOT_DIR / "test_data" / "levels" / "standard_config.yaml"
```
"""
