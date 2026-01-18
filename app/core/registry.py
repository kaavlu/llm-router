import yaml
from pathlib import Path

class ModelRegistry:
    def __init__(self, path: str):
        self._path = Path(path)
        self._registry = self._load()

    def _load(self):
        with open(self._path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("models", {})

    def resolve(self, model_name: str) -> str:
        """
        If model_name is a virtual name (e.g. 'default'), return its target.
        Otherwise return model_name unchanged (passthrough).
        """
        return self._registry.get(model_name, model_name)
