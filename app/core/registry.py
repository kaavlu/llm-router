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

    def resolve_full(self, model_name: str) -> str:
        """
        Fully resolve aliases (e.g. 'auto' -> 'default' -> 'ollama/llama3:latest').
        Stops when the result is not an alias. Detects circular aliases.
        """
        seen = set()
        current = model_name
        while current in self._registry:
            if current in seen:
                raise ValueError(f"Circular alias in model registry: {current}")
            seen.add(current)
            current = self._registry[current]
        return current
