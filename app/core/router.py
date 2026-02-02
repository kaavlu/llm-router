from app.core.registry import ModelRegistry

class ModelRouter:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def select(self, requested_model: str) -> str:
        """Resolve alias to concrete model (e.g. auto -> default -> ollama/llama3:latest)."""
        return self.registry.resolve_full(requested_model)
