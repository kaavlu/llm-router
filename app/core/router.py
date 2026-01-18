from app.core.registry import ModelRegistry

class ModelRouter:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def select(self, requested_model: str) -> str:
        return self.registry.resolve(requested_model)
