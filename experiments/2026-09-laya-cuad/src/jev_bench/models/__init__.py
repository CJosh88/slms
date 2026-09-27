from .base import ModelClient, Prediction


def get_client(name: str, cfg: dict) -> ModelClient:
    if name == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(**cfg)
    if name == "laya":
        from .laya_client import LayaClient

        return LayaClient(**cfg)
    raise ValueError(f"Unknown model: {name}")


__all__ = ["ModelClient", "Prediction", "get_client"]
