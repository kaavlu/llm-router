from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatCompletionRequest
from app.core.registry import ModelRegistry
from app.core.router import ModelRouter
from litellm import completion
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize model registry and router
_registry_path = Path(__file__).parent.parent / "config" / "model_registry.yaml"
_model_registry = ModelRegistry(str(_registry_path))
_model_router = ModelRouter(_model_registry)

# Global model-to-provider mapping
# Maps model names to their provider
MODEL_TO_PROVIDER = {
    # Ollama models
    "llava:latest": "ollama",
    "gemma3:latest": "ollama",
    "Qwen3-14B-AWQ": "vllm",
}

# Configuration for different providers
# All values MUST be set in .env file - no hardcoded defaults to avoid exposing sensitive data
def _get_provider_config():
    """Load provider configuration from environment variables."""
    ollama_base = os.getenv("OLLAMA_API_BASE")
    vllm_base = os.getenv("VLLM_API_BASE")
    
    if not ollama_base:
        raise ValueError("OLLAMA_API_BASE environment variable is required. Set it in .env file.")
    if not vllm_base:
        raise ValueError("VLLM_API_BASE environment variable is required. Set it in .env file.")
    
    return {
        "ollama": {
            "api_base": ollama_base,
            "env_var": "OLLAMA_API_BASE",
            "api_key": os.getenv("OLLAMA_API_KEY")
        },
        "vllm": {
            "api_base": vllm_base,
            "env_var": "VLLM_API_BASE",
            "api_key": os.getenv("VLLM_API_KEY", "EMPTY")
        }
    }

PROVIDER_CONFIG = _get_provider_config()


def _detect_provider(model: str) -> str:
    """
    Detect the provider from the model name.
    First checks MODEL_TO_PROVIDER mapping, then checks for provider prefix.
    Returns None if provider cannot be determined.
    """
    # Remove provider prefix if present for lookup
    model_lookup = model.split("/", 1)[-1] if "/" in model else model
    
    # Check global model-to-provider mapping
    if model_lookup in MODEL_TO_PROVIDER:
        return MODEL_TO_PROVIDER[model_lookup]
    
    # If model has a provider prefix, extract it
    if "/" in model:
        provider = model.split("/")[0]
        # Check if it's a known provider in config
        if provider in PROVIDER_CONFIG:
            return provider
        # Return the prefix even if not in config (might be a valid LiteLLM provider)
        return provider
    
    # Provider not found
    return None


def _prepare_model_and_config(model: str):
    """
    Prepare the model name and set appropriate environment variables.
    Returns a tuple of (formatted_model_name, api_base, api_key).
    Raises ValueError if provider cannot be determined or is not configured.
    """
    provider = _detect_provider(model)
    
    if provider is None:
        raise ValueError(
            f"Could not determine provider for model '{model}'. "
            f"Please add it to MODEL_TO_PROVIDER mapping."
        )
    
    api_base = None
    api_key = None
    # If provider is configured, set the API base
    if provider in PROVIDER_CONFIG:
        config = PROVIDER_CONFIG[provider]
        api_base = config["api_base"]
        api_key = config.get("api_key")  # Get API key if configured
        # Set environment variable if not already set
        if config["env_var"] not in os.environ:
            os.environ[config["env_var"]] = api_base
        # For vllm, also set OPENAI_API_KEY to empty string if not set (LiteLLM requires it)
        if provider == "vllm" and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = ""
        
        # Extract the base model name (without provider prefix)
        model_name = model.split("/", 1)[-1] if "/" in model else model
        
        # For Ollama and vllm, when api_base is provided, LiteLLM expects just the model name
        # vllm provides an OpenAI-compatible API, so it uses the model name directly
        if provider == "ollama" or provider == "vllm":
            model = model_name  # Just use the model name for Ollama and vllm
        else:
            # For other providers, ensure provider prefix is present
            if not model.startswith(f"{provider}/"):
                model = f"{provider}/{model_name}"
    else:
        # For providers not in config, keep the model as-is (might have provider prefix)
        pass
    
    return model, api_base, api_key


@router.post("/chat/completions")
async def create_chat_completion(payload: ChatCompletionRequest):
    messages = [m.dict() for m in payload.messages]
    
    try:
        # Resolve virtual model name (e.g., "default", "smart", "cheap") to actual model name
        resolved_model = _model_router.select(payload.model)
        logger.info(f"Resolved model '{payload.model}' to '{resolved_model}'")
        
        # Detect provider before preparing model (we need it for custom_llm_provider)
        provider = _detect_provider(resolved_model)
        if provider is None:
            raise ValueError(
                f"Could not determine provider for model '{resolved_model}'. "
                f"Please add it to MODEL_TO_PROVIDER mapping."
            )
        
        # Prepare model name and set API base URL
        model, api_base, api_key = _prepare_model_and_config(resolved_model)
        logger.info(f"Final model name: '{model}', api_base: '{api_base}', provider: '{provider}', api_key: '{api_key}'")
    except ValueError as e:
        # Return error if provider cannot be determined
        logger.error(f"Error preparing model: {e}")
        return {"error": {"message": str(e), "type": "invalid_request_error"}}

    try:
        # Prepare completion kwargs
        completion_kwargs = {
            "model": model,
            "messages": messages,
        }
        if api_base:
            completion_kwargs["api_base"] = api_base
        # Set API key if configured (needed for vllm/OpenAI-compatible endpoints)
        # For vllm, use empty string since local instances don't require auth
        if api_key is not None:
            # Use empty string for vllm, None means don't set it
            if api_key == "EMPTY":
                completion_kwargs["api_key"] = ""
            else:
                completion_kwargs["api_key"] = api_key
        # Explicitly set custom_llm_provider based on detected provider
        # vllm provides an OpenAI-compatible API, so use "openai" as provider
        if provider and provider in PROVIDER_CONFIG:
            if provider == "vllm":
                completion_kwargs["custom_llm_provider"] = "openai"
            else:
                completion_kwargs["custom_llm_provider"] = provider
        
        logger.info(f"Completion kwargs: model={completion_kwargs.get('model')}, api_base={completion_kwargs.get('api_base')}, custom_llm_provider={completion_kwargs.get('custom_llm_provider')}, has_api_key={'api_key' in completion_kwargs}")
        
        # --- STREAMING PATH ---
        if payload.stream:
            completion_kwargs["stream"] = True
            def event_stream():
                # LiteLLM yields ModelResponseStream objects that need to be converted to dict
                try:
                    for chunk in completion(**completion_kwargs):
                        # Convert ModelResponseStream to dict (handles both Pydantic v1 and v2)
                        if hasattr(chunk, "model_dump"):
                            chunk_dict = chunk.model_dump(exclude_none=True)
                        elif hasattr(chunk, "dict"):
                            chunk_dict = chunk.dict(exclude_none=True)
                        elif hasattr(chunk, "json"):
                            # json() method returns dict
                            chunk_dict = chunk.json()
                        else:
                            # Fallback: try to convert to dict directly
                            chunk_dict = dict(chunk) if hasattr(chunk, "__dict__") else chunk
                        
                        # Convert to JSON string
                        yield f"data: {json.dumps(chunk_dict)}\n\n"

                    # End-of-stream sentinel required by OpenAI protocol
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    # Send error in streaming format
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "api_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream"
            )

        # --- NON-STREAMING PATH ---
        completion_kwargs["stream"] = False
        resp = completion(**completion_kwargs)
        return resp
    except Exception as e:
        # Return error for non-streaming requests
        logger.error(f"Error in completion: {e}", exc_info=True)
        return {"error": {"message": str(e), "type": "api_error"}}
