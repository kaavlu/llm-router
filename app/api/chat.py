from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatCompletionRequest
from litellm import completion
import json
import os

router = APIRouter()

# Global model-to-provider mapping
# Maps model names to their provider
MODEL_TO_PROVIDER = {
    # Ollama models
    "llava:latest": "ollama",
    "gemma3:latest": "ollama",
}

# Configuration for different providers
PROVIDER_CONFIG = {
    "ollama": {
        "api_base": "http://127.0.0.1:11434",
        "env_var": "OLLAMA_API_BASE"
    }
}


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
    Returns the formatted model name.
    Raises ValueError if provider cannot be determined or is not configured.
    """
    provider = _detect_provider(model)
    
    if provider is None:
        raise ValueError(
            f"Could not determine provider for model '{model}'. "
            f"Please add it to MODEL_TO_PROVIDER mapping."
        )
    
    # If provider is configured, set the API base
    if provider in PROVIDER_CONFIG:
        config = PROVIDER_CONFIG[provider]
        # Set environment variable if not already set
        if config["env_var"] not in os.environ:
            os.environ[config["env_var"]] = config["api_base"]
        
        # If model doesn't have provider prefix, add it for LiteLLM
        # Remove any existing prefix first to avoid duplicates
        model_name = model.split("/", 1)[-1] if "/" in model else model
        if not model.startswith(f"{provider}/"):
            model = f"{provider}/{model_name}"
    
    return model


@router.post("/chat/completions")
async def create_chat_completion(payload: ChatCompletionRequest):
    messages = [m.dict() for m in payload.messages]
    
    try:
        # Prepare model name and set API base URL
        model = _prepare_model_and_config(payload.model)
    except ValueError as e:
        # Return error if provider cannot be determined
        return {"error": {"message": str(e), "type": "invalid_request_error"}}

    try:
        # --- STREAMING PATH ---
        if payload.stream:
            def event_stream():
                # LiteLLM yields ModelResponseStream objects that need to be converted to dict
                try:
                    for chunk in completion(
                        model=model,
                        messages=messages,
                        stream=True
                    ):
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
        resp = completion(
            model=model,
            messages=messages,
            stream=False
        )
        return resp
    except Exception as e:
        # Return error for non-streaming requests
        return {"error": {"message": str(e), "type": "api_error"}}
