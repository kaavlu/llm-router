# LLM Router

Routes chat requests to **default**, **smart**, or **cheap** models via aliases. Supports an **auto** alias that resolves to the default model (extend router logic to pick by prompt/load if needed).

## Model registry

`app/config/model_registry.yaml`:

- **default** → `ollama/llama3:latest`
- **smart** → `vllm/Qwen/Qwen3-14B-AWQ`
- **cheap** → `ollama/gemma3:latest`
- **auto** → `default` (full resolution: auto → default → ollama/llama3:latest)

## Running

1. Set `.env`: `OLLAMA_API_BASE`, `VLLM_API_BASE` (and optional API keys).
2. Start backends (Ollama, vLLM) as needed.
3. Start the API:

   ```bash
   uvicorn app.server:app --reload
   ```

## Testing

### Unit + API tests (no backends)

From project root (with venv activated or `pip install -r requirements.txt`):

```bash
pytest tests/ -v
```

- **Router/registry**: alias resolution for `default`, `smart`, `cheap`, `auto` and passthrough.
- **Chat API**: each alias is resolved and LiteLLM is called with the correct model (LiteLLM is mocked).

### Live test (server + backends running)

1. Start the server and your backends.
2. Run the live script:

```bash
python scripts/test_models_live.py
# optional: --base http://localhost:8000  --stream
```

### cURL examples

Server base: `http://localhost:8000`. Replace if different.

**Default (Llama 3):**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello."}],"stream":false}'
```

**Smart (Qwen via vLLM):**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smart","messages":[{"role":"user","content":"Say hello."}],"stream":false}'
```

**Cheap (Gemma 3):**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cheap","messages":[{"role":"user","content":"Say hello."}],"stream":false}'
```

**Auto (resolves to default):**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello."}],"stream":false}'
```

**Streaming (any alias):**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello."}],"stream":true}'
```
