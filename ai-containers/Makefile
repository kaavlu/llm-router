.PHONY: help vllm-up vllm-down vllm-logs vllm-restart \
        openwebui-up openwebui-down openwebui-logs openwebui-restart \
        ollama-up ollama-down ollama-logs ollama-restart ollama-stop-host \
        all-up all-down all-restart all-logs status \
        app app-bg app-stop up

# Project root: parent of ai-containers (use when running make from ai-containers)
ROOT := $(CURDIR)/..

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "  Ollama (port 11434):"
	@echo "    make ollama-stop-host - Stop Ollama if running via 'ollama serve' on host"
	@echo "    make ollama-up        - Start Ollama container"
	@echo "    make ollama-down      - Stop and remove Ollama container"
	@echo "    make ollama-restart   - Restart Ollama container"
	@echo "    make ollama-logs      - View Ollama logs (follow mode)"
	@echo ""
	@echo "  vLLM:"
	@echo "    make vllm-up        - Start vLLM container"
	@echo "    make vllm-down      - Stop and remove vLLM container"
	@echo "    make vllm-restart   - Restart vLLM container"
	@echo "    make vllm-logs      - View vLLM logs (follow mode)"
	@echo ""
	@echo "  Open WebUI:"
	@echo "    make openwebui-up      - Start Open WebUI container"
	@echo "    make openwebui-down    - Stop and remove Open WebUI container"
	@echo "    make openwebui-restart - Restart Open WebUI container"
	@echo "    make openwebui-logs    - View Open WebUI logs (follow mode)"
	@echo ""
	@echo "  LLM Router app (port 8000):"
	@echo "    make app            - Start router app (foreground)"
	@echo "    make app-bg         - Start router app in background"
	@echo "    make app-stop       - Stop router app (if started with app-bg)"
	@echo ""
	@echo "  All Services:"
	@echo "    make all-up         - Start all containers (Ollama + vLLM + Open WebUI)"
	@echo "    make up             - Start all containers + router app (background)"
	@echo "    make all-down       - Stop and remove all containers"
	@echo "    make all-restart    - Restart all containers"
	@echo "    make all-logs      - View all logs (follow mode)"
	@echo ""
	@echo "  Utility:"
	@echo "    make status         - Show running container status"

# Ollama targets (switch from host 'ollama serve' to container)
ollama-stop-host:
	@echo "Stopping Ollama if running on host (ollama serve)..."
	@pkill -f "ollama serve" 2>/dev/null || true
	@echo "Done. Start container with: make ollama-up"

ollama-up:
	docker compose -f Ollama/ollama-compose.yaml up -d

ollama-down:
	docker compose -f Ollama/ollama-compose.yaml down

ollama-restart: ollama-down ollama-up

ollama-logs:
	docker compose -f Ollama/ollama-compose.yaml logs -f

# vLLM targets
vllm-up:
	docker compose -f vLLM/vllm-compose.yaml up -d

vllm-down:
	docker compose -f vLLM/vllm-compose.yaml down

vllm-restart: vllm-down vllm-up

vllm-logs:
	docker compose -f vLLM/vllm-compose.yaml logs -f

# Open WebUI targets
openwebui-up:
	docker compose -f Open-WebUI/openwebui-compose.yaml up -d

openwebui-down:
	docker compose -f Open-WebUI/openwebui-compose.yaml down

openwebui-restart: openwebui-down openwebui-up

openwebui-logs:
	docker compose -f Open-WebUI/openwebui-compose.yaml logs -f

# LLM Router app (runs from project root)
app:
	cd $(ROOT) && .venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000

app-bg:
	cd $(ROOT) && (.venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000 & echo $$! > .app.pid) && echo "Router app started in background (PID $$(cat $(ROOT)/.app.pid)), http://localhost:8000"

app-stop:
	@if [ -f $(ROOT)/.app.pid ]; then kill $$(cat $(ROOT)/.app.pid) 2>/dev/null; rm -f $(ROOT)/.app.pid; echo "Router app stopped."; else echo "No .app.pid found (app was not started with make app-bg)."; fi

# Start everything: containers + router app in background
up: all-up app-bg

# All services targets
all-up: ollama-up vllm-up openwebui-up

all-down: ollama-down vllm-down openwebui-down

all-restart: all-down all-up

all-logs:
	docker compose -f Ollama/ollama-compose.yaml -f vLLM/vllm-compose.yaml -f Open-WebUI/openwebui-compose.yaml logs -f

# Utility targets
status:
	@echo "=== Container Status ==="
	@docker ps --filter "name=ollama" --filter "name=vllm" --filter "name=openwebui" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
