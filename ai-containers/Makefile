.PHONY: help vllm-up vllm-down vllm-logs vllm-restart \
        openwebui-up openwebui-down openwebui-logs openwebui-restart \
        all-up all-down all-restart all-logs status

# Default target
help:
	@echo "Available commands:"
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
	@echo "  All Services:"
	@echo "    make all-up         - Start all containers"
	@echo "    make all-down       - Stop and remove all containers"
	@echo "    make all-restart    - Restart all containers"
	@echo "    make all-logs       - View all logs (follow mode)"
	@echo ""
	@echo "  Utility:"
	@echo "    make status         - Show running container status"

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

# All services targets
all-up: vllm-up openwebui-up

all-down: vllm-down openwebui-down

all-restart: all-down all-up

all-logs:
	docker compose -f vLLM/vllm-compose.yaml -f Open-WebUI/openwebui-compose.yaml logs -f

# Utility targets
status:
	@echo "=== Container Status ==="
	@docker ps --filter "name=vllm" --filter "name=openwebui" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
