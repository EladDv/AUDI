# =============================================================================
# AUDI — Type A Makefile
# =============================================================================

.PHONY: help build run stop logs check clean shell restart install-service install-kiosk

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build the Docker image
	docker compose -f docker/docker-compose.yml build

run: ## Start the container (foreground)
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.pi.yml up

start: ## Start the container (detached/daemon)
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.pi.yml up -d

stop: ## Stop the container
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.pi.yml down

logs: ## Tail container logs
	docker compose -f docker/docker-compose.yml logs -f

check: ## Run system check inside container
	docker compose -f docker/docker-compose.yml run --rm audio-guard python3 src/main.py --check

clean: ## Remove container, image, and data volume
	docker compose -f docker/docker-compose.yml down -v
	docker rmi audio-guard:latest 2>/dev/null || true

shell: ## Open a shell in the running container
	docker exec -it audio-guard bash

restart: stop start ## Restart the container

# --- Systemd Service Installation (for auto-start on boot) ---
install-service: ## Install systemd service for auto-start on boot
	sudo cp docker/audio-guard.service /etc/systemd/system/audio-guard.service
	sudo systemctl daemon-reload
	sudo systemctl enable audio-guard.service
	@echo "Service installed. Start with: sudo systemctl start audio-guard"

install-kiosk: ## Install browser kiosk autostart and desktop launcher
	sudo bash scripts/install-kiosk.sh

uninstall-service: ## Remove systemd service
	sudo systemctl stop audio-guard.service 2>/dev/null || true
	sudo systemctl disable audio-guard.service 2>/dev/null || true
	sudo rm -f /etc/systemd/system/audio-guard.service
	sudo systemctl daemon-reload
	@echo "Service uninstalled."

status: ## Check service and container status
	@echo "=== Systemd Service ==="
	-sudo systemctl status audio-guard.service 2>/dev/null | head -20
	@echo ""
	@echo "=== Docker Container ==="
	-docker ps --filter name=audio-guard --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== Web UI ==="
	@echo "http://localhost:8080 (or http://<pi-ip>:8080)"

# --- Dev shortcuts ---
dev-install: ## Install dependencies locally (for dev/test)
	pip install -r requirements.txt

dev-run: ## Run locally without Docker (dev mode)
	python3 src/main.py

dev-check: ## Run system check locally
	python3 src/main.py --check
