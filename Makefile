.PHONY: proto setup test test-py test-go test-integration build inference server frontend docker-up docker-down lint clean

# Locate Go 1.25: system package, user SDK install, or PATH fallback
GO ?= $(shell \
  if [ -x /usr/lib/go-1.25/bin/go ]; then echo /usr/lib/go-1.25/bin/go; \
  elif [ -x $(HOME)/sdk/go1.25.9/bin/go ]; then echo $(HOME)/sdk/go1.25.9/bin/go; \
  else echo go; fi)

# C library paths from conda env (opus, soxr, etc. required by livekit media-sdk)
CONDA_ENV     ?= $(HOME)/miniconda3/envs/cyberverse
CONDA_LIB      = $(CONDA_ENV)/lib
CONDA_PKG_CFG  = $(CONDA_LIB)/pkgconfig

# Locate Node 22+ via nvm or PATH fallback
NVM_NODE      := $(HOME)/.nvm/versions/node
NODE_BIN      ?= $(shell \
  found=$$(ls -d $(NVM_NODE)/v22.*/bin/node 2>/dev/null | sort -V | tail -1); \
  if [ -x "$$found" ]; then dirname "$$found"; \
  else echo ""; fi)

# Proto generation (Python + Go)
proto:
	./scripts/generate_proto.sh

# First-time setup (install Python deps before proto so grpc_tools is available)
setup:
	pip install -e ".[dev,inference]"
	$(MAKE) proto
	cd frontend && npm install

# Testing
test: test-py test-go

test-py:
	python -m pytest tests/unit -v

test-go:
	cd server && $(GO) test ./... -v

# 真实 FlashHead 出视频（需 CUDA + 本地 checkpoints，见 tests/integration）
test-integration:
	python -m pytest tests/integration/ -m integration -v -s

# Development servers
#   Reads avatar runtime GPU settings from cyberverse_config.yaml; auto-selects python vs torchrun.
#   Override with env vars for ad-hoc testing:
#     WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 make inference
inference:
	@bash ./scripts/inference.sh

server:
	# Load .env and start livekit-enabled server.
	@PKG_CONFIG_PATH=$(CONDA_PKG_CFG):$$PKG_CONFIG_PATH \
	  LD_LIBRARY_PATH=$(CONDA_LIB):$$LD_LIBRARY_PATH \
	  set -a; [ -f ./.env ] && . ./.env; set +a; \
	  cd server && PKG_CONFIG_PATH=$(CONDA_PKG_CFG):$$PKG_CONFIG_PATH \
	    LD_LIBRARY_PATH=$(CONDA_LIB):$$LD_LIBRARY_PATH \
	    $(GO) run -tags livekit ./cmd/cyberverse-server/ --config ../cyberverse_config.yaml

frontend:
	@if [ -n "$(NODE_BIN)" ]; then export PATH=$(NODE_BIN):$$PATH; fi; cd frontend && npm run dev

# Build
build-go:
	# Build with the "livekit" tag so LiveKit functionality is enabled in production too.
	PKG_CONFIG_PATH=$(CONDA_PKG_CFG):$$PKG_CONFIG_PATH \
	  LD_LIBRARY_PATH=$(CONDA_LIB):$$LD_LIBRARY_PATH \
	  cd server && $(GO) build -tags livekit -o ../bin/cyberverse-server ./cmd/cyberverse-server/

# Docker
docker-up:
	cd infra && docker compose up --build

docker-down:
	cd infra && docker compose down

# Clean generated files
clean:
	rm -f inference/generated/*_pb2*.py
	rm -f server/internal/pb/*.go
	rm -rf bin/
