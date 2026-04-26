#!/usr/bin/env bash
# Unified inference launcher.
# Reads shared avatar runtime GPU settings from cyberverse_config.yaml,
# then auto-selects plain python (single GPU) or torchrun (multi GPU).
# Environment variables still override config values for ad-hoc debugging.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${1:-cyberverse_config.yaml}"

# ── Source .env ──────────────────────────────────────────────────────────────
if [[ -f ./.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# ── Read YAML values (env still wins when set) ──────────────────────────────
_yaml_first_val() {
  python3 - "$CONFIG" "$@" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
paths = sys.argv[2:-1]
default = sys.argv[-1]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}


def lookup(root, dotted):
    value = root
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


for path in paths:
    value = lookup(cfg, path)
    if value is None:
        continue
    if isinstance(value, str) and value.strip() == "":
        continue
    print(value)
    break
else:
    print(default)
PY
}

# Detect active avatar model from config
AVATAR_MODEL="$(_yaml_first_val 'inference.avatar.default' 'flash_head')"
echo "[inference] Active avatar model: ${AVATAR_MODEL}"

if [[ -z "${WORLD_SIZE:-}" ]]; then
  WORLD_SIZE="$(_yaml_first_val \
    "inference.avatar.${AVATAR_MODEL}.world_size" \
    "inference.avatar.runtime.world_size" \
    '1')"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(_yaml_first_val \
    "inference.avatar.${AVATAR_MODEL}.cuda_visible_devices" \
    "inference.avatar.runtime.cuda_visible_devices" \
    '')"
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export WORLD_SIZE CUDA_VISIBLE_DEVICES MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=1
: "${TORCH_CPP_LOG_LEVEL:=WARNING}"
export TORCH_CPP_LOG_LEVEL

# ── torch.compile inductor cache (persists across restarts) ────────────────
: "${TORCHINDUCTOR_CACHE_DIR:=/root/autodl-tmp/cache/torch_inductor}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

# ── Generate protobuf if needed ─────────────────────────────────────────────
./scripts/generate_proto.sh

# ── Single GPU: plain python ────────────────────────────────────────────────
if [[ "${WORLD_SIZE}" -le 1 ]]; then
  echo "[inference] Single GPU mode (world_size=${WORLD_SIZE})"
  [[ -n "${CUDA_VISIBLE_DEVICES}" ]] && echo "[inference] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exec python -m inference.server --config "${CONFIG}"
fi

# ── Multi GPU: torchrun ─────────────────────────────────────────────────────
echo "[inference] Multi-GPU mode: WORLD_SIZE=${WORLD_SIZE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[inference] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# Model-specific distributed tuning defaults
if [[ "${AVATAR_MODEL}" == "flash_head" ]]; then
  : "${FLASHHEAD_MIN_NEW_AUDIO_RATIO:=0.6}"
  if [[ -z "${FLASHHEAD_COMPILE_MODEL:-}" ]]; then
    FLASHHEAD_COMPILE_MODEL="$(_yaml_first_val \
      "inference.avatar.flash_head.compile_model" \
      '1')"
  fi
  if [[ -z "${FLASHHEAD_COMPILE_VAE:-}" ]]; then
    FLASHHEAD_COMPILE_VAE="$(_yaml_first_val \
      "inference.avatar.flash_head.compile_vae" \
      '1')"
  fi
  if [[ -z "${FLASHHEAD_DIST_WORKER_MAIN_THREAD:-}" ]]; then
    FLASHHEAD_DIST_WORKER_MAIN_THREAD="$(_yaml_first_val \
      "inference.avatar.flash_head.dist_worker_main_thread" \
      '1')"
  fi
  export FLASHHEAD_MIN_NEW_AUDIO_RATIO
  export FLASHHEAD_COMPILE_MODEL FLASHHEAD_COMPILE_VAE
  export FLASHHEAD_DIST_WORKER_MAIN_THREAD
  echo "[inference] FLASHHEAD_MIN_NEW_AUDIO_RATIO=${FLASHHEAD_MIN_NEW_AUDIO_RATIO}"
  echo "[inference] Warmup control: ${CONFIG} -> warmup.*"
  echo "[inference] FLASHHEAD_COMPILE_MODEL=${FLASHHEAD_COMPILE_MODEL}"
  echo "[inference] FLASHHEAD_COMPILE_VAE=${FLASHHEAD_COMPILE_VAE}"
  echo "[inference] FLASHHEAD_DIST_WORKER_MAIN_THREAD=${FLASHHEAD_DIST_WORKER_MAIN_THREAD}"
elif [[ "${AVATAR_MODEL}" == "live_act" ]]; then
  # torch.compile currently produces noisy Dynamo/Inductor fallback logs on
  # LiveAct's dynamic-shape warmup path. Keep it off by default in dev mode.
  if [[ -z "${LIVEACT_COMPILE_WAN_MODEL:-}" ]]; then
    LIVEACT_COMPILE_WAN_MODEL="$(_yaml_first_val \
      "inference.avatar.live_act.compile_wan_model" \
      '1')"
  fi
  if [[ -z "${LIVEACT_COMPILE_VAE_DECODE:-}" ]]; then
    LIVEACT_COMPILE_VAE_DECODE="$(_yaml_first_val \
      "inference.avatar.live_act.compile_vae_decode" \
      '1')"
  fi
  if [[ -z "${LIVEACT_DIST_WORKER_MAIN_THREAD:-}" ]]; then
    LIVEACT_DIST_WORKER_MAIN_THREAD="$(_yaml_first_val \
      "inference.avatar.live_act.dist_worker_main_thread" \
      '1')"
  fi
  export LIVEACT_COMPILE_WAN_MODEL LIVEACT_COMPILE_VAE_DECODE
  export LIVEACT_DIST_WORKER_MAIN_THREAD
  echo "[inference] LIVEACT_COMPILE_WAN_MODEL=${LIVEACT_COMPILE_WAN_MODEL}"
  echo "[inference] LIVEACT_COMPILE_VAE_DECODE=${LIVEACT_COMPILE_VAE_DECODE}"
  echo "[inference] LIVEACT_DIST_WORKER_MAIN_THREAD=${LIVEACT_DIST_WORKER_MAIN_THREAD}"
fi

exec torchrun \
  --nnodes=1 \
  --nproc_per_node="${WORLD_SIZE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m inference.server --config "${CONFIG}"
