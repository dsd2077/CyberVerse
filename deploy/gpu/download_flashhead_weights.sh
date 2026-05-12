#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

mkdir -p "$CHECKPOINT_DIR"

python3 -m pip install --user -U "huggingface_hub[cli]"
export PATH="$HOME/.local/bin:$PATH"

hf download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir "$CHECKPOINT_DIR/SoulX-FlashHead-1_3B"

hf download facebook/wav2vec2-base-960h \
  --local-dir "$CHECKPOINT_DIR/wav2vec2-base-960h"
