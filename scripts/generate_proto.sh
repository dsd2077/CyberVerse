#!/bin/bash
# Generate Python gRPC code from proto files
# Works on both macOS and Linux
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_DIR="$REPO_ROOT/server"
PROTO_DIR="$REPO_ROOT/proto"
OUT_DIR="$REPO_ROOT/inference/generated"
GO_OUT_DIR="$REPO_ROOT/server/internal/pb"

# protoc is not a Go module; pin for reproducible server/internal/pb headers.
# protobuf 5.29.x prints "libprotoc 29.3" from `protoc --version`.
REQUIRED_LIBPROTOC_VERSION="29.3"

verify_protoc_for_go() {
    if ! command -v protoc &>/dev/null; then
        echo "ERROR: protoc not found in PATH (required for Go proto generation)." >&2
        exit 1
    fi
    local pv
    pv=$(protoc --version 2>&1 | tr -d '\r')
    if [[ "$pv" != "libprotoc ${REQUIRED_LIBPROTOC_VERSION}" ]]; then
        echo "ERROR: protoc version mismatch for reproducible Go codegen." >&2
        echo "  Expected: libprotoc ${REQUIRED_LIBPROTOC_VERSION} (protobuf 5.29.3)" >&2
        echo "  Got:      ${pv}" >&2
        exit 1
    fi
}

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python python3; do
        if command -v "$candidate" &> /dev/null && "$candidate" -c 'import grpc_tools.protoc' &> /dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "grpc_tools is required: install grpcio-tools or set PYTHON=/path/to/python"
    exit 1
fi

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR"/*.proto

# Fix imports to use absolute paths (cross-platform sed)
fix_imports() {
    local file="$1"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/^import \([a-z_]*\)_pb2/from inference.generated import \1_pb2/' "$file"
    else
        sed -i 's/^import \([a-z_]*\)_pb2/from inference.generated import \1_pb2/' "$file"
    fi
}

for f in "$OUT_DIR"/*_pb2_grpc.py "$OUT_DIR"/*_pb2.py; do
    fix_imports "$f"
done

echo "Python proto generation complete: $OUT_DIR"

# Generate Go gRPC code (plugin versions: server/go.mod tool block)
mkdir -p "$GO_OUT_DIR"

if command -v go &> /dev/null && [[ -f "$SERVER_DIR/go.mod" ]]; then
    verify_protoc_for_go
    PLUGIN_GEN_GO=$(go -C "$SERVER_DIR" tool -n protoc-gen-go)
    PLUGIN_GEN_GO_GRPC=$(go -C "$SERVER_DIR" tool -n protoc-gen-go-grpc)
    protoc -I "$PROTO_DIR" \
        --plugin=protoc-gen-go="$PLUGIN_GEN_GO" \
        --plugin=protoc-gen-go-grpc="$PLUGIN_GEN_GO_GRPC" \
        --go_out="$GO_OUT_DIR" --go_opt=paths=source_relative \
        --go-grpc_out="$GO_OUT_DIR" --go-grpc_opt=paths=source_relative \
        "$PROTO_DIR"/*.proto
    echo "Go proto generation complete: $GO_OUT_DIR"
else
    echo "Skipping Go proto generation: go not found or missing $SERVER_DIR/go.mod"
    echo "Go plugins are pinned in server/go.mod (tool); with Go installed, versions follow that file."
    echo "protoc for Go codegen: protobuf 5.29.3 (protoc --version => libprotoc ${REQUIRED_LIBPROTOC_VERSION})"
fi
