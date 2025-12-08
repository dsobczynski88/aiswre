
#!/usr/bin/env bash
set -euo pipefail

# Usage and examples
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <number_of_instances> [verify]"
  echo "Example: $0 6"
  echo "Example (with basic verification): $0 4 verify"
  exit 1
fi

NUM_INSTANCES="$1"
VERIFY_MODE="${2:-}"

if ! [[ "$NUM_INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: Number of instances must be a positive integer" >&2
  exit 1
fi

echo "Setting up Ollama with $NUM_INSTANCES instances..."

# --- Detect GPUs ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver/toolkit and retry." >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | awk '{print $1}')"
echo "Detected GPUs: $GPU_COUNT"

# Strategy when instances > GPUs:
#   round_robin=true  -> cycle across available GPUs
#   round_robin=false -> extra instances run CPU-only (CUDA_VISIBLE_DEVICES=-1)
round_robin=true

# --- Install Ollama (if not present) ---
if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Optional OS deps (already fine on most systems)
echo "Updating system packages & utilities..."
sudo apt update -y
sudo apt install -y pciutils lshw curl jq

# --- Pull the model once (will be reused across instances) ---
# Start a temporary server to ensure the CLI can reach it, then pull and stop.
echo "Starting temporary Ollama server to pull models..."
OLLAMA_HOST=127.0.0.1:11433 ollama serve >/dev/null 2>&1 &
TEMP_PID=$!
sleep 3

echo "Pulling base model llama3.1..."
OLLAMA_HOST=127.0.0.1:11433 ollama pull llama3.1 || true

echo "Stopping temporary server..."
kill "$TEMP_PID" || true
sleep 1

# --- Start multiple instances pinned to GPUs ---
BASE_PORT=11434
PIDS=()
PORTS=()
MAP_REPORT=()

echo "Starting $NUM_INSTANCES Ollama instances on different ports with GPU pinning..."

for i in $(seq 1 "$NUM_INSTANCES"); do
  PORT=$((BASE_PORT + i - 1))
  PORTS+=("$PORT")

  # Choose GPU for this instance
  GPU_ID="-1"  # default CPU-only
  if (( GPU_COUNT > 0 )); then
    if (( i <= GPU_COUNT )); then
      GPU_ID=$((i - 1))  # 0..GPU_COUNT-1 (one instance per GPU)
    else
      if [[ "$round_robin" == "true" ]]; then
        GPU_ID=$(( (i - 1) % GPU_COUNT ))  # cycle GPUs
      else
        GPU_ID="-1"  # CPU-only for extra instances
      fi
    fi
  fi

  # Start the server pinned to selected GPU (or CPU with -1)
  echo "Starting instance $i on port $PORT (CUDA_VISIBLE_DEVICES=${GPU_ID}) ..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  OLLAMA_HOST="127.0.0.1:${PORT}" \
  OLLAMA_KEEP_ALIVE="30m" \
  nohup ollama serve >"/tmp/ollama_${PORT}.log" 2>&1 &

  PID=$!
  PIDS+=("$PID")
  MAP_REPORT+=("instance=$i port=$PORT gpu=${GPU_ID} pid=$PID")
done

echo "All instances started."
printf '%s\n' "${MAP_REPORT[@]}"

echo
echo "Instances are running on ports ${BASE_PORT} to $((BASE_PORT + NUM_INSTANCES - 1))."
echo "To use a specific instance, set OLLAMA_HOST and run a model, e.g.:"
echo "  OLLAMA_HOST=127.0.0.1:${BASE_PORT} ollama run llama3.1 'hello'"
echo

# --- Optional basic verification per port ---
if [[ "$VERIFY_MODE" == "verify" ]]; then
  echo "Verifying ports with /api/version and reporting processor via 'ollama ps'..."
  for PORT in "${PORTS[@]}"; do
    echo "  Port ${PORT}:"
    if curl -sS "http://127.0.0.1:${PORT}/api/version" >/dev/null; then
      echo "    Reachable ✓"
      # Show processor allocation if any model is loaded
      OLLAMA_HOST="127.0.0.1:${PORT}" ollama run llama3.1 >/dev/null || true
      PROC_LINE="$(OLLAMA_HOST="127.0.0.1:${PORT}" ollama ps 2>/dev/null | sed -n '2p' || true)"
      if [[ -n "$PROC_LINE" ]]; then
        echo "    ps: $PROC_LINE"
      else
        echo "    ps: (no models loaded yet)"
      fi
    else
      echo "    Reachable ✗"
    fi
  done
fi

echo
echo "To stop all instances: pkill -f 'ollama serve'"
exit 0