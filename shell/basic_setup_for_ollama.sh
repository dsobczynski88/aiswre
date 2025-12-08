#!/bin/bash

# Check if number of instances is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <number_of_instances>"
    echo "Example: $0 6"
    exit 1
fi

# Store the number of instances
NUM_INSTANCES=$1

# Validate input is a positive integer
if ! [[ "$NUM_INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of instances must be a positive integer"
    exit 1
fi

echo "Setting up Ollama with $NUM_INSTANCES instances..."

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install hardware utilities
echo "Installing hardware utilities..."
sudo apt install -y pciutils lshw

# Start the default Ollama instance (this will be stopped later)
echo "Starting default Ollama instance temporarily..."
ollama serve &
DEFAULT_PID=$!

# Wait for the service to start
sleep 5

# Pull the model once (will be shared across instances)
echo "Pulling llama3.1 model..."
ollama pull llama3.1

# Stop the default instance
echo "Stopping default instance..."
kill $DEFAULT_PID
sleep 2

# Start multiple instances on different ports
echo "Starting $NUM_INSTANCES Ollama instances on different ports..."

for i in $(seq 1 $NUM_INSTANCES); do
    PORT=$((11433 + $i))
    echo "Starting instance $i on port $PORT..."
    OLLAMA_HOST=127.0.0.1:$PORT ollama serve > /dev/null 2>&1 &
    echo "Instance $i started with PID $!"
done

echo "All instances started successfully!"
echo "Instances are running on ports 11434 to $((11433 + $NUM_INSTANCES))"
echo ""
echo "To use a specific instance, set the OLLAMA_HOST environment variable:"
echo "Example: OLLAMA_HOST=127.0.0.1:11435 ollama run llama3.1"
echo ""
echo "To stop all instances, run: pkill -f 'ollama serve'"

exit 0