#! usr/bin/bash
curl -fsSL https://ollama.com/install.sh | sh
sudo apt update
sudo apt install pciutils lshw
ollama serve