#! usr/bin/bash
ollama pull llama3.1
python -m pip install -r requirements.txt
python -m pip install langchain-ollama