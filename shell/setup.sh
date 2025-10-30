#!/bin/bash

# USAGE EXAMPLES (when run from current directory aiswre/shell):
#
# Run with default settings 
#   --> ./setup.sh
# Disable requirements processing 
#   --> ./setup.sh --requirements false
# Specify custom input and output for requirements processor:
#   --> ./setup.sh --input custom_requirements.xlsx --output ./custom_output
# Full example with multiple options
#   --> ./setup.sh --venv-dir custom_env --make-dataset --input my_reqs.xlsx --output ./results
# Show help
#   --> ./setup.sh --help

# Default values
VENV_DIR="venv"
MAKE_DATASET=false
RUN_REQUIREMENTS=true
REQ_INPUT_FILE="sample_requirements.xlsx"
REQ_OUTPUT_DIR="output"

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -v, --venv-dir DIR         Specify virtual environment directory (default: venv)"
    echo "  -d, --make-dataset         Run the make_dataset script"
    echo "  -r, --requirements BOOL    Run the requirements processor (default: true)"
    echo "  -i, --input FILE           Input file for requirements processor (default: sample_requirements.xlsx)"
    echo "  -o, --output DIR           Output directory for requirements processor (default: ./output)"
    echo "  -h, --help                 Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        -d|--make-dataset)
            MAKE_DATASET=true
            shift
            ;;
        -r|--requirements)
            if [[ "$2" == "false" || "$2" == "no" || "$2" == "0" ]]; then
                RUN_REQUIREMENTS=false
            fi
            shift 2
            ;;
        -i|--input)
            REQ_INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            REQ_OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Change to parent directory
cd ..

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists at $VENV_DIR."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install requirements
echo "Installing requirements..."
python -m pip install -r requirements.txt
python -m pip install pandas openpyxl flatdict ollama
python -m pip install -e .

# Create output directory if it doesn't exist
mkdir -p scripts/output

# Run make_dataset script if requested
if [ "$MAKE_DATASET" = true ]; then
    echo "Running make_dataset script..."
    python -m scripts.make_dataset
fi

# Run requirements processor if requested
if [ "$RUN_REQUIREMENTS" = true ]; then
    echo "Running requirements processor..."
    echo "Input file: $REQ_INPUT_FILE"
    echo "Output directory: $REQ_OUTPUT_DIR"
    python -m scripts.requirements_processor "$REQ_INPUT_FILE" "../$REQ_OUTPUT_DIR"
fi

echo "Script completed successfully."