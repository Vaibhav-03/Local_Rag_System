#!/bin/bash
# =============================================================================
# Download Model Script
# =============================================================================
# Downloads recommended GGUF models for the Local RAG System.
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
MODEL_DIR="$PROJECT_DIR/models"

mkdir -p "$MODEL_DIR"

echo "Available models:"
echo ""
echo "  1. Mistral 7B Instruct v0.2 Q4_K_M (~4.4GB) - Recommended"
echo "     Best balance of quality and speed"
echo ""
echo "  2. TinyLlama 1.1B Chat Q4_K_M (~670MB) - Lightweight"
echo "     Faster but lower quality"
echo ""
echo "  3. Phi-2 Q4_K_M (~1.6GB) - Small but capable"
echo "     Good for resource-constrained systems"
echo ""

read -p "Select model (1-3): " choice

case $choice in
    1)
        MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        MODEL_NAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ;;
    2)
        MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        MODEL_NAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ;;
    3)
        MODEL_URL="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
        MODEL_NAME="phi-2.Q4_K_M.gguf"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Downloading $MODEL_NAME..."
echo "This may take several minutes depending on your connection."
echo ""

if command -v wget &> /dev/null; then
    wget -c -P "$MODEL_DIR" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -C - -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
else
    echo "Error: Neither wget nor curl found. Please install one."
    exit 1
fi

echo ""
echo "✓ Model downloaded to: $MODEL_DIR/$MODEL_NAME"
echo ""
echo "To use this model, run:"
echo "  python main.py --model models/$MODEL_NAME"

