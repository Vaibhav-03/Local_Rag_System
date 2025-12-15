#!/bin/bash
# =============================================================================
# Setup Script for 8GB RAM M1 Mac
# =============================================================================
# Optimized for memory-constrained Apple Silicon Macs
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         LOCAL RAG SYSTEM - 8GB M1 MAC SETUP                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Apple Silicon (M1/M2/M3)${NC}"
fi

# Check Python
echo -e "${YELLOW}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found${NC}"

# Create venv
echo -e "${YELLOW}[2/5] Creating virtual environment...${NC}"
cd "$PROJECT_DIR"
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
echo -e "${GREEN}✓ Virtual environment ready${NC}"

# Install dependencies
echo -e "${YELLOW}[3/5] Installing dependencies...${NC}"
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directories
echo -e "${YELLOW}[4/5] Creating directories...${NC}"
mkdir -p "$PROJECT_DIR/corpus"
mkdir -p "$PROJECT_DIR/models"
echo -e "${GREEN}✓ Directories created${NC}"

# Download TinyLlama (small model for 8GB)
echo -e "${YELLOW}[5/5] Downloading TinyLlama model (~670MB)...${NC}"
MODEL_FILE="$PROJECT_DIR/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo -e "${GREEN}✓ Model already downloaded${NC}"
else
    echo "Downloading TinyLlama (optimized for 8GB RAM)..."
    if command -v curl &> /dev/null; then
        curl -L -o "$MODEL_FILE" \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    elif command -v wget &> /dev/null; then
        wget -O "$MODEL_FILE" \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    else
        echo -e "${RED}Error: Neither curl nor wget found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Model downloaded${NC}"
fi

# Create sample corpus if not exists
if [ ! -f "$PROJECT_DIR/corpus/sample_knowledge_base.txt" ]; then
    cp "$PROJECT_DIR/corpus/sample_knowledge_base.txt" "$PROJECT_DIR/corpus/" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE! ✓                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "To start the RAG system:"
echo ""
echo -e "  ${BLUE}source venv/bin/activate${NC}"
echo -e "  ${BLUE}python main.py index${NC}"
echo -e "  ${BLUE}python main.py --config config-8gb.yaml${NC}"
echo ""
echo -e "${YELLOW}Tip: Close other apps to free up RAM before running.${NC}"
echo ""

