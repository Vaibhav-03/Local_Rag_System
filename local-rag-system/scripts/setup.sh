#!/bin/bash
# =============================================================================
# Local RAG System - Setup Script
# =============================================================================
# This script sets up the complete environment for running the Local RAG System.
# 
# Prerequisites:
#   - Python 3.9+ installed
#   - pip installed
#   - ~4GB disk space for model
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
#
# Tested on:
#   - Ubuntu 22.04 LTS
#   - macOS 13+ (Intel and Apple Silicon)
# =============================================================================

set -e  # Exit on error


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              LOCAL RAG SYSTEM - SETUP SCRIPT                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# -----------------------------------------------------------------------------
# 1. Check Python version
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}Error: Python not found. Please install Python 3.9+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# -----------------------------------------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"

cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    echo -e "${BLUE}Virtual environment already exists, skipping creation${NC}"
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi


source venv/bin/activate


pip install --upgrade pip -q

# -----------------------------------------------------------------------------
# 3. Install dependencies
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/6] Installing Python dependencies...${NC}"


pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

# -----------------------------------------------------------------------------
# 4. Create necessary directories
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/6] Creating directories...${NC}"

mkdir -p "$PROJECT_DIR/corpus"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/tests"

echo -e "${GREEN}✓ Directories created${NC}"

# -----------------------------------------------------------------------------
# 5. Download sample model (optional)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/6] Model setup...${NC}"

MODEL_DIR="$PROJECT_DIR/models"
MODEL_FILE="$MODEL_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo -e "${GREEN}✓ Model already downloaded${NC}"
else
    echo -e "${BLUE}No model found. You have two options:${NC}"
    echo ""
    echo "  Option A: Download Mistral 7B Instruct (recommended, ~4.4GB)"
    echo "    wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    echo ""
    echo "  Option B: Download a smaller model (~2GB)"
    echo "    wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    echo ""
    echo -e "${YELLOW}Would you like to download Mistral 7B Instruct now? (y/n)${NC}"
    
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Downloading model... (this may take several minutes)"
        if command -v wget &> /dev/null; then
            wget -P "$MODEL_DIR" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        elif command -v curl &> /dev/null; then
            curl -L -o "$MODEL_FILE" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        else
            echo -e "${RED}Neither wget nor curl found. Please download manually.${NC}"
        fi
        
        if [ -f "$MODEL_FILE" ]; then
            echo -e "${GREEN}✓ Model downloaded successfully${NC}"
        fi
    else
        echo -e "${BLUE}Skipping model download. You can run with --mock flag to test.${NC}"
    fi
fi

# -----------------------------------------------------------------------------
# 6. Create sample corpus
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[6/6] Setting up sample corpus...${NC}"

SAMPLE_FILE="$PROJECT_DIR/corpus/sample_knowledge_base.txt"

if [ ! -f "$SAMPLE_FILE" ]; then
    cat > "$SAMPLE_FILE" << 'EOF'
# Sample Knowledge Base for Local RAG System

## Introduction to RAG (Retrieval-Augmented Generation)

Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge retrieval. Instead of relying solely on the knowledge encoded in model weights, RAG systems retrieve relevant information from a corpus before generating responses.

### How RAG Works

1. **Query Processing**: The user's query is first processed and potentially refined.
2. **Retrieval**: Relevant documents are retrieved from a knowledge base using similarity search.
3. **Context Assembly**: Retrieved documents are combined into a context for the language model.
4. **Generation**: The language model generates a response using both the query and retrieved context.

### Benefits of RAG

- **Up-to-date information**: The knowledge base can be updated without retraining the model.
- **Verifiable sources**: Responses can cite specific documents.
- **Domain specificity**: The system can be specialized by changing the corpus.
- **Reduced hallucination**: Grounding responses in retrieved documents improves accuracy.

## Vector Embeddings

Vector embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling efficient similarity search.

### Popular Embedding Models

- **all-MiniLM-L6-v2**: Small, fast, and efficient (384 dimensions)
- **all-mpnet-base-v2**: Higher quality, more compute (768 dimensions)
- **text-embedding-ada-002**: OpenAI's embedding model (1536 dimensions)

### FAISS (Facebook AI Similarity Search)

FAISS is a library for efficient similarity search. It provides:
- Fast nearest neighbor search
- Support for billion-scale datasets
- Both exact and approximate search algorithms
- CPU and GPU implementations

## Quantization

Quantization reduces the precision of model weights to decrease memory usage and improve speed.

### Types of Quantization

- **INT8**: 8-bit integer quantization (~4x smaller)
- **INT4**: 4-bit integer quantization (~8x smaller)
- **Mixed precision**: Different precision for different layers

### GGUF Format

GGUF (GPT-Generated Unified Format) is a file format for quantized LLMs:
- Supported by llama.cpp
- Efficient memory mapping
- Various quantization levels (Q4, Q5, Q8, etc.)
- Metadata for model configuration

## Language Models

### Mistral 7B

Mistral 7B is a 7 billion parameter language model known for:
- Strong performance for its size
- Efficient inference
- Good instruction following
- Open weights and permissive license

### Llama 2

Llama 2 is Meta's open-source language model family:
- Available in 7B, 13B, and 70B sizes
- Chat-optimized variants available
- Commercial use permitted

## Best Practices for RAG

1. **Chunk size**: Balance between context and specificity (300-1000 tokens)
2. **Overlap**: Use overlapping chunks to preserve context
3. **Retrieval count**: Retrieve 3-5 documents typically
4. **Relevance threshold**: Filter out low-relevance results
5. **Prompt engineering**: Structure prompts to use context effectively
EOF
    echo -e "${GREEN}✓ Sample corpus created${NC}"
else
    echo -e "${BLUE}Sample corpus already exists${NC}"
fi

# -----------------------------------------------------------------------------
# Setup complete
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE! ✓                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "Next steps:"
echo ""
echo -e "  1. Activate the virtual environment:"
echo -e "     ${BLUE}source venv/bin/activate${NC}"
echo ""
echo -e "  2. Build the document index:"
echo -e "     ${BLUE}python main.py index${NC}"
echo ""
echo -e "  3. Start the interactive CLI:"
if [ -f "$MODEL_FILE" ]; then
    echo -e "     ${BLUE}python main.py --model models/mistral-7b-instruct-v0.2.Q4_K_M.gguf${NC}"
else
    echo -e "     ${BLUE}python main.py --mock${NC}  (test mode, no real model)"
fi
echo ""
echo -e "  For more options:"
echo -e "     ${BLUE}python main.py --help${NC}"
echo ""

