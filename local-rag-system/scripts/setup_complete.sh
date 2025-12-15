#!/bin/bash
# =============================================================================
# Local RAG System - Complete End-to-End Setup Script
# =============================================================================
# 
# This script performs a fully automated setup of the Local RAG System,
# including environment configuration, dependency installation, model download,
# index building, and optional evaluation.
#
# Usage:
#   chmod +x scripts/setup_complete.sh
#   ./scripts/setup_complete.sh [OPTIONS]
#
# Options:
#   --full          Full setup including BioASQ evaluation dataset
#   --lightweight   Use TinyLlama (smaller, faster, lower quality)
#   --skip-model    Skip model download (if you have your own)
#   --run-eval      Run evaluation after setup
#   --help          Show this help message
#
# Prerequisites:
#   - Python 3.9+ installed
#   - pip installed
#   - ~5-10GB disk space (depending on model choice)
#   - Internet connection for downloads
#
# Tested on:
#   - Ubuntu 22.04 LTS
#   - macOS 13+ (Intel and Apple Silicon)
#   - WSL2 (Windows)
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default options
FULL_SETUP=false
LIGHTWEIGHT=false
SKIP_MODEL=false
RUN_EVAL=false

# Model options
MISTRAL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MISTRAL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
TINYLLAMA_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TINYLLAMA_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║           🔍 LOCAL RAG SYSTEM - COMPLETE SETUP 🔍                  ║"
    echo "║      Retrieval-Augmented Generation on Your Laptop                 ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[$1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

show_help() {
    cat << EOF
Local RAG System - Complete Setup Script

Usage: ./scripts/setup_complete.sh [OPTIONS]

Options:
  --full          Full setup including BioASQ evaluation dataset and run evaluation
  --lightweight   Use TinyLlama model (~670MB) instead of Mistral (~4.4GB)
  --skip-model    Skip model download (use if you have your own model)
  --run-eval      Run BioASQ evaluation after setup
  --help          Show this help message

Examples:
  # Standard setup with Mistral 7B
  ./scripts/setup_complete.sh

  # Quick setup with smaller model
  ./scripts/setup_complete.sh --lightweight

  # Full setup with evaluation
  ./scripts/setup_complete.sh --full

  # Setup without model download
  ./scripts/setup_complete.sh --skip-model
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_SETUP=true
            RUN_EVAL=true
            shift
            ;;
        --lightweight)
            LIGHTWEIGHT=true
            shift
            ;;
        --skip-model)
            SKIP_MODEL=true
            shift
            ;;
        --run-eval)
            RUN_EVAL=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Calculate total steps
TOTAL_STEPS=7
if [ "$FULL_SETUP" = true ]; then
    TOTAL_STEPS=9
fi
if [ "$RUN_EVAL" = true ]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi

# =============================================================================
# Main Setup
# =============================================================================

print_banner

echo -e "${MAGENTA}Setup Configuration:${NC}"
echo -e "  • Full setup: ${BOLD}$FULL_SETUP${NC}"
echo -e "  • Lightweight mode: ${BOLD}$LIGHTWEIGHT${NC}"
echo -e "  • Skip model: ${BOLD}$SKIP_MODEL${NC}"
echo -e "  • Run evaluation: ${BOLD}$RUN_EVAL${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check System Requirements
# -----------------------------------------------------------------------------
print_step 1 "Checking system requirements"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    print_error "Python not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    print_error "Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi
print_success "Python $PYTHON_VERSION found"

# Check pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip not found. Please install pip"
    exit 1
fi
print_success "pip found"

# Check wget or curl
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -c -P"
    print_success "wget found"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -C - -o"
    print_success "curl found"
else
    print_warning "Neither wget nor curl found. Model download may fail."
fi

# Check available memory
if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        print_warning "Only ${TOTAL_MEM}GB RAM detected. 8GB+ recommended."
        if [ "$LIGHTWEIGHT" = false ]; then
            print_info "Consider using --lightweight flag for better performance"
        fi
    else
        print_success "${TOTAL_MEM}GB RAM detected"
    fi
elif [ "$(uname)" = "Darwin" ]; then
    TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    print_success "${TOTAL_MEM}GB RAM detected"
fi

# Check available disk space
AVAILABLE_SPACE=$(df -BG "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G' || echo "unknown")
if [ "$AVAILABLE_SPACE" != "unknown" ]; then
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        print_warning "Only ${AVAILABLE_SPACE}GB disk space available. 10GB+ recommended."
    else
        print_success "${AVAILABLE_SPACE}GB disk space available"
    fi
fi

# -----------------------------------------------------------------------------
# Step 2: Create Virtual Environment
# -----------------------------------------------------------------------------
print_step 2 "Creating Python virtual environment"

cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Activated existing virtual environment"
    else
        rm -rf venv
        $PYTHON_CMD -m venv venv
        source venv/bin/activate
        print_success "Recreated and activated virtual environment"
    fi
else
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    print_success "Created and activated virtual environment"
fi

# Upgrade pip
pip install --upgrade pip -q
print_success "pip upgraded"

# -----------------------------------------------------------------------------
# Step 3: Install Dependencies
# -----------------------------------------------------------------------------
print_step 3 "Installing Python dependencies"

print_info "This may take a few minutes..."

# Install main dependencies
pip install -r requirements.txt

print_success "All dependencies installed"

# Verify key packages
python -c "import sentence_transformers; print('  sentence-transformers:', sentence_transformers.__version__)" 2>/dev/null || print_warning "sentence-transformers not found"
python -c "import faiss; print('  faiss-cpu: installed')" 2>/dev/null || print_warning "faiss-cpu not found"
python -c "import llama_cpp; print('  llama-cpp-python:', llama_cpp.__version__)" 2>/dev/null || print_warning "llama-cpp-python not found"

# -----------------------------------------------------------------------------
# Step 4: Create Directory Structure
# -----------------------------------------------------------------------------
print_step 4 "Creating directory structure"

mkdir -p "$PROJECT_DIR/corpus"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/data/bioasq"
mkdir -p "$PROJECT_DIR/tests"

print_success "Directory structure created:"
echo "  ├── corpus/       (your documents)"
echo "  ├── models/       (GGUF models + indices)"
echo "  ├── data/bioasq/  (evaluation dataset)"
echo "  └── tests/        (test files)"

# -----------------------------------------------------------------------------
# Step 5: Download Model
# -----------------------------------------------------------------------------
print_step 5 "Setting up language model"

MODEL_DIR="$PROJECT_DIR/models"

if [ "$SKIP_MODEL" = true ]; then
    print_info "Skipping model download (--skip-model flag)"
    print_info "Make sure to specify your model path in config.yaml"
else
    if [ "$LIGHTWEIGHT" = true ]; then
        MODEL_URL="$TINYLLAMA_URL"
        MODEL_FILE="$TINYLLAMA_FILE"
        MODEL_SIZE="670MB"
    else
        MODEL_URL="$MISTRAL_URL"
        MODEL_FILE="$MISTRAL_FILE"
        MODEL_SIZE="4.4GB"
    fi
    
    if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
        print_success "Model already exists: $MODEL_FILE"
    else
        print_info "Downloading $MODEL_FILE ($MODEL_SIZE)..."
        print_info "This may take several minutes depending on your connection."
        
        if command -v wget &> /dev/null; then
            wget -c -P "$MODEL_DIR" "$MODEL_URL"
        elif command -v curl &> /dev/null; then
            curl -L -C - -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
        else
            print_error "Cannot download model: neither wget nor curl available"
            print_info "Please download manually from: $MODEL_URL"
            exit 1
        fi
        
        if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
            print_success "Model downloaded successfully"
        else
            print_error "Model download failed"
            exit 1
        fi
    fi
    
    # Update config.yaml with model path
    if [ -f "$PROJECT_DIR/config.yaml" ]; then
        sed -i.bak "s|model_path:.*|model_path: \"models/$MODEL_FILE\"|g" "$PROJECT_DIR/config.yaml"
        rm -f "$PROJECT_DIR/config.yaml.bak"
        print_success "Updated config.yaml with model path"
    fi
fi

# -----------------------------------------------------------------------------
# Step 6: Create Sample Corpus
# -----------------------------------------------------------------------------
print_step 6 "Setting up sample corpus"

SAMPLE_FILE="$PROJECT_DIR/corpus/sample_knowledge_base.txt"

if [ ! -f "$SAMPLE_FILE" ]; then
    cat > "$SAMPLE_FILE" << 'CORPUS_EOF'
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

- **all-MiniLM-L6-v2**: Small, fast, and efficient (384 dimensions). Best for local deployment.
- **all-mpnet-base-v2**: Higher quality, more compute (768 dimensions). Better for accuracy.
- **text-embedding-ada-002**: OpenAI's embedding model (1536 dimensions). Cloud-based.

### FAISS (Facebook AI Similarity Search)

FAISS is a library developed by Facebook AI Research for efficient similarity search. It provides:
- Fast nearest neighbor search algorithms
- Support for billion-scale datasets
- Both exact and approximate search algorithms
- CPU and GPU implementations

FAISS supports multiple index types:
- IndexFlatIP: Exact inner product search, best for small datasets
- IndexIVFFlat: Inverted file index, faster for medium datasets
- IndexHNSW: Hierarchical navigable small world graphs, very fast approximate search

## Quantization

Quantization reduces the precision of model weights to decrease memory usage and improve inference speed.

### Types of Quantization

- **INT8**: 8-bit integer quantization (~4x smaller than FP32)
- **INT4**: 4-bit integer quantization (~8x smaller than FP32)
- **Mixed precision**: Different precision for different layers

Quantization reduces memory but may slightly decrease quality. The Q4_K_M quantization typically retains 98-99% of original model quality.

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
- Good instruction following (Instruct variant)
- Open weights and permissive license

### TinyLlama 1.1B

TinyLlama is a smaller language model (1.1 billion parameters):
- Very fast inference even on CPUs
- Good for resource-constrained environments
- Trained on 3 trillion tokens
- Chat variant available for conversational tasks

### Llama 2

Llama 2 is Meta's open-source language model family:
- Available in 7B, 13B, and 70B sizes
- Chat-optimized variants available
- Commercial use permitted with license

## Best Practices for RAG Systems

1. **Chunk size**: Balance between context and specificity (300-1000 tokens recommended)
2. **Overlap**: Use overlapping chunks (10-20%) to preserve context across boundaries
3. **Retrieval count**: Retrieve 3-5 documents typically
4. **Relevance threshold**: Filter out low-relevance results (threshold 0.3-0.5)
5. **Prompt engineering**: Structure prompts to use context effectively
6. **Source attribution**: Always cite sources for verifiability

## System Architecture Considerations

### Embedding Model Selection
Choose based on your needs:
- Speed priority: all-MiniLM-L6-v2
- Quality priority: all-mpnet-base-v2
- Multilingual: paraphrase-multilingual-MiniLM-L12-v2

### Index Selection
Choose based on dataset size:
- <100K docs: IndexFlatIP (exact search)
- 100K-1M docs: IndexIVFFlat
- >1M docs: IndexHNSW or IndexIVFPQ

### LLM Selection
Choose based on hardware:
- 8GB RAM: TinyLlama 1.1B Q4
- 16GB RAM: Mistral 7B Q4
- 32GB+ RAM: Llama 2 13B Q4

CORPUS_EOF
    print_success "Sample corpus created"
else
    print_info "Sample corpus already exists"
fi

# Count corpus files
CORPUS_COUNT=$(find "$PROJECT_DIR/corpus" -type f \( -name "*.txt" -o -name "*.md" -o -name "*.pdf" -o -name "*.docx" \) | wc -l)
print_success "Corpus contains $CORPUS_COUNT document(s)"

# -----------------------------------------------------------------------------
# Step 7: Build Vector Index
# -----------------------------------------------------------------------------
print_step 7 "Building vector index"

print_info "Loading embedding model and indexing documents..."

cd "$PROJECT_DIR"
python -c "
from src.config import get_default_config
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever, load_corpus_from_directory
from src.embeddings import DocumentChunker

config = get_default_config()
embedding_model = EmbeddingModel(config.embedding)
retriever = VectorRetriever(config.retriever, embedding_model)

chunker = DocumentChunker(
    chunk_size=config.retriever.chunk_size,
    chunk_overlap=config.retriever.chunk_overlap,
)
documents = load_corpus_from_directory(config.corpus_dir, chunker)

if documents:
    retriever.build_index(documents)
    retriever.save_index()
    print(f'Index built with {len(documents)} chunks')
else:
    print('No documents found in corpus directory')
"

print_success "Vector index built and saved"

# -----------------------------------------------------------------------------
# Step 8: Setup BioASQ (Full setup only)
# -----------------------------------------------------------------------------
if [ "$FULL_SETUP" = true ]; then
    print_step 8 "Setting up BioASQ evaluation dataset"
    
    print_info "Downloading and indexing BioASQ dataset..."
    
    python scripts/setup_bioasq.py
    
    print_success "BioASQ dataset ready"
fi

# -----------------------------------------------------------------------------
# Step 9: Verify Installation
# -----------------------------------------------------------------------------
VERIFY_STEP=$((TOTAL_STEPS - 1))
if [ "$RUN_EVAL" = true ]; then
    VERIFY_STEP=$((TOTAL_STEPS - 2))
fi

print_step $VERIFY_STEP "Verifying installation"

# Test imports
echo "Testing Python imports..."
python -c "
from src.config import RAGConfig, get_default_config
from src.embeddings import EmbeddingModel, DocumentChunker
from src.retriever import VectorRetriever, Document
from src.guardrails import ContentGuardrails, GuardrailAction
from src.llm import LocalLLM
from src.rag import RAGPipeline
from src.cli import cli
print('All imports successful!')
"
print_success "All modules import correctly"

# Check index
python -c "
from src.config import get_default_config
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever

config = get_default_config()
embedding_model = EmbeddingModel(config.embedding)
retriever = VectorRetriever(config.retriever, embedding_model)
stats = retriever.get_stats()
print(f'Index contains {stats[\"num_documents\"]} documents')
"
print_success "Index verification passed"

# Check model (if not skipped)
if [ "$SKIP_MODEL" = false ]; then
    if [ "$LIGHTWEIGHT" = true ]; then
        MODEL_PATH="$MODEL_DIR/$TINYLLAMA_FILE"
    else
        MODEL_PATH="$MODEL_DIR/$MISTRAL_FILE"
    fi
    
    if [ -f "$MODEL_PATH" ]; then
        MODEL_SIZE_ACTUAL=$(du -h "$MODEL_PATH" | cut -f1)
        print_success "Model file exists: $MODEL_SIZE_ACTUAL"
    else
        print_warning "Model file not found at expected location"
    fi
fi

# -----------------------------------------------------------------------------
# Step 10: Run Evaluation (if requested)
# -----------------------------------------------------------------------------
if [ "$RUN_EVAL" = true ]; then
    print_step $TOTAL_STEPS "Running BioASQ evaluation"
    
    print_info "This may take 30-60 minutes depending on your hardware..."
    
    python scripts/evaluate_bioasq.py --num-questions 20
    
    if [ -f "$PROJECT_DIR/evaluation_results.json" ]; then
        print_success "Evaluation complete! Results saved to evaluation_results.json"
        echo ""
        cat "$PROJECT_DIR/evaluation_results.json"
    fi
fi

# =============================================================================
# Setup Complete
# =============================================================================

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ✓ SETUP COMPLETE!                               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BOLD}Quick Start Commands:${NC}"
echo ""
echo -e "  ${CYAN}1. Activate the virtual environment:${NC}"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "  ${CYAN}2. Start interactive mode:${NC}"
if [ "$SKIP_MODEL" = false ]; then
    if [ "$LIGHTWEIGHT" = true ]; then
        echo -e "     ${YELLOW}python main.py --config config.yaml${NC}"
    else
        echo -e "     ${YELLOW}python main.py --config config.yaml${NC}"
    fi
else
    echo -e "     ${YELLOW}python main.py --model /path/to/your/model.gguf${NC}"
fi
echo ""
echo -e "  ${CYAN}3. Ask a single question:${NC}"
echo -e "     ${YELLOW}python main.py query \"What is RAG?\"${NC}"
echo ""
echo -e "  ${CYAN}4. Run tests:${NC}"
echo -e "     ${YELLOW}pytest tests/ -v${NC}"
echo ""

if [ "$FULL_SETUP" = true ]; then
    echo -e "  ${CYAN}5. Run BioASQ evaluation:${NC}"
    echo -e "     ${YELLOW}python scripts/evaluate_bioasq.py${NC}"
    echo ""
fi

echo -e "${BOLD}Documentation:${NC}"
echo -e "  See ${CYAN}DOCUMENTATION.md${NC} for comprehensive documentation"
echo -e "  See ${CYAN}README.md${NC} for quick reference"
echo ""

echo -e "${MAGENTA}Happy RAG-ing! 🔍${NC}"
echo ""

