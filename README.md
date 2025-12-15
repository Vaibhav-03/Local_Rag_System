# 🔍 Local RAG System

A fully local **Retrieval-Augmented Generation (RAG)** system that runs entirely on your laptop. No cloud resources required.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 📖 **For comprehensive documentation**, see [**DOCUMENTATION.md**](DOCUMENTATION.md) - includes architecture diagrams, evaluation results, and discussion topics.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Evaluation](#-evaluation)
- [BioASQ Benchmark Results](#-bioasq-benchmark-results)
- [Design Decisions](#-design-decisions)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🏠 Fully Local** | No cloud APIs or internet required for inference |
| **⚡ Quantized LLM** | 4-bit quantized models for efficient CPU inference |
| **🔍 Semantic Search** | FAISS-powered vector retrieval with sentence embeddings |
| **📚 Source Citations** | Every response includes references to source documents |
| **🛡️ Guardrails** | Content safety filters and prompt injection protection |
| **💬 Interactive CLI** | Beautiful terminal interface with streaming responses |
| **📊 Evaluation Suite** | Built-in testing and quality metrics |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GUARDRAILS CHECK                            │
│  • Content filtering    • Prompt injection detection            │
│  • Query validation     • Topic restrictions                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EMBEDDING & RETRIEVAL                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ all-MiniLM   │───▶│    FAISS     │───▶│  Top-K Docs  │      │
│  │  Embeddings  │    │    Index     │    │   Retrieved  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CONTEXT ASSEMBLY & PROMPT                      │
│  System prompt + Retrieved context + User query                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   QUANTIZED LLM (GGUF)                          │
│  Mistral 7B / Llama 2 / TinyLlama via llama.cpp                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESPONSE WITH SOURCE CITATIONS                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### One-Command Setup (Linux/macOS)

```bash
# Clone and setup
git clone <repository-url>
cd local-rag-system

# Option 1: Complete automated setup (recommended)
chmod +x scripts/setup_complete.sh
./scripts/setup_complete.sh

# Option 2: Full setup with BioASQ evaluation
./scripts/setup_complete.sh --full

# Option 3: Lightweight setup (smaller model, faster)
./scripts/setup_complete.sh --lightweight

# Option 4: Basic setup (original script)
chmod +x scripts/setup.sh
./scripts/setup.sh
```

#### Setup Script Options

| Option | Description |
|--------|-------------|
| `--full` | Download BioASQ dataset and run evaluation |
| `--lightweight` | Use TinyLlama (~670MB) instead of Mistral (~4.4GB) |
| `--skip-model` | Skip model download if you have your own |
| `--run-eval` | Run evaluation after setup |

### Manual Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download a model (choose one)
# Option A: Mistral 7B (~4.4GB, recommended)
wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Option B: TinyLlama (~670MB, lightweight)
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 4. Setup BioASQ dataset (recommended demo)
python scripts/setup_bioasq.py

# 5. Start the system (works out of the box!)
python main.py

# Alternative: Use your own documents
# cp your-documents/*.txt corpus/
# python main.py index
# python main.py
```

---

## 📦 Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **CPU** | Intel i5 / M1 | Intel i7 / M2+ |
| **RAM** | 8GB | 16GB |
| **Storage** | 10GB free | 20GB free |
| **Python** | 3.9 | 3.10+ |

### Step-by-Step Installation

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a quantized model**
   
   | Model | Size | Quality | Speed | RAM Needed | Download |
   |-------|------|---------|-------|------------|----------|
   | TinyLlama Q4 | 670MB | ⭐⭐ | ⭐⭐⭐⭐⭐ | 4GB | [Link](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
   | Phi-2 Q4 | 1.6GB | ⭐⭐⭐ | ⭐⭐⭐⭐ | 5GB | [Link](https://huggingface.co/TheBloke/phi-2-GGUF) |
   | Mistral 7B Q4 | 4.4GB | ⭐⭐⭐⭐ | ⭐⭐⭐ | 8GB | [Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) |

   ```bash
   # Download TinyLlama (default, fast)
   wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

   # Or Phi-2 (recommended for better quality on 8GB RAM)
   wget -P models/ https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf
   ```

4. **Prepare your corpus**
   ```bash
   # Add text files to the corpus directory
   cp /path/to/your/documents/*.txt corpus/
   cp /path/to/your/documents/*.md corpus/
   cp /path/to/your/documents/*.pdf corpus/
   ```

5. **Build the index**
   ```bash
   python main.py index
   ```

6. **Start the system**
   ```bash
   python main.py
   ```

---

## 💻 Usage

### Interactive Mode

```bash
# Default: Uses BioASQ dataset + TinyLlama model (works out of the box)
python main.py

# Or with custom config
python main.py --config config-bioasq.yaml
```

Interactive commands:
- Type your question and press Enter
- `help` - Show help
- `stats` - Show system statistics  
- `clear` - Clear screen
- `quit` or `exit` - Exit

### Single Query Mode

```bash
# Ask a single question (BioASQ examples)
python main.py query "Is Hirschsprung disease a mendelian or multifactorial disorder?"

# With streaming output
python main.py query "Is the protein Papilin secreted?" --stream

# Output as JSON
python main.py query "Has Denosumab been approved by FDA?" --json
```

### Index Management

```bash
# Build/rebuild index from corpus directory
python main.py index

# Index from a specific directory
python main.py index --corpus /path/to/documents
```

### Configuration

```bash
# Generate default config file
python main.py init-config

# Use custom config
python main.py --config my-config.yaml
```

---

## ⚙️ Configuration

### Configuration File (`config.yaml`)

```yaml
# Language Model
llm:
  model_path: "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  n_ctx: 4096          # Context window
  max_tokens: 512      # Max response length
  temperature: 0.7     # Creativity (0-1)
  n_threads: 0         # CPU threads (0=auto)
  n_gpu_layers: 0      # GPU layers (0=CPU only)

# Embedding Model  
embedding:
  model_name: "all-MiniLM-L6-v2"
  device: "cpu"        # cpu, cuda, or mps

# Retriever
retriever:
  top_k: 5             # Documents to retrieve
  similarity_threshold: 0.3
  chunk_size: 500      # Characters per chunk
  chunk_overlap: 50

# Guardrails
guardrails:
  enabled: true
  blocked_topics:
    - "violence"
    - "illegal activities"
  max_query_length: 2000
```

### Environment Variables

```bash
export RAG_MODEL_PATH=/path/to/model.gguf
export RAG_CORPUS_DIR=/path/to/corpus
export RAG_VERBOSE=true
```

---

## 📊 Evaluation

### Run Evaluation Suite

```bash
# Full evaluation
python tests/test_evaluation.py --full

# Quick sanity check
python tests/test_evaluation.py --quick
```

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Topic Coverage** | % of expected topics in response | > 40% |
| **ROUGE-1** | Unigram overlap with reference | > 0.3 |
| **ROUGE-L** | Longest common subsequence | > 0.25 |
| **Retrieval Time** | Time to find relevant docs | < 100ms |
| **Generation Time** | Time to generate response | < 5s |
| **Total Time** | End-to-end latency | < 10s |

### Sample Evaluation Results (BioASQ)

**Model Comparison on 100 Questions:**

| Metric | TinyLlama 1.1B | Phi-2 2.7B |
|--------|---------------|------------|
| Precision@5 | 53.3% | 46.9% |
| **Recall@5** | 19.0% | **25.3% (+33%)** |
| MRR | 0.702 | 0.710 |
| **Hit Rate** | 79% | **83%** |
| **ROUGE-1** | 0.186 | **0.254 (+37%)** |
| **ROUGE-L** | 0.132 | **0.204 (+55%)** |
| Avg Time | 22s | 56s |

### Running the Evaluation

```bash
# Setup BioASQ dataset
python scripts/setup_bioasq.py

# Run evaluation with TinyLlama (default)
python scripts/evaluate_bioasq.py

# Run evaluation with Phi-2 (better quality)
python scripts/evaluate_bioasq.py --config config-phi2.yaml

# Run subset evaluation (faster)
python scripts/evaluate_bioasq.py --num-questions 20
```

> 📖 See [DOCUMENTATION.md](DOCUMENTATION.md#6-evaluation--experimental-results) for detailed analysis.

---

## 🎯 Design Decisions

### Why GGUF + llama.cpp?

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| **PyTorch FP32** | Full precision | 28GB+ for 7B model | ❌ Too large |
| **ONNX Runtime** | Optimized | Limited model support | ❌ Compatibility |
| **llama.cpp GGUF** | 4GB for 7B, CPU optimized | Minor quality loss | ✅ **Selected** |

**Quantization Trade-off Analysis:**

```
Model: Mistral 7B
┌────────────────┬─────────┬───────────┬──────────┐
│ Quantization   │ Size    │ Speed     │ Quality  │
├────────────────┼─────────┼───────────┼──────────┤
│ FP16           │ 14GB    │ Baseline  │ 100%     │
│ Q8_0           │ 7.7GB   │ 1.2x      │ 99.5%    │
│ Q4_K_M         │ 4.4GB   │ 1.5x      │ 98.5%    │  ← Selected
│ Q4_0           │ 4.0GB   │ 1.6x      │ 97%      │
└────────────────┴─────────┴───────────┴──────────┘
```

### Why all-MiniLM-L6-v2 for Embeddings?

| Model | Dimensions | Size | Speed | Quality |
|-------|------------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | 80MB | 14K/s | Good | ✅ **Selected** |
| all-mpnet-base-v2 | 768 | 420MB | 2.8K/s | Better |
| text-embedding-ada-002 | 1536 | API | Fast | Best | ❌ Cloud |

**Decision:** MiniLM provides the best balance of speed, size, and quality for local deployment.

### Why FAISS over alternatives?

| Solution | Speed | Scalability | Memory | Decision |
|----------|-------|-------------|--------|----------|
| Brute Force | Slow | Poor | Low | ❌ |
| Annoy | Good | Good | Medium | Consider |
| **FAISS** | Excellent | Excellent | Low | ✅ **Selected** |
| Pinecone | Excellent | Excellent | N/A | ❌ Cloud |

---

## ⚡ Performance

### Benchmarks (M2 MacBook Air, 16GB RAM)

| Operation | Time | Memory |
|-----------|------|--------|
| Model Load | 3-5s | 4.5GB |
| Embedding (query) | 5ms | 200MB |
| FAISS Search (10K docs) | 2ms | 50MB |
| Generation (100 tokens) | 2-4s | - |
| **Total Query** | **3-5s** | **~5GB** |

### Optimization Tips

1. **Reduce context window** (`n_ctx: 2048`) for faster inference
2. **Use TinyLlama** for resource-constrained systems
3. **Enable Metal** on Apple Silicon: `n_gpu_layers: 32`
4. **Increase threads** on multi-core CPUs: `n_threads: 8`

---

## 🔧 Troubleshooting

### Common Issues

**❌ "Model file not found"**
```bash
# Download a model first
./scripts/download_model.sh
```

**❌ "Out of memory"**
```bash
# Use a smaller model
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Or reduce context window in config.yaml
n_ctx: 2048
```

**❌ "Slow generation"**
```yaml
# config.yaml - increase threads
llm:
  n_threads: 8
  
# On Apple Silicon, enable GPU
  n_gpu_layers: 32
```

**❌ "No documents found"**
```bash
# Add documents to corpus/
echo "Your knowledge content here" > corpus/my_doc.txt
python main.py index
```

**❌ "Import errors"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📁 Project Structure

```
local-rag-system/
├── src/
│   ├── __init__.py      # Package init
│   ├── config.py        # Configuration management
│   ├── llm.py           # LLM wrapper (llama.cpp)
│   ├── embeddings.py    # Embedding model & chunking
│   ├── retriever.py     # FAISS vector retrieval
│   ├── guardrails.py    # Content safety filters
│   ├── rag.py           # Main RAG pipeline
│   └── cli.py           # Command line interface
├── corpus/              # Your custom documents (optional)
├── data/bioasq/         # BioASQ dataset (default demo)
├── models/              # Downloaded GGUF models
├── tests/
│   ├── test_rag.py      # Unit tests
│   └── test_evaluation.py  # Evaluation suite
├── scripts/
│   ├── setup.sh         # One-command setup
│   └── download_model.sh
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── main.py              # Entry point
└── README.md
```

---

---

## 📋 Limitations & Future Work

### Current Limitations

| Limitation | Impact | Future Solution |
|------------|--------|-----------------|
| Single-turn only | No conversation memory | Chat history management |
| English only | No multilingual support | Multilingual embeddings |
| CPU-focused | Slower than GPU | GPU layer offloading |
| Exact search | O(n) for large corpora | HNSW index for >1M docs |
| No reranking | First-pass results only | Cross-encoder reranker |

### Future Directions

- **v2.0**: Conversation memory, hybrid search (semantic + BM25)
- **v3.0**: Cross-encoder reranking, query expansion
- **v4.0**: Agent capabilities, tool use

> 📖 See [DOCUMENTATION.md](DOCUMENTATION.md#10-limitations--future-directions) for detailed roadmap.

---

## 🎤 Discussion Topics

Key areas for technical discussion:

1. **Architecture Decisions**: Embedding model choice, FAISS index type, quantization level
2. **Trade-offs**: Quality vs. speed, memory vs. accuracy
3. **Scaling**: Changes needed for 1M+ documents, multi-user deployment
4. **Quality**: Retrieval improvements, generation enhancements

> 📖 See [DOCUMENTATION.md](DOCUMENTATION.md#11-whiteboard-discussion-topics) for comprehensive discussion guide.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal output
- [TheBloke](https://huggingface.co/TheBloke) - Quantized models

