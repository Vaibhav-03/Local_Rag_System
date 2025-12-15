# 📚 Local RAG System - Comprehensive Documentation

> **A Fully Local Retrieval-Augmented Generation System for Laptop-Scale Inference**

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Environment Setup](#3-environment-setup)
4. [CLI Tool Usage](#4-cli-tool-usage)
5. [Usage Examples with Expected Outputs](#5-usage-examples-with-expected-outputs)
6. [Evaluation & Experimental Results](#6-evaluation--experimental-results)
7. [Retrieval Mechanism Deep Dive](#7-retrieval-mechanism-deep-dive)
8. [LLM Integration](#8-llm-integration)
9. [Additional Features](#9-additional-features)
10. [Limitations & Future Directions](#10-limitations--future-directions)
11. [Whiteboard Discussion Topics](#11-whiteboard-discussion-topics)

---

## 1. Overview

### What is this system?

This is a **fully local RAG (Retrieval-Augmented Generation) system** that runs entirely on consumer hardware without requiring cloud APIs or internet connectivity. It combines:

- **Semantic document retrieval** using vector embeddings and FAISS
- **Local LLM inference** using quantized models via llama.cpp
- **Content safety guardrails** with prompt injection protection
- **Interactive CLI** with rich terminal formatting

### Key Differentiators

| Feature | Cloud RAG (OpenAI + Pinecone) | This System |
|---------|-------------------------------|-------------|
| **Privacy** | Data sent to cloud | 100% local |
| **Cost** | Pay per API call | One-time model download |
| **Latency** | Network dependent | 3-5s total |
| **Offline** | ❌ | ✅ |
| **Hardware** | Any | 16GB RAM recommended |

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                 LOCAL RAG SYSTEM                                │
│                        Complete Data Flow Architecture                          │
└────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │  USER QUERY  │
                                    │   (string)   │
                                    └──────┬───────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               1. GUARDRAILS LAYER                                │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │   Content Filter    │  │  Prompt Injection   │  │   Query Validation      │  │
│  │   (blocked topics)  │  │    Detection        │  │   (length, format)      │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────────┘  │
│                                                                                  │
│  Output: ALLOW / BLOCK / WARN                                                   │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │ If ALLOW
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              2. EMBEDDING LAYER                                  │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                     Sentence-Transformers                                   │ │
│  │                     (all-MiniLM-L6-v2)                                     │ │
│  │                                                                            │ │
│  │  "Is Hirschsprung disease mendelian?"  ──▶  [0.12, -0.34, ...]  (384-dim)│ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Model Details:                                                                 │
│  • Size: 80MB                                                                   │
│  • Speed: ~14,000 sentences/second                                              │
│  • Normalized for cosine similarity                                             │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             3. RETRIEVAL LAYER                                   │
│                                                                                  │
│  ┌────────────────────┐      ┌────────────────────┐      ┌──────────────────┐   │
│  │   Query Embedding  │──▶   │   FAISS Index      │──▶   │  Top-K Results   │   │
│  │   (384-dim vector) │      │   (IndexFlatIP)    │      │  (ranked docs)   │   │
│  └────────────────────┘      └────────────────────┘      └──────────────────┘   │
│                                       │                                         │
│                              ┌────────┴────────┐                               │
│                              │ Document Store  │                               │
│                              │  (documents.pkl)│                               │
│                              └─────────────────┘                               │
│                                                                                  │
│  Retrieval Flow:                                                                │
│  1. Encode query → 384-dim vector                                               │
│  2. Inner product similarity search in FAISS                                    │
│  3. Filter by similarity_threshold (default: 0.3)                               │
│  4. Return top_k documents (default: 5)                                         │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           4. CONTEXT ASSEMBLY                                    │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  System Prompt:                                                            │ │
│  │  "You are a helpful AI assistant with access to a knowledge base..."      │ │
│  │                                                                            │ │
│  │  Retrieved Context:                                                        │ │
│  │  [Source: bioasq_passage_20598273] Coding sequence mutations in RET...    │ │
│  │  [Source: bioasq_passage_6650562] Hirschsprung disease genetics...        │ │
│  │                                                                            │ │
│  │  User Question: Is Hirschsprung disease mendelian?                         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Prompt Template (TinyLlama/Llama format):                                        │
│  [INST] <<SYS>> {system_prompt} <</SYS>> {context + question} [/INST]          │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          5. LLM GENERATION LAYER                                 │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    llama.cpp (GGUF Runtime)                                │ │
│  │                                                                            │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Quantized Model: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf              │  │ │
│  │  │  • Original: 2.2GB FP16 → Quantized: 670MB Q4                      │  │ │
│  │  │  • Context Window: 4096 tokens                                      │  │ │
│  │  │  • SIMD-optimized CPU inference                                     │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                            │ │
│  │  Generation Parameters:                                                    │ │
│  │  • max_tokens: 512      • temperature: 0.7                                │ │
│  │  • top_p: 0.9           • repeat_penalty: 1.1                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          6. RESPONSE ASSEMBLY                                    │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  RAGResponse:                                                              │ │
│  │  {                                                                         │ │
│  │    "answer": "Hirschsprung disease shows both Mendelian and complex...",  │ │
│  │    "sources": [                                                            │ │
│  │      { "rank": 1, "source": "bioasq_passage_20598273", "score": 0.82 },  │ │
│  │      { "rank": 2, "source": "bioasq_passage_6650562", "score": 0.76 }    │ │
│  │    ],                                                                      │ │
│  │    "retrieval_time": 1.469,                                               │ │
│  │    "generation_time": 20.7,                                               │ │
│  │    "tokens_used": 1247                                                     │ │
│  │  }                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT DEPENDENCIES                              │
└──────────────────────────────────────────────────────────────────────────────┘

                                ┌───────────────┐
                                │   config.py   │
                                │ (RAGConfig)   │
                                └───────┬───────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │                            │                            │
           ▼                            ▼                            ▼
   ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
   │   llm.py      │           │ embeddings.py │           │ guardrails.py │
   │  (LocalLLM)   │           │(EmbeddingModel│           │(ContentGuard- │
   │               │           │ DocChunker)   │           │   rails)      │
   └───────┬───────┘           └───────┬───────┘           └───────┬───────┘
           │                            │                            │
           │                            ▼                            │
           │                   ┌───────────────┐                     │
           │                   │ retriever.py  │                     │
           │                   │(VectorRetriev-│                     │
           │                   │    er)        │                     │
           │                   └───────┬───────┘                     │
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        │
                                        ▼
                               ┌───────────────┐
                               │    rag.py     │
                               │ (RAGPipeline) │
                               └───────┬───────┘
                                        │
                                        ▼
                               ┌───────────────┐
                               │    cli.py     │
                               │  (User CLI)   │
                               └───────────────┘
```

### 2.3 Data Flow During Indexing

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING PIPELINE                               │
└──────────────────────────────────────────────────────────────────────────────┘

  corpus/                                                      models/
    │                                                            │
    ├── doc1.txt ─┐                                              ├── faiss_index
    ├── doc2.md  ─┼──▶ DocumentChunker ──▶ EmbeddingModel ──▶    │
    ├── doc3.pdf ─┘         │                    │               └── documents.pkl
                            │                    │
                            ▼                    ▼
                   ┌─────────────────┐   ┌─────────────────┐
                   │   Chunks:       │   │   Embeddings:   │
                   │   - chunk_0     │   │   - [0.1, ...]  │
                   │   - chunk_1     │   │   - [0.2, ...]  │
                   │   - chunk_2     │   │   - [0.3, ...]  │
                   └─────────────────┘   └─────────────────┘

Chunking Parameters:
• chunk_size: 500 characters
• chunk_overlap: 50 characters (preserves context across boundaries)

Supported File Types:
• .txt, .md (native)
• .pdf (via pypdf)
• .docx (via python-docx)
```

---

## 3. Environment Setup

### 3.1 Automated End-to-End Setup (Recommended)

The system includes a comprehensive setup script that handles everything:

```bash
# Clone the repository
git clone <repository-url>
cd local-rag-system

# Make the setup script executable
chmod +x scripts/setup.sh

# Run the automated setup
./scripts/setup.sh
```

#### What the setup script does:

| Step | Action | Details |
|------|--------|---------|
| 1 | Python check | Verifies Python 3.9+ is installed |
| 2 | Virtual environment | Creates `venv/` directory |
| 3 | Dependencies | Installs all packages from `requirements.txt` |
| 4 | Directories | Creates `corpus/`, `models/`, `tests/` |
| 5 | Model download | Optionally downloads TinyLlama 1.1B (670MB) or Phi-2 2.7B (1.6GB) |
| 6 | Sample corpus | Creates a sample knowledge base document |

### 3.2 Manual Setup Steps

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create necessary directories
mkdir -p corpus models

# 4. Download a model (choose ONE)

# Option A: TinyLlama 1.1B (default, ~670MB, fast)
wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Option B: Phi-2 2.7B (better quality, ~1.6GB, recommended for 8GB RAM)
wget -P models/ https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf

# Option C: Mistral 7B Instruct (highest quality, ~4.4GB, needs 16GB RAM)
wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# 5. Add your documents to corpus/
cp /path/to/your/documents/*.txt corpus/
cp /path/to/your/documents/*.md corpus/
cp /path/to/your/documents/*.pdf corpus/

# 6. Build the vector index
python main.py index

# 7. Start the system
python main.py --config config.yaml
```

### 3.3 Dependencies Breakdown

```
# requirements.txt explained:

# LLM Backend
llama-cpp-python>=0.2.20      # GGUF model inference

# Embeddings and Vector Search
sentence-transformers>=2.2.2  # Text embeddings
faiss-cpu>=1.7.4              # Vector similarity search

# Text Processing
langchain>=0.1.0              # Document processing utilities
tiktoken>=0.5.1               # Token counting

# CLI and UX
rich>=13.7.0                  # Beautiful terminal output
click>=8.1.7                  # Command-line interface

# Document Processing
pypdf>=3.17.0                 # PDF reading
python-docx>=1.1.0            # DOCX reading

# Evaluation
pytest>=7.4.3                 # Testing
rouge-score>=0.1.2            # Answer quality metrics
```

### 3.4 Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | Intel i5 / M1 | Intel i7+ / M2+ | More cores = faster generation |
| **RAM** | 8GB | 16GB+ | Model loaded fully into RAM |
| **Storage** | 10GB | 20GB | For models + corpus |
| **Python** | 3.9 | 3.10+ | Type hints compatibility |
| **OS** | Linux/macOS | Any | Windows requires extra setup |
---

## 4. CLI Tool Usage

The CLI is built with [Click](https://click.palletsprojects.com/) for command handling and [Rich](https://rich.readthedocs.io/) for beautiful terminal output.

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │   main.py       │
                        │   Entry Point   │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   cli.py        │
                        │   Click Group   │
                        └────────┬────────┘
                                 │
         ┌───────────┬───────────┼───────────┬───────────┐
         ▼           ▼           ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
    │ query   │ │ index   │ │ stats   │ │init-    │ │interactive│
    │ command │ │ command │ │ command │ │config   │ │   mode   │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────────┘
```

### 4.2 Global Options

These options apply to ALL commands:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--config` | `-c` | PATH | Path to YAML config file |
| `--model` | `-m` | PATH | Path to GGUF model file (overrides config) |
| `--corpus` | `-d` | PATH | Path to corpus directory (overrides config) |
| `--verbose` | `-v` | FLAG | Enable verbose output |

```bash
# Examples of global options usage
python main.py --config config-phi2.yaml                    # Use Phi-2 config
python main.py --model models/phi-2.Q4_K_M.gguf            # Override model path
python main.py --corpus /path/to/my/docs                    # Override corpus dir
python main.py -v                                           # Verbose mode
python main.py -c config.yaml -m models/custom.gguf -v      # Combine options
```

### 4.3 Commands Reference

#### 4.3.1 Interactive Mode (Default)

When no command is specified, the CLI enters interactive chat mode.

```bash
python main.py                                # Default config (TinyLlama + BioASQ)
python main.py --config config-phi2.yaml      # Use Phi-2 model
```

**Interactive Mode Flow:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                     INTERACTIVE MODE FLOW                            │
└──────────────────────────────────────────────────────────────────────┘

    Start
      │
      ▼
┌────────────┐
│ Print      │
│ Banner     │
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Load RAG   │◄──────────────────────────────────────────┐
│ Pipeline   │  (Embedding model, LLM, FAISS index)       │
└─────┬──────┘                                            │
      │                                                   │
      ▼                                                   │
┌────────────┐     ┌────────────────┐                    │
│ Wait for   │────▶│ quit/exit/q    │────▶ Exit          │
│ User Input │     └────────────────┘                    │
└─────┬──────┘                                            │
      │             ┌────────────────┐                    │
      ├────────────▶│ help           │────▶ Show Help ───┤
      │             └────────────────┘                    │
      │             ┌────────────────┐                    │
      ├────────────▶│ stats          │────▶ Show Stats ──┤
      │             └────────────────┘                    │
      │             ┌────────────────┐                    │
      ├────────────▶│ clear          │────▶ Clear Screen─┤
      │             └────────────────┘                    │
      ▼                                                   │
┌────────────┐     ┌────────────────┐     ┌──────────┐   │
│ Any other  │────▶│ Query Pipeline │────▶│ Print    │───┘
│ text       │     │ (RAG Process)  │     │ Response │
└────────────┘     └────────────────┘     └──────────┘
```

**In-chat Commands:**

| Command | Description |
|---------|-------------|
| `help` | Display available commands and tips |
| `stats` | Show indexed documents count, embedding dimension, guardrails status |
| `clear` | Clear the terminal screen |
| `quit`, `exit`, `q` | Exit the program |

**Example Session:**

```
╔═══════════════════════════════════════════════════════════════╗
║                   🔍 LOCAL RAG SYSTEM 🔍                       ║
║         Retrieval-Augmented Generation on Your Laptop         ║
╚═══════════════════════════════════════════════════════════════╝

Loading RAG system... (this may take a moment)
✓ 40221 documents ready for retrieval

Commands: 'quit' to exit, 'help' for help, 'stats' for statistics

You: Is Hirschsprung disease mendelian?

╭─────────────────────────────────────────────────────────────────╮
│ 💬 Response                                                      │
├─────────────────────────────────────────────────────────────────┤
│ Hirschsprung disease shows both Mendelian and multifactorial    │
│ inheritance patterns...                                          │
╰─────────────────────────────────────────────────────────────────╯

📚 Sources
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Rank ┃ Source                     ┃ Relevance  ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1    │ bioasq_passage_20598273    │ 82%        │
│ 2    │ bioasq_passage_6650562     │ 76%        │
└──────┴────────────────────────────┴────────────┘

⏱️  Retrieval: 1469ms | Generation: 20687ms | Total: 22156ms

You: stats

📊 Statistics
  Documents indexed: 40221
  Embedding dimension: 384
  Guardrails: enabled

You: quit
Goodbye! 👋
```

---

#### 4.3.2 Query Command

Ask a single question without entering interactive mode.

**Syntax:**
```bash
python main.py query "YOUR QUESTION" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--stream` | `-s` | Stream response token-by-token |
| `--json` | | Output response as JSON |

**Examples:**

```bash
# Basic query
python main.py query "Is the protein Papilin secreted?"

# With Phi-2 model
python main.py query "What is Hirschsprung disease?" --config config-phi2.yaml

# Stream output (see tokens as they generate)
python main.py query "Are long non coding RNAs spliced?" --stream

# JSON output (for integration with other tools)
python main.py query "Has Denosumab been approved by FDA?" --json
```

**JSON Output Format:**

```json
{
  "answer": "Yes, Denosumab (Prolia) has been approved by FDA...",
  "sources": [
    {
      "rank": 1,
      "score": 0.823,
      "source": "bioasq_passage_21784067",
      "text_preview": "Denosumab is a fully human monoclonal..."
    }
  ],
  "query": "Has Denosumab been approved by FDA?",
  "timing": {
    "retrieval_ms": 1256,
    "generation_ms": 18542,
    "total_ms": 19798
  },
  "tokens_used": 823
}
```

---

#### 4.3.3 Index Command

Build or rebuild the FAISS vector index from documents.

**Syntax:**
```bash
python main.py index [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--corpus` | `-d` | Path to corpus directory (overrides default) |

**Examples:**

```bash
# Index documents from default directory (data/bioasq or corpus/)
python main.py index

# Index from custom directory
python main.py index --corpus /path/to/my/documents

# With custom config
python main.py index --config config.yaml
```

**Expected Output:**

```
📁 Indexing corpus from: /path/to/corpus

Loading embedding model: all-MiniLM-L6-v2
Indexing documents...

✅ Successfully indexed 156 document chunks!
```

**Supported File Types:**

| Extension | Format |
|-----------|--------|
| `.txt` | Plain text |
| `.md` | Markdown |
| `.pdf` | PDF documents |
| `.docx` | Microsoft Word |

---

#### 4.3.4 Stats Command

Display detailed system statistics.

**Syntax:**
```bash
python main.py stats
```

**Expected Output:**

```
📊 System Statistics

           📚 Retriever            
┌────────────────────┬─────────────┐
│ num_documents      │ 40221       │
│ index_type         │ IndexFlatIP │
│ embedding_dim      │ 384         │
└────────────────────┴─────────────┘

        🧮 Embedding Model         
┌────────────────────┬─────────────┐
│ model_name         │ all-MiniL.. │
│ dimension          │ 384         │
│ device             │ cpu         │
└────────────────────┴─────────────┘

         🤖 Language Model         
┌────────────────────┬─────────────┐
│ model_path         │ models/ti.. │
│ context_length     │ 4096        │
│ n_gpu_layers       │ 0           │
└────────────────────┴─────────────┘
```

---

#### 4.3.5 Init-Config Command

Generate a default configuration file.

**Syntax:**
```bash
python main.py init-config [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `config.yaml` | Output file path |

**Examples:**

```bash
# Generate default config.yaml
python main.py init-config

# Custom output path
python main.py init-config --output my-custom-config.yaml
```

---

### 4.4 Configuration Files

The system supports multiple config files for different models:

| Config File | Model | Use Case |
|-------------|-------|----------|
| (default) | TinyLlama 1.1B | Fast inference, 4GB RAM |
| `config-phi2.yaml` | Phi-2 2.7B | Better quality, 8GB RAM |
| `config-bioasq.yaml` | TinyLlama 1.1B | BioASQ-specific settings |

```bash
# Use TinyLlama (default)
python main.py

# Use Phi-2 for better quality
python main.py --config config-phi2.yaml

# Use BioASQ config
python main.py --config config-bioasq.yaml
```

### 4.5 Error Handling

The CLI provides helpful error messages:

| Error | Cause | Solution |
|-------|-------|----------|
| "Model path not specified" | No model file found | Download a GGUF model to `models/` |
| "Corpus directory not found" | Invalid corpus path | Check path exists |
| "No files found in corpus" | Empty corpus directory | Add text files to index |
| "exceed context window" | Prompt too long | System auto-truncates (fixed) |

### 4.6 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Interrupt current operation |
| `Ctrl+D` | Exit interactive mode |
| `↑`/`↓` | Navigate command history (if `readline` enabled) |
| `quit`, `exit`, `q` | Exit the program |

---

## 5. Usage Examples with Expected Outputs

### 5.1 Starting the System

```bash
$ python main.py --config config.yaml

╔═══════════════════════════════════════════════════════════════╗
║                   🔍 LOCAL RAG SYSTEM 🔍                       ║
║         Retrieval-Augmented Generation on Your Laptop         ║
╚═══════════════════════════════════════════════════════════════╝

Loading RAG system... (this may take a moment)

Initializing RAG pipeline...
Loading embedding model: all-MiniLM-L6-v2
Embedding dimension: 384
Loading existing index from /path/to/models/faiss_index
Loaded 156 documents
Loading model from models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf...
Using 8 CPU threads
Model loaded in 2.34 seconds
RAG pipeline initialized!

✓ 156 documents ready for retrieval

Commands: 'quit' to exit, 'help' for help, 'stats' for statistics

You: 
```

### 5.2 Example Query and Response (BioASQ Dataset)

```
You: Is Hirschsprung disease a mendelian or a multifactorial disorder?

╭─────────────────────────────────────────────────────────────────────────────╮
│ 💬 Response                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Based on the retrieved context, Hirschsprung disease shows both Mendelian  │
│ and multifactorial inheritance patterns depending on the form:              │
│                                                                             │
│ **Mendelian forms**: Coding sequence mutations in genes like RET, GDNF,    │
│ EDNRB, EDN3, and SOX10 are involved in syndromic forms of Hirschsprung     │
│ disease, which follow Mendelian inheritance patterns.                       │
│                                                                             │
│ **Multifactorial forms**: The non-Mendelian inheritance of sporadic        │
│ non-syndromic Hirschsprung disease is complex, with involvement of         │
│ multiple loci demonstrated in a multiplicative model.                       │
│                                                                             │
│ In summary, syndromic forms are Mendelian while sporadic non-syndromic     │
│ cases show complex multifactorial inheritance.                              │
╰─────────────────────────────────────────────────────────────────────────────╯

                        📚 Sources                         
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃ Source                          ┃ Relevance  ┃ Preview                   ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ bioasq_passage_20598273         │ 82%        │ Coding sequence mutatio...│
│ 2    │ bioasq_passage_6650562          │ 76%        │ Hirschsprung disease is...│
│ 3    │ bioasq_passage_15829955         │ 71%        │ The genetics of Hirschs...│
│ 4    │ bioasq_passage_15617541         │ 65%        │ Non-syndromic Hirschspr...│
│ 5    │ bioasq_passage_23001136         │ 58%        │ Multiple loci involved ...│
└──────┴─────────────────────────────────┴────────────┴───────────────────────────┘

⏱️  Retrieval: 1469ms | Generation: 20687ms | Total: 22156ms | Tokens: 1247
```

### 5.3 Single Query JSON Output

```bash
$ python main.py query "Is the protein Papilin secreted?" --json

{
  "answer": "Yes, papilin is a secreted protein. Based on the retrieved passages, papilin is an extracellular matrix glycoprotein that is secreted and plays a role in tissue morphogenesis and cell migration.",
  "sources": [
    {
      "rank": 1,
      "score": 0.8234,
      "source": "bioasq_passage_21784067",
      "text_preview": "Papilin is a secreted extracellular matrix glycoprotein that..."
    },
    {
      "rank": 2,
      "score": 0.7892,
      "source": "bioasq_passage_19297413", 
      "text_preview": "The secreted protein papilin is involved in basement membrane..."
    }
  ],
  "query": "Is the protein Papilin secreted?",
  "refined_query": null,
  "timing": {
    "generation_ms": 18542,
    "retrieval_ms": 1256,
    "total_ms": 19798
  },
  "tokens_used": 823
}
```

### 5.4 Guardrail Blocking Example

```
You: How to hack into systems

╭─────────────────────────────────────────────────────────────────────────────╮
│ 💬 Response                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ I'm sorry, but I cannot assist with that topic. Please ask about           │
│ something else.                                                              │
╰─────────────────────────────────────────────────────────────────────────────╯

⚠️  Query contains blocked topic: hack
```

### 5.5 Statistics Output

```
You: stats

📊 Statistics
  Documents indexed: 156
  Embedding dimension: 384
  Guardrails: enabled
```

---

## 6. Evaluation & Experimental Results

### 6.1 BioASQ Benchmark Results

The system was evaluated on the **rag-mini-bioasq** dataset from Hugging Face, a biomedical question-answering benchmark.

#### Dataset Statistics (rag-mini-bioasq from Hugging Face)

| Metric | Value |
|--------|-------|
| Total passages | ~4,700+ |
| Total questions | 100 |
| Questions with ground truth passages | 100 |
| Questions with reference answers | 100 |
| Average passage length | ~300 characters |
| Average relevant passages per question | ~10 |

**Sample Questions from BioASQ:**
- "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
- "List signaling molecules (ligands) that interact with the receptor EGFR?"
- "Is the protein Papilin secreted?"
- "Are long non coding RNAs spliced?"
- "Has Denosumab (Prolia) been approved by FDA?"

#### Model Comparison: TinyLlama vs Phi-2 (100 Questions)

| Metric | TinyLlama 1.1B | Phi-2 2.7B | Improvement |
|--------|---------------|------------|-------------|
| **Precision@5** | 53.3% | 46.9% | - |
| **Recall@5** | 19.0% | 25.3% | **+33%** |
| **MRR** | 0.702 | 0.710 | +1% |
| **Hit Rate** | 79% | 83% | **+5%** |
| **ROUGE-1** | 0.186 | 0.254 | **+37%** |
| **ROUGE-L** | 0.132 | 0.204 | **+55%** |
| **Avg Total Time** | 22.2s | 55.7s | Slower |

#### TinyLlama 1.1B Results

```json
{
  "num_questions": 100,
  "retrieval": {
    "precision_at_k": 0.533,
    "recall_at_k": 0.190,
    "mrr": 0.702,
    "hit_rate": 0.79
  },
  "generation": {
    "rouge_1": 0.186,
    "rouge_l": 0.132
  },
  "timing": {
    "avg_retrieval_ms": 1469,
    "avg_generation_ms": 20687,
    "avg_total_ms": 22156
  }
}
```

#### Phi-2 2.7B Results

```json
{
  "num_questions": 100,
  "retrieval": {
    "precision_at_k": 0.469,
    "recall_at_k": 0.253,
    "mrr": 0.710,
    "hit_rate": 0.83
  },
  "generation": {
    "rouge_1": 0.254,
    "rouge_l": 0.204
  },
  "timing": {
    "avg_retrieval_ms": 1842,
    "avg_generation_ms": 53825,
    "avg_total_ms": 55667
  }
}
```

#### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Precision@5** | % of retrieved docs that are relevant |
| **Recall@5** | % of all relevant docs that were retrieved |
| **MRR** | Mean Reciprocal Rank (how early relevant doc appears) |
| **Hit Rate** | % of queries with ≥1 relevant doc retrieved |
| **ROUGE-1** | Word overlap with reference answers |
| **ROUGE-L** | Longest common subsequence similarity |

### 6.2 Retrieval Quality Analysis

```
Retrieval Performance Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           Hit Rate by Rank
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank 1 (Top Result):    ████████████████████░░░░░ 62%
Rank 2:                 ████████████████░░░░░░░░░ 52%  
Rank 3:                 ██████████████░░░░░░░░░░░ 45%
Rank 4:                 ████████████░░░░░░░░░░░░░ 38%
Rank 5:                 ██████████░░░░░░░░░░░░░░░ 32%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 6.3 Running the Evaluation

```bash
# Setup BioASQ dataset and index
python scripts/setup_bioasq.py

# Run evaluation with TinyLlama (default)
python scripts/evaluate_bioasq.py

# Run evaluation with Phi-2
python scripts/evaluate_bioasq.py --config config-phi2.yaml

# Run evaluation on subset (faster)
python scripts/evaluate_bioasq.py --num-questions 20

# Run with reduced output
python scripts/evaluate_bioasq.py --quiet
```

#### Expected Evaluation Output

```
======================================================================
BioASQ RAG EVALUATION
======================================================================

[1/100] Is Hirschsprung disease a mendelian or a multifactorial disorder?...
  ✓ P@K: 0.60 | R@K: 0.22 | ROUGE-1: 0.24 | Time: 21.3s
[2/100] List signaling molecules (ligands) that interact with the receptor EGFR?...
  ✓ P@K: 0.40 | R@K: 0.15 | ROUGE-1: 0.18 | Time: 19.8s
[3/100] Is the protein Papilin secreted?...
  ✓ P@K: 0.80 | R@K: 0.30 | ROUGE-1: 0.31 | Time: 20.5s
[4/100] Are long non coding RNAs spliced?...
  ✓ P@K: 0.60 | R@K: 0.25 | ROUGE-1: 0.22 | Time: 21.8s
...
[100/100] Has Denosumab (Prolia) been approved by FDA?...
  ✓ P@K: 0.80 | R@K: 0.25 | ROUGE-1: 0.21 | Time: 22.1s

======================================================================
EVALUATION SUMMARY
======================================================================

📊 Retrieval Metrics (on 100 questions with ground truth):
   Precision@K:  0.533
   Recall@K:     0.190
   MRR:          0.702
   Hit Rate:     79.0%

📝 Generation Metrics (on 100 questions with answers):
   ROUGE-1:      0.186
   ROUGE-L:      0.132

⏱️  Timing:
   Avg Retrieval: 1469ms
   Avg Generation: 20687ms
   Avg Total:     22156ms

======================================================================
Results saved to evaluation_results.json
```

### 6.4 Unit Test Results

```bash
$ pytest tests/ -v

========================= test session starts ==========================
tests/test_rag.py::TestConfig::test_default_config PASSED
tests/test_rag.py::TestConfig::test_config_defaults PASSED
tests/test_rag.py::TestConfig::test_config_yaml_roundtrip PASSED
tests/test_rag.py::TestGuardrails::test_allow_safe_query PASSED
tests/test_rag.py::TestGuardrails::test_block_harmful_query PASSED
tests/test_rag.py::TestGuardrails::test_block_long_query PASSED
tests/test_rag.py::TestGuardrails::test_block_empty_query PASSED
tests/test_rag.py::TestGuardrails::test_block_prompt_injection PASSED
tests/test_rag.py::TestGuardrails::test_sanitize_query PASSED
tests/test_rag.py::TestQueryRefiner::test_detect_ambiguous_query PASSED
tests/test_rag.py::TestQueryRefiner::test_detect_short_query PASSED
tests/test_rag.py::TestQueryRefiner::test_accept_good_query PASSED
tests/test_rag.py::TestDocumentChunker::test_chunk_short_text PASSED
tests/test_rag.py::TestDocumentChunker::test_chunk_long_text PASSED
tests/test_rag.py::TestDocumentChunker::test_chunk_with_metadata PASSED
tests/test_rag.py::TestDocumentChunker::test_chunk_empty_text PASSED
tests/test_rag.py::TestDocument::test_document_creation PASSED
tests/test_rag.py::TestDocument::test_document_serialization PASSED
tests/test_rag.py::TestRetrievalResult::test_format_for_context PASSED
========================= 19 passed in 2.34s ===========================
```

---

## 7. Retrieval Mechanism Deep Dive

### 7.1 Embedding Process

```
                        EMBEDDING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Text                    Tokenization               Encoding
─────────────                 ────────────               ────────
"Is papilin secreted?"  ──▶  [is, papilin, ...]  ──────▶  SentenceTransformer
                                   │                         │
                                   ▼                         ▼
                             Token IDs              Transformer Layers
                            [2054, 2003,                    │
                             14751, 136]                    │
                                                           ▼
                                                    Mean Pooling
                                                           │
                                                           ▼
                                                    L2 Normalization
                                                           │
                                                           ▼
                                                    [0.12, -0.34, 0.56, ...]
                                                    (384 dimensions)
```

#### Model Details: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Architecture | BERT-based transformer |
| Layers | 6 |
| Hidden size | 384 |
| Parameters | 22M |
| Training | Contrastive learning on 1B+ pairs |
| Speed | ~14,000 sentences/sec (CPU) |

### 7.2 FAISS Index Structure

```
                          FAISS IndexFlatIP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure: Flat (brute-force) Inner Product index

┌─────────────────────────────────────────────────────────┐
│                     Index Matrix                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Doc 0: [0.12, -0.34, 0.56, ..., 0.08]  (384-d) │   │
│  │ Doc 1: [0.23, 0.45, -0.67, ..., 0.15]  (384-d) │   │
│  │ Doc 2: [-0.18, 0.29, 0.41, ..., -0.22] (384-d) │   │
│  │ ...                                              │   │
│  │ Doc N: [0.09, -0.56, 0.33, ..., 0.44]  (384-d) │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Search: query · doc = similarity score                 │
│  (Inner product = cosine similarity for normalized vecs)│
└─────────────────────────────────────────────────────────┘

Why IndexFlatIP?
• Exact search (no approximation error)
• Fast for < 1M documents
• Simple to implement and debug
• Memory efficient for our scale
```

### 7.3 Retrieval Algorithm

```python
# Simplified retrieval flow (see src/retriever.py for full implementation)

def retrieve(query: str, top_k: int = 5) -> List[RetrievalResult]:
    # Step 1: Encode query
    query_embedding = embedding_model.encode(query)  # Shape: (384,)
    
    # Step 2: FAISS search
    scores, indices = faiss_index.search(
        query_embedding.reshape(1, -1),  # Shape: (1, 384)
        k=top_k
    )
    # scores: [[0.87, 0.72, 0.65, 0.58, 0.45]]
    # indices: [[42, 156, 89, 23, 201]]
    
    # Step 3: Filter by threshold
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= similarity_threshold:  # Default: 0.3
            results.append(RetrievalResult(
                document=documents[idx],
                score=score,
                rank=len(results) + 1
            ))
    
    return results
```

### 7.4 Document Chunking Strategy

The system supports two chunking approaches depending on the data source:

#### BioASQ Dataset (Pre-chunked Passages)

For the BioASQ benchmark dataset, passages are **pre-chunked** from Hugging Face:

```
BioASQ Passage Structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source: rag-datasets/rag-mini-bioasq from Hugging Face

┌─────────────────────────────────────────────────────────┐
│  Passage ID: 20598273                                    │
│  Text: "Coding sequence mutations in RET and EDNRB      │
│         are involved in Hirschsprung disease..."        │
│  Avg Length: ~300 characters                             │
└─────────────────────────────────────────────────────────┘

Dataset Statistics:
• Total passages: ~4,700+
• Each passage = 1 indexed document (no additional chunking)
• Pre-processed for biomedical domain
• Passage IDs map to ground truth for evaluation
```

#### Custom Documents (Fixed-size Chunking)

For your own documents in `corpus/`, the system uses fixed-size character chunking:

```
                        CHUNKING EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original Document (1500 chars):
┌─────────────────────────────────────────────────────────┐
│ Machine learning is a subset of artificial              │
│ intelligence. It focuses on building systems that       │
│ learn from data. Deep learning is a subset of ML.       │
│ [... more content ...]                                  │
│ Neural networks are inspired by the human brain.        │
│ They consist of layers of interconnected nodes.         │
└─────────────────────────────────────────────────────────┘

Chunking Parameters (configurable in config.yaml):
• chunk_size: 500 characters
• chunk_overlap: 50 characters

Result:
┌─────────────────┐
│     Chunk 0     │ chars 0-500
│ "Machine learn- │
│  ing is a sub-  │
│  set of..."     │
└────────┬────────┘
         │ 50 char overlap
┌────────▼────────┐
│     Chunk 1     │ chars 450-950
│ "...of ML. Deep │
│  learning is a  │
│  subset of..."  │
└────────┬────────┘
         │ 50 char overlap
┌────────▼────────┐
│     Chunk 2     │ chars 900-1400
│ "...Neural net- │
│  works are in-  │
│  spired by..."  │
└─────────────────┘

Benefits of Overlap:
• Preserves context at chunk boundaries
• Prevents splitting important phrases
• Improves retrieval for queries spanning chunk edges
```

---

## 8. LLM Integration

### 8.1 Quantization Deep Dive

```
                    QUANTIZATION COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: TinyLlama 1.1B (1.1 billion parameters) - Current Default

Precision     Memory      Speed     Quality    Selected
────────────────────────────────────────────────────────
FP32          4.4 GB      1.0x      100%       ❌ Large
FP16          2.2 GB      1.2x      99.9%      Consider
Q8_0          1.2 GB      1.5x      99.5%      Consider
Q5_K_M        850 MB      1.8x      99.0%      Consider
Q4_K_M        670 MB      2.0x      98.5%      ✅ SELECTED
Q4_0          600 MB      2.1x      97.0%      ❌ Quality

Why TinyLlama with Q4_K_M?
• Extremely lightweight: only 670MB on disk
• Fast inference even on modest CPUs
• Good for 8GB RAM systems
• Trained on 3 trillion tokens for its size
• Best choice for resource-constrained laptops

Recommended: Phi-2 Q4_K_M (1.6GB)
• Better quality than TinyLlama
• Works on 8GB RAM machines
• Optimal for laptop inference

Alternative: Mistral 7B Q4_K_M (4.4GB)
• Highest quality
• Requires 16GB RAM
• Higher quality responses
• Requires 16GB+ RAM
• ~3x slower inference
• Recommended for production use with sufficient hardware
```

### 8.2 llama.cpp Integration

```
                    LLAMA.CPP ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────┐
│                    Python Layer                          │
│                 (llama-cpp-python)                       │
└─────────────────────────┬───────────────────────────────┘
                          │ ctypes bindings
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    C++ Layer                             │
│                   (llama.cpp)                            │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │    GGML     │  │   Memory    │  │   Inference     │ │
│  │   Tensors   │  │   Mapping   │  │   Engine        │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                                                          │
│  Optimizations:                                          │
│  • SIMD (AVX2, AVX-512, ARM NEON)                       │
│  • Batch processing                                      │
│  • KV cache for context reuse                           │
│  • Optional GPU offload (CUDA, Metal, OpenCL)           │
└─────────────────────────────────────────────────────────┘
```

### 8.3 Prompt Template

```python
# Chat format for TinyLlama/Mistral/Llama instruction-tuned models

def build_prompt(user_query: str, context: str, system_prompt: str) -> str:
    """
    Build prompt in Llama/Mistral instruction format (used by TinyLlama).
    Note: <s> token is added automatically by llama.cpp
    """
    return f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

Based on the following context, please answer the question.
If the context doesn't contain enough information, say so clearly.

**Context:**
{context}

**Question:** {user_query}

**Answer:** [/INST]"""
```

### 8.4 Generation Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_ctx` | 4096 | Context window (prompt + response) |
| `max_tokens` | 512 | Maximum response length |
| `temperature` | 0.7 | Creativity (0=deterministic, 1=random) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `repeat_penalty` | 1.1 | Reduces repetition |
| `stop_sequences` | `["Human:", "User:"]` | Response terminators |

---

## 9. Additional Features

### 9.1 Content Guardrails

The system includes multi-layer safety features:

```
                        GUARDRAILS PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query Input
     │
     ▼
┌─────────────────────┐
│   Length Check      │──▶ Block if > 2000 chars
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Empty Check       │──▶ Block if empty/whitespace
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Prompt Injection   │──▶ Block patterns like:
│     Detection       │    "ignore previous instructions"
└──────────┬──────────┘    "you are now a different AI"
           ▼               "[INST]" embedded in query
┌─────────────────────┐
│   Topic Filter      │──▶ Block keywords:
│                     │    violence, illegal, hack, etc.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Domain Check      │──▶ Optional: restrict to allowed topics
└──────────┬──────────┘
           ▼
      ✅ ALLOW
```

### 9.2 Query Refinement

Detects and handles ambiguous queries:

```python
# Examples of query analysis

"what?"           → "Your question seems incomplete. Could you provide more details?"
"help"            → "I'd be happy to help! What would you like to know?"
"it"              → "Could you clarify what you're referring to?"
"maybe something" → "Let me help clarify - what specific aspect would you like to know?"
```

### 9.3 Source Attribution

Every response includes verifiable sources:

```
Sources Table:
┌──────┬───────────────────┬───────────┬────────────────────────────┐
│ Rank │ Source            │ Relevance │ Preview                    │
├──────┼───────────────────┼───────────┼────────────────────────────┤
│ 1    │ rag_basics.txt    │ 87%       │ RAG combines retrieval...  │
│ 2    │ ml_concepts.md    │ 72%       │ Machine learning is...     │
│ 3    │ ai_overview.pdf   │ 65%       │ Artificial intelligence... │
└──────┴───────────────────┴───────────┴────────────────────────────┘
```

### 9.4 Streaming Responses

Real-time token generation for interactive experience:

```python
# Enable streaming mode
python main.py query "Which miRNAs are biomarkers for ovarian cancer?" --stream

# Tokens appear as they're generated:
# The... The following... The following miRNAs... could be used... as biomarkers...
```

### 9.5 Multi-Format Document Support

| Format | Library | Notes |
|--------|---------|-------|
| `.txt` | Built-in | Plain text |
| `.md` | Built-in | Markdown |
| `.pdf` | pypdf | Text extraction |
| `.docx` | python-docx | Word documents |

### 9.6 BioASQ Benchmark Integration

Built-in support for the BioASQ biomedical QA dataset:

```bash
# Setup BioASQ dataset
python scripts/setup_bioasq.py

# Run evaluation
python scripts/evaluate_bioasq.py --num-questions 50
```

---

## Appendix A: Configuration Reference

```yaml
# Complete config.yaml with all options

llm:
  model_path: "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  n_ctx: 4096                    # Context window
  max_tokens: 512                # Max response tokens
  temperature: 0.7               # 0-1, higher = more creative
  top_p: 0.9                     # Nucleus sampling
  n_threads: 0                   # 0 = auto-detect
  n_gpu_layers: 0                # GPU layers (0 = CPU only)
  repeat_penalty: 1.1            # Reduce repetition
  stop_sequences:
    - "Human:"
    - "User:"
    - "\n\n\n"

embedding:
  model_name: "all-MiniLM-L6-v2" # Or "all-mpnet-base-v2"
  device: "cpu"                  # "cpu", "cuda", "mps"
  batch_size: 32                 # Documents per batch
  normalize: true                # L2 normalize embeddings

retriever:
  top_k: 5                       # Documents to retrieve
  similarity_threshold: 0.3     # Minimum relevance score
  chunk_size: 500                # Characters per chunk
  chunk_overlap: 50              # Overlap between chunks
  index_path: "models/faiss_index"
  documents_path: "models/documents.pkl"

guardrails:
  enabled: true
  blocked_topics:
    - "illegal activities"
    - "violence"
    - "hate speech"
    - "malware"
    - "exploit"
    - "hack"
    - "weapon"
    - "drug synthesis"
    - "terrorism"
  allowed_domains: []            # Empty = all allowed
  max_query_length: 2000
  rejection_message: "I'm sorry, but I cannot assist with that topic."

system_prompt: |
  You are a helpful AI assistant with access to a knowledge base.
  When answering questions:
  1. Use the provided context to give accurate, sourced answers
  2. If the context doesn't contain relevant information, say so clearly
  3. Cite your sources by referencing the document chunks provided
  4. Be concise but thorough
  5. If you're uncertain, express that uncertainty

verbose: false
corpus_dir: "corpus"
```

---

## Appendix B: Troubleshooting Guide

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `Model file not found` | Model not downloaded | Run `./scripts/download_model.sh` |
| `Out of memory` | Model too large | Use TinyLlama or reduce n_ctx |
| `Slow generation` | CPU bottleneck | Enable GPU layers or use smaller model |
| `No documents found` | Empty corpus | Add files to `corpus/` and run `index` |
| `Import errors` | Missing dependencies | `pip install -r requirements.txt` |
| `Low relevance scores` | Poor chunking | Adjust chunk_size/overlap |
| `Garbled output` | Model mismatch | Verify correct GGUF format |

---

*Document Version: 1.0*  
*Last Updated: December 2024*

