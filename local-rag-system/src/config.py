"""
Configuration Management
========================

Centralized configuration for the RAG system with sensible defaults
optimized for laptop performance (Intel i7/Apple M1-M3, 16GB RAM).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class LLMConfig:
    """Configuration for the language model."""
    
    # Model path (GGUF format for llama.cpp)
    model_path: str = ""
    
    # Context window size (tokens)
    n_ctx: int = 4096
    
    # Number of tokens to generate
    max_tokens: int = 512
    
    # Temperature for generation (lower = more focused)
    temperature: float = 0.7
    
    # Top-p sampling
    top_p: float = 0.9
    
    # Number of CPU threads (0 = auto-detect)
    n_threads: int = 0
    
    # Number of GPU layers to offload (0 for CPU-only)
    n_gpu_layers: int = 0
    
    # Repeat penalty to reduce repetition
    repeat_penalty: float = 1.1
    
    # Stop sequences
    stop_sequences: List[str] = field(default_factory=lambda: ["Human:", "User:", "\n\n\n"])


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""
    
    # Model name from HuggingFace (small, efficient model)
    model_name: str = "all-MiniLM-L6-v2"
    
    # Device to run on ('cpu', 'cuda', or 'mps' for Apple Silicon)
    device: str = "cpu"
    
    # Batch size for encoding
    batch_size: int = 32
    
    # Normalize embeddings
    normalize: bool = True


@dataclass
class RetrieverConfig:
    """Configuration for the retrieval system."""
    
    # Number of documents to retrieve
    top_k: int = 5
    
    # Minimum similarity threshold (0-1)
    similarity_threshold: float = 0.3
    
    # Chunk size for document splitting (characters)
    chunk_size: int = 500
    
    # Chunk overlap for context continuity
    chunk_overlap: int = 50
    
    # Path to FAISS index
    index_path: str = ""
    
    # Path to document store
    documents_path: str = ""


@dataclass
class GuardrailsConfig:
    """Configuration for content guardrails."""
    
    # Enable guardrails
    enabled: bool = True
    
    # Blocked topics (keywords that trigger rejection)
    blocked_topics: List[str] = field(default_factory=lambda: [
        "illegal activities",
        "violence",
        "hate speech",
        "malware",
        "exploit",
        "hack",
        "weapon",
        "drug synthesis",
        "terrorism"
    ])
    
    # Allowed domains (empty = all allowed except blocked)
    allowed_domains: List[str] = field(default_factory=list)
    
    # Maximum query length (characters)
    max_query_length: int = 2000
    
    # Rejection message
    rejection_message: str = "I'm sorry, but I cannot assist with that topic. Please ask about something else."


@dataclass
class RAGConfig:
    """Main configuration class combining all components."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    
    # System prompt for the LLM
    system_prompt: str = """You are a helpful AI assistant with access to a knowledge base. 
When answering questions:
1. Use the provided context to give accurate, sourced answers
2. If the context doesn't contain relevant information, say so clearly
3. Cite your sources by referencing the document chunks provided
4. Be concise but thorough
5. If you're uncertain, express that uncertainty"""
    
    # Enable verbose logging
    verbose: bool = False
    
    # Corpus directory
    corpus_dir: str = "data/bioasq"
    
    @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'llm' in data:
            for key, value in data['llm'].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)
        
        if 'embedding' in data:
            for key, value in data['embedding'].items():
                if hasattr(config.embedding, key):
                    setattr(config.embedding, key, value)
        
        if 'retriever' in data:
            for key, value in data['retriever'].items():
                if hasattr(config.retriever, key):
                    setattr(config.retriever, key, value)
        
        if 'guardrails' in data:
            for key, value in data['guardrails'].items():
                if hasattr(config.guardrails, key):
                    setattr(config.guardrails, key, value)
        
        for key in ['system_prompt', 'verbose', 'corpus_dir']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            'llm': {
                'model_path': self.llm.model_path,
                'n_ctx': self.llm.n_ctx,
                'max_tokens': self.llm.max_tokens,
                'temperature': self.llm.temperature,
                'top_p': self.llm.top_p,
                'n_threads': self.llm.n_threads,
                'n_gpu_layers': self.llm.n_gpu_layers,
                'repeat_penalty': self.llm.repeat_penalty,
                'stop_sequences': self.llm.stop_sequences,
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'device': self.embedding.device,
                'batch_size': self.embedding.batch_size,
                'normalize': self.embedding.normalize,
            },
            'retriever': {
                'top_k': self.retriever.top_k,
                'similarity_threshold': self.retriever.similarity_threshold,
                'chunk_size': self.retriever.chunk_size,
                'chunk_overlap': self.retriever.chunk_overlap,
                'index_path': self.retriever.index_path,
                'documents_path': self.retriever.documents_path,
            },
            'guardrails': {
                'enabled': self.guardrails.enabled,
                'blocked_topics': self.guardrails.blocked_topics,
                'allowed_domains': self.guardrails.allowed_domains,
                'max_query_length': self.guardrails.max_query_length,
                'rejection_message': self.guardrails.rejection_message,
            },
            'system_prompt': self.system_prompt,
            'verbose': self.verbose,
            'corpus_dir': self.corpus_dir,
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> RAGConfig:
    """Get default configuration optimized for laptop usage."""
    config = RAGConfig()
    
    # Auto-detect optimal settings
    import platform
    
    # Set device based on platform
    if platform.system() == "Darwin" and platform.processor() == "arm":
        config.embedding.device = "mps"  # Apple Silicon
    else:
        config.embedding.device = "cpu"
    
    # Set paths relative to project root (use BioASQ pre-built index by default)
    project_root = Path(__file__).parent.parent
    config.retriever.index_path = str(project_root / "models" / "bioasq_faiss_index")
    config.retriever.documents_path = str(project_root / "models" / "bioasq_documents.pkl")
    config.corpus_dir = str(project_root / "data" / "bioasq")
    
    # Set default model path (TinyLlama)
    config.llm.model_path = str(project_root / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    return config

