#!/usr/bin/env python3
"""
Setup BioASQ Dataset for RAG System
====================================

Downloads the rag-mini-bioasq dataset and prepares it for use
with the Local RAG System.

Usage:
    python scripts/setup_bioasq.py
    
This will:
1. Download the dataset from Hugging Face
2. Process passages and questions
3. Build the FAISS index
4. Save everything for use with the RAG system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset_loader import BioASQDatasetLoader
from src.config import get_default_config
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever


def main():
    print("=" * 60)
    print("BioASQ Dataset Setup for Local RAG System")
    print("=" * 60)
    print()
    
    # Initialize loader
    print("[1/4] Downloading BioASQ dataset...")
    loader = BioASQDatasetLoader(cache_dir=str(project_root / "data" / "bioasq"))
    loader.download()
    
    # Print stats
    stats = loader.get_stats()
    print()
    print("📊 Dataset Statistics:")
    print(f"   Passages: {stats['num_passages']}")
    print(f"   Questions: {stats['num_questions']}")
    print(f"   Questions with answers: {stats['questions_with_answer']}")
    print(f"   Questions with relevant passages: {stats['questions_with_relevant_passages']}")
    print(f"   Avg passage length: {stats['avg_passage_length']:.0f} chars")
    print(f"   Avg relevant passages per question: {stats['avg_relevant_passages']:.1f}")
    print()
    
    # Convert to documents
    print("[2/4] Converting passages to documents...")
    documents = loader.to_documents()
    print(f"   Created {len(documents)} documents")
    print()
    
    # Initialize embedding model and retriever
    print("[3/4] Building FAISS index...")
    config = get_default_config()
    
    # Update paths for BioASQ
    config.retriever.index_path = str(project_root / "models" / "bioasq_faiss_index")
    config.retriever.documents_path = str(project_root / "models" / "bioasq_documents.pkl")
    
    embedding_model = EmbeddingModel(config.embedding)
    retriever = VectorRetriever(config.retriever, embedding_model)
    
    # Build index
    retriever.build_index(documents)
    retriever.save_index()
    print()
    
    # Create config file for BioASQ
    print("[4/4] Creating BioASQ config file...")
    config.retriever.index_path = "models/bioasq_faiss_index"
    config.retriever.documents_path = "models/bioasq_documents.pkl"
    config.corpus_dir = "data/bioasq"
    
    config_path = project_root / "config-bioasq.yaml"
    config.to_yaml(str(config_path))
    print(f"   Saved config to {config_path}")
    print()
    
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print()
    print("To use the BioASQ dataset:")
    print()
    print("  1. Run the RAG system with BioASQ config:")
    print("     python main.py --config config-bioasq.yaml")
    print()
    print("  2. Run evaluation on BioASQ:")
    print("     python scripts/evaluate_bioasq.py")
    print()
    print("  3. Ask medical/biomedical questions!")
    print()
    
    # Show sample questions
    questions = loader.get_questions()[:5]
    print("Sample questions from the dataset:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q.question[:80]}...")
        if q.answer:
            print(f"     Answer: {q.answer[:60]}...")
    print()


if __name__ == "__main__":
    main()

