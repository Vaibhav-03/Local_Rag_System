"""
Embedding Model
===============

Handles text embedding using sentence-transformers.
Uses efficient, lightweight models suitable for laptop inference.

Model Choice: all-MiniLM-L6-v2
------------------------------
- Size: ~80MB
- Embedding dimension: 384
- Speed: ~14,000 sentences/second on CPU
- Quality: Strong performance on semantic similarity tasks
- Why: Best balance of speed, size, and quality for local deployment

Alternative options:
- all-mpnet-base-v2: Higher quality, slower (~420MB)
- paraphrase-MiniLM-L3-v2: Faster, lower quality (~60MB)
"""

import os
import pickle
from typing import List, Optional, Union
from pathlib import Path
import numpy as np

from .config import EmbeddingConfig


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    
    Provides efficient text encoding for both queries and documents.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding model.
        
        Args:
            config: Embedding configuration object
        """
        self.config = config
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {self.config.model_name}")
        

        device = self.config.device
        if device == "mps":

            import torch
            if not torch.backends.mps.is_available():
                print("MPS not available, falling back to CPU")
                device = "cpu"
        
        self.model = SentenceTransformer(
            self.config.model_name,
            device=device,
        )
        

        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            show_progress: Show progress bar for batch encoding
            
        Returns:
            Numpy array of embeddings, shape (n_texts, dimension)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        

        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query for retrieval.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding, shape (dimension,)
        """
        return self.encode(query)[0]
    
    def encode_documents(
        self,
        documents: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple documents for indexing.
        
        Args:
            documents: List of document texts
            show_progress: Show progress bar
            
        Returns:
            Document embeddings, shape (n_documents, dimension)
        """
        return self.encode(documents, show_progress=show_progress)
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding, shape (dimension,)
            document_embeddings: Document embeddings, shape (n_docs, dimension)
            
        Returns:
            Similarity scores, shape (n_docs,)
        """
        # If embeddings are normalized, dot product equals cosine similarity
        if self.config.normalize:
            return np.dot(document_embeddings, query_embedding)
        
        # Otherwise compute cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        
        return np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.model_name,
            "dimension": self.dimension,
            "device": self.config.device,
            "normalized": self.config.normalize,
        }


class DocumentChunker:
    """
    Utility for splitting documents into chunks for embedding.
    
    Uses overlapping chunks to maintain context across boundaries.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation
                for sep in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                }
                
                if metadata:
                    chunk_data["metadata"] = metadata
                
                chunks.append(chunk_data)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= chunks[-1]["start_char"] if chunks else 0:
                start = end  # Prevent infinite loop
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[dict],
        text_key: str = "text",
    ) -> List[dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dicts with text content
            text_key: Key for text content in document dict
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_id, doc in enumerate(documents):
            text = doc.get(text_key, "")
            metadata = {k: v for k, v in doc.items() if k != text_key}
            metadata["doc_id"] = doc_id
            
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks

