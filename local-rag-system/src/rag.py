"""
RAG Pipeline
============

The core Retrieval-Augmented Generation pipeline that combines:
1. Query processing and guardrails
2. Document retrieval
3. Context assembly
4. LLM generation with sources

Pipeline Flow:
--------------
Query → Guardrails Check → Query Refinement (optional) → Retrieval
      → Context Assembly → Prompt Construction → LLM Generation
      → Response Guardrails → Output with Sources

Source Attribution:
-------------------
Each generated response includes references to the retrieved documents,
allowing users to verify information and trace back to original sources.
"""

import time
from typing import List, Optional, Tuple, Generator, Dict, Any
from dataclasses import dataclass

from .config import RAGConfig
from .llm import LocalLLM, create_llm
from .embeddings import EmbeddingModel, DocumentChunker
from .retriever import VectorRetriever, RetrievalResult, Document, load_corpus_from_directory
from .guardrails import ContentGuardrails, QueryRefiner, GuardrailAction


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    sources: List[RetrievalResult]
    query: str
    refined_query: Optional[str]
    generation_time: float
    retrieval_time: float
    total_time: float
    tokens_used: int
    guardrail_warning: Optional[str] = None
    
    def format_with_sources(self) -> str:
        """Format the response with source citations."""
        output = [self.answer]
        
        if self.sources:
            output.append("\n\n📚 **Sources:**")
            for result in self.sources:
                source_name = result.document.source or f"Document {result.document.doc_id}"
                score_pct = int(result.score * 100)
                output.append(f"  [{result.rank}] {source_name} (relevance: {score_pct}%)")
        
        return "\n".join(output)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "sources": [
                {
                    "rank": r.rank,
                    "score": r.score,
                    "source": r.document.source,
                    "text_preview": r.document.text[:200],
                }
                for r in self.sources
            ],
            "query": self.query,
            "refined_query": self.refined_query,
            "timing": {
                "generation_ms": int(self.generation_time * 1000),
                "retrieval_ms": int(self.retrieval_time * 1000),
                "total_ms": int(self.total_time * 1000),
            },
            "tokens_used": self.tokens_used,
        }


class RAGPipeline:
    """
    Main RAG pipeline orchestrating retrieval and generation.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.verbose = config.verbose
        
        # Initialize components
        print("Initializing RAG pipeline...")
        
        # Embedding model (always needed)
        self.embedding_model = EmbeddingModel(config.embedding)
        
        # Retriever
        self.retriever = VectorRetriever(config.retriever, self.embedding_model)
        
        # LLM
        self.llm = create_llm(config.llm)
        
        # Guardrails
        self.guardrails = ContentGuardrails(config.guardrails)
        
        # Query refinement
        self.query_refiner = QueryRefiner()
        
        print("RAG pipeline initialized!")
    
    def index_corpus(self, corpus_dir: Optional[str] = None, save: bool = True) -> int:
        """
        Index documents from the corpus directory.
        
        Args:
            corpus_dir: Directory containing documents (default from config)
            save: Whether to save the index to disk
            
        Returns:
            Number of document chunks indexed
        """
        corpus_dir = corpus_dir or self.config.corpus_dir
        
        print(f"Indexing corpus from: {corpus_dir}")
        
        chunker = DocumentChunker(
            chunk_size=self.config.retriever.chunk_size,
            chunk_overlap=self.config.retriever.chunk_overlap,
        )
        
        documents = load_corpus_from_directory(corpus_dir, chunker)
        
        if not documents:
            print("No documents found to index!")
            return 0
        
        self.retriever.build_index(documents)
        
        if save:
            self.retriever.save_index()
        
        return len(documents)
    
    def query(
        self,
        query: str,
        stream: bool = False,
        skip_guardrails: bool = False,
    ) -> RAGResponse | Generator[str, None, RAGResponse]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            stream: If True, stream the response tokens
            skip_guardrails: Skip guardrail checks (for testing)
            
        Returns:
            RAGResponse or generator yielding tokens then RAGResponse
        """
        start_time = time.time()
        refined_query = None
        guardrail_warning = None
        
        # Step 1: Guardrails check
        if not skip_guardrails:
            guardrail_result = self.guardrails.check_query(query)
            
            if guardrail_result.action == GuardrailAction.BLOCK:
                return RAGResponse(
                    answer=self.guardrails.get_rejection_message(),
                    sources=[],
                    query=query,
                    refined_query=None,
                    generation_time=0,
                    retrieval_time=0,
                    total_time=time.time() - start_time,
                    tokens_used=0,
                    guardrail_warning=guardrail_result.reason,
                )
            
            if guardrail_result.action == GuardrailAction.WARN:
                guardrail_warning = guardrail_result.reason
        
        # Step 2: Query refinement check
        needs_refinement, clarification = self.query_refiner.analyze_query(query)
        if needs_refinement and clarification:
            # In interactive mode, this would prompt the user
            # For now, we proceed but note the potential issue
            if self.verbose:
                print(f"Note: {clarification}")
        
        # Step 3: Retrieval
        retrieval_start = time.time()
        search_query = refined_query or query
        results = self.retriever.retrieve(search_query)
        retrieval_time = time.time() - retrieval_start
        
        if self.verbose:
            print(f"Retrieved {len(results)} relevant documents in {retrieval_time:.2f}s")
        
        # Step 4: Build context
        context = self._build_context(results)
        
        # Step 5: Build prompt
        prompt = self._build_prompt(query, context, results)
        
        # Step 6: Generate response
        generation_start = time.time()
        
        if stream:
            return self._stream_response(
                query=query,
                refined_query=refined_query,
                prompt=prompt,
                results=results,
                start_time=start_time,
                retrieval_time=retrieval_time,
                guardrail_warning=guardrail_warning,
            )
        else:
            response_text = self.llm.generate(
                prompt,
                system_prompt=self.config.system_prompt,
                stream=False,
            )
            generation_time = time.time() - generation_start
            
            # Check response guardrails
            if not skip_guardrails:
                response_check = self.guardrails.check_response(response_text)
                if response_check.action == GuardrailAction.BLOCK:
                    response_text = "I apologize, but I cannot provide that response."
            
            return RAGResponse(
                answer=response_text,
                sources=results,
                query=query,
                refined_query=refined_query,
                generation_time=generation_time,
                retrieval_time=retrieval_time,
                total_time=time.time() - start_time,
                tokens_used=self.llm.count_tokens(prompt + response_text),
                guardrail_warning=guardrail_warning,
            )
    
    def _stream_response(
        self,
        query: str,
        refined_query: Optional[str],
        prompt: str,
        results: List[RetrievalResult],
        start_time: float,
        retrieval_time: float,
        guardrail_warning: Optional[str],
    ) -> Generator[str, None, RAGResponse]:
        """Stream response tokens and return final RAGResponse."""
        generation_start = time.time()
        response_parts = []
        
        for token in self.llm.generate(
            prompt,
            system_prompt=self.config.system_prompt,
            stream=True,
        ):
            response_parts.append(token)
            yield token
        
        response_text = "".join(response_parts)
        generation_time = time.time() - generation_start
        
        return RAGResponse(
            answer=response_text,
            sources=results,
            query=query,
            refined_query=refined_query,
            generation_time=generation_time,
            retrieval_time=retrieval_time,
            total_time=time.time() - start_time,
            tokens_used=self.llm.count_tokens(prompt + response_text),
            guardrail_warning=guardrail_warning,
        )
    
    def _build_context(self, results: List[RetrievalResult]) -> str:
        """Build context string from retrieval results."""
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for result in results:
            source_label = result.document.source or f"Document {result.document.doc_id}"
            context_parts.append(
                f"[Source: {source_label}]\n{result.document.text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        results: List[RetrievalResult],
    ) -> str:
        """Build the full prompt for the LLM."""
        # Include source information for citation
        source_list = ", ".join(
            r.document.source or f"Doc {r.document.doc_id}"
            for r in results[:3]
        ) if results else "None"
        
        prompt = f"""Based on the following context, please answer the question. 
If the context doesn't contain enough information to fully answer the question, say so clearly.
Cite specific sources when possible.

**Available Sources:** {source_list}

**Context:**
{context}

**Question:** {query}

**Answer:**"""
        
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "retriever": self.retriever.get_stats(),
            "embedding_model": self.embedding_model.get_model_info(),
            "llm": self.llm.get_model_info(),
            "guardrails_enabled": self.config.guardrails.enabled,
        }
    
    def interactive_query(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Process a query with interactive refinement support.
        
        Returns:
            Tuple of (response_or_clarification, None if final response)
        """
        # Check if query needs refinement
        needs_refinement, clarification = self.query_refiner.analyze_query(query)
        
        if needs_refinement and clarification:
            return clarification, "needs_refinement"
        
        # Process normally
        response = self.query(query)
        return response.format_with_sources(), None

