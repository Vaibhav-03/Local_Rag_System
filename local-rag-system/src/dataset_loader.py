"""
Dataset Loader for RAG Benchmarks
==================================

Loads datasets from Hugging Face for RAG evaluation.
Supports the rag-mini-bioasq dataset and similar formats.

Dataset Structure:
- text-corpus: Passages with IDs for retrieval
- question-answer-passages: Questions with correct answers and relevant passage IDs
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .retriever import Document


@dataclass
class BioASQPassage:
    """A passage from the BioASQ corpus."""
    passage_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BioASQQuestion:
    """A question from the BioASQ test set."""
    question_id: str
    question: str
    answer: Optional[str] = None
    relevant_passage_ids: List[str] = field(default_factory=list)
    
    def has_ground_truth(self) -> bool:
        """Check if this question has ground truth for evaluation."""
        return bool(self.relevant_passage_ids) or bool(self.answer)


class BioASQDatasetLoader:
    """
    Loader for the rag-mini-bioasq dataset from Hugging Face.
    
    Dataset: https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq
    
    Usage:
        loader = BioASQDatasetLoader()
        loader.download()
        passages = loader.get_passages()
        questions = loader.get_questions()
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/bioasq")
        self.corpus: List[BioASQPassage] = []
        self.questions: List[BioASQQuestion] = []
        self._loaded = False
    
    def download(self, force: bool = False) -> None:
        """
        Download the dataset from Hugging Face.
        
        Args:
            force: Re-download even if cached
        """
        from datasets import load_dataset
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        corpus_cache = self.cache_dir / "corpus.json"
        questions_cache = self.cache_dir / "questions.json"
        
        # Check cache
        if not force and corpus_cache.exists() and questions_cache.exists():
            print("Loading from cache...")
            self._load_from_cache()
            return
        
        print("Downloading rag-mini-bioasq dataset from Hugging Face...")
        
        # Load text corpus
        print("Loading text corpus...")
        corpus_ds = load_dataset(
            "rag-datasets/rag-mini-bioasq",
            "text-corpus",
            split="passages"
        )
        
        # Load question-answer-passages
        print("Loading questions and answers...")
        qa_ds = load_dataset(
            "rag-datasets/rag-mini-bioasq",
            "question-answer-passages",
            split="test"
        )
        
        # Process corpus
        print(f"Processing {len(corpus_ds)} passages...")
        self.corpus = []
        for item in corpus_ds:
            passage = BioASQPassage(
                passage_id=str(item.get("id", item.get("passage_id", ""))),
                text=item.get("passage", item.get("text", "")),
                metadata={k: v for k, v in item.items() if k not in ["id", "passage_id", "passage", "text"]}
            )
            self.corpus.append(passage)
        
        # Process questions
        print(f"Processing {len(qa_ds)} questions...")
        self.questions = []
        for item in qa_ds:
            # Handle different possible field names
            question_text = item.get("question", item.get("query", ""))
            answer = item.get("answer", item.get("answers", None))
            
            # Handle answer being a list
            if isinstance(answer, list):
                answer = answer[0] if answer else None
            
            # Get relevant passage IDs
            relevant_ids = item.get("relevant_passage_ids", 
                                   item.get("positive_passages", 
                                           item.get("passages", [])))
            
            # Handle different formats
            if relevant_ids:
                # If it's a string representation of a list, parse it
                if isinstance(relevant_ids, str):
                    import ast
                    try:
                        relevant_ids = ast.literal_eval(relevant_ids)
                    except:
                        relevant_ids = []
                
                # If it's a dict with 'id' field (list of dicts)
                if relevant_ids and isinstance(relevant_ids[0], dict):
                    relevant_ids = [str(p.get("id", p.get("passage_id", ""))) for p in relevant_ids]
                else:
                    # Convert to list of strings
                    relevant_ids = [str(pid) for pid in relevant_ids]
            else:
                relevant_ids = []
            
            question = BioASQQuestion(
                question_id=str(item.get("id", item.get("question_id", len(self.questions)))),
                question=question_text,
                answer=answer,
                relevant_passage_ids=relevant_ids
            )
            self.questions.append(question)
        
        # Cache the processed data
        self._save_to_cache()
        self._loaded = True
        
        print(f"✓ Loaded {len(self.corpus)} passages and {len(self.questions)} questions")
    
    def _save_to_cache(self) -> None:
        """Save processed data to cache."""
        corpus_cache = self.cache_dir / "corpus.json"
        questions_cache = self.cache_dir / "questions.json"
        
        with open(corpus_cache, 'w') as f:
            json.dump([{
                "passage_id": p.passage_id,
                "text": p.text,
                "metadata": p.metadata
            } for p in self.corpus], f)
        
        with open(questions_cache, 'w') as f:
            json.dump([{
                "question_id": q.question_id,
                "question": q.question,
                "answer": q.answer,
                "relevant_passage_ids": q.relevant_passage_ids
            } for q in self.questions], f)
    
    def _load_from_cache(self) -> None:
        """Load data from cache."""
        corpus_cache = self.cache_dir / "corpus.json"
        questions_cache = self.cache_dir / "questions.json"
        
        with open(corpus_cache, 'r') as f:
            corpus_data = json.load(f)
            self.corpus = [
                BioASQPassage(
                    passage_id=p["passage_id"],
                    text=p["text"],
                    metadata=p.get("metadata", {})
                ) for p in corpus_data
            ]
        
        with open(questions_cache, 'r') as f:
            questions_data = json.load(f)
            self.questions = [
                BioASQQuestion(
                    question_id=q["question_id"],
                    question=q["question"],
                    answer=q.get("answer"),
                    relevant_passage_ids=q.get("relevant_passage_ids", [])
                ) for q in questions_data
            ]
        
        self._loaded = True
        print(f"✓ Loaded {len(self.corpus)} passages and {len(self.questions)} questions from cache")
    
    def get_passages(self) -> List[BioASQPassage]:
        """Get all passages from the corpus."""
        if not self._loaded:
            self.download()
        return self.corpus
    
    def get_questions(self) -> List[BioASQQuestion]:
        """Get all test questions."""
        if not self._loaded:
            self.download()
        return self.questions
    
    def get_passage_by_id(self, passage_id: str) -> Optional[BioASQPassage]:
        """Get a specific passage by ID."""
        if not self._loaded:
            self.download()
        for passage in self.corpus:
            if passage.passage_id == passage_id:
                return passage
        return None
    
    def to_documents(self) -> List[Document]:
        """
        Convert passages to Document objects for the retriever.
        
        Returns:
            List of Document objects ready for indexing
        """
        if not self._loaded:
            self.download()
        
        documents = []
        for i, passage in enumerate(self.corpus):
            doc = Document(
                text=passage.text,
                doc_id=i,
                chunk_id=0,  # Each passage is already a chunk
                source=f"bioasq_passage_{passage.passage_id}",
                metadata={
                    "passage_id": passage.passage_id,
                    "dataset": "rag-mini-bioasq",
                    **passage.metadata
                }
            )
            documents.append(doc)
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._loaded:
            self.download()
        
        questions_with_answer = sum(1 for q in self.questions if q.answer)
        questions_with_passages = sum(1 for q in self.questions if q.relevant_passage_ids)
        
        return {
            "num_passages": len(self.corpus),
            "num_questions": len(self.questions),
            "questions_with_answer": questions_with_answer,
            "questions_with_relevant_passages": questions_with_passages,
            "avg_passage_length": sum(len(p.text) for p in self.corpus) / len(self.corpus) if self.corpus else 0,
            "avg_relevant_passages": sum(len(q.relevant_passage_ids) for q in self.questions) / len(self.questions) if self.questions else 0,
        }


def export_corpus_to_files(loader: BioASQDatasetLoader, output_dir: str) -> int:
    """
    Export the corpus to text files in the corpus directory.
    
    Args:
        loader: BioASQDatasetLoader instance
        output_dir: Directory to save files
        
    Returns:
        Number of files created
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    passages = loader.get_passages()
    
    # Group passages into files (100 per file for manageability)
    batch_size = 100
    file_count = 0
    
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        file_path = output_path / f"bioasq_passages_{i:05d}.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for passage in batch:
                f.write(f"[Passage ID: {passage.passage_id}]\n")
                f.write(passage.text)
                f.write("\n\n---\n\n")
        
        file_count += 1
    
    print(f"Exported {len(passages)} passages to {file_count} files in {output_dir}")
    return file_count

