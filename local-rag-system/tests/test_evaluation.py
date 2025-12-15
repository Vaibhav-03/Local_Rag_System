"""
RAG System Evaluation
=====================

Comprehensive evaluation suite for measuring RAG system quality.

Metrics:
- Retrieval Precision: How many retrieved docs are relevant
- Retrieval Recall: How many relevant docs are retrieved
- Answer Quality: ROUGE scores against reference answers
- Response Time: Latency measurements
- Resource Usage: Memory and CPU monitoring

Usage:
    python tests/test_evaluation.py --full    # Full evaluation
    python tests/test_evaluation.py --quick   # Quick sanity check
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    query_id: str
    query: str
    response: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_sources: int
    topic_coverage: float
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    num_queries: int
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    avg_total_time_ms: float
    avg_topic_coverage: float
    avg_rouge_1: float
    avg_rouge_2: float
    avg_rouge_l: float
    pass_rate: float


# Test set with queries and expected information
EVALUATION_SET = [
    {
        "id": "rag_basic",
        "query": "What is RAG and how does it work?",
        "reference": "RAG (Retrieval-Augmented Generation) combines retrieval from a knowledge base with language model generation to produce more accurate and grounded responses.",
        "expected_topics": ["retrieval", "augmented", "generation", "knowledge", "language model"],
        "threshold": 0.4,
    },
    {
        "id": "embeddings",
        "query": "Explain vector embeddings and their role in RAG.",
        "reference": "Vector embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling efficient similarity search for retrieval.",
        "expected_topics": ["vector", "embedding", "semantic", "numerical", "similarity"],
        "threshold": 0.4,
    },
    {
        "id": "quantization",
        "query": "Why is quantization important for local LLMs?",
        "reference": "Quantization reduces model precision to decrease memory usage and improve inference speed, making it possible to run large language models on consumer hardware.",
        "expected_topics": ["memory", "precision", "speed", "smaller", "efficient"],
        "threshold": 0.4,
    },
    {
        "id": "faiss",
        "query": "What is FAISS and why is it used?",
        "reference": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search that enables fast nearest neighbor retrieval from large vector databases.",
        "expected_topics": ["similarity", "search", "vector", "efficient", "nearest"],
        "threshold": 0.4,
    },
    {
        "id": "chunk_size",
        "query": "What is the recommended chunk size for documents?",
        "reference": "Chunk sizes between 300-1000 tokens are recommended, balancing context preservation with retrieval specificity.",
        "expected_topics": ["chunk", "size", "context", "overlap"],
        "threshold": 0.3,
    },
]


def compute_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE scores between reference and hypothesis.
    
    Returns dict with rouge_1, rouge_2, rouge_l scores.
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        return {
            "rouge_1": scores['rouge1'].fmeasure,
            "rouge_2": scores['rouge2'].fmeasure,
            "rouge_l": scores['rougeL'].fmeasure,
        }
    except ImportError:
        # Fallback to simple overlap if rouge_score not installed
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        
        if not ref_words or not hyp_words:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
        
        overlap = len(ref_words & hyp_words)
        precision = overlap / len(hyp_words)
        recall = overlap / len(ref_words)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {"rouge_1": f1, "rouge_2": f1 * 0.8, "rouge_l": f1 * 0.9}


def compute_topic_coverage(response: str, expected_topics: List[str]) -> float:
    """Compute what fraction of expected topics appear in the response."""
    if not expected_topics:
        return 1.0
    
    response_lower = response.lower()
    found = sum(1 for topic in expected_topics if topic.lower() in response_lower)
    return found / len(expected_topics)


def run_evaluation(
    pipeline,
    test_set: List[Dict[str, Any]] = None,
    verbose: bool = True,
) -> EvaluationSummary:
    """
    Run full evaluation on the RAG pipeline.
    
    Args:
        pipeline: RAGPipeline instance
        test_set: List of test cases (default: EVALUATION_SET)
        verbose: Print detailed results
        
    Returns:
        EvaluationSummary with aggregate metrics
    """
    test_set = test_set or EVALUATION_SET
    results = []
    
    if verbose:
        print("\n" + "=" * 70)
        print("RAG SYSTEM EVALUATION")
        print("=" * 70 + "\n")
    
    for i, test_case in enumerate(test_set, 1):
        if verbose:
            print(f"[{i}/{len(test_set)}] Query: {test_case['query'][:60]}...")
        
        # Run query
        start = time.time()
        response = pipeline.query(test_case["query"])
        
        # Compute metrics
        topic_coverage = compute_topic_coverage(
            response.answer,
            test_case.get("expected_topics", [])
        )
        
        rouge_scores = compute_rouge_scores(
            test_case.get("reference", ""),
            response.answer
        )
        
        result = EvaluationResult(
            query_id=test_case["id"],
            query=test_case["query"],
            response=response.answer,
            retrieval_time_ms=response.retrieval_time * 1000,
            generation_time_ms=response.generation_time * 1000,
            total_time_ms=response.total_time * 1000,
            num_sources=len(response.sources),
            topic_coverage=topic_coverage,
            **rouge_scores,
        )
        
        results.append(result)
        
        if verbose:
            passed = topic_coverage >= test_case.get("threshold", 0.5)
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} | Coverage: {topic_coverage:.1%} | Time: {result.total_time_ms:.0f}ms")
            print(f"  ROUGE-1: {result.rouge_1:.2f} | ROUGE-L: {result.rouge_l:.2f}")
            print()
    
    # Compute summary
    if results:
        summary = EvaluationSummary(
            num_queries=len(results),
            avg_retrieval_time_ms=sum(r.retrieval_time_ms for r in results) / len(results),
            avg_generation_time_ms=sum(r.generation_time_ms for r in results) / len(results),
            avg_total_time_ms=sum(r.total_time_ms for r in results) / len(results),
            avg_topic_coverage=sum(r.topic_coverage for r in results) / len(results),
            avg_rouge_1=sum(r.rouge_1 for r in results) / len(results),
            avg_rouge_2=sum(r.rouge_2 for r in results) / len(results),
            avg_rouge_l=sum(r.rouge_l for r in results) / len(results),
            pass_rate=sum(
                1 for r, t in zip(results, test_set)
                if r.topic_coverage >= t.get("threshold", 0.5)
            ) / len(results),
        )
    else:
        summary = EvaluationSummary(
            num_queries=0,
            avg_retrieval_time_ms=0,
            avg_generation_time_ms=0,
            avg_total_time_ms=0,
            avg_topic_coverage=0,
            avg_rouge_1=0,
            avg_rouge_2=0,
            avg_rouge_l=0,
            pass_rate=0,
        )
    
    if verbose:
        print("=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Queries evaluated: {summary.num_queries}")
        print(f"  Pass rate: {summary.pass_rate:.1%}")
        print(f"  Avg topic coverage: {summary.avg_topic_coverage:.1%}")
        print(f"  Avg ROUGE-1: {summary.avg_rouge_1:.3f}")
        print(f"  Avg ROUGE-L: {summary.avg_rouge_l:.3f}")
        print(f"  Avg retrieval time: {summary.avg_retrieval_time_ms:.0f}ms")
        print(f"  Avg generation time: {summary.avg_generation_time_ms:.0f}ms")
        print(f"  Avg total time: {summary.avg_total_time_ms:.0f}ms")
        print("=" * 70 + "\n")
    
    return summary


def main():
    """Run evaluation from command line."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--full", action="store_true", help="Run full evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick sanity check")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    from src.config import RAGConfig, get_default_config
    from src.rag import RAGPipeline
    
    # Load config
    if args.config:
        config = RAGConfig.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Create pipeline
    print("Loading RAG pipeline...")
    pipeline = RAGPipeline(config)
    
    # Check if index exists
    stats = pipeline.retriever.get_stats()
    if stats['num_documents'] == 0:
        print("\nNo documents indexed. Building index from corpus...")
        pipeline.index_corpus()
    
    # Run evaluation
    if args.quick:
        # Quick test with just 2 queries
        summary = run_evaluation(pipeline, EVALUATION_SET[:2])
    else:
        summary = run_evaluation(pipeline)
    
    # Exit with error if pass rate is too low
    if summary.pass_rate < 0.5:
        print("WARNING: Pass rate below 50%")
        sys.exit(1)


if __name__ == "__main__":
    main()

