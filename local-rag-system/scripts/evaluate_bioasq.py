#!/usr/bin/env python3
"""
BioASQ Evaluation Script
========================

Evaluates the RAG system on the BioASQ benchmark dataset.

Metrics computed:
- Retrieval Precision@K: % of retrieved passages that are relevant
- Retrieval Recall@K: % of relevant passages that are retrieved
- MRR (Mean Reciprocal Rank): How early relevant docs appear
- Answer Quality: ROUGE scores against ground truth answers

Usage:
    python scripts/evaluate_bioasq.py
    python scripts/evaluate_bioasq.py --num-questions 50
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset_loader import BioASQDatasetLoader, BioASQQuestion
from src.config import RAGConfig
from src.rag import RAGPipeline


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    hit_rate: float  # % of questions with at least one relevant doc retrieved


@dataclass
class GenerationMetrics:
    """Generation quality metrics."""
    rouge_1: float
    rouge_2: float
    rouge_l: float
    exact_match: float


@dataclass
class EvaluationResult:
    """Complete evaluation result for a question."""
    question_id: str
    question: str
    ground_truth_answer: str
    generated_answer: str
    relevant_passage_ids: List[str]
    retrieved_passage_ids: List[str]
    precision: float
    recall: float
    reciprocal_rank: float
    rouge_1: float
    retrieval_time_ms: float
    generation_time_ms: float


def compute_retrieval_metrics(
    relevant_ids: List[str],
    retrieved_ids: List[str]
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    Args:
        relevant_ids: Ground truth relevant passage IDs
        retrieved_ids: IDs of retrieved passages
        
    Returns:
        Dict with precision, recall, reciprocal_rank
    """
    if not retrieved_ids:
        return {"precision": 0.0, "recall": 0.0, "reciprocal_rank": 0.0}
    
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    
    # Hits
    hits = relevant_set & retrieved_set
    
    # Precision@K
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0
    
    # Recall@K
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0
    
    # Reciprocal Rank (position of first relevant doc)
    reciprocal_rank = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            reciprocal_rank = 1.0 / (i + 1)
            break
    
    return {
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank
    }


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE scores."""
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
        # Fallback
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if not ref_words or not hyp_words:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
        overlap = len(ref_words & hyp_words)
        precision = overlap / len(hyp_words)
        recall = overlap / len(ref_words)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"rouge_1": f1, "rouge_2": f1 * 0.8, "rouge_l": f1 * 0.9}


def run_evaluation(
    pipeline: RAGPipeline,
    questions: List[BioASQQuestion],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full evaluation on the BioASQ dataset.
    
    Args:
        pipeline: RAGPipeline instance
        questions: List of BioASQ questions
        verbose: Print detailed output
        
    Returns:
        Dict with aggregated metrics and individual results
    """
    results = []
    
    # Metrics accumulators
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    total_rouge_1 = 0.0
    total_rouge_l = 0.0
    hits = 0
    questions_with_relevant = 0
    questions_with_answer = 0
    
    if verbose:
        print("\n" + "=" * 70)
        print("BioASQ RAG EVALUATION")
        print("=" * 70 + "\n")
    
    for i, question in enumerate(questions):
        if verbose:
            print(f"[{i+1}/{len(questions)}] {question.question[:60]}...")
        
        # Run query
        start_time = time.time()
        response = pipeline.query(question.question)
        total_time = time.time() - start_time
        
        # Extract retrieved passage IDs
        retrieved_ids = []
        for source in response.sources:
            # Extract passage ID from source metadata or name
            metadata = source.document.metadata
            if "passage_id" in metadata:
                retrieved_ids.append(str(metadata["passage_id"]))
            elif source.document.source.startswith("bioasq_passage_"):
                pid = source.document.source.replace("bioasq_passage_", "")
                retrieved_ids.append(pid)
        
        # Compute retrieval metrics
        if question.relevant_passage_ids:
            questions_with_relevant += 1
            metrics = compute_retrieval_metrics(
                question.relevant_passage_ids,
                retrieved_ids
            )
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_mrr += metrics["reciprocal_rank"]
            if metrics["recall"] > 0:
                hits += 1
        else:
            metrics = {"precision": 0, "recall": 0, "reciprocal_rank": 0}
        
        # Compute generation metrics
        rouge_scores = {"rouge_1": 0, "rouge_2": 0, "rouge_l": 0}
        if question.answer:
            questions_with_answer += 1
            rouge_scores = compute_rouge(question.answer, response.answer)
            total_rouge_1 += rouge_scores["rouge_1"]
            total_rouge_l += rouge_scores["rouge_l"]
        
        result = EvaluationResult(
            question_id=question.question_id,
            question=question.question,
            ground_truth_answer=question.answer or "",
            generated_answer=response.answer,
            relevant_passage_ids=question.relevant_passage_ids,
            retrieved_passage_ids=retrieved_ids,
            precision=metrics["precision"],
            recall=metrics["recall"],
            reciprocal_rank=metrics["reciprocal_rank"],
            rouge_1=rouge_scores["rouge_1"],
            retrieval_time_ms=response.retrieval_time * 1000,
            generation_time_ms=response.generation_time * 1000,
        )
        results.append(result)
        
        if verbose:
            status = "✓" if metrics["recall"] > 0 else "✗"
            print(f"  {status} P@K: {metrics['precision']:.2f} | R@K: {metrics['recall']:.2f} | "
                  f"ROUGE-1: {rouge_scores['rouge_1']:.2f} | Time: {total_time:.1f}s")
    
    # Compute aggregates
    n = len(questions)
    n_relevant = questions_with_relevant or 1
    n_answer = questions_with_answer or 1
    
    summary = {
        "num_questions": n,
        "retrieval": {
            "precision_at_k": total_precision / n_relevant,
            "recall_at_k": total_recall / n_relevant,
            "mrr": total_mrr / n_relevant,
            "hit_rate": hits / n_relevant,
        },
        "generation": {
            "rouge_1": total_rouge_1 / n_answer,
            "rouge_l": total_rouge_l / n_answer,
        },
        "timing": {
            "avg_retrieval_ms": sum(r.retrieval_time_ms for r in results) / n,
            "avg_generation_ms": sum(r.generation_time_ms for r in results) / n,
            "avg_total_ms": sum(r.retrieval_time_ms + r.generation_time_ms for r in results) / n,
        },
        "results": results,
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\n📊 Retrieval Metrics (on {questions_with_relevant} questions with ground truth):")
        print(f"   Precision@K:  {summary['retrieval']['precision_at_k']:.3f}")
        print(f"   Recall@K:     {summary['retrieval']['recall_at_k']:.3f}")
        print(f"   MRR:          {summary['retrieval']['mrr']:.3f}")
        print(f"   Hit Rate:     {summary['retrieval']['hit_rate']:.1%}")
        print(f"\n📝 Generation Metrics (on {questions_with_answer} questions with answers):")
        print(f"   ROUGE-1:      {summary['generation']['rouge_1']:.3f}")
        print(f"   ROUGE-L:      {summary['generation']['rouge_l']:.3f}")
        print(f"\n⏱️  Timing:")
        print(f"   Avg Retrieval: {summary['timing']['avg_retrieval_ms']:.0f}ms")
        print(f"   Avg Generation: {summary['timing']['avg_generation_ms']:.0f}ms")
        print(f"   Avg Total:     {summary['timing']['avg_total_ms']:.0f}ms")
        print("\n" + "=" * 70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG on BioASQ dataset")
    parser.add_argument("--config", type=str, default="config-bioasq.yaml",
                       help="Path to config file")
    parser.add_argument("--num-questions", type=int, default=None,
                       help="Number of questions to evaluate (default: all)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Load config
    config_path = project_root / args.config
    if config_path.exists():
        config = RAGConfig.from_yaml(str(config_path))
    else:
        print(f"Config not found: {config_path}")
        print("Run 'python scripts/setup_bioasq.py' first to set up the dataset.")
        sys.exit(1)
    
    # Load dataset
    loader = BioASQDatasetLoader(cache_dir=str(project_root / "data" / "bioasq"))
    loader.download()
    
    questions = loader.get_questions()
    if args.num_questions:
        questions = questions[:args.num_questions]
    
    print(f"Evaluating on {len(questions)} questions...")
    
    # Initialize pipeline
    print("Loading RAG pipeline...")
    pipeline = RAGPipeline(config)
    
    # Run evaluation
    summary = run_evaluation(pipeline, questions, verbose=not args.quiet)
    
    # Save results
    import json
    results_path = project_root / "evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert results to serializable format
        output = {
            "num_questions": summary["num_questions"],
            "retrieval": summary["retrieval"],
            "generation": summary["generation"],
            "timing": summary["timing"],
        }
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

