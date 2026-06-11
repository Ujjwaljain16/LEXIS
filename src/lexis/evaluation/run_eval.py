import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from lexis.retrieval.hybrid_retriever import RetrievalEngine
from lexis.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def calculate_metrics(expected_ids: set, retrieved_ids: list):
    """Calculates Recall@K and MRR."""
    # Recall@5
    retrieved_5 = set(retrieved_ids[:5])
    recall_5 = len(expected_ids.intersection(retrieved_5)) / len(expected_ids) if expected_ids else 0.0
    
    # Recall@30
    retrieved_30 = set(retrieved_ids[:30])
    recall_30 = len(expected_ids.intersection(retrieved_30)) / len(expected_ids) if expected_ids else 0.0

    # MRR
    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in expected_ids:
            mrr = 1.0 / rank
            break

    return recall_5, recall_30, mrr

async def run_evaluation(dataset_path: str, output_path: str):
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset.get("questions", [])
    if not questions:
        logger.error("No questions found in dataset.")
        return

    retriever = RetrievalEngine()
    
    total_recall_5 = 0.0
    total_recall_30 = 0.0
    total_mrr = 0.0
    
    results = []

    for idx, q_item in enumerate(questions):
        query = q_item["query"]
        expected = set(q_item["expected_chunk_ids"])
        
        logger.info(f"[{idx+1}/{len(questions)}] Query: {query}")
        
        # Retrieve Top 30 for evaluation
        candidates = await retriever.retrieve(query, top_k_per_path=30, top_n_rrf=30)
        retrieved_ids = [c["id"] for c in candidates]
        
        r5, r30, mrr = await calculate_metrics(expected, retrieved_ids)
        
        total_recall_5 += r5
        total_recall_30 += r30
        total_mrr += mrr
        
        results.append({
            "query": query,
            "expected": list(expected),
            "retrieved_top_5": retrieved_ids[:5],
            "recall_5": r5,
            "recall_30": r30,
            "mrr": mrr
        })

    avg_recall_5 = total_recall_5 / len(questions)
    avg_recall_30 = total_recall_30 / len(questions)
    avg_mrr = total_mrr / len(questions)
    
    logger.info("========== FOUNDATION EVALUATION ==========")
    logger.info(f"Queries: {len(questions)}")
    logger.info(f"Recall@5:  {avg_recall_5:.4f}")
    logger.info(f"Recall@30: {avg_recall_30:.4f}")
    logger.info(f"MRR:       {avg_mrr:.4f}")
    
    status = "PASS" if avg_recall_30 > 0.70 and avg_mrr > 0.60 else "FAIL"
    logger.info(f"STATUS: {status}")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "git_commit": "foundation-remediation",  # Can be injected via git rev-parse
        "embedding_model": settings.embedding_model,
        "retriever_config": {
            "rrf_k": settings.rrf_k,
            "paths": ["path_b_global", "path_d_bm25"]
        },
        "metrics": {
            "recall_5": avg_recall_5,
            "recall_30": avg_recall_30,
            "mrr": avg_mrr
        },
        "status": status,
        "details": results
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation(
        "evaluation/datasets/cuad_foundation_v1.json",
        "evaluation/reports/foundation_v1_results.json"
    ))
